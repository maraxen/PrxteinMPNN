"""Decode-path completeness + differential suite (260716, EPIC #1541 P4, following aminx#110).

Two concerns, both about the SAME root cause: `make_inference_plan` (the single-structure/
scoring path) and `sample_multistate_poe_bead` (the genuine multi-state PoE path) each used to
independently decide which decode class to build, with nothing forcing agreement -- that's how
aminx#110 happened (fixed_mask/temperature/random_seed/num_samples were all silently inert on
the ConditionalDecode path because no branch ever built AutoregressiveDecode there).

1. TestRunSpecCompleteness -- static/AST guard: catches a future silent reversion to flat
   getattr(spec, ...) reads for fields that now have a RunSpec.sampling home. This is exactly
   the class of regression that let aminx#110 happen undetected in the first place, and the
   same class of regression that deleted 3 RunSpec sub-configs (`dd0e952`) as incidental
   cleanup during an unrelated refactor with nothing to catch it.

2. TestDecodePathDifferential -- live, real-checkpoint two-armed differentials (same pattern
   proven throughout the 260716 audit) re-verifying the actual fix, marked `slow` since they
   load a real proteinmpnn_v_48_020 checkpoint and are not cheap enough for default CI.

Local compute note: every real-model test here uses max_length=128 (>= 1ubq's ~76-residue
native length) to avoid aminx's own truncation_strategy="random_crop" default silently firing
on a too-small max_length -- exactly the confound that produced a false "sample() has unseeded
non-determinism" signal during this same audit before being traced to its real cause.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields as dc_fields

import jax
import numpy as np
import pytest

from aminx.host.plan import make_inference_plan
from aminx.run.spec import SamplingConfig

pytestmark = pytest.mark.slow

_MAX_LENGTH = 128
_STRUCTURE = "tests/data/1ubq.pdb"
_CHECKPOINT = "proteinmpnn_v_48_020"


class TestRunSpecCompleteness:
  """Guard against silently reverting to flat spec reads for RunSpec-migrated fields."""

  def test_make_inference_plan_does_not_flat_read_migrated_fields(self) -> None:
    """make_inference_plan must read migrated fields via spec.run_spec.sampling, not flat.

    AST-scans the function source for `getattr(spec, "<field>", ...)` calls naming any field
    that RunSpec.SamplingConfig now carries (added 260716) -- a flat read of one of these is
    not a bug in the traditional sense (it would likely still work, since flat and run_spec
    values agree today), but it is exactly the kind of silent divergence-in-waiting that
    produced aminx#110: two code paths reading the "same" value through different routes with
    nothing keeping them honest if one drifts.
    """
    migrated_fields = {f.name for f in dc_fields(SamplingConfig)}

    source = inspect.getsource(make_inference_plan)
    tree = ast.parse(source)

    flat_reads = []
    for node in ast.walk(tree):
      if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "spec"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value in migrated_fields
      ):
        flat_reads.append(node.args[1].value)

    assert not flat_reads, (
      f"make_inference_plan reads {flat_reads} via flat getattr(spec, ...) despite these "
      f"fields having a RunSpec.sampling home -- read spec.run_spec.sampling.<field> instead. "
      f"This is exactly the silent-regression pattern that let aminx#110 go undetected."
    )

  def test_sampling_strategy_is_a_real_runspec_field(self) -> None:
    """sampling_strategy -- the field that decides which decode class gets built at all --
    must never silently disappear from RunSpec again. Before 260716 it was never migrated in
    the first place (a genuine blind spot in the original RS-1 inventory, not a deliberate
    decision); after 260716's fix, its disappearance would silently break resolve_decode_mode
    for every caller. A plain field-presence assertion, not a differential -- cheap and
    always-on.
    """
    field_names = {f.name for f in dc_fields(SamplingConfig)}
    assert "sampling_strategy" in field_names, (
      "RunSpec.SamplingConfig lost its sampling_strategy field -- resolve_decode_mode "
      "(host/plan.py) depends on spec.run_spec.sampling.sampling_strategy existing; without "
      "it, decode-mode resolution has no signal to route STEMode at all."
    )


def _sample_sequences(*, num_samples: int, temperature: float, random_seed: int, **kwargs):
  from aminx.host.runner import sample

  r = sample(
    inputs=[_STRUCTURE],
    checkpoint_id=_CHECKPOINT,
    num_samples=num_samples,
    temperature=temperature,
    random_seed=random_seed,
    max_length=_MAX_LENGTH,
    **kwargs,
  )
  return np.asarray(r["sequences"]).reshape(num_samples, -1)[:, :_MAX_LENGTH]


class TestDecodePathDifferential:
  """Live re-verification of the aminx#110 fix on the ConditionalDecode (sample()) path.

  Mutation-tested: reverting host/plan.py's resolve_decode_mode call to the pre-fix
  `decode_mode = ConditionalMode()` (unconditional) makes every test in this class fail --
  confirmed manually before landing this file, per this project's standing rule that a test
  which cannot fail is worse than no test.
  """

  def test_fixed_mask_holds_on_sample(self) -> None:
    from aminx.utils.aa_convert import MPNN_ALPHABET

    w_idx = MPNN_ALPHABET.index("W")
    mask = np.zeros(_MAX_LENGTH, dtype=np.float32)
    mask[:20] = 1.0
    tok = np.zeros(_MAX_LENGTH, dtype=np.int32)
    tok[:20] = w_idx

    masked = _sample_sequences(
      num_samples=2, temperature=0.5, random_seed=0, fixed_mask=mask, fixed_tokens=tok,
    )
    unmasked = _sample_sequences(num_samples=2, temperature=0.5, random_seed=0)

    frac_w_masked = float((masked[:, :20] == w_idx).mean())
    frac_w_unmasked = float((unmasked[:, :20] == w_idx).mean())
    assert frac_w_masked > 0.9, (
      f"fixed_mask did not hold on sample(): only {frac_w_masked:.2f} of the frozen region "
      f"was the declared token. aminx#110 regression."
    )
    assert frac_w_unmasked < 0.3, (
      "unmasked control unexpectedly high at the same positions -- background W rate should "
      "be low; if this fires, the differential's own control arm is broken, not just the fix."
    )
    assert not np.array_equal(masked, unmasked), "masked and unmasked arms must not be identical"

  def test_num_samples_produces_real_diversity(self) -> None:
    seqs = _sample_sequences(num_samples=4, temperature=0.5, random_seed=0)
    assert not bool(np.all(seqs == seqs[0])), (
      "all 4 samples from one sample() call are byte-identical -- ConditionalDecode's "
      "no-stochastic-sampling regression (aminx#110 escalation) is back."
    )

  def test_random_seed_changes_output(self) -> None:
    outputs = [
      _sample_sequences(num_samples=1, temperature=0.5, random_seed=seed)[0]
      for seed in (0, 1, 2)
    ]
    assert not all(np.array_equal(outputs[0], o) for o in outputs[1:]), (
      "random_seed has no effect on sample() output -- aminx#110 escalation regression."
    )

  def test_bias_affects_sample_output(self) -> None:
    from aminx.utils.aa_convert import MPNN_ALPHABET

    w_idx = MPNN_ALPHABET.index("W")
    bias_off = np.zeros((_MAX_LENGTH, 21), dtype=np.float32)
    bias_on = np.zeros((_MAX_LENGTH, 21), dtype=np.float32)
    bias_on[:, w_idx] = 50.0

    off = _sample_sequences(num_samples=1, temperature=0.5, random_seed=0, bias=bias_off)
    on = _sample_sequences(num_samples=1, temperature=0.5, random_seed=0, bias=bias_on)
    assert float((on == w_idx).mean()) > float((off == w_idx).mean()), (
      "bias had no effect on sample() output."
    )

  def test_tie_group_map_ties_positions_on_sample(self) -> None:
    tie_map = np.arange(_MAX_LENGTH, dtype=np.int32)
    tie_map[5] = tie_map[0]

    untied = _sample_sequences(num_samples=1, temperature=0.5, random_seed=0)[0]
    tied = _sample_sequences(
      num_samples=1, temperature=0.5, random_seed=0, tie_group_map=tie_map,
    )[0]
    assert untied[0] != untied[5], "fixture assumption violated: positions 0/5 already equal untied"
    assert tied[0] == tied[5], "tie_group_map did not force positions 0 and 5 to match"


class TestCrossPathAgreement:
  """The property that was structurally impossible to test before resolve_decode_mode existed:
  do the two real decode paths agree on which DecodeMode a given sampling_strategy means?
  """

  def test_default_sampling_strategy_resolves_to_the_same_mode_class_on_both_paths(self) -> None:
    from aminx.host.plan import resolve_decode_mode
    from aminx.inference.decode.mode import AutoregressiveMode
    from aminx.run.specs import SamplingSpecification

    single_spec = SamplingSpecification(
      inputs=[_STRUCTURE], checkpoint_id=_CHECKPOINT, max_length=_MAX_LENGTH,
    )
    multi_spec = SamplingSpecification(
      inputs=[_STRUCTURE, _STRUCTURE],
      checkpoint_id=_CHECKPOINT,
      batch_size=2,
      max_length=_MAX_LENGTH,
    )

    single_mode = resolve_decode_mode(single_spec.run_spec, purpose="sample")
    multi_mode = resolve_decode_mode(multi_spec.run_spec, purpose="sample")

    assert type(single_mode) is type(multi_mode), (
      f"the same sampling_strategy resolved to different DecodeMode classes depending on "
      f"whether the spec was single- or multi-state ({type(single_mode).__name__} vs "
      f"{type(multi_mode).__name__}) -- exactly the divergence resolve_decode_mode exists "
      f"to prevent."
    )
    assert isinstance(single_mode, AutoregressiveMode), (
      f"purpose='sample' at the default sampling_strategy should resolve to AutoregressiveMode, "
      f"got {type(single_mode).__name__}"
    )

  def test_straight_through_is_honestly_rejected_by_the_poe_path(self) -> None:
    """sample_multistate_poe_bead has no STE-for-PoE implementation (out of scope, see the
    260716 migration plan) -- it must raise, not silently sample via AutoregressiveMode anyway.
    """
    from aminx.run.specs import SamplingSpecification
    from aminx.sampling.multistate_poe import sample_multistate_poe_bead

    spec = SamplingSpecification(
      inputs=[_STRUCTURE, _STRUCTURE],
      checkpoint_id=_CHECKPOINT,
      batch_size=2,
      max_length=_MAX_LENGTH,
      sampling_strategy="straight_through",
      iterations=1,
      learning_rate=0.1,
    )
    with pytest.raises(NotImplementedError, match="straight_through"):
      sample_multistate_poe_bead(spec, jax.random.PRNGKey(0), 1)
