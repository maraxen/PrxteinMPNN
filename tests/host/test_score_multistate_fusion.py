"""Regression tests: runner.score() genuine multi-state PoE fusion (260719).

Root cause (found live, dogfooded from tev_design's necklace campaign trying to build a
Stage 1 fusion-strategy comparison): `runner.score()`'s independent-per-structure loop
always called `score_fn` with a single (L, 4, 3) structure (S implicitly 1), so
`multi_state_strategy` had nothing to fuse and `state_position_map` was never even read
-- confirmed via a live differential (arithmetic_mean vs product produced byte-identical
logits) before this fix, not assumed from a source read alone. `score_sequence` itself
(aminx/scoring/score.py) was already correctly wired for genuine S>1 fusion via
`build_inference_bundle` + `score_conditional.kernel` -- only the caller in
`host/runner.py::score()` never invoked it that way.

Fix: `spec.state_position_map is not None` is an unambiguous signal the caller wants ONE
fused score across every `spec.inputs` structure (its only sensible purpose is cross-state
alignment for fusion) -- `_score_fused_multistate` handles that branch; the default
(`state_position_map=None`) independent-per-structure loop is untouched, preserving exact
prior behavior for every existing caller.
"""

from __future__ import annotations

import numpy as np
import pytest

from aminx.host.runner import score
from aminx.io.parsing import parse_structure
from aminx.run.specs import ScoringSpecification

_STRUCTURE = "tests/data/1ubq.pdb"
_STRUCTURE_2 = "tests/data/1mbn.pdb"
_CHECKPOINT = "proteinmpnn_v_48_020"


def _native_sequence(path: str) -> str:
  from proxide.chem.residues import restype_order_with_x

  structure = parse_structure(path)
  idx_to_aa = {i: aa for aa, i in restype_order_with_x.items()}
  return "".join(idx_to_aa.get(int(i), "X") for i in np.asarray(structure.aatype))


@pytest.mark.slow
class TestDefaultPathUnaffected:
  """state_position_map=None (the default) must be byte-identical to pre-fix behavior."""

  def test_multiple_inputs_without_state_position_map_scored_independently(self) -> None:
    seq = _native_sequence(_STRUCTURE)
    spec = ScoringSpecification(
      inputs=[_STRUCTURE, _STRUCTURE_2],
      checkpoint_id=_CHECKPOINT,
      model_family="proteinmpnn",
      sequences_to_score=[seq],
      return_logits=True,
      max_length=200,
    )
    result = score(spec)
    # Two INDEPENDENT scores, one per input -- not fused. This is the pre-existing,
    # correct behavior for "score this sequence against N unrelated candidate
    # structures" and must not change just because this file exists.
    assert np.asarray(result["scores"]).shape == (2, 1)
    assert result["metadata"]["structure_ids"] == ["1ubq", "1mbn"]


@pytest.mark.slow
class TestFusedMultistateScoring:
  """state_position_map set -> genuine cross-state PoE fusion, one score per bead."""

  def test_fused_result_has_single_leading_dimension(self) -> None:
    seq = _native_sequence(_STRUCTURE)
    L = len(seq)
    identity_map = np.tile(np.arange(L), (2, 1))  # (S=2, L), same structure twice
    spec = ScoringSpecification(
      inputs=[_STRUCTURE, _STRUCTURE],
      state_position_map=identity_map,
      checkpoint_id=_CHECKPOINT,
      model_family="proteinmpnn",
      sequences_to_score=[seq],
      multi_state_strategy="arithmetic_mean",
      return_logits=True,
      max_length=L,
    )
    result = score(spec)
    # ONE fused "structure", not len(inputs)=2 independent ones -- the defining
    # behavioral difference from the default path above.
    assert np.asarray(result["scores"]).shape == (1, 1)
    assert np.asarray(result["logits"]).shape == (1, 1, L, 21)
    assert result["metadata"]["structure_ids"] == ["fused_multistate"]
    assert len(result["metadata"]["fused_structure_ids"]) == 2

  def test_multi_state_strategy_actually_changes_the_result(self) -> None:
    """The exact regression this fix closes: arithmetic_mean vs product must differ."""
    seq = _native_sequence(_STRUCTURE)
    L = len(seq)
    identity_map = np.tile(np.arange(L), (2, 1))

    def _run(strategy: str):
      spec = ScoringSpecification(
        inputs=[_STRUCTURE, _STRUCTURE],
        state_position_map=identity_map,
        checkpoint_id=_CHECKPOINT,
        model_family="proteinmpnn",
        sequences_to_score=[seq],
        multi_state_strategy=strategy,
        return_logits=True,
        max_length=L,
      )
      return score(spec)

    arithmetic = _run("arithmetic_mean")
    product = _run("product")
    am_logits = np.asarray(arithmetic["logits"])
    pr_logits = np.asarray(product["logits"])

    # Before this fix: byte-identical (confirmed live, not assumed) regardless of
    # strategy, because S was always 1 at the score_fn call site. PoE (product, raw
    # logit sum pre-softmax) of two IDENTICAL per-state distributions is strictly
    # sharper than their arithmetic mean (which reproduces the single-state
    # distribution unchanged) -- a real, predictable, non-trivial difference, not an
    # arbitrary numerical fluke.
    assert not np.allclose(am_logits, pr_logits), (
      "multi_state_strategy had no effect -- the fusion no-op regression is back"
    )

  def test_state_position_map_state_count_mismatch_raises(self) -> None:
    seq = _native_sequence(_STRUCTURE)
    L = len(seq)
    wrong_shape_map = np.tile(np.arange(L), (3, 1))  # claims 3 states, only 2 inputs given
    spec = ScoringSpecification(
      inputs=[_STRUCTURE, _STRUCTURE],
      state_position_map=wrong_shape_map,
      checkpoint_id=_CHECKPOINT,
      model_family="proteinmpnn",
      sequences_to_score=[seq],
      return_logits=True,
      max_length=L,
    )
    with pytest.raises(ValueError, match="state_position_map has 3 states but 2 structures"):
      score(spec)
