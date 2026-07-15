"""A fixed policy must reach the model, or it is decoration.

`--fixed-policies catalytic_triad` was DECORATIVE for the life of the flag. The planner wrote
the policy NAME into the manifest row and never into the spec:

    for fixed_policy in fixed_policies:            # campaign.py:612
        spec_variant = replace(base_spec, ...)     # <- fixed_policy NEVER touches this
        row = build_manifest_row(..., fixed_policy=fixed_policy)   # -> row dict only
        row["sampling_spec"] = campaign_sampling_spec_payload(spec_variant, ...)

The worker reconstructs from `row["sampling_spec"]` alone, so the policy could not reach it.
Consequence: **no residue was ever held fixed in any produced design** -- 882/882 beads of a
real library, silently violating a locked preregistration.

The gate that should have caught it could not:

    policies_seen = {row["fixed_policy"] for row in rows}      # campaign.py:210
    missing = set(required_fixed_policies) - policies_seen

`campaign.py:671` passes the same tuple the loop consumed, so this compares the planner's
output against the planner's own input. It cannot fail. It checked that a *string label was
present* -- nothing about whether a residue was fixed.

**Why the policy now carries its own referent.** A bare name is unresolvable by construction:
aminx cannot know that `catalytic_triad` means canonical [38, 73, 143] for TEV, or that
`active_site` is an 84-residue shell -- those live in the consumer's own bundle. That is why
the shipped default (`catalytic_triad,active_site`) was unresolvable out of the box. A
resolver callback is not an option either: callables cannot cross the manifest's JSON
boundary, and `spec_partition._assert_no_callable_knobs` refuses them outright.

**The mask is 1-D, in the canonical reference frame.** Not per-state. `bundles.py:138` types
it `Float[Array, L]` while `state_position_map` beside it is `"S L"`; `multistate_poe.py`
RAISES on a genuinely per-state mask; decode broadcasts it over the group axis because one
sequence is sampled and a position is either designed or it isn't. Cross-state index shifts
(canonical 38 is His in 1LVB but Ser in 1LVM) are what `state_position_map` is for, and the
consumer already computes and injects it.

task_id: 260715_aminx-campaign-control-knob-audit
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from aminx.host.campaign import plan_campaign_manifest
from aminx.run.specs import SamplingSpecification

pytestmark = pytest.mark.knob_differential

_CKPT = "ligandmpnn_v_32_020_25"

# `SamplingSpecification.max_length`'s default (run/specs.py:270). The mask must match the
# PARSED structure's padded length -- `_prepare_fixed_controls` takes seq_len from
# `batched_ensemble.coordinates.shape[1]`, and `prep.py:117` pads the dataset to
# `spec.max_length`. `campaign plan` exposes neither --max-length nor --truncation-strategy,
# so a campaign always runs this default and 512 is always right *there*.
#
# An earlier version of this comment called it CAMPAIGN_MAX_LENGTH. **No such constant exists
# anywhere in aminx** -- the name was carried out of a stale plan and written here as though
# it were real. It is an ordinary spec field with an ordinary default. Left as a note because
# an invented citation in a test is exactly the kind of confident-looking wrong thing this
# audit keeps finding.
_L = 512
_TRIAD = (38, 73, 143)  # canonical indices: His46/Asp81/Cys151 == tev_num - 8


def _mask(*positions: int) -> np.ndarray:
  m = np.zeros(_L, dtype=np.float32)
  for p in positions:
    m[p] = 1.0
  return m


def _plan(**kwargs: Any) -> list[dict[str, Any]]:
  """Drive the REAL planner. Not a reconstruction of it."""
  return plan_campaign_manifest(
    base_spec=SamplingSpecification(inputs=["ref.pdb"], checkpoint_id=_CKPT),
    campaign_id="fixed-arms",
    output_root="/tmp/fixed-arms",
    designs_per_library_type=1,
    samples_chunk_size=1,
    **kwargs,
  )


def test_arm_mask_reaches_the_sampling_spec() -> None:
  """THE bug. The arm's mask must land in `sampling_spec`, which is all the worker reads.

  This is the whole epic in one assertion. Before the fix the planner produced a row whose
  `sampling_spec` had `fixed_mask: None` while the row cheerfully advertised
  `fixed_policy: "catalytic_triad"` -- and every downstream check looked at the label.
  """
  rows = _plan(fixed_arms={"catalytic_triad": _mask(*_TRIAD)})

  spec = rows[0]["sampling_spec"]
  assert spec.get("fixed_mask") is not None, (
    "The arm's mask did not reach sampling_spec. The worker reconstructs its spec from "
    "sampling_spec alone, so a mask that is not here does not exist as far as the model is "
    "concerned -- which is exactly how 882/882 beads shipped with nothing held fixed."
  )
  got = np.asarray(spec["fixed_mask"])
  assert sorted(np.nonzero(got)[0].tolist()) == list(_TRIAD)


def test_the_arm_s_mask_SURVIVES_TO_THE_WORKER_not_just_into_json() -> None:
  """Plan -> manifest JSON -> reconstructed spec -> `_prepare_fixed_controls`. End to end.

  **This is the test cycle 1 should have had and did not.** Every other assertion here stops
  at the manifest JSON, which proves the planner wrote a key -- not that the mask is intact
  and correctly shaped by the time anything could act on it. "It's in the JSON" was the same
  confidence level as "the row says catalytic_triad", which is what the whole epic is about.

  Shape is the live risk: `_prepare_fixed_controls` derives seq_len from the PARSED
  structure's padded length and hard-errors on a mismatch, so a mask sized against the wrong
  assumption fails here and nowhere earlier.
  """
  import jax.numpy as jnp

  from aminx.host._sampling_helper import _prepare_fixed_controls
  from aminx.run.spec_json import _coerce_field_value
  from aminx.utils.data_structures import Protein

  rows = _plan(fixed_arms={"catalytic_triad": _mask(*_TRIAD)})

  # The worker's REAL reconstruction path (campaign.py:838-856): coerce the JSON payload
  # back through the plain constructor. Not a stand-in for it.
  payload = {
    k: _coerce_field_value(SamplingSpecification, k, v)
    for k, v in dict(rows[0]["sampling_spec"]).items()
  }
  spec = SamplingSpecification(**payload)

  ensemble = Protein(
    coordinates=jnp.zeros((1, _L, 4, 3), dtype=jnp.float32),
    aatype=jnp.zeros((1, _L), dtype=jnp.int32),
    atom_mask=jnp.ones((1, _L, 37), dtype=jnp.float32),
    residue_index=jnp.arange(_L, dtype=jnp.int32)[None, :],
    chain_index=jnp.zeros((1, _L), dtype=jnp.int32),
    mask=jnp.ones((1, _L), dtype=jnp.float32),
    mapping=None,
  )
  # fixed_tokens must accompany the mask or the Alanine guard refuses the pair -- freezing
  # without saying to what locks every frozen position to token 0 (Ala).
  spec = replace(spec, fixed_tokens=np.zeros(_L, dtype=np.int32))

  fixed_mask, _tokens = _prepare_fixed_controls(spec, batched_ensemble=ensemble)

  frozen = sorted(np.nonzero(np.asarray(fixed_mask)[0])[0].tolist())
  assert frozen == list(_TRIAD), (
    f"The arm's mask did not survive to the worker: frozen positions are {frozen}, expected "
    f"{list(_TRIAD)}. It reached the manifest but not the thing that acts on it -- which is "
    f"the entire failure mode this audit exists to close."
  )


def test_arm_label_still_travels_for_provenance() -> None:
  """The label is still written -- it just is no longer the ONLY thing written."""
  rows = _plan(fixed_arms={"catalytic_triad": _mask(*_TRIAD)})
  assert rows[0]["fixed_policy"] == "catalytic_triad"


def test_distinct_arms_produce_DISTINCT_specs() -> None:
  """Two arms must differ in the spec, not merely in the label.

  Today they do not: every policy produces a byte-identical `sampling_spec`, so the grid axis
  manufactures exact duplicates -- the same defect the consumer already found on the
  ligand x sidechain axis (3 of every 4 groups identical after patching). A grid whose arms
  are indistinguishable is not a grid; it is the same run counted twice.
  """
  rows = _plan(
    fixed_arms={"catalytic_triad": _mask(*_TRIAD), "wider": _mask(*_TRIAD, 100, 101)},
  )

  by_label = {r["fixed_policy"]: np.asarray(r["sampling_spec"]["fixed_mask"]) for r in rows}
  assert set(by_label) == {"catalytic_triad", "wider"}
  assert not np.array_equal(by_label["catalytic_triad"], by_label["wider"]), (
    "Two arms produced identical fixed_masks. The arms are not distinct, so the grid is "
    "generating duplicate work and calling it diversity."
  )


def test_no_arms_means_design_everything_NOT_zero_rows() -> None:
  """Absent arms must yield a full row-set with nothing fixed. Never zero rows.

  `fixed_policies=()` used to silently produce ZERO rows: the policy loop was the outermost
  one, so an empty tuple meant the body never executed and `rows` stayed `[]`. Then
  `validate_manifest_rows(rows=[], required_fixed_policies=())` skipped every check (both
  loops iterate nothing; the final check is gated behind `if required_fixed_policies:`) and
  returned cleanly. An empty manifest is not a plan -- it is a no-op wearing a plan's clothes.

  "Fix nothing" is the only default aminx can honestly ship: it cannot know any protein's
  catalytic triad, so fixing must be an explicit opt-in.
  """
  rows = _plan()
  assert rows, "No arms produced ZERO rows. An empty manifest must never be a silent success."
  assert rows[0]["sampling_spec"].get("fixed_mask") is None
  assert rows[0]["fixed_policy"] == "none"


def test_explicitly_empty_arms_raises() -> None:
  """`fixed_arms={}` states an intent that cannot be honoured, so it must not be guessed at.

  Distinct from `fixed_arms=None` (above), which means "I did not ask for any fixing".
  Passing an empty mapping means "here are my arms" followed by no arms -- a contradiction.
  Silently treating it as "design everything" is how the original zero-rows bug read.
  """
  with pytest.raises(ValueError, match="fixed_arms"):
    _plan(fixed_arms={})


def test_the_old_decorative_flag_is_a_hard_error() -> None:
  """`fixed_policies=` must fail loudly, not be quietly ignored or silently accepted.

  Accepting a bare policy name and doing nothing with it IS the original sin. A deprecation
  warning would preserve the exact behaviour that voided a library, so this raises instead --
  and the message has to name the replacement, or we have only moved the confusion.
  """
  with pytest.raises(TypeError, match="fixed_polic"):
    _plan(fixed_policies=("catalytic_triad",))


# ---------------------------------------------------------------------------
# The evidence channel: can a reader PROVE what was frozen, after the fact?
# ---------------------------------------------------------------------------


def test_provenance_outputs_carry_the_mask_and_name_their_alphabet() -> None:
  """`fixed_mask` must reach the OUTPUT, not just the model.

  Before this, the mask reached the model and vanished: no `fixed_mask` anywhere in `io/` or
  any sink. So a finished campaign could not be interrogated -- 882/882 rows, every done
  marker valid, every content digest intact, and not one residue held. The completion record
  was perfect precisely because completion was the only thing recorded.
  """
  from aminx.host._sampling_helper import fixed_provenance_outputs

  mask = _mask(*_TRIAD)
  spec = SamplingSpecification(
    inputs=["ref.pdb"],
    checkpoint_id=_CKPT,
    fixed_mask=mask,
    fixed_tokens=np.zeros(_L, dtype=np.int32),
  )
  arrays, attrs = fixed_provenance_outputs(spec, seq_len=_L)

  assert sorted(np.nonzero(arrays["fixed_mask"])[0].tolist()) == list(_TRIAD)
  assert attrs["n_fixed"] == len(_TRIAD)
  # The alphabet travels as DATA. The consumer-side bug this audit found was a rename
  # stripping an `_af` suffix and taking the convention with it; a rename cannot strip a value.
  assert attrs["fixed_tokens_alphabet"] == "MPNN"


def test_an_unfixed_row_says_so_rather_than_staying_silent() -> None:
  """Nothing-fixed must be a WRITTEN zero, not an absent key.

  "No mask" and "a mask that froze nothing" are the same silence to a reader, and this whole
  epic is what silence costs. n_fixed == 0 is a fact someone can see in the metadata and act
  on; a missing key is something they have to notice.
  """
  from aminx.host._sampling_helper import fixed_provenance_outputs

  spec = SamplingSpecification(inputs=["ref.pdb"], checkpoint_id=_CKPT)
  arrays, attrs = fixed_provenance_outputs(spec, seq_len=_L)

  assert "fixed_mask" in arrays, "an unfixed row must still SAY it fixed nothing"
  assert arrays["fixed_mask"].shape == (_L,)
  assert attrs["n_fixed"] == 0
