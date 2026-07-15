"""Differential harness: does a control knob a caller sets actually reach the model?

Ten times an aminx control knob has been declared on the config surface, appeared to work, and
silently never reached the model. Every one was found incidentally; none by a systematic sweep.
Existing conditioning tests miss them because they build `Protein`/bundle objects directly and
bypass the CLI -> manifest -> spec -> runner layer where all the bugs actually live.

This converts "is this knob wired?" from a code-reading judgment (which has failed every time)
into an executable assertion, driven through the REAL path:

    plan_campaign_manifest -> row["sampling_spec"] JSON -> SamplingSpecification(**payload)
                           -> [Tier B] real runner -> spy at load_model / build_inference_bundle

Tier A needs no model and no structure and runs in milliseconds; it covers the manifest-drop
class, which is 6 of the 10 known occurrences. Tier B spies at the model boundary for the hops
Tier A cannot see.

Expected behavior per field lives in `knob_observations.py` -- see that module for why declared
intent is required rather than a bare value-diff.

task_id: 260715_aminx-campaign-control-knob-audit
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import pytest

from aminx.host.campaign import plan_campaign_manifest
from aminx.run.specs import SamplingSpecification, pop_deprecated_spec_kwargs
from tests.host.knob_observations import (
  KNOWN_BROKEN_AT_A17,
  OBSERVATIONS,
  Bundle,
  Internal,
  LoadModel,
  NonSerializable,
  NotApplicable,
  Overridden,
  SpecOnly,
)

pytestmark = pytest.mark.knob_differential

_LIGAND_CKPT = "ligandmpnn_v_32_020_25"


def _plan_one_row(**spec_kwargs: Any) -> dict[str, Any]:
  """Drive the REAL planner and return the row's serialized sampling_spec."""
  base = SamplingSpecification(
    inputs=["ref.pdb"],
    checkpoint_id=_LIGAND_CKPT,
    **spec_kwargs,
  )
  rows = plan_campaign_manifest(
    base_spec=base,
    campaign_id="knob-differential",
    output_root="/tmp/knob-differential",
    designs_per_library_type=1,
    samples_chunk_size=1,
    fixed_policies=("catalytic_triad",),
  )
  return dict(rows[0]["sampling_spec"])


def _reconstruct(payload: dict[str, Any]) -> SamplingSpecification:
  """Reconstruct exactly as the worker does (campaign.py:853-856).

  Deliberately mirrors the real worker rather than using spec_json: the harness must not
  diverge from the path under test, or it stops testing anything real.
  """
  worker_payload = dict(payload)
  pop_deprecated_spec_kwargs(worker_payload)
  return SamplingSpecification(**worker_payload)


def _round_trip(field: str, value: Any) -> Any:
  """Set `field` on base_spec, drive the real path, return the reconstructed value."""
  spec = _reconstruct(_plan_one_row(**{field: value}))
  return getattr(spec, field)


def _equal(a: Any, b: Any) -> bool:
  if a is None or b is None:
    return a is b
  try:
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))
  except (TypeError, ValueError):
    return a == b


# Differential pairs: (field, value_a, value_b). Values must be legal on their own so a
# failure means "the knob was dropped", never "the spec rejected my input".
DIFFERENTIAL_PAIRS: dict[str, tuple[Any, Any]] = {
  "fixed_mask": (np.zeros(4, dtype=np.float32), np.array([1, 0, 1, 0], dtype=np.float32)),
  "fixed_positions": (np.zeros(4, dtype=np.float32), np.array([0, 1, 0, 1], dtype=np.float32)),
  "fixed_tokens": (np.zeros(4, dtype=np.int32), np.array([3, 3, 3, 3], dtype=np.int32)),
  "state_position_map": (np.zeros((1, 4), dtype=np.int32), np.arange(4, dtype=np.int32)[None, :]),
  "state_weights": (np.array([1.0], dtype=np.float32), np.array([0.5], dtype=np.float32)),
  "bias": (np.zeros((4, 21), dtype=np.float32), np.ones((4, 21), dtype=np.float32)),
  "tie_group_map": (np.zeros(4, dtype=np.int32), np.array([0, 0, 1, 1], dtype=np.int32)),
  "structure_mapping": (np.zeros(4, dtype=np.int32), np.arange(4, dtype=np.int32)),
  "multi_state_strategy": ("arithmetic_mean", "product"),
  "ligand_mpnn_use_side_chain_context": (False, True),
  "sidechain_conditioning": (False, True),
  "ligand_conditioning": (False, True),
  "use_electrostatics": (False, True),
  "use_vdw": (False, True),
  "temperature": (0.1, 0.9),
  "backbone_noise": (0.0, 0.25),
  "batch_size": (2, 8),
  "model_weights": ("original", "soluble"),
  "chain_id": ("A", "B"),
  "model_family": ("ligandmpnn", "proteinmpnn"),
}

_TIER_A_OBSERVABLE = (Bundle, LoadModel, SpecOnly)


def _tier_a_fields() -> list[str]:
  """Fields whose declared verdict implies the value must survive the manifest."""
  return sorted(
    name
    for name, verdict in OBSERVATIONS.items()
    if isinstance(verdict, _TIER_A_OBSERVABLE) and name in DIFFERENTIAL_PAIRS
  )


def _maybe_xfail(field: str) -> None:
  if field in KNOWN_BROKEN_AT_A17:
    pytest.xfail(KNOWN_BROKEN_AT_A17[field])


class TestDeclarationCompleteness:
  """The drift guard: a new spec field with no declared verdict is a red build.

  This is what makes occurrence #11 structurally impossible. It is the same failure the audit
  exists to answer for -- `plan_campaign_manifest`'s hand-written literal drifting from an
  88-field dataclass with nothing enforcing the sync.
  """

  def test_every_spec_field_has_a_declared_observation(self) -> None:
    declared = set(OBSERVATIONS)
    actual = {f.name for f in fields(SamplingSpecification)}

    undeclared = actual - declared
    assert not undeclared, (
      f"Spec field(s) with no declared observation: {sorted(undeclared)}. "
      "Add an entry to tests/host/knob_observations.py declaring how this knob behaves in "
      "campaign mode and where it is observable. An undeclared field is exactly how the "
      "previous ten silent-knob bugs reached production."
    )

    stale = declared - actual
    assert not stale, f"Declared field(s) that no longer exist on the spec: {sorted(stale)}"

  def test_every_reason_cites_something(self) -> None:
    """A verdict whose reason asserts a conclusion without evidence is not a verdict."""
    vague = [
      name
      for name, verdict in OBSERVATIONS.items()
      if len(verdict.reason) < 20
    ]
    assert not vague, f"Verdicts needing an evidenced reason: {vague}"

  def test_known_broken_entries_are_declared(self) -> None:
    undeclared = set(KNOWN_BROKEN_AT_A17) - set(OBSERVATIONS)
    assert not undeclared, f"KNOWN_BROKEN_AT_A17 names not in OBSERVATIONS: {sorted(undeclared)}"


class TestTierAManifestSurvival:
  """Does a caller's value survive plan -> manifest JSON -> reconstructed spec?

  No model, no structure, milliseconds. This is where 6 of the 10 known occurrences live.
  """

  @pytest.mark.parametrize("field", _tier_a_fields())
  def test_value_survives_the_real_round_trip(self, field: str) -> None:
    _maybe_xfail(field)
    value_a, value_b = DIFFERENTIAL_PAIRS[field]

    got_a = _round_trip(field, value_a)
    got_b = _round_trip(field, value_b)

    assert not _equal(got_a, got_b), (
      f"{field!r} is a SILENT NO-OP: caller set {value_a!r} vs {value_b!r}, but both "
      f"reconstruct to {got_a!r}. The value never survives the manifest, so the campaign "
      f"runs with the dataclass default regardless of what the caller asked for. "
      f"Declared verdict: {OBSERVATIONS[field]}"
    )


class TestOverriddenKnobsAreOverridden:
  """Campaign deliberately overrides these. Assert the override, not the caller's input.

  Without this, a naive differential reports all four as "not wired" -- false alarms that
  would discredit the harness immediately.
  """

  @pytest.mark.parametrize(
    "field",
    sorted(n for n, v in OBSERVATIONS.items() if isinstance(v, Overridden)),
  )
  def test_campaign_overrides_the_caller(self, field: str) -> None:
    verdict = OBSERVATIONS[field]
    assert isinstance(verdict, Overridden)

    payload = _plan_one_row()
    if field not in payload:
      pytest.skip(f"{field} is not carried by the manifest literal at all")

    spec = _reconstruct(payload)
    got = getattr(spec, field)

    if verdict.value is None:
      return  # per-row planner value; asserted by the planner's own tests
    assert got == verdict.value, (
      f"{field!r} should be overridden to {verdict.value!r} by the campaign planner "
      f"({verdict.reason}), got {got!r}. Either the override changed or the declaration is stale."
    )


class TestInertKnobsStayInert:
  """NOT_APPLICABLE knobs must remain absent from the manifest.

  If one starts being carried, the declaration is stale and must be re-adjudicated. This is
  the guard against a fix silently invalidating a triage verdict.
  """

  @pytest.mark.parametrize(
    "field",
    sorted(
      n for n, v in OBSERVATIONS.items()
      if isinstance(v, NotApplicable) and n in DIFFERENTIAL_PAIRS
    ),
  )
  def test_declared_inert_knob_is_still_inert(self, field: str) -> None:
    value_a, value_b = DIFFERENTIAL_PAIRS[field]
    got_a = _round_trip(field, value_a)
    got_b = _round_trip(field, value_b)

    assert _equal(got_a, got_b), (
      f"{field!r} is declared NOT_APPLICABLE in campaign mode "
      f"({OBSERVATIONS[field].reason}) but its value now SURVIVES the round trip "
      f"({got_a!r} vs {got_b!r}). The declaration is stale -- re-adjudicate it."
    )


class TestNonSerializableKnobsRejected:
  """Callables must not silently vanish into the manifest."""

  @pytest.mark.parametrize(
    "field",
    sorted(n for n, v in OBSERVATIONS.items() if isinstance(v, (NonSerializable, Internal))),
  )
  def test_not_carried_by_the_manifest(self, field: str) -> None:
    payload = _plan_one_row()
    assert field not in payload, (
      f"{field!r} is declared non-serializable/internal but appears in the manifest literal. "
      "Either it is now serializable (update the declaration) or the literal is wrong."
    )
