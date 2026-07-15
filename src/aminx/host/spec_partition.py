"""Serialize a SamplingSpecification exhaustively, or fail loudly.

Twelve control knobs were declared on aminx's config surface, appeared to work, and silently
never reached the model. All twelve had one root cause: ``plan_campaign_manifest`` serialized
the spec through a **hand-written dict literal of 23 keys**, with no coupling to
``dataclasses.fields(SamplingSpecification)`` (88 fields). Nothing enforced the sync, so a
field's absence was indistinguishable from a caller wanting the default. Over ~5 weeks, eleven
separate bugs; every one found incidentally, none by a sweep. The worst -- ``fixed_mask`` --
meant no residue was ever held fixed in any produced design, silently violating a locked
preregistration.

The fix is not a better literal. It is **removing the opportunity to write one**: build the
payload from the dataclass itself, and require every field to be classified. A field nobody
classified fails the build instead of vanishing.

Three classifications, and the distinction is load-bearing:

- **serialized** -- dumped from the spec; the caller's value reaches the worker.
- **campaign-owned** -- the planner owns this per row, so the caller's value is *replaced*, not
  dropped (``job_id``, ``grid_mode``, ...). A naive differential reports these as "not wired";
  they are not bugs.
- **excluded** -- genuinely cannot cross the JSON boundary, *with a cited reason*. Today only
  callables and nested dataclasses.

task_id: 260715_aminx-campaign-control-knob-audit
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any

from aminx.run.spec_json import run_specification_to_json_dict
from aminx.run.specs import SamplingSpecification

if TYPE_CHECKING:
  from collections.abc import Mapping

# Keys the PLANNER owns per row. The caller's value is deliberately replaced, not dropped --
# these describe the row's place in the campaign, not the caller's intent.
#
# Only 7. `campaign_mode`/`return_logits`/`ligand_conditioning`/`sidechain_conditioning` are NOT
# here: `plan_campaign_manifest` already sets them on `spec_variant` via `replace()`, so the
# field-driven dump reproduces them. Of the hardcoded trio only `grid_mode` needs an override,
# because `specs.py`'s default is False and `replace()` never touches it.
CAMPAIGN_OWNED_KEYS: frozenset[str] = frozenset(
  {
    "grid_mode",
    "samples_chunk_size",
    "job_id",
    "chunk_id",
    "sample_start",
    "sample_count",
    "output_h5_path",
  },
)

# Fields that genuinely cannot cross the manifest's JSON boundary. Each entry is the REASON,
# and it must name a mechanism -- "it contains a callable" is a reason; "not applicable" is not.
#
# `run_specification_to_json_dict` already refuses these by raising SpecJSONEncodeError, so this
# set does not suppress anything; it documents *why* the raise is correct and gives the
# completeness check something to check against. Setting one of these on a campaign base_spec is
# an error, not a silent no-op -- that change (silent-drop -> loud error) is the entire point.
EXCLUDED_WITH_REASON: Mapping[str, str] = {
  "run_spec": "init=False; rebuilt by __post_init__ from the other fields",
  "_run_spec_synced": "init=False; an internal guard flag",
  "decoding_order_fn": "a callable; cannot cross a JSON boundary",
  "encoding_fusion": "a callable; cannot cross a JSON boundary",
  "decoding_fusion": "a callable; cannot cross a JSON boundary",
  "foldcomp_database": "a live database handle; cannot cross a JSON boundary",
  "conformational_states": (
    "an arbitrary object satisfying the ConformationalStates protocol; _to_json_value has no "
    "branch for it and raises"
  ),
  "carry_specs": (
    "list[CarrySpec]; CarrySpec.transition is a callable and .init holds JAX arrays. A campaign "
    "row genuinely cannot carry a Scan axis -- setting this is an error, not a dropped knob"
  ),
  "dedup_specs": (
    "list[DedupSpec]; dedup_fn/gather_fn are callables. A campaign row genuinely cannot carry a "
    "dedup axis -- setting this is an error, not a dropped knob"
  ),
}


# Fields __post_init__ DERIVES from other, serialized fields. Not carried, and losslessly so --
# verified in BOTH directions, because "derived" is only safe if the derivation is invertible:
#   backbone_noise=0.25             -> noise == [FeatureNoiseBundle(feature_type='backbone', ...)]
#   noise=[FeatureNoiseBundle(...)] -> backbone_noise == (0.3,)   (back-populated)
# The second direction is the load-bearing one: without it, a caller using the unified `noise`
# API and never touching `backbone_noise` would silently lose their setting.
DERIVED_FIELDS: frozenset[str] = frozenset({"noise"})


class SpecPartitionError(RuntimeError):
  """A spec field is classified nowhere, so its fate would be silent.

  This is the guard the manifest literal never had. It fires at import, not at sampling time --
  a knob that silently fails to reach the model is only discoverable weeks later, in results.
  """


def _spec_field_names() -> set[str]:
  return {f.name for f in fields(SamplingSpecification)}


def assert_partition_is_exhaustive() -> None:
  """Every SamplingSpecification field must be classified. Raise if any is not.

  Runs at import so a newly-added spec field breaks the build immediately, rather than being
  silently omitted from every manifest until someone notices the science is wrong.
  """
  actual = _spec_field_names()

  # A field is classified if it is excluded, planner-owned, or dumped by spec_json. The dump is
  # the source of truth for "serialized" -- asking it is what keeps this honest, versus keeping
  # a third hand-written list that could drift exactly like the literal did.
  #
  # The probe must RESEMBLE A REAL CAMPAIGN SPEC, not take defaults. An earlier version used
  # bare defaults and passed -- but `noise` is empty at defaults and only populates once
  # `backbone_noise` is set (which every campaign does), at which point the dump raises on a
  # nested dataclass. A probe that cannot reach the failing path is a false green, which is the
  # exact bug class this module exists to prevent.
  probe = SamplingSpecification(
    inputs=["_partition_probe.pdb"],
    checkpoint_id="ligandmpnn_v_32_020_25",
    backbone_noise=0.25,
    temperature=0.1,
    campaign_mode=True,
    return_logits=False,
  )
  dumped = set(run_specification_to_json_dict(probe)) - {"_spec_class"}

  classified = dumped | set(EXCLUDED_WITH_REASON) | CAMPAIGN_OWNED_KEYS | DERIVED_FIELDS
  unclassified = actual - classified
  if unclassified:
    msg = (
      f"SamplingSpecification field(s) classified nowhere: {sorted(unclassified)}. Every field "
      f"must be serialized by spec_json, listed in CAMPAIGN_OWNED_KEYS, or in "
      f"EXCLUDED_WITH_REASON with a reason naming a mechanism. An unclassified field is exactly "
      f"how twelve knobs silently never reached the model."
    )
    raise SpecPartitionError(msg)

  stale = (set(EXCLUDED_WITH_REASON) | CAMPAIGN_OWNED_KEYS) - actual
  if stale:
    msg = f"Classified name(s) that are no longer SamplingSpecification fields: {sorted(stale)}"
    raise SpecPartitionError(msg)

  vague = [name for name, reason in EXCLUDED_WITH_REASON.items() if len(reason.split()) < 4]
  if vague:
    msg = f"EXCLUDED_WITH_REASON entries needing a real reason: {vague}"
    raise SpecPartitionError(msg)


def campaign_sampling_spec_payload(
  spec_variant: SamplingSpecification,
  *,
  campaign_owned: Mapping[str, Any],
) -> dict[str, Any]:
  """Build a manifest row's ``sampling_spec`` from the dataclass, not by hand.

  Args:
    spec_variant: The per-variant spec the planner built via ``replace()``.
    campaign_owned: Per-row values the planner owns; keys must be ``CAMPAIGN_OWNED_KEYS``.

  Returns:
    A JSON-safe payload that reconstructs via ``SamplingSpecification(**payload)`` -- the
    worker's real path. ``_spec_class`` is removed deliberately: the worker uses the plain
    constructor, which rejects unknown keys with a TypeError. ``run_specification_from_json_dict``
    would instead *silently ignore* them, which would trade one silent-failure class for another.

  Raises:
    SpecPartitionError: ``campaign_owned`` does not match ``CAMPAIGN_OWNED_KEYS`` exactly.
    SpecJSONEncodeError: a field cannot cross the JSON boundary (see EXCLUDED_WITH_REASON) --
      loud by design; these were previously dropped in silence.

  """
  supplied = set(campaign_owned)
  if supplied != CAMPAIGN_OWNED_KEYS:
    missing, extra = CAMPAIGN_OWNED_KEYS - supplied, supplied - CAMPAIGN_OWNED_KEYS
    msg = (
      f"campaign_owned must supply exactly CAMPAIGN_OWNED_KEYS; missing={sorted(missing)}, "
      f"unexpected={sorted(extra)}"
    )
    raise SpecPartitionError(msg)

  payload = run_specification_to_json_dict(spec_variant)
  payload.pop("_spec_class", None)
  payload.update(campaign_owned)
  return payload


assert_partition_is_exhaustive()
