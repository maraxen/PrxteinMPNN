"""JSON-safe round-trip for a **subset** of :class:`RunSpec` static fields.

Full Equinox / PyTree serialization is out of scope. Only ``version``,
``multistate`` (``mode``, ``n_states``, ``combine_strategy``), ``resource``
(``n_devices``, ``sample_batch_size``, ``structure_batch_size``), and
``precision.compute`` are read from or written to JSON-native dicts. Other
sub-configs on :func:`run_spec_portable_from_dict` are **placeholders** aligned
with :func:`build_run_spec` defaults for a minimal single-GPU
:class:`~prxteinmpnn.run.specs.RunSpecification`-style inference stub.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from .spec import (
  AveragingConfig,
  BatchingConfig,
  GridLineageConfig,
  IOConfig,
  LigandConfig,
  MultistateConfig,
  PrecisionConfig,
  ResourceConfig,
  RunSpec,
  TiedPositionsConfig,
)

PORTABLE_RUN_SPEC_VERSION = 1

_PrecisionLabel = Literal["fp32", "fp16", "bf16"]

_INTRA = "intra"


def _as_int(path: str, value: object) -> int:
  if type(value) is not int:
    msg = f"[portable_run_spec] {path}: expected int, got {type(value).__name__}"
    raise TypeError(msg)
  return value


def _as_str(path: str, value: object) -> str:
  if not isinstance(value, str):
    msg = f"[portable_run_spec] {path}: expected str, got {type(value).__name__}"
    raise TypeError(msg)
  return value


def _as_mapping(path: str, value: object) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    msg = f"[portable_run_spec] {path}: expected object, got {type(value).__name__}"
    raise TypeError(msg)
  return cast("Mapping[str, Any]", value)


def _placeholder_run_spec(
  *,
  multistate: MultistateConfig,
  resource: ResourceConfig,
  precision: PrecisionConfig,
) -> RunSpec:
  """Defaults mirror :func:`build_run_spec` for a bare ``RunSpecification``-like spec."""
  io = IOConfig(output_dir=None, sink_kind="none", manifest_path=None)
  ligand = LigandConfig(
    model_family="proteinmpnn",
    use_side_chain_context=None,
    ligand_conditioning=False,
    sidechain_conditioning=False,
    context_path=None,
  )
  tied = TiedPositionsConfig(
    tied_positions=None,
    pass_mode=_INTRA,
    multi_state_temperature=1.0,
    tie_group_map=None,
    structure_mapping=None,
  )
  grid = GridLineageConfig(
    grid_mode=False,
    campaign_mode=False,
    job_id=None,
    chunk_id=None,
    sample_start=None,
    sample_count=None,
  )
  batching = BatchingConfig(
    batch_size=32,
    samples_batch_size=None,
    samples_chunk_size=None,
    noise_batch_size=None,
    temperature_batch_size=None,
    jacobian_batch_size=None,
    combine_batch_size=None,
    apc_batch_size=None,
    apc_residue_batch_size=None,
  )
  averaging = AveragingConfig(
    average_node_features=False,
    average_encoding_mode="inputs_and_noise",
    average_encodings=None,
    state_weights=None,
  )
  tree = RunSpec(
    io=io,
    resource=resource,
    multistate=multistate,
    ligand=ligand,
    tied=tied,
    grid=grid,
    batching=batching,
    averaging=averaging,
    precision=precision,
  )
  return cast("RunSpec", tree)


def run_spec_portable_to_dict(run_spec: RunSpec) -> dict[str, Any]:
  """Export JSON-native fields from ``run_spec`` (static scalar subset only)."""
  return {
    "version": PORTABLE_RUN_SPEC_VERSION,
    "multistate": {
      "mode": run_spec.multistate.mode,
      "n_states": run_spec.multistate.n_states,
      "combine_strategy": run_spec.multistate.combine_strategy,
    },
    "resource": {
      "n_devices": run_spec.resource.n_devices,
      "sample_batch_size": run_spec.resource.sample_batch_size,
      "structure_batch_size": run_spec.resource.structure_batch_size,
    },
    "precision": {"compute": run_spec.precision.compute},
  }


def run_spec_portable_from_dict(data: Mapping[str, Any]) -> RunSpec:
  """Parse portable dict; unknown top-level keys are ignored."""
  raw_ver = data.get("version", None)
  if type(raw_ver) is not int or raw_ver != PORTABLE_RUN_SPEC_VERSION:
    msg = f"[portable_run_spec] version: expected int {PORTABLE_RUN_SPEC_VERSION}, got {raw_ver!r}"
    raise ValueError(msg)

  for key in ("multistate", "resource", "precision"):
    if key not in data:
      msg = f"[portable_run_spec] missing required key {key!r}"
      raise ValueError(msg)

  ms = _as_mapping("multistate", data["multistate"])
  for sub in ("mode", "n_states", "combine_strategy"):
    if sub not in ms:
      msg = f"[portable_run_spec] multistate.{sub}: required key missing"
      raise ValueError(msg)
  multistate = cast(
    "MultistateConfig",
    MultistateConfig(
      mode=_as_str("multistate.mode", ms["mode"]),
      n_states=_as_int("multistate.n_states", ms["n_states"]),
      combine_strategy=_as_str("multistate.combine_strategy", ms["combine_strategy"]),
    ),
  )

  res = _as_mapping("resource", data["resource"])
  for sub in ("n_devices", "sample_batch_size", "structure_batch_size"):
    if sub not in res:
      msg = f"[portable_run_spec] resource.{sub}: required key missing"
      raise ValueError(msg)
  resource = cast(
    "ResourceConfig",
    ResourceConfig(
      n_devices=_as_int("resource.n_devices", res["n_devices"]),
      sample_batch_size=_as_int("resource.sample_batch_size", res["sample_batch_size"]),
      structure_batch_size=_as_int("resource.structure_batch_size", res["structure_batch_size"]),
    ),
  )

  prec = _as_mapping("precision", data["precision"])
  if "compute" not in prec:
    msg = "[portable_run_spec] precision.compute: required key missing"
    raise ValueError(msg)
  compute_raw = prec["compute"]
  compute = _as_str("precision.compute", compute_raw)
  if compute not in ("fp32", "fp16", "bf16"):
    msg = f"[portable_run_spec] precision.compute: invalid {compute!r}"
    raise ValueError(msg)
  precision = cast("PrecisionConfig", PrecisionConfig(compute=cast("_PrecisionLabel", compute)))

  return _placeholder_run_spec(
    multistate=multistate,
    resource=resource,
    precision=precision,
  )
