"""Composed run configuration (`RunSpec`) and sub-config modules.

See `.agents/REFACTOR_ROADMAP.md` §3.5. This module is the Equinox/pytree representation;
`run.specs.RunSpecification` remains the public dataclass façade for one minor version.
"""

from __future__ import annotations

from os import PathLike, fspath
from pathlib import Path
from typing import Any, Literal, cast

import equinox as eqx


class IOConfig(eqx.Module):
  """Output / manifest / sink selection (host-side)."""

  output_dir: Path | None = eqx.field(static=True)
  sink_kind: str = eqx.field(static=True)
  manifest_path: Path | None = eqx.field(static=True)


class ResourceConfig(eqx.Module):
  """Device and batching caps (structure vs sample streams)."""

  n_devices: int = eqx.field(static=True)
  sample_batch_size: int = eqx.field(static=True)
  structure_batch_size: int = eqx.field(static=True)


class MultistateConfig(eqx.Module):
  """Multistate / combine-strategy knobs (registry keys are strings, not int indices)."""

  mode: str = eqx.field(static=True)
  n_states: int = eqx.field(static=True)
  combine_strategy: str = eqx.field(static=True)


class LigandConfig(eqx.Module):
  """LigandMPNN-specific options layered on `RunSpecification`."""

  model_family: str = eqx.field(static=True)
  use_side_chain_context: bool | None = eqx.field(static=True)
  ligand_conditioning: bool = eqx.field(static=True)
  sidechain_conditioning: bool = eqx.field(static=True)
  context_path: Path | None = eqx.field(static=True)


class TiedPositionsConfig(eqx.Module):
  """Tied-position and structure-mapping payloads (arrays may be traced)."""

  tied_positions: object | None = eqx.field(static=True)
  pass_mode: str = eqx.field(static=True)
  multi_state_temperature: float = eqx.field(static=True)
  tie_group_map: Any = None
  structure_mapping: Any = None


class GridLineageConfig(eqx.Module):
  """Grid / campaign lineage metadata (mostly static host fields)."""

  grid_mode: bool = eqx.field(static=True)
  campaign_mode: bool = eqx.field(static=True)
  job_id: str | None = eqx.field(static=True)
  chunk_id: int | None = eqx.field(static=True)
  sample_start: int | None = eqx.field(static=True)
  sample_count: int | None = eqx.field(static=True)


class BatchingConfig(eqx.Module):
  """Per-task batch and chunk sizes; `None` means “not applicable” for this task."""

  batch_size: int = eqx.field(static=True)
  samples_batch_size: int | None = eqx.field(static=True)
  samples_chunk_size: int | None = eqx.field(static=True)
  noise_batch_size: int | None = eqx.field(static=True)
  temperature_batch_size: int | None = eqx.field(static=True)
  jacobian_batch_size: int | None = eqx.field(static=True)
  combine_batch_size: int | None = eqx.field(static=True)
  apc_batch_size: int | None = eqx.field(static=True)
  apc_residue_batch_size: int | None = eqx.field(static=True)


class AveragingConfig(eqx.Module):
  """Feature / encoding averaging options."""

  average_node_features: bool = eqx.field(static=True)
  average_encoding_mode: str = eqx.field(static=True)
  average_encodings: bool | None = eqx.field(static=True)
  state_weights: Any = None


class PrecisionConfig(eqx.Module):
  """JAX compute dtype policy for training and inference (mirrors dataclass ``precision`` when set)."""

  compute: Literal["fp32", "fp16", "bf16"] = eqx.field(static=True)


class RunSpec(eqx.Module):
  """Composed configuration for run/prep pipelines."""

  io: IOConfig
  resource: ResourceConfig
  multistate: MultistateConfig
  ligand: LigandConfig
  tied: TiedPositionsConfig
  grid: GridLineageConfig
  batching: BatchingConfig
  averaging: AveragingConfig
  precision: PrecisionConfig


def _optional_path(value: object | None) -> Path | None:
  if value is None:
    return None
  if isinstance(value, Path):
    return value
  if isinstance(value, str):
    return Path(value)
  return Path(fspath(cast("PathLike[str]", value)))


def _output_h5_path(spec: object) -> Path | None:
  raw = getattr(spec, "output_h5_path", None)
  return _optional_path(raw)


def _infer_output_dir(spec: object) -> Path | None:
  h5 = _output_h5_path(spec)
  if h5 is not None:
    return h5.parent
  cache = getattr(spec, "cache_path", None)
  if cache is not None:
    p = _optional_path(cache)
    return p.parent if p is not None and p.suffix else p
  # TODO(REFACTOR_ROADMAP §3.5): unify with explicit `output_dir` once call-sites migrate.
  return None


def _run_spec_precision_compute(spec: object) -> Literal["fp32", "fp16", "bf16"]:
  """Resolve compute precision: training specs use ``precision``; otherwise default fp32."""
  raw = getattr(spec, "precision", "fp32")
  if raw in ("fp32", "fp16", "bf16"):
    return cast("Literal['fp32', 'fp16', 'bf16']", raw)
  return "fp32"


def _infer_sink_kind(spec: object) -> str:
  if getattr(spec, "use_arrayrecord", False):
    return "arrayrecord"
  if _output_h5_path(spec) is not None:
    return "hdf5"
  # TODO(REFACTOR_ROADMAP §3.5): align with `OUTPUT_SINKS` registry keys when Phase 5 lands.
  return "none"


def _coerce_n_states(spec: object) -> int:
  cs = getattr(spec, "conformational_states", None)
  if cs is None:
    return 1
  ns = getattr(cs, "n_states", None)
  if ns is None:
    return 1
  try:
    if hasattr(ns, "item"):
      return int(ns.item())
    return int(ns)
  except (TypeError, ValueError):
    # TODO(REFACTOR_ROADMAP §3.5): stable int from `MultistateStackPayload` / stack shape.
    return 1


def _infer_multistate_mode(spec: object) -> str:
  # TODO(REFACTOR_ROADMAP §3.3 `MULTISTATE_MODES`): replace placeholder with registry key
  # derived from prep/model capabilities once `multistate_mode` leaves ad-hoc if-ladders.
  if getattr(spec, "conformational_states", None) is not None:
    return "conformational_context"
  if _coerce_n_states(spec) > 1:
    return "multi_state"
  return "single"


def build_run_spec(spec: object) -> RunSpec:
  """Build a :class:`RunSpec` view from a :class:`RunSpecification` (or subclass) instance."""
  pre_idx = getattr(spec, "preprocessed_index_path", None)
  io = IOConfig(
    output_dir=_infer_output_dir(spec),
    sink_kind=_infer_sink_kind(spec),
    manifest_path=_optional_path(pre_idx),
  )

  batch_size = int(getattr(spec, "batch_size", 32))
  samples_batch = getattr(spec, "samples_batch_size", None)
  resource = ResourceConfig(
    # TODO(REFACTOR_ROADMAP Phase 4): thread `jax.local_device_count()` / user override.
    n_devices=1,
    sample_batch_size=int(samples_batch) if samples_batch is not None else batch_size,
    structure_batch_size=batch_size,
  )

  combine = getattr(spec, "multi_state_strategy", "arithmetic_mean")
  multistate = MultistateConfig(
    mode=_infer_multistate_mode(spec),
    n_states=_coerce_n_states(spec),
    combine_strategy=str(combine),
  )

  ctx_path = getattr(spec, "ligand_context_path", None)
  ligand = LigandConfig(
    model_family=str(getattr(spec, "model_family", "proteinmpnn")),
    use_side_chain_context=getattr(spec, "ligand_mpnn_use_side_chain_context", None),
    ligand_conditioning=bool(getattr(spec, "ligand_conditioning", False)),
    sidechain_conditioning=bool(getattr(spec, "sidechain_conditioning", False)),
    context_path=_optional_path(ctx_path),
  )

  tied = TiedPositionsConfig(
    tied_positions=getattr(spec, "tied_positions", None),
    pass_mode=str(getattr(spec, "pass_mode", "intra")),
    multi_state_temperature=float(getattr(spec, "multi_state_temperature", 1.0)),
    tie_group_map=getattr(spec, "tie_group_map", None),
    structure_mapping=getattr(spec, "structure_mapping", None),
  )

  grid = GridLineageConfig(
    grid_mode=bool(getattr(spec, "grid_mode", False)),
    campaign_mode=bool(getattr(spec, "campaign_mode", False)),
    job_id=getattr(spec, "job_id", None),
    chunk_id=getattr(spec, "chunk_id", None),
    sample_start=getattr(spec, "sample_start", None),
    sample_count=getattr(spec, "sample_count", None),
  )

  jacobian_batch = getattr(spec, "jacobian_batch_size", None)
  combine_bs = getattr(spec, "combine_batch_size", None)
  apc_bs = getattr(spec, "apc_batch_size", None)
  apc_res_bs = getattr(spec, "apc_residue_batch_size", None)
  samples_chunk = getattr(spec, "samples_chunk_size", None)
  temperature_bs = getattr(spec, "temperature_batch_size", None)
  noise_bs = getattr(spec, "noise_batch_size", None)
  batching = BatchingConfig(
    batch_size=batch_size,
    samples_batch_size=int(samples_batch) if samples_batch is not None else None,
    samples_chunk_size=int(samples_chunk) if samples_chunk is not None else None,
    noise_batch_size=int(noise_bs) if noise_bs is not None else None,
    temperature_batch_size=int(temperature_bs) if temperature_bs is not None else None,
    jacobian_batch_size=int(jacobian_batch) if jacobian_batch is not None else None,
    combine_batch_size=int(combine_bs) if combine_bs is not None else None,
    apc_batch_size=int(apc_bs) if apc_bs is not None else None,
    apc_residue_batch_size=int(apc_res_bs) if apc_res_bs is not None else None,
  )

  avg_enc = getattr(spec, "average_encodings", None)
  averaging = AveragingConfig(
    average_node_features=bool(getattr(spec, "average_node_features", False)),
    average_encoding_mode=str(getattr(spec, "average_encoding_mode", "inputs_and_noise")),
    average_encodings=(bool(avg_enc) if avg_enc is not None else None),
    state_weights=getattr(spec, "state_weights", None),
  )

  precision = PrecisionConfig(compute=_run_spec_precision_compute(spec))

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
