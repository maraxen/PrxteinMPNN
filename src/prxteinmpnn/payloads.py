"""Equinox payload carriers for multistate / sampling (Phase 3).

Field groupings follow ``state_vmap_prep``, ``sampling/sample.py`` stack kwargs,
``protocols.SamplerFn`` / ``StateVmapExactScoreFn``, ``io/designs`` design records,
and ``run/sampling`` grid lineage. Shapes are caller-defined; see roadmap §3.2.
"""

from __future__ import annotations

from typing import Any, TypeVar

import equinox as eqx
from jaxtyping import Array, Bool, Float, Int

T_payload = TypeVar("T_payload", bound=eqx.Module)


def _replace_payload(
  obj: T_payload,
  cls: type[T_payload],
  fields: tuple[str, ...],
  static: frozenset[str],
  **kw: Any,
) -> T_payload:
  """``eqx.tree_at`` for dynamic-only updates; full ``cls(**…)`` if any static field changes."""
  if not kw:
    return obj
  allowed = set(fields)
  bad = set(kw) - allowed
  if bad:
    msg = f"{cls.__name__}.replace: unknown field(s) {sorted(bad)}"
    raise TypeError(msg)
  if static & kw.keys():
    merged = {name: kw[name] if name in kw else getattr(obj, name) for name in fields}
    return cls(**merged)  # type: ignore[arg-type]
  out: Any = obj
  for key, val in kw.items():
    out = eqx.tree_at(lambda o, k=key: getattr(o, k), out, val)
  return out


class MultistateStackPayload(eqx.Module):
  """Stacked backbone / token geometry for ``state_vmap_exact`` (numpy prep → JAX arrays)."""

  coords_stack: Float[Array, ...]
  mask_stack: Float[Array, ...]
  residue_index_stack: Int[Array, ...]
  chain_index_stack: Int[Array, ...]
  tie_group_map_stack: Int[Array, ...]
  fixed_mask_stack: Float[Array, ...]
  fixed_tokens_stack: Int[Array, ...]
  state_flat_rows: Int[Array, ...]
  flat_row_offsets: Int[Array, ...]
  n_states: int = eqx.field(static=True)
  n_canonical: int = eqx.field(static=True)
  n_flat: int = eqx.field(static=True)

  def replace(self, **kw: Any) -> MultistateStackPayload:
    fields = (
      "coords_stack",
      "mask_stack",
      "residue_index_stack",
      "chain_index_stack",
      "tie_group_map_stack",
      "fixed_mask_stack",
      "fixed_tokens_stack",
      "state_flat_rows",
      "flat_row_offsets",
      "n_states",
      "n_canonical",
      "n_flat",
    )
    static = frozenset({"n_states", "n_canonical", "n_flat"})
    return _replace_payload(self, MultistateStackPayload, fields, static, **kw)


class LigandStack(eqx.Module):
  """Ligand ``Y`` / ``Y_t`` / ``Y_m`` stacks for LigandMPNN ``state_vmap_exact`` paths."""

  y_stack: Float[Array, ...]
  y_t_stack: Float[Array, ...]
  y_m_stack: Float[Array, ...]

  def replace(self, **kw: Any) -> LigandStack:
    fields = ("y_stack", "y_t_stack", "y_m_stack")
    return _replace_payload(self, LigandStack, fields, frozenset(), **kw)


class LigandContext(eqx.Module):
  """Ligand-side scoring / chunking metadata beside :class:`LigandStack`."""

  state_mapping: Float[Array, ...]
  states_chunk_size: int = eqx.field(static=True)

  def replace(self, **kw: Any) -> LigandContext:
    fields = ("state_mapping", "states_chunk_size")
    static = frozenset({"states_chunk_size"})
    return _replace_payload(self, LigandContext, fields, static, **kw)


class SamplingControls(eqx.Module):
  """Temperature, fusion index, and optional stacked bias for multistate sampling."""

  temperature: Float[Array, ...]
  multi_state_strategy_idx: Int[Array, ...]
  multi_state_temperature: Float[Array, ...]
  state_weights: Float[Array, ...]
  bias_stack: Float[Array, ...]

  def replace(self, **kw: Any) -> SamplingControls:
    fields = (
      "temperature",
      "multi_state_strategy_idx",
      "multi_state_temperature",
      "state_weights",
      "bias_stack",
    )
    return _replace_payload(self, SamplingControls, fields, frozenset(), **kw)


class MultistateContext(eqx.Module):
  """Wave-schedule tensors for stacked autoregressive decode."""

  wave_group_ids_local: Int[Array, ...]
  wave_group_positions_local: Int[Array, ...]
  wave_group_valid_local: Bool[Array, ...]
  wave_position_valid_local: Bool[Array, ...]
  autoregressive_mask_stack: Float[Array, ...]

  def replace(self, **kw: Any) -> MultistateContext:
    fields = (
      "wave_group_ids_local",
      "wave_group_positions_local",
      "wave_group_valid_local",
      "wave_position_valid_local",
      "autoregressive_mask_stack",
    )
    return _replace_payload(self, MultistateContext, fields, frozenset(), **kw)


class EncodedFeatures(eqx.Module):
  """Encoder outputs carried across encode / decode boundaries (single or batched)."""

  node_features: Float[Array, ...]
  edge_features: Float[Array, ...]
  neighbor_indices: Int[Array, ...]

  def replace(self, **kw: Any) -> EncodedFeatures:
    fields = ("node_features", "edge_features", "neighbor_indices")
    return _replace_payload(self, EncodedFeatures, fields, frozenset(), **kw)


class SampleResult(eqx.Module):
  """Sequence + logits from a sampler or autoregressive pass."""

  sequence: Int[Array, ...]
  logits: Float[Array, ...]

  def replace(self, **kw: Any) -> SampleResult:
    fields = ("sequence", "logits")
    return _replace_payload(self, SampleResult, fields, frozenset(), **kw)


class GridLineage(eqx.Module):
  """Design-grid manifest fields (host metadata; all ``static=True``)."""

  job_id: str = eqx.field(static=True)
  chunk_id: int = eqx.field(static=True)
  sample_start: int = eqx.field(static=True)
  sample_count: int = eqx.field(static=True)

  def replace(self, **kw: Any) -> GridLineage:
    fields = ("job_id", "chunk_id", "sample_start", "sample_count")
    static = frozenset(fields)
    return _replace_payload(self, GridLineage, fields, static, **kw)


__all__ = [
  "EncodedFeatures",
  "GridLineage",
  "LigandContext",
  "LigandStack",
  "MultistateContext",
  "MultistateStackPayload",
  "SampleResult",
  "SamplingControls",
]
