"""Equinox payload carriers for multistate / sampling (Phase 3).

Field groupings follow ``state_vmap_prep``, ``sampling/sample.py`` stack kwargs,
``protocols.SamplerFn`` / ``StateVmapExactScoreFn``, ``io/designs`` design records,
and ``run/sampling`` grid lineage. Shapes are caller-defined; see roadmap §3.2.
"""

from __future__ import annotations

from typing import Any, TypeVar

import equinox as eqx
from jaxtyping import Array, Bool, Float, Int

T_bundle = TypeVar("T_bundle", bound=eqx.Module)


def _replace_bundle(
  obj: T_bundle,
  cls: type[T_bundle],
  fields: tuple[str, ...],
  static: frozenset[str],
  **kw: Any,
) -> T_bundle:
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


class ProteinBundle(eqx.Module):
  """Stacked backbone / token geometry for ``state_vmap_exact`` (numpy prep → JAX arrays)."""

  coords: Float[Array, ...] # TODO: we need to use our Batch and Axis primiatives, raw stacks like this are highly memory efficient
  mask: Float[Array, ...]
  residue_index: Int[Array, ...]
  chain_index: Int[Array, ...]
  tie_group_map: Int[Array, ...]
  fixed_mask: Float[Array, ...]
  fixed_tokens: Int[Array, ...]
  state_index: Int[Array, ...]
  state_embedding: Float[Array, ...]
  n_states: int = eqx.field(static=True)
  n_canonical: int = eqx.field(static=True)

  def replace(self, **kw: Any) -> ProteinBundle:
    fields = (
      "coords",
      "mask",
      "residue_index",
      "chain_index",
      "tie_group_map",
      "fixed_mask",
      "fixed_tokens",
      "state_index",
      "state_embedding",
      "n_states",
      "n_canonical",
    )
    static = frozenset({"n_states", "n_canonical"})
    return _replace_bundle(self, ProteinBundle, fields, static, **kw)

  def slice(self, start: int, count: int) -> "ProteinBundle":
    """Return a new payload with states [start, start+count).

    Updates n_states = count.
    n_flat = count * n_canonical (assumes uniform state lengths — valid for tied multistate).
    All (S, ...) arrays are sliced on axis 0.
    flat_row_offsets is rebased: flat_row_offsets[start:start+count] - flat_row_offsets[start].

    Raises ValueError if start < 0 or start + count > n_states.
    """
    if start < 0 or count <= 0 or start + count > self.n_states:
      raise ValueError(
          f"slice out of range: start={start}, count={count}, n_states={self.n_states}"
      )

    # Slice arrays on axis 0 (the S dimension)
    sliced_coords = self.coords[start:start+count]
    sliced_mask = self.mask[start:start+count]
    sliced_ri = self.residue_index[start:start+count]
    sliced_ci = self.chain_index[start:start+count]
    sliced_tie = self.tie_group_map[start:start+count]
    sliced_fixed_mask = self.fixed_mask[start:start+count]
    sliced_fixed_tokens = self.fixed_tokens[start:start+count]
    sliced_state_idx = self.state_index[start:start+count]
    sliced_state_emb = self.state_embedding[start:start+count]

    return ProteinBundle(
        coords=sliced_coords,
        mask=sliced_mask,
        residue_index=sliced_ri,
        chain_index=sliced_ci,
        tie_group_map=sliced_tie,
        fixed_mask=sliced_fixed_mask,
        fixed_tokens=sliced_fixed_tokens,
        state_index=sliced_state_idx,
        state_embedding=sliced_state_emb,
        n_states=count,
        n_canonical=self.n_canonical,
    )


class WaveColorBundle(eqx.Module):
  """Wave-parallel autoregressive decode schedule (graph-coloring output, host-prepped)."""

  wave_group_ids: Int[Array, ...]
  wave_group_positions: Int[Array, ...]
  wave_group_valid: Bool[Array, ...]
  wave_position_valid: Bool[Array, ...]

  def replace(self, **kw: Any) -> WaveColorBundle:
    fields = (
      "wave_group_ids",
      "wave_group_positions",
      "wave_group_valid",
      "wave_position_valid",
    )
    return _replace_bundle(self, WaveColorBundle, fields, frozenset(), **kw)


class LigandBundle(eqx.Module):
  """Ligand ``Y`` / ``Y_t`` / ``Y_m`` stacks for LigandMPNN ``state_vmap_exact`` paths."""

  y: Float[Array, ...]
  y_t: Float[Array, ...]
  y_m: Float[Array, ...]

  def replace(self, **kw: Any) -> LigandBundle:
    fields = ("y", "y_t", "y_m")
    return _replace_bundle(self, LigandBundle, fields, frozenset(), **kw)


class LigandContext(eqx.Module):
  """Ligand-side scoring / chunking metadata beside :class:`LigandBundle`."""

  state_mapping: Float[Array, ...]
  states_chunk_size: int = eqx.field(static=True)

  def replace(self, **kw: Any) -> LigandContext:
    fields = ("state_mapping", "states_chunk_size")
    static = frozenset({"states_chunk_size"})
    return _replace_bundle(self, LigandContext, fields, static, **kw)


class SamplingControls(eqx.Module):
  """Temperature, fusion index, and optional stacked bias for multistate sampling."""

  temperature: Float[Array, ...]
  multi_state_strategy_idx: Int[Array, ...]
  multi_state_temperature: Float[Array, ...]
  state_weights: Float[Array, ...]
  bias: Float[Array, ...]

  def replace(self, **kw: Any) -> SamplingControls:
    fields = (
      "temperature",
      "multi_state_strategy_idx",
      "multi_state_temperature",
      "state_weights",
      "bias",
    )
    return _replace_bundle(self, SamplingControls, fields, frozenset(), **kw)


class MultistateBundle(eqx.Module):
  """Wave-schedule tensors for stacked autoregressive decode."""

  wave_group_ids_local: Int[Array, ...]
  wave_group_positions_local: Int[Array, ...]
  wave_group_valid_local: Bool[Array, ...]
  wave_position_valid_local: Bool[Array, ...]
  autoregressive_mask: Float[Array, ...]

  def replace(self, **kw: Any) -> MultistateBundle:
    fields = (
      "wave_group_ids_local",
      "wave_group_positions_local",
      "wave_group_valid_local",
      "wave_position_valid_local",
      "autoregressive_mask",
    )
    return _replace_bundle(self, MultistateBundle, fields, frozenset(), **kw)


class EncodedFeatures(eqx.Module):
  """Encoder outputs carried across encode / decode boundaries (single or batched)."""

  node_features: Float[Array, ...]
  edge_features: Float[Array, ...]
  neighbor_indices: Int[Array, ...]

  def replace(self, **kw: Any) -> EncodedFeatures:
    fields = ("node_features", "edge_features", "neighbor_indices")
    return _replace_bundle(self, EncodedFeatures, fields, frozenset(), **kw)


class EncoderOutput(eqx.Module):
  """Multi-state encoder output (S states) for EncoderPostFn hook injection.

  Produced by jax.vmap(encode_one) over a state stack; passed to EncoderPostFn
  before decoding. Use this type in EncoderPostFn signatures (not EncodedFeatures).
  """

  node_features: Float[Array, ...]
  edge_features: Float[Array, ...]
  neighbor_indices: Int[Array, ...]
  mask: Float[Array, ...]

  def replace(self, **kw: Any) -> EncoderOutput:
    fields = ("node_features", "edge_features", "neighbor_indices", "mask")
    return _replace_bundle(self, EncoderOutput, fields, frozenset(), **kw)


class SampleResult(eqx.Module):
  """Sequence + logits from a sampler or autoregressive pass."""

  sequence: Int[Array, ...]
  logits: Float[Array, ...]

  def replace(self, **kw: Any) -> SampleResult:
    fields = ("sequence", "logits")
    return _replace_bundle(self, SampleResult, fields, frozenset(), **kw)


class GridLineage(eqx.Module):
  """Design-grid manifest fields (host metadata; all ``static=True``)."""

  job_id: str = eqx.field(static=True)
  chunk_id: int = eqx.field(static=True)
  sample_start: int = eqx.field(static=True)
  sample_count: int = eqx.field(static=True)

  def replace(self, **kw: Any) -> GridLineage:
    fields = ("job_id", "chunk_id", "sample_start", "sample_count")
    static = frozenset(fields)
    return _replace_bundle(self, GridLineage, fields, static, **kw)


__all__ = [
  "EncodedFeatures",
  "EncoderOutput",
  "GridLineage",
  "LigandBundle",
  "LigandContext",
  "MultistateBundle",
  "ProteinBundle",
  "SampleResult",
  "SamplingControls",
  "WaveColorBundle",
]
