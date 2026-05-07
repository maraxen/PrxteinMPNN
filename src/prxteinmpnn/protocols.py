"""Runtime-checkable callback protocols for sampling, scoring, and logits factories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import jax
from jaxtyping import Float, Int, PRNGKeyArray

if TYPE_CHECKING:
  from prxteinmpnn.payloads import LigandStack, MultistateStackPayload
  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    AutoRegressiveMask,
    BackboneNoise,
    ChainIndex,
    DecodingOrder,
    InputBias,
    Logits,
    OneHotProteinSequence,
    ProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
  )


@runtime_checkable
class ConditionalLogitsFn(Protocol):
  def __call__(
    self,
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    sequence: ProteinSequence,
    ar_mask: AutoRegressiveMask | None = None,
    backbone_noise: BackboneNoise | None = None,
    structure_mapping: jax.Array | None = None,
  ) -> Logits:
    """Conditional logits (single graph); optional ``structure_mapping`` for multistate encode."""


@runtime_checkable
class UnconditionalLogitsFn(Protocol):
  def __call__(
    self,
    prng_key: jax.Array,
    structure_coordinates: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    ar_mask: jax.Array | None = None,
    backbone_noise: jax.Array | None = None,
  ) -> jax.Array:
    """Dense single-graph unconditional logits."""


@runtime_checkable
class StateVmapExactLogitsFn(Protocol):
  """Stacked multistate exact logits from conditional / unconditional state-vmap factories.

  Protein vs ligand and conditional vs unconditional callables differ (extra ``sequence``,
  ``y_*``, ``states_chunk_size``, etc.); factories use ``cast`` at return. This protocol
  only fixes the return as a JAX array of logits.
  """

  def __call__(self, *args: object, **kwargs: object) -> jax.Array:
    ...


@runtime_checkable
class SamplerFn(Protocol):
  def __call__(
    self,
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    bias: InputBias | None = None,
    fixed_positions: jax.Array | None = None,
    fixed_mask: jax.Array | None = None,
    fixed_tokens: jax.Array | None = None,
    backbone_noise: BackboneNoise | None = None,
    iterations: Int | None = None,
    learning_rate: Float | None = None,
    temperature: Float | None = None,
    tie_group_map: jax.Array | None = None,
    num_groups: int | None = None,
    multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
    multistate_mode: Literal["flat", "state_vmap", "state_vmap_exact"] = "flat",
    structure_mapping: jax.Array | None = None,
    multi_state_temperature: Float = 1.0,
    max_group_size: int = 16,
    precomputed_node_features: jax.Array | None = None,
    precomputed_edge_features: jax.Array | None = None,
    precomputed_neighbor_indices: jax.Array | None = None,
    state_vmap_n_canonical: int = 0,
    coords_stack: jax.Array | None = None,
    mask_stack: jax.Array | None = None,
    residue_index_stack: jax.Array | None = None,
    chain_index_stack: jax.Array | None = None,
    tie_group_map_stack: jax.Array | None = None,
    fixed_mask_stack: jax.Array | None = None,
    fixed_tokens_stack: jax.Array | None = None,
    bias_stack: jax.Array | None = None,
    state_flat_rows: jax.Array | None = None,
    wave_group_ids_local: jax.Array | None = None,
    wave_group_positions_local: jax.Array | None = None,
    wave_group_valid_local: jax.Array | None = None,
    wave_position_valid_local: jax.Array | None = None,
    y_stack: jax.Array | None = None,
    y_t_stack: jax.Array | None = None,
    y_m_stack: jax.Array | None = None,
    multistate_stack: MultistateStackPayload | None = None,
    ligand_stack: LigandStack | None = None,
    **kwargs: object,
  ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
    ...


@runtime_checkable
class ScoreFn(Protocol):
  """Flat-graph sequence scoring (``multistate_mode=\"flat\"``)."""

  def __call__(
    self,
    prng_key: PRNGKeyArray,
    sequence: ProteinSequence | OneHotProteinSequence,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    backbone_noise: BackboneNoise | None = None,
    ar_mask: AutoRegressiveMask | None = None,
    structure_mapping: jax.Array | None = None,
    tie_group_map: jax.Array | None = None,
    multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
    multi_state_temperature: Float = 1.0,
    **kwargs: object,
  ) -> tuple[Float, Logits, DecodingOrder]:
    ...


@runtime_checkable
class StateVmapExactScoreFn(Protocol):
  """Stacked ``state_vmap_exact`` scoring; stack kwargs are keyword-only.

  Pass either individual ``coords_stack=`` … ``n_flat=`` tensors or a single
  ``multistate_stack=`` :class:`~prxteinmpnn.payloads.MultistateStackPayload`` (and
  ``ligand_stack=`` for ligand models); when ``multistate_stack`` is set it overrides
  the unpacked stack keyword arguments.
  """

  def __call__(
    self,
    prng_key: PRNGKeyArray,
    sequence: ProteinSequence | OneHotProteinSequence,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    backbone_noise: BackboneNoise | None = None,
    ar_mask: AutoRegressiveMask | None = None,
    structure_mapping: jax.Array | None = None,
    tie_group_map: jax.Array | None = None,
    multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
    multi_state_temperature: Float = 1.0,
    *,
    coords_stack: jax.Array | None = None,
    mask_stack: jax.Array | None = None,
    residue_index_stack: jax.Array | None = None,
    chain_index_stack: jax.Array | None = None,
    state_flat_rows: jax.Array | None = None,
    n_flat: int | None = None,
    state_weights: jax.Array | None = None,
    y_stack: jax.Array | None = None,
    y_t_stack: jax.Array | None = None,
    y_m_stack: jax.Array | None = None,
    ar_mask_stack: jax.Array | None = None,
    bias_flat: jax.Array | None = None,
    states_chunk_size: int = 0,
    multistate_stack: MultistateStackPayload | None = None,
    ligand_stack: LigandStack | None = None,
    **kwargs: object,
  ) -> tuple[Float, Logits, DecodingOrder]:
    ...


@runtime_checkable
class DesignSink(Protocol):
  """Host-side consumer for design tensor payloads emitted via ``io_callback`` (Phase 5).

  Implementations are registered under :data:`~prxteinmpnn.registry.OUTPUT_SINKS` and
  selected by host sessions (e.g. streaming HDF5 / ArrayRecord). Traced code calls these
  only through ``jax.experimental.io_callback`` host targets with ``ordered=False``.
  """

  def on_sampling_sequences_logits(
    self,
    batch_idx: object,
    batch_count: object,
    chunk_start: object,
    chunk_count: object,
    sequences_host: object,
    logits_host: object,
  ) -> None:
    """Record per-``_sample_batch`` sequences/logits host tensors (streaming sampling)."""

  def on_scoring_scores_logits(
    self,
    batch_idx: object,
    batch_count: object,
    scores_host: object,
    logits_host: object,
  ) -> None:
    """Record per-batch scores/logits host tensors (streaming scoring)."""
