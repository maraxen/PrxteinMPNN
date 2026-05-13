"""Runtime-checkable callback protocols for sampling, scoring, and logits factories.

Tier 1 Protocols (generic, reusable):
- TransformFn[In, Out] — stateless transformation
- RollingFn[Carry, In, Out] — scan-body with init_carry
- FuseFn[PerItem, Combined] — reduce-across-axis

Tier 2 Aliases (MPNN-specific):
- FeaturizeFn, EncoderStepFn, EncoderStateFn, ProteinEncodeFn, LigandEncodeFn
- ConditionalDecodeFn, UnconditionalDecodeFn
- LogitTransformFn (as FuseFn), ARLogitTransformFn (as FuseFn)

ModelProtocol — structural seam over model implementations, unifying protein and ligand variants.
See §12 in 2026-05-13-model-protocol-seam-spec.md for architecture notes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import jax
from jaxtyping import Array, Float, Int, PRNGKeyArray

# Tier 1/2 protocols and aliases
from prxteinmpnn.model_inputs import (
  ARLogitTransformFn,
  ConditionalDecodeFn,
  EncoderStateFn,
  EncoderStepFn,
  FeaturizeFn,
  FuseFn,
  LigandEncodeFn,
  LogitTransformFn,
  ProteinEncodeFn,
  RollingFn,
  TransformFn,
  UnconditionalDecodeFn,
)
from prxteinmpnn.model.capabilities import ModelCapabilities

if TYPE_CHECKING:
  from prxteinmpnn.model_inputs import BackboneGeometry
  from prxteinmpnn.payloads import EncoderOutput, LigandStack, MultistateStackPayload
  from prxteinmpnn.pipeline_registry import StageSet
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
    TieGroupMap,
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


@runtime_checkable
class EncoderPreFn(Protocol):
  # DEPRECATED: Use EncoderStateFn instead. Will be removed in a future sprint.
  """Hook called before the encoder on each state batch.

  Returns a dict of supplemental feature arrays keyed by feature name
  (e.g. ``"initial_node_features"``, ``"rbf_features"``), or ``None``
  to use default feature extraction. Must use only jnp ops.

  NOT yet wired inside the model encoder — wiring is a follow-up task.
  """

  def __call__(
    self,
    backbone: BackboneGeometry,
    state_index: Int[Array, "S"],
  ) -> dict[str, Any] | None: ...


@runtime_checkable
class EncoderPostFn(Protocol):
  # DEPRECATED: Use EncoderStateFn instead. Will be removed in a future sprint.
  """Hook called after jax.vmap(encode_one) on the full state batch.

  Receives the stacked encoder output (S states) and may return a modified
  EncoderOutput (e.g. cosine-similarity re-weighting across states). Must
  use only jnp ops — no Python branching on traced values.

  NOT yet wired inside the model encoder — wiring is a follow-up task.
  """

  def __call__(
    self,
    encoded: EncoderOutput,
    state_index: Int[Array, "S"],
  ) -> EncoderOutput: ...


@runtime_checkable
class ModelProtocol(Protocol):
  """Structural protocol over prxteinmpnn model implementations.

  Concrete implementations: PrxteinMPNN, PrxteinLigandMPNN, DiffusionPrxteinMPNN.

  Equinox compatibility: all concrete implementations are eqx.Module subclasses
  and are valid JAX pytrees. Protocol satisfaction is structural — no ABC
  registration. jit/vmap/scan work unchanged because they trace through
  eqx.Module leaves, not through the protocol.

  Do NOT add __abstractmethods__ or ABCMeta — that breaks eqx.Module.

  The _from_payload methods form the stable seam: sampling and scoring layers
  must call only these, never raw stack-kwargs or concrete-class methods.
  """

  features: Any
  encoder: Any
  decoder: Any
  w_out: Any
  w_s_embed: Any
  capabilities: ModelCapabilities

  # NOTE: __call__ is intentionally NOT part of the protocol.
  # Its signature differs fundamentally between protein (coords, mask, ri, ci, ...)
  # and ligand (coords, mask, ri, ci, Y, Y_t, Y_m, ...). Callers that need the raw
  # forward pass should depend on the concrete class; callers using the seam should
  # go through the *_from_payload methods declared below.

  @classmethod
  def stage_schema(cls) -> dict[str, type | None]: ...

  def score_unconditional_from_payload(
    self, prng_key: PRNGKeyArray, stack: MultistateStackPayload,
    *, ligand: LigandStack | None = None,
    tie_group_map: TieGroupMap | None, multi_state_strategy_idx: Int,
    state_weights: jax.Array | None, state_mapping: jax.Array | None,
    inference: bool = True,
    logit_transform_fn: LogitTransformFn | None = None,
    encoder_state_fn: EncoderStateFn | None = None,
  ) -> Logits: ...

  def score_conditional_from_payload(
    self, prng_key, stack: MultistateStackPayload,
    seq_oh_stack: jax.Array, ar_mask_stack: jax.Array,
    *, ligand: LigandStack | None = None,
    tie_group_map, multi_state_strategy_idx,
    state_weights, state_mapping,
    bias_flat: jax.Array | None = None,
    inference: bool = True,
    logit_transform_fn: LogitTransformFn | None = None,
    encoder_state_fn: EncoderStateFn | None = None,
  ) -> Logits: ...

  def sample_autoregressive_state_vmap_exact_from_payload(
    self, prng_key, stack: MultistateStackPayload,
    autoregressive_mask_stack: jax.Array, bias_stack: jax.Array,
    temperature: float, multi_state_strategy_idx,
    state_weights, wave_group_ids_local, wave_group_positions_local,
    wave_group_valid_local, wave_position_valid_local,
    *, ligand: LigandStack | None = None,
    ar_logit_transform_fn: ARLogitTransformFn | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]: ...


@runtime_checkable
class Pipeline(Protocol):
  """Callable protocol for model pipeline implementations.

  Each concrete Pipeline (Unconditional, Conditional, Autoregressive, STE) implements
  this by calling the appropriate model method with StageSet hooks resolved.
  """

  def __call__(
    self,
    module: ModelProtocol,
    key: PRNGKeyArray,
    inputs: Any,
    *,
    stage_set: StageSet,
  ) -> Any: ...


__all__ = [
  # Tier 1 Protocols
  "TransformFn",
  "RollingFn",
  "FuseFn",
  # Tier 2 Aliases
  "FeaturizeFn",
  "EncoderStepFn",
  "EncoderStateFn",
  "ProteinEncodeFn",
  "LigandEncodeFn",
  "ConditionalDecodeFn",
  "UnconditionalDecodeFn",
  "LogitTransformFn",
  "ARLogitTransformFn",
  # Legacy/Current Protocols
  "ConditionalLogitsFn",
  "DesignSink",
  "EncoderPostFn",
  "EncoderPreFn",
  "ModelProtocol",
  "Pipeline",
  "SamplerFn",
  "ScoreFn",
  "StateVmapExactLogitsFn",
  "StateVmapExactScoreFn",
  "UnconditionalLogitsFn",
]
