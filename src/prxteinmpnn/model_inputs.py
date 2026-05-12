"""ModelInputs / ModelStaticConfig type hierarchy for prxteinmpnn.

ModelInputs (eqx.Module pytrees) carry array data through JIT.
ModelStaticConfig (frozen dataclasses) carry compile-time constants as static_argnames.
LogitTransformFn (Protocol) defines the JAX-traceable post-processing contract.

Design rules:
- NO Optional[Array] on any eqx.Module field — resolve on host before JIT.
- state_embedding resolves to zeros(n_states, embed_dim) when absent.
- SamplingStaticConfig.decode_fn_uid resolves to callable via DecodeFnRegistry.

Tier 1 Protocols (generic, reusable across projects):
- TransformFn[In, Out] — stateless transformation
- RollingFn[Carry, In, Out] — scan-body with init_carry()
- FuseFn[PerItem, Combined] — reduce-across-axis

Tier 2 Aliases (MPNN-specific type aliases built from Tier 1):
- FeaturizeFn, EncoderStepFn, EncoderStateFn, ProteinEncodeFn, LigandEncodeFn
- ConditionalDecodeFn, UnconditionalDecodeFn
- LogitTransformFn, ARLogitTransformFn (now FuseFn specializations)
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

import equinox as eqx
from jaxtyping import Array, Float, Int

from prxteinmpnn.payloads import MultistateStackPayload, WaveParallelPayload

# ==============================================================================
# Concrete Model Input Types
# ==============================================================================


class BackboneGeometry(eqx.Module):
  """Single-state backbone geometry for encoding."""

  coords: Float[Array, ...]
  mask: Float[Array, ...]
  residue_index: Int[Array, ...]
  chain_index: Int[Array, ...]


class ConditioningFeatures(eqx.Module):
  """Fixed tokens, logit bias, and autoregressive mask for decoding."""

  fixed_tokens: Int[Array, ...]
  bias: Float[Array, ...]
  ar_mask: Float[Array, ...]


class SamplingInputs(eqx.Module):
  """Full pytree input for tied multistate autoregressive sampling.

  All sub-payloads must be resolved to concrete arrays before JIT.
  state_stack.state_embedding must be zeros((n_states, D)) when no embedding.
  """

  backbone: BackboneGeometry
  state_stack: MultistateStackPayload
  wave_parallel: WaveParallelPayload
  conditioning: ConditioningFeatures

  def slice_states(self, start: int, count: int) -> "SamplingInputs":
    """Return a SamplingInputs with state_stack sliced to [start, start+count).

    backbone, wave_parallel, and conditioning are passed through unchanged
    (they carry no n_states axis at the SamplingInputs level).
    """
    return SamplingInputs(
        backbone=self.backbone,
        state_stack=self.state_stack.slice(start, count),
        wave_parallel=self.wave_parallel,
        conditioning=self.conditioning,
    )


class ScoringInputs(eqx.Module):
  """Pytree input for sequence scoring."""

  backbone: BackboneGeometry
  sequences: Int[Array, ...]


# ==============================================================================
# Tier 1 Protocols: Generic, Reusable Abstractions
# ==============================================================================

In = TypeVar("In")
Out = TypeVar("Out")
Carry = TypeVar("Carry")
PerItem = TypeVar("PerItem")
Combined = TypeVar("Combined")


@runtime_checkable
class TransformFn(Protocol[In, Out]):
  """Stateless transformation: In → Out.

  Generic function protocol for any host- or JAX-traceable transformation.
  Examples: featurization, encoding, decoding.
  """

  def __call__(self, input: In) -> Out:
    """Apply the transformation.

    Args:
      input: Input of type In.

    Returns:
      Output of type Out.
    """
    ...


@runtime_checkable
class RollingFn(Protocol[Carry, In, Out]):
  """Carry-based scan body: (carry, input) → (carry, output).

  Used for state-accumulating transformations (e.g. encoder state threading).
  Carry structure must be fixed at JAX trace time.
  """

  def init_carry(self) -> Carry:
    """Return the initial carry pytree.

    Must return a valid JAX pytree with fixed structure/shapes at trace time.

    Returns:
      Initial carry value.
    """
    ...

  def __call__(self, carry: Carry, state_idx: Int[Array, ""], input: In) -> tuple[Carry, Out]:
    """Process one step of the scan.

    Args:
      carry: Current carry pytree.
      state_idx: Scalar traced int32 index in the sequence.
      input: Input value of type In.

    Returns:
      (new_carry, output): Updated carry and output of type Out.
    """
    ...


@runtime_checkable
class FuseFn(Protocol[PerItem, Combined]):
  """Reduce-across-axis fusion: per_item → combined.

  Used for combining per-item (e.g. per-state) results into a single combined result.
  Examples: logit stacking and reduction, state aggregation.
  """

  def __call__(self, per_item: PerItem) -> Combined:
    """Fuse per-item values into a single combined result.

    Args:
      per_item: Per-item value (often stacked across some axis).

    Returns:
      Combined result (reduced across that axis).
    """
    ...


# ==============================================================================
# Tier 2 Aliases: MPNN-Specific Type Specializations
# ==============================================================================

# Featurization and encoding aliases
FeaturizeFn = TransformFn[BackboneGeometry, Any]
"""Transform from BackboneGeometry to FeaturizedGraph."""

EncoderStepFn = TransformFn[Any, Any]
"""Transform from FeaturizedGraph to EncoderOutput (single step)."""

EncoderStateFn = RollingFn[Any, BackboneGeometry, Any]
"""Carry-based scan for encoding with cross-state accumulation."""

ProteinEncodeFn = TransformFn[Any, Any]
"""Transform from FeaturizedGraph to EncoderOutput (protein variant)."""

LigandEncodeFn = TransformFn[Any, Any]
"""Transform from FeaturizedGraph to EncoderOutput (ligand variant)."""

# Decoding aliases (asymmetric, kept distinct)
ConditionalDecodeFn = TransformFn[Any, Any]
"""Conditional decode signature (sequence-conditioned)."""

UnconditionalDecodeFn = TransformFn[Any, Any]
"""Unconditional decode signature (structure-only)."""

# Logit aggregation aliases (FuseFn specializations)
LogitTransformFn = FuseFn[Float[Array, "S L V"], Float[Array, "L V"]]
"""Reduce per-state logits (S, L, V) across states S to (L, V)."""

ARLogitTransformFn = FuseFn[Float[Array, "S V"], Float[Array, "V"]]
"""Reduce per-state AR logits (S, V) across states S to (V)."""


# ==============================================================================
# Concrete Stage Type Wrappers
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class StageInput:
  """Wrapper for stage input data (polymorphic across stages)."""

  value: Any
  """Wrapped input value (type depends on stage)."""


@dataclasses.dataclass(frozen=True)
class StageOutput:
  """Wrapper for stage output data (polymorphic across stages)."""

  value: Any
  """Wrapped output value (type depends on stage)."""


@dataclasses.dataclass(frozen=True)
class StageSchema:
  """Mapping of stage names to their expected type aliases.

  Frozen dataclass for immutability and hashability.
  Used by executors for host-time validation of StageSet against model.
  """

  featurize: type | None = None
  encode: type | None = None
  decode: type | None = None
  logit_transform: type | None = None
  ar_logit_transform: type | None = None
  encoder_state_fn: type | None = None

  def __getitem__(self, key: str) -> type | None:
    """Dict-like access for stage type by name."""
    if not hasattr(self, key):
      raise KeyError(f"Stage '{key}' not in schema")
    return getattr(self, key)


# ==============================================================================
# Static Configuration Dataclasses
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class SamplingStaticConfig:
  """Compile-time constants for sampling — passed as static_argnames to JIT."""

  decode_fn_uid: str
  n_samples: int
  temperature: float
  multistate_mode: Literal["tied", "independent"] = "tied"
  max_group_size: int = 1


@dataclasses.dataclass(frozen=True)
class ScoringStaticConfig:
  """Compile-time constants for scoring — passed as static_argnames to JIT."""

  pass_mode: Literal["unconditional", "conditional"] = "unconditional"
  ar_mask_is_eye: bool = False


# ==============================================================================
# Legacy Protocol Definitions (maintained for backwards compatibility)
# ==============================================================================
# These are now subsumed by Tier 1/2 but kept in place to avoid churn.
# New code should use FeaturizeFn, EncoderStepFn, LogitTransformFn, etc.

class _LegacyLogitTransformFn(Protocol):
  """LEGACY: JAX-traceable fn combining per-state logits into a single flat distribution.

  DEPRECATED: Use LogitTransformFn (now FuseFn[Float[Array, "S L V"], Float[Array, "L V"]]) instead.

  Passed as static_argnames to the outer JIT and inlined at jax.export time.
  Must use only jnp ops — no Python branching on traced values.

  state_weights is always a concrete array (uniform 1/S resolved on host if absent).
  """

  def __call__(
    self,
    state_logits: Float[Array, "S L V"],
    state_index: Int[Array, "S"],
    state_weights: Float[Array, "S"],
  ) -> Float[Array, "L V"]: ...


class _LegacyARLogitTransformFn(Protocol):
  """LEGACY: JAX-traceable fn combining per-state logits for ONE decode position into a single vector.

  DEPRECATED: Use ARLogitTransformFn (now FuseFn[Float[Array, "S V"], Float[Array, "V"]]) instead.

  Called per decode step inside the AR wave-parallel scan, where logits are accumulated
  one position at a time (shape (S, V)), not across the full sequence.
  Contrast with LogitTransformFn which operates on (S, L, V).

  Must use only jnp ops — no Python branching on traced values.
  state_weights is always a concrete array (uniform 1/S if absent).
  """

  def __call__(
    self,
    state_logits: Float[Array, "S V"],
    state_index: Int[Array, "S"],
    state_weights: Float[Array, "S"],
  ) -> Float[Array, "V"]: ...


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
  # Concrete Types
  "StageInput",
  "StageOutput",
  "StageSchema",
  # Input/Output & Config
  "BackboneGeometry",
  "ConditioningFeatures",
  "SamplingInputs",
  "SamplingStaticConfig",
  "ScoringInputs",
  "ScoringStaticConfig",
]
