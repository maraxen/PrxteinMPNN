"""ModelInputs / ModelStaticConfig type hierarchy for prxteinmpnn.

ModelInputs (eqx.Module pytrees) carry array data through JIT.
ModelStaticConfig (frozen dataclasses) carry compile-time constants as static_argnames.
LogitTransformFn (Protocol) defines the JAX-traceable post-processing contract.

Design rules:
- NO Optional[Array] on any eqx.Module field — resolve on host before JIT.
- state_embedding resolves to zeros(n_states, embed_dim) when absent.
- SamplingStaticConfig.decode_fn_uid resolves to callable via DecodeFnRegistry.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Protocol

import equinox as eqx
from jaxtyping import Array, Float, Int

from prxteinmpnn.payloads import MultistateStackPayload, WaveParallelPayload


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


class LogitTransformFn(Protocol):
  """JAX-traceable fn combining per-state logits into a single flat distribution.

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


class ARLogitTransformFn(Protocol):
  """JAX-traceable fn combining per-state logits for ONE decode position into a single vector.

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
  "ARLogitTransformFn",
  "BackboneGeometry",
  "LogitTransformFn",
  "ConditioningFeatures",
  "SamplingInputs",
  "SamplingStaticConfig",
  "ScoringInputs",
  "ScoringStaticConfig",
]
