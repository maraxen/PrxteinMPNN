"""Stage protocols and type aliases for logic injection.

Tier 1 Protocols (generic, reusable):
- TransformFn[In, Out] — stateless transformation
- RollingFn[Carry, In, Out] — scan-body with init_carry
- FuseFn[PerItem, Combined] — reduce-across-axis

Tier 2 Aliases (MPNN-specific):
- FeaturizeFn, EncoderStepFn, EncoderStateFn, ProteinEncodeFn, LigandEncodeFn
- ConditionalDecodeFn, UnconditionalDecodeFn
- LogitTransformFn (as FuseFn), ARLogitTransformFn (as FuseFn)
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

import equinox as eqx
from jaxtyping import Array, Float, Int

from prxteinmpnn.inference.logits import (
    ArithmeticMeanLogits,
    BatchLogitFn,
    GeometricMeanLogits,
    ProductOfProbabilities,
    TieGroupFuseFn,
)

# Type variables for generic protocols
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
        ...


@runtime_checkable
class RollingFn(Protocol[Carry, In, Out]):
    """Carry-based scan body: (carry, input) → (carry, output).

    Used for state-accumulating transformations (e.g. encoder state threading).
    Carry structure must be fixed at JAX trace time.
    """

    def init_carry(self) -> Carry:
        ...

    def __call__(
        self, carry: Carry, state_idx: Int[Array, ""], input: In
    ) -> tuple[Carry, Out]:
        ...


@runtime_checkable
class FuseFn(Protocol[PerItem, Combined]):
    """Reduce-across-axis fusion: per_item → combined.

    Used for combining per-item (e.g. per-state) results into a single combined result.
    Examples: logit stacking and reduction, state aggregation.
    """

    def __call__(self, per_item: PerItem, bias: PerItem | None = None) -> Combined:
        ...


# Tier 2 Aliases: MPNN-Specific Type Specializations
# In practice these take specific Pytree bundles, but are typed flexibly for now.

FeaturizeFn = TransformFn[Any, Any]
EncoderStepFn = TransformFn[Any, Any]
EncoderStateFn = RollingFn[Any, Any, Any]
ProteinEncodeFn = TransformFn[Any, Any]
LigandEncodeFn = TransformFn[Any, Any]

ConditionalDecodeFn = TransformFn[Any, Any]
UnconditionalDecodeFn = TransformFn[Any, Any]

LogitTransformFn = FuseFn[Float[Array, "S L V"], Float[Array, "L V"]]
# ARLogitTransformFn: concrete signature (S, V) + (V,) -> (V,) — bias always passed, never None
# Callers provide jnp.zeros_like(bias_shape) for no-op bias
ARLogitTransformFn = FuseFn[Float[Array, "S V"], Float[Array, "V"]]


class ConditionalDecodeStep(eqx.Module):
    """Wraps decoder.call_conditional for use as a stage_set.decode_step.

    Encapsulates the conditional decoding operation in a composable module
    for clean wiring into inference kernels.
    """
    decoder: Any  # model.decoder
    w_s_embed: Any  # model.w_s_embed.weight

    def __call__(
        self,
        node_f: Any,
        edge_f: Any,
        nei: Any,
        mask: Any,
        ar_mask: Any,
        seq_oh: Any,
        *,
        key: Any,
        inference: Any
    ) -> Any:
        """Decode conditional given node features, edges, and sequence one-hot."""
        return self.decoder.call_conditional(
            node_f, edge_f, nei, mask, ar_mask, seq_oh, self.w_s_embed,
            key=key, inference=inference
        )


class UnconditionalDecodeStep(eqx.Module):
    """Wraps decoder.__call__ for unconditional scoring.

    Encapsulates the unconditional decoding operation in a composable module
    for clean wiring into inference kernels.
    """
    decoder: Any  # model.decoder

    def __call__(
        self,
        node_f: Any,
        edge_f: Any,
        nei: Any,
        mask: Any,
        *,
        key: Any,
        inference: Any
    ) -> Any:
        """Decode unconditional given node features and edges."""
        return self.decoder(node_f, edge_f, nei, mask, key=key, inference=inference)


class StageSet(eqx.Module):
    """Typed bag of composable pipeline stages.

    eqx.Module makes this a JAX PyTree — weight arrays inside
    FuseFn implementations are traced through JIT correctly.
    """

    logit_transform: (
        BatchLogitFn | None
    ) = None
    ar_logit_transform: (
        BatchLogitFn | None
    ) = None
    decode_step: (
        ConditionalDecodeStep | UnconditionalDecodeStep | None
    ) = None
    sample_step: Any | None = None  # None = scoring mode; categorical/gumbel/ste = sampling
    tie_group_fuse: TieGroupFuseFn | None = None


__all__ = [
    "TransformFn",
    "RollingFn",
    "FuseFn",
    "FeaturizeFn",
    "EncoderStepFn",
    "EncoderStateFn",
    "ProteinEncodeFn",
    "LigandEncodeFn",
    "ConditionalDecodeFn",
    "UnconditionalDecodeFn",
    "ConditionalDecodeStep",
    "UnconditionalDecodeStep",
    "LogitTransformFn",
    "ARLogitTransformFn",
    "StageSet",
    "TieGroupFuseFn",
]
