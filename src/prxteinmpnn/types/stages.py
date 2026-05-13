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

from jaxtyping import Array, Float, Int

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
    def __call__(self, input: In) -> Out: ...


@runtime_checkable
class RollingFn(Protocol[Carry, In, Out]):
    """Carry-based scan body: (carry, input) → (carry, output).

    Used for state-accumulating transformations (e.g. encoder state threading).
    Carry structure must be fixed at JAX trace time.
    """
    def init_carry(self) -> Carry: ...
    def __call__(self, carry: Carry, state_idx: Int[Array, ""], input: In) -> tuple[Carry, Out]: ...


@runtime_checkable
class FuseFn(Protocol[PerItem, Combined]):
    """Reduce-across-axis fusion: per_item → combined.

    Used for combining per-item (e.g. per-state) results into a single combined result.
    Examples: logit stacking and reduction, state aggregation.
    """
    def __call__(self, per_item: PerItem) -> Combined: ...


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
ARLogitTransformFn = FuseFn[Float[Array, "S V"], Float[Array, "V"]]

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
    "LogitTransformFn",
    "ARLogitTransformFn",
]
