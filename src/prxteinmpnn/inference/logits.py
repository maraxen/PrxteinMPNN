"""Composable logit combination strategies as BatchLogitFn implementations.

These modules operate on stacked tensors of shape (S, ..., V) where S is the 
number of states and V is the vocabulary size (usually 21). They reduce 
the S dimension to produce a single logit set.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from prxteinmpnn.registry import Registry


@runtime_checkable
class BatchLogitFn(Protocol):
    """Protocol for fusing stacked state logits into a single canonical set.
    
    Equivalent to the generic FuseFn specialized for logit tensors.
    """
    def __call__(self, per_state: Float[Array, "S ... V"]) -> Float[Array, "... V"]:
        ...


# Registry for logit combination strategies
LOGIT_STRATEGIES: Registry[type[BatchLogitFn]] = Registry[type[BatchLogitFn]]("logit_strategies")


@LOGIT_STRATEGIES.register("arithmetic_mean")
class ArithmeticMeanLogits(eqx.Module):
    """Weighted arithmetic mean in log-space across states.

    Implements: log(Σ(wᵢ · exp(Lᵢ)) / Σwᵢ) (numerically stable)
    Input:  (S, ..., V)  →  Output: (..., V)
    """

    weights: Float[Array, "S"]

    def __call__(self, per_state: Float[Array, "S ... V"]) -> Float[Array, "... V"]:
        # log_sum_exp(L_i + log(w_i)) - log(sum(w_i))
        # Safety for zero weights: log(1e-9) if weight is 0
        log_w = jnp.log(jnp.where(self.weights > 0, self.weights, 1e-9))
        
        # Add log_w to logits (broadcasting over ..., V)
        # per_state is (S, ..., V), log_w is (S,)
        # We need to reshape log_w to (S, 1, ..., 1)
        dims_to_add = per_state.ndim - 1
        log_w_reshaped = log_w.reshape((per_state.shape[0],) + (1,) * dims_to_add)
        
        weighted_logits = per_state + log_w_reshaped
        
        # Numerically stable logsumexp over S
        max_logits = jnp.max(weighted_logits, axis=0)
        shifted = weighted_logits - max_logits[None, ...]
        sum_exp = jnp.sum(jnp.exp(shifted), axis=0)
        
        log_sum_w = jnp.log(jnp.sum(self.weights))
        
        return max_logits + jnp.log(sum_exp) - log_sum_w


@LOGIT_STRATEGIES.register("geometric_mean")
class GeometricMeanLogits(eqx.Module):
    """Weighted geometric mean across states.

    Implements: Σ(wᵢ · Lᵢ) / (T · Σwᵢ)
    """

    weights: Float[Array, "S"]
    temperature: float = eqx.field(static=True, default=1.0)

    def __call__(self, per_state: Float[Array, "S ... V"]) -> Float[Array, "... V"]:
        dims_to_add = per_state.ndim - 1
        w_reshaped = self.weights.reshape((per_state.shape[0],) + (1,) * dims_to_add)
        
        weighted_logits = per_state * w_reshaped
        sum_weighted = jnp.sum(weighted_logits, axis=0)
        sum_w = jnp.sum(self.weights)
        
        return sum_weighted / (self.temperature * sum_w)


@LOGIT_STRATEGIES.register("product")
class ProductOfProbabilities(eqx.Module):
    """Sum of weighted log-probabilities across states.

    Implements: Σ(wᵢ · Lᵢ)
    """

    weights: Float[Array, "S"]

    def __call__(self, per_state: Float[Array, "S ... V"]) -> Float[Array, "... V"]:
        dims_to_add = per_state.ndim - 1
        w_reshaped = self.weights.reshape((per_state.shape[0],) + (1,) * dims_to_add)
        
        weighted_logits = per_state * w_reshaped
        return jnp.sum(weighted_logits, axis=0)
