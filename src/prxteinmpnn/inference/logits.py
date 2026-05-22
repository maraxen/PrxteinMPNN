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
    Accepts optional bias to be applied after fusion.
    """
    def __call__(
        self,
        per_state: Float[Array, "S ... V"],
        bias: Float[Array, "... V"] | None = None,
    ) -> Float[Array, "... V"]:
        """Fuse per-state logits and optional bias into canonical logits.

        Parameters
        ----------
        per_state : Float[Array, "S ... V"]
            Per-state logits. S = number of states, V = vocabulary size (21 amino acids).
        bias : Float[Array, "... V"] | None
            Optional per-position logit bias to add after fusion. If None, no bias applied.

        Returns
        -------
        Float[Array, "... V"]
            Fused logits with state dimension reduced.
        """
        ...


# Registry for logit combination strategies
LOGIT_STRATEGIES: Registry[type[BatchLogitFn]] = Registry[type[BatchLogitFn]]("logit_strategies")


@LOGIT_STRATEGIES.register("arithmetic_mean")
class ArithmeticMeanLogits(eqx.Module):
    """Weighted arithmetic mean in log-space across states.

    Implements: log(Σ(wᵢ · exp(Lᵢ)) / Σwᵢ) via numerically stable logsumexp.
    Registered strategy name: ``"arithmetic_mean"``.

    Parameters
    ----------
    weights : Float[Array, "S"]
        Per-state weights. S = number of states. Traced JAX leaf.

    References
    ----------
    .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
       sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
       https://doi.org/10.1126/science.add2187
    """

    weights: Float[Array, S]

    def __call__(
        self,
        per_state: Float[Array, "S ... V"],
        bias: Float[Array, "... V"] | None = None,
    ) -> Float[Array, "... V"]:
        """Fuse per-state logits via weighted arithmetic mean in log-space.

        Parameters
        ----------
        per_state : Float[Array, "S ... V"]
            Per-state logits. S = number of states, V = vocabulary size.
        bias : Float[Array, "... V"] | None
            Optional per-position logit bias to add after fusion.

        Returns
        -------
        Float[Array, "... V"]
            Fused logits, shape ``(..., V)``.
        """
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

        result = max_logits + jnp.log(sum_exp) - log_sum_w

        if bias is not None:
            result = result + bias

        return result


@LOGIT_STRATEGIES.register("geometric_mean")
class GeometricMeanLogits(eqx.Module):
    """Weighted geometric mean across states.

    Implements: Σ(wᵢ · Lᵢ) / (T · Σwᵢ) where T is temperature.
    Registered strategy name: ``"geometric_mean"``.

    Parameters
    ----------
    weights : Float[Array, "S"]
        Per-state weights. S = number of states. Traced JAX leaf.
    temperature : float
        Temperature scaling factor. Static (not a JAX array). Default: 1.0.

    References
    ----------
    .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
       sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
       https://doi.org/10.1126/science.add2187
    """

    weights: Float[Array, S]
    temperature: float = eqx.field(static=True, default=1.0)

    def __call__(
        self,
        per_state: Float[Array, "S ... V"],
        bias: Float[Array, "... V"] | None = None,
    ) -> Float[Array, "... V"]:
        """Fuse per-state logits via weighted geometric mean.

        Parameters
        ----------
        per_state : Float[Array, "S ... V"]
            Per-state logits. S = number of states, V = vocabulary size.
        bias : Float[Array, "... V"] | None
            Optional per-position logit bias to add after fusion.

        Returns
        -------
        Float[Array, "... V"]
            Fused logits, shape ``(..., V)``.
        """
        dims_to_add = per_state.ndim - 1
        w_reshaped = self.weights.reshape((per_state.shape[0],) + (1,) * dims_to_add)

        weighted_logits = per_state * w_reshaped
        sum_weighted = jnp.sum(weighted_logits, axis=0)
        sum_w = jnp.sum(self.weights)

        result = sum_weighted / (self.temperature * sum_w)

        if bias is not None:
            result = result + bias

        return result


@LOGIT_STRATEGIES.register("product")
class ProductOfProbabilities(eqx.Module):
    """Sum of weighted log-probabilities across states.

    Implements: Σ(wᵢ · Lᵢ) — weighted sum of log-probabilities (no normalization).
    Registered strategy name: ``"product"``.

    Parameters
    ----------
    weights : Float[Array, "S"]
        Per-state weights. S = number of states. Traced JAX leaf.

    References
    ----------
    .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
       sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
       https://doi.org/10.1126/science.add2187
    """

    weights: Float[Array, S]

    def __call__(
        self,
        per_state: Float[Array, "S ... V"],
        bias: Float[Array, "... V"] | None = None,
    ) -> Float[Array, "... V"]:
        """Fuse per-state logits via weighted sum.

        Parameters
        ----------
        per_state : Float[Array, "S ... V"]
            Per-state logits. S = number of states, V = vocabulary size.
        bias : Float[Array, "... V"] | None
            Optional per-position logit bias to add after fusion.

        Returns
        -------
        Float[Array, "... V"]
            Fused logits, shape ``(..., V)``.
        """
        dims_to_add = per_state.ndim - 1
        w_reshaped = self.weights.reshape((per_state.shape[0],) + (1,) * dims_to_add)

        weighted_logits = per_state * w_reshaped
        result = jnp.sum(weighted_logits, axis=0)

        if bias is not None:
            result = result + bias

        return result


class ARLogitFuse(eqx.Module):
    """Default per-step logit fuser for autoregressive decode.

    Reduces per-state logits to a single canonical set via arithmetic mean,
    then applies bias for sampling/scoring. Called once per position during
    autoregressive decode.

    Notes
    -----
    Bias is always a concrete array (use ``jnp.zeros(V)`` for no-op).
    This is used in ``sample_autoregressive.kernel`` as the per-position fuser
    when ``stage_set.ar_logit_transform`` is set.

    References
    ----------
    .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
       sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
       https://doi.org/10.1126/science.add2187
    """

    def __call__(self, logits: Float[Array, "S V"], bias: Float[Array, V]) -> Float[Array, V]:
        """Fuse per-state logits and add bias.

        Parameters
        ----------
        logits : Float[Array, "S V"]
            Per-state logits. S = number of states, V = vocabulary size (21).
        bias : Float[Array, "V"]
            Per-position logit bias to add after fusion.

        Returns
        -------
        Float[Array, "V"]
            Fused logits with bias applied.
        """
        return jnp.mean(logits, axis=0) + bias


# ---------------------------------------------------------------------------
# Tie-group fuse strategies
# ---------------------------------------------------------------------------

from jaxtyping import Bool


@runtime_checkable
class TieGroupFuseFn(Protocol):
    """Protocol for fusing tied-position logits into a single canonical set.

    Signature: (logits: (L, V), mask: (L,)) -> (V,)
    where L is the sequence length (positions in group selected by mask)
    and V is the vocabulary size (usually 21).
    """
    def __call__(
        self,
        logits: Float[Array, "L V"],
        mask: Bool[Array, L],
    ) -> Float[Array, V]:
        """Fuse tied-position logits into a single canonical set.

        Parameters
        ----------
        logits : Float[Array, "L V"]
            Per-position logits for all positions in the group. L = sequence length.
        mask : Bool[Array, "L"]
            Boolean mask selecting tied positions (True = in group).

        Returns
        -------
        Float[Array, "V"]
            Fused logits for the tied group.
        """
        ...


TIE_GROUP_STRATEGIES: Registry = Registry("tie_group_strategies")


@TIE_GROUP_STRATEGIES.register("logsumexp_mean")
class TieGroupLogsumexpMean(eqx.Module):
    """Logsumexp mean across tied positions (legacy behavior).

    Computes: ``logsumexp(logits[group], axis=0) - log(n_tied)``.
    Registered strategy name: ``"logsumexp_mean"``.
    """
    def __call__(
        self,
        logits: Float[Array, "L V"],
        mask: Bool[Array, L],
    ) -> Float[Array, V]:
        """Fuse tied-position logits via logsumexp mean.

        Parameters
        ----------
        logits : Float[Array, "L V"]
            Per-position logits for all positions in the group.
        mask : Bool[Array, "L"]
            Boolean mask selecting tied positions.

        Returns
        -------
        Float[Array, "V"]
            Fused logits for the tied group.
        """
        group = jnp.where(mask[:, None], logits, -jnp.inf)
        n = jnp.sum(mask)
        return jax.scipy.special.logsumexp(group, axis=0) - jnp.log(jnp.maximum(n, 1))


@TIE_GROUP_STRATEGIES.register("product_of_experts")
class TieGroupProductOfExperts(eqx.Module):
    """Product-of-experts across tied positions (matches LigandMPNN reference).

    Computes: ``sum(log_softmax(logits[group]), axis=0)``.
    Registered strategy name: ``"product_of_experts"``.
    This matches the PyTorch reference's ``_combine_reference_tied_log_probs``.

    References
    ----------
    .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
       sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
       https://doi.org/10.1038/s41592-025-02626-1

    .. [LigandMPNN-code] Dauparas, J. LigandMPNN source code (commit 3870631).
       https://github.com/dauparas/LigandMPNN
    """
    def __call__(
        self,
        logits: Float[Array, "L V"],
        mask: Bool[Array, L],
    ) -> Float[Array, V]:
        """Fuse tied-position logits via product-of-experts.

        Parameters
        ----------
        logits : Float[Array, "L V"]
            Per-position logits for all positions in the group.
        mask : Bool[Array, "L"]
            Boolean mask selecting tied positions.

        Returns
        -------
        Float[Array, "V"]
            Fused logits for the tied group.
        """
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return jnp.sum(jnp.where(mask[:, None], log_probs, 0.0), axis=0)


__all__ = [
    "LOGIT_STRATEGIES",
    "TIE_GROUP_STRATEGIES",
    "ARLogitFuse",
    "ArithmeticMeanLogits",
    "BatchLogitFn",
    "GeometricMeanLogits",
    "ProductOfProbabilities",
    "TieGroupFuseFn",
    "TieGroupLogsumexpMean",
    "TieGroupProductOfExperts",
]
