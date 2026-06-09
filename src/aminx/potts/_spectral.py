"""Spectral helpers for weighted graph Laplacians and UST edge leverage scores."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def laplacian_from_adjacency(W: Float[Array, "n n"]) -> Array:  # noqa: N803
    """Return the combinatorial Laplacian L = D - W for symmetric nonnegative weights W."""
    W = 0.5 * (W + W.T)  # noqa: N806
    d = jnp.sum(W, axis=-1)
    d_mat = jnp.diag(d)
    return d_mat - W
