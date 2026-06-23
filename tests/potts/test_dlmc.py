"""Tests for DLMC (Discrete Langevin Monte Carlo) sampler and LCP regularizer."""

import jax
import jax.numpy as jnp
import pytest

from aminx.potts.sampling import dlmc_sweep
from aminx.potts._trw import lcp_regularizer


def test_dlmc_sweep_3site_shape():
    """DLMC sweep on 3-site binary Potts model returns valid shape and state."""
    L, q = 3, 2
    key = jax.random.PRNGKey(0)
    J = jnp.zeros((L, L, q, q))  # no couplings → uniform
    h = jnp.zeros((L, q))
    w = jnp.zeros((L, L))
    mask = jnp.ones((L,))
    x = jnp.zeros((L,), dtype=jnp.int32)

    x_new = dlmc_sweep(key, x, h, J, w, mask, n_steps=10)

    assert x_new.shape == (L,), f"Expected shape ({L},), got {x_new.shape}"
    assert jnp.all((x_new >= 0) & (x_new < q)), \
        f"All states should be in [0, {q}), got min={jnp.min(x_new)}, max={jnp.max(x_new)}"


def test_lcp_regularizer_zero_for_zeros():
    """LCP regularizer returns zero for zero coupling matrix."""
    J = jnp.zeros((3, 3, 2, 2))
    result = float(lcp_regularizer(J))
    assert result == pytest.approx(0.0, abs=1e-10), \
        f"LCP regularizer on zero matrix should be 0, got {result}"


def test_lcp_regularizer_positive_for_nonzero():
    """LCP regularizer returns positive value for nonzero coupling matrix."""
    J = jnp.ones((3, 3, 2, 2))
    result = float(lcp_regularizer(J))
    assert result > 0.0, f"LCP regularizer on ones matrix should be positive, got {result}"
    # Verify it matches the expected formula: lambda * sum(J^2)
    expected = 1e-3 * jnp.sum(J**2)
    assert result == pytest.approx(float(expected), rel=1e-6)
