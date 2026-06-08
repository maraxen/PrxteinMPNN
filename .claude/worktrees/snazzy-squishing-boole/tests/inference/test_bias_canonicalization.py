"""COMP-4: Tests for bias canonicalization into stage_set.logit_transform.

RED phase — tests fail until:
  - BatchLogitFn protocol and all implementations accept an optional bias param
  - All three kernels call logit_transform(logits, bias=cond.bias)
  - No kernel has a standalone `+ cond.bias` before logit_transform
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# 1. BatchLogitFn implementations accept bias kwarg
# ---------------------------------------------------------------------------

def test_arithmetic_mean_accepts_bias():
    """ArithmeticMeanLogits.__call__ applies bias when provided."""
    import jax.numpy as jnp
    from prxteinmpnn.inference.logits import ArithmeticMeanLogits

    S, L, V = 2, 4, 21
    weights = jnp.ones(S)
    fn = ArithmeticMeanLogits(weights=weights)
    logits = jnp.zeros((S, L, V))
    bias = jnp.ones((L, V))

    out_no_bias = fn(logits)
    out_with_bias = fn(logits, bias=bias)

    assert out_with_bias.shape == (L, V)
    assert jnp.allclose(out_with_bias, out_no_bias + bias, atol=1e-5)


def test_geometric_mean_accepts_bias():
    """GeometricMeanLogits.__call__ applies bias when provided."""
    from prxteinmpnn.inference.logits import GeometricMeanLogits

    S, L, V = 2, 4, 21
    weights = jnp.ones(S)
    fn = GeometricMeanLogits(weights=weights)
    logits = jnp.zeros((S, L, V))
    bias = jnp.ones((L, V)) * 2.0

    out_no_bias = fn(logits)
    out_with_bias = fn(logits, bias=bias)

    assert out_with_bias.shape == (L, V)
    assert jnp.allclose(out_with_bias, out_no_bias + bias, atol=1e-5)


def test_product_accepts_bias():
    """ProductOfProbabilities.__call__ applies bias when provided."""
    from prxteinmpnn.inference.logits import ProductOfProbabilities

    S, L, V = 2, 4, 21
    weights = jnp.ones(S)
    fn = ProductOfProbabilities(weights=weights)
    logits = jnp.zeros((S, L, V))
    bias = jnp.full((L, V), 3.0)

    out_no_bias = fn(logits)
    out_with_bias = fn(logits, bias=bias)

    assert out_with_bias.shape == (L, V)
    assert jnp.allclose(out_with_bias, out_no_bias + bias, atol=1e-5)


def test_implementations_accept_none_bias():
    """All logit strategies accept bias=None without error (backward compat)."""
    from prxteinmpnn.inference.logits import (
        ArithmeticMeanLogits,
        GeometricMeanLogits,
        ProductOfProbabilities,
    )

    S, L, V = 2, 4, 21
    weights = jnp.ones(S)
    logits = jnp.zeros((S, L, V))

    for cls in (ArithmeticMeanLogits, GeometricMeanLogits, ProductOfProbabilities):
        fn = cls(weights=weights)
        out = fn(logits, bias=None)
        assert out.shape == (L, V)


def test_bias_applied_after_fusion_numerically():
    """Bias is added after state fusion, not before — numerical verification."""
    from prxteinmpnn.inference.logits import ArithmeticMeanLogits

    S, L, V = 2, 4, 21
    weights = jnp.ones(S)
    fn = ArithmeticMeanLogits(weights=weights)
    logits = jnp.zeros((S, L, V))
    bias = jnp.ones((L, V))

    # Without bias
    fused_no_bias = fn(logits, bias=None)
    # With bias
    fused_with_bias = fn(logits, bias=bias)

    # Bias should increase the output
    assert jnp.allclose(fused_with_bias, fused_no_bias + bias, atol=1e-5)


