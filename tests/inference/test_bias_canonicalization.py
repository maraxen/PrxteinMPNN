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


# ---------------------------------------------------------------------------
# 2. Kernel call sites: bias passed to logit_transform, not inline
# ---------------------------------------------------------------------------

def test_score_conditional_no_standalone_bias_before_transform():
    """score_conditional does not add bias before calling logit_transform."""
    import prxteinmpnn.inference.score_conditional as sc

    source = inspect.getsource(sc)
    # Bias must not appear on a line before logit_transform as a standalone addition
    lines = source.splitlines()
    in_bias_zone = False
    for line in lines:
        stripped = line.strip()
        if "logit_transform" in stripped:
            break
        # The old pattern was: logits_stack = logits_stack + cond.bias[None, ...]
        if "cond.bias" in stripped and "+=" in stripped or (
            "cond.bias" in stripped and stripped.startswith("logits_stack")
        ):
            pytest.fail(
                f"score_conditional still adds bias before logit_transform: {line!r}"
            )


def test_score_unconditional_no_standalone_bias_before_transform():
    """score_unconditional does not add bias before calling logit_transform."""
    import prxteinmpnn.inference.score_unconditional as su

    source = inspect.getsource(su)
    lines = source.splitlines()
    for line in lines:
        stripped = line.strip()
        if "logit_transform" in stripped:
            break
        if "cond.bias" in stripped and stripped.startswith("logits_stack"):
            pytest.fail(
                f"score_unconditional still adds bias before logit_transform: {line!r}"
            )


def test_sample_ar_no_standalone_bias_after_transform():
    """sample_autoregressive does not add bias separately after logit_transform."""
    import prxteinmpnn.inference.sample_autoregressive as sar

    source = inspect.getsource(sar)
    # Old pattern: combined_logits = combined_logits + cond.bias
    assert "combined_logits + cond.bias" not in source and \
           "combined_logits += cond.bias" not in source, (
        "sample_autoregressive still has a standalone bias addition after logit_transform"
    )
