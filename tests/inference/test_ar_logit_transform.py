"""COMP-2: Tests for ar_logit_transform wired into sample_autoregressive.kernel.

RED phase — tests fail until:
  - A default ARLogitFuse implementation is added to inference/logits.py
  - sample_autoregressive.kernel calls stage_set.ar_logit_transform for
    tied-group averaging instead of inlining it
  - bundle_builder.py wires ar_logit_transform=ARLogitFuse() by default
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# 1. ARLogitFuse existence and import
# ---------------------------------------------------------------------------

def test_ar_logit_fuse_importable():
    """ARLogitFuse is importable from inference.logits."""
    from prxteinmpnn.inference.logits import ARLogitFuse  # noqa: F401


def test_ar_logit_fuse_is_eqx_module():
    """ARLogitFuse is an equinox Module (JAX pytree)."""
    import equinox as eqx
    from prxteinmpnn.inference.logits import ARLogitFuse
    assert issubclass(ARLogitFuse, eqx.Module)


# ---------------------------------------------------------------------------
# 2. ARLogitFuse callable contract
# ---------------------------------------------------------------------------

def test_ar_logit_fuse_reduces_state_dim():
    """ARLogitFuse(logits_S_V, bias_V) -> (V,) reduces the S (states) dimension."""
    from prxteinmpnn.inference.logits import ARLogitFuse

    fuse = ARLogitFuse()
    S, V = 3, 21
    logits = jnp.ones((S, V))
    bias = jnp.zeros(V)
    out = fuse(logits, bias)
    assert out.shape == (V,), f"Expected ({V},), got {out.shape}"


def test_ar_logit_fuse_default_is_mean():
    """Default ARLogitFuse applies arithmetic mean across states (dim 0), then adds bias."""
    from prxteinmpnn.inference.logits import ARLogitFuse

    fuse = ARLogitFuse()
    S, V = 4, 21
    # Make each state have distinct values so mean is non-trivial
    logits = jnp.arange(S * V, dtype=jnp.float32).reshape(S, V)
    bias = jnp.ones(V) * 2.0
    out = fuse(logits, bias)
    expected = jnp.mean(logits, axis=0) + bias
    assert jnp.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. StageSet.ar_logit_transform accepts ARLogitFuse
# ---------------------------------------------------------------------------

def test_stage_set_accepts_ar_logit_fuse():
    """StageSet can be constructed with ar_logit_transform=ARLogitFuse()."""
    from prxteinmpnn.inference.logits import ARLogitFuse
    from prxteinmpnn.types.stages import StageSet

    ss = StageSet(ar_logit_transform=ARLogitFuse())
    assert ss.ar_logit_transform is not None


def test_stage_set_ar_logit_fuse_survives_jit():
    """StageSet with ARLogitFuse passes through jax.jit as a pytree."""
    import jax
    from prxteinmpnn.inference.logits import ARLogitFuse
    from prxteinmpnn.types.stages import StageSet

    ss = StageSet(ar_logit_transform=ARLogitFuse())

    @jax.jit
    def apply(stage_set, logits, bias):
        return stage_set.ar_logit_transform(logits, bias)

    S, V = 3, 21
    logits = jnp.zeros((S, V))
    bias = jnp.zeros(V)
    out = apply(ss, logits, bias)
    assert out.shape == (V,)


# ---------------------------------------------------------------------------
# TieGroupFuseFn strategies
# ---------------------------------------------------------------------------

def test_tie_group_product_of_experts_n1_matches_log_softmax():
    """TieGroupProductOfExperts with n_tied=1 equals log_softmax of the single position.

    For a group of size 1, PoE = sum(log_softmax(logits[group])) = log_softmax(logits[pos]).
    This verifies the n_tied=1 case reduces correctly without double-softmax.
    """
    from prxteinmpnn.inference.logits import TieGroupProductOfExperts

    V, L = 21, 5
    key = jax.random.PRNGKey(0)
    logits = jax.random.normal(key, (L, V))
    mask = jnp.zeros(L, dtype=jnp.bool_).at[2].set(True)  # single position

    poe = TieGroupProductOfExperts()
    fused = poe(logits, mask)  # (V,)

    expected = jax.nn.log_softmax(logits[2])
    assert fused.shape == (V,)
    assert jnp.allclose(fused, expected, atol=1e-5), f"max diff: {jnp.max(jnp.abs(fused - expected))}"


def test_tie_group_product_of_experts_importable_from_logits():
    from prxteinmpnn.inference.logits import TieGroupProductOfExperts, TieGroupLogsumexpMean, TIE_GROUP_STRATEGIES  # noqa: F401


def test_tie_group_strategies_registry_has_both():
    from prxteinmpnn.inference.logits import TIE_GROUP_STRATEGIES
    assert TIE_GROUP_STRATEGIES.get("product_of_experts") is not None
    assert TIE_GROUP_STRATEGIES.get("logsumexp_mean") is not None


def test_stage_set_has_tie_group_fuse_slot():
    from prxteinmpnn.types.stages import StageSet
    from prxteinmpnn.inference.logits import TieGroupProductOfExperts
    ss = StageSet(tie_group_fuse=TieGroupProductOfExperts())
    assert ss.tie_group_fuse is not None


def test_bundle_builder_defaults_tie_group_fuse():
    """make_stage_set wires TieGroupProductOfExperts by default."""
    import jax.numpy as jnp
    from prxteinmpnn.inference.logits import make_stage_set, TieGroupProductOfExperts

    stage_set = make_stage_set()
    assert isinstance(stage_set.tie_group_fuse, TieGroupProductOfExperts)


