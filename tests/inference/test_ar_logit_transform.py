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
    """ARLogitFuse(logits_S_V) -> (V,) reduces the S (states) dimension."""
    from prxteinmpnn.inference.logits import ARLogitFuse

    fuse = ARLogitFuse()
    S, V = 3, 21
    logits = jnp.ones((S, V))
    out = fuse(logits)
    assert out.shape == (V,), f"Expected ({V},), got {out.shape}"


def test_ar_logit_fuse_default_is_mean():
    """Default ARLogitFuse applies arithmetic mean across states (dim 0)."""
    from prxteinmpnn.inference.logits import ARLogitFuse

    fuse = ARLogitFuse()
    S, V = 4, 21
    # Make each state have distinct values so mean is non-trivial
    logits = jnp.arange(S * V, dtype=jnp.float32).reshape(S, V)
    out = fuse(logits)
    expected = jnp.mean(logits, axis=0)
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
    def apply(stage_set, x):
        return stage_set.ar_logit_transform(x)

    S, V = 3, 21
    logits = jnp.zeros((S, V))
    out = apply(ss, logits)
    assert out.shape == (V,)


