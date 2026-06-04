"""COMP-9: Tests for LOGIT_STRATEGIES + decode_registry consolidation.

RED phase — tests fail until:
  - _default_arithmetic_mean plain-function duplicate is removed from decode_registry
  - DEFAULT_DECODE_FN_UID is removed or replaced with a LOGIT_STRATEGIES-backed entry
  - LOGIT_STRATEGIES is the single source of truth for logit combination strategies
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# 1. No duplicate arithmetic mean in decode_registry
# ---------------------------------------------------------------------------

def test_default_decode_fn_uid_removed_or_backed_by_logit_strategies():
    """DEFAULT_DECODE_FN_UID is gone or backed by a LOGIT_STRATEGIES eqx.Module instance."""
    import aminx.run.decode_registry as dr

    if not hasattr(dr, "DEFAULT_DECODE_FN_UID"):
        return  # removed entirely — pass

    # If it still exists, it must resolve to an eqx.Module (not a plain function)
    import equinox as eqx
    uid = dr.DEFAULT_DECODE_FN_UID
    fn = dr.resolve_decode_fn(uid)
    assert isinstance(fn, eqx.Module), (
        f"DEFAULT_DECODE_FN_UID must resolve to an eqx.Module, got {type(fn)}"
    )


# ---------------------------------------------------------------------------
# 2. LOGIT_STRATEGIES is the single source of truth
# ---------------------------------------------------------------------------

def test_logit_strategies_has_arithmetic_mean():
    """LOGIT_STRATEGIES registry contains 'arithmetic_mean'."""
    from aminx.inference.logits import LOGIT_STRATEGIES
    cls = LOGIT_STRATEGIES.get("arithmetic_mean")
    assert cls is not None, "LOGIT_STRATEGIES must have 'arithmetic_mean'"


def test_logit_strategies_arithmetic_mean_is_eqx_module():
    """LOGIT_STRATEGIES['arithmetic_mean'] is an eqx.Module subclass."""
    import equinox as eqx
    from aminx.inference.logits import LOGIT_STRATEGIES
    cls = LOGIT_STRATEGIES.get("arithmetic_mean")
    assert issubclass(cls, eqx.Module), (
        f"ArithmeticMeanLogits must be an eqx.Module, got {cls}"
    )


# ---------------------------------------------------------------------------
# 3. decode_registry infrastructure (DecodeFnEntry, register/resolve) survives
# ---------------------------------------------------------------------------

def test_decode_registry_infrastructure_intact():
    """DecodeFnEntry, register_decode_fn, resolve_decode_fn still exist."""
    from aminx.run.decode_registry import (  # noqa: F401
        DecodeFnEntry,
        register_decode_fn,
        resolve_decode_fn,
    )


def test_register_resolve_roundtrip_with_eqx_module():
    """register_decode_fn + resolve_decode_fn roundtrip works with an eqx.Module."""
    import jax.numpy as jnp
    from aminx.inference.logits import ArithmeticMeanLogits
    from aminx.run.decode_registry import register_decode_fn, resolve_decode_fn

    weights = jnp.ones(2)
    fn = ArithmeticMeanLogits(weights=weights)
    uid = register_decode_fn(fn, name="test_arith_mean")
    resolved = resolve_decode_fn(uid)
    assert resolved is fn, "resolve_decode_fn must return the registered instance"


# ---------------------------------------------------------------------------
# 4. Error handling: unknown UID raises KeyError
# ---------------------------------------------------------------------------

def test_resolve_decode_fn_unknown_uid_raises_key_error():
    """resolve_decode_fn raises KeyError for unknown UID."""
    import pytest
    from aminx.run.decode_registry import resolve_decode_fn

    with pytest.raises(KeyError, match="No decode fn registered"):
        resolve_decode_fn("nonexistent_uid_xyz_12345")


def test_resolve_decode_fn_error_message_includes_uid():
    """KeyError message from resolve_decode_fn includes the requested UID."""
    import pytest
    from aminx.run.decode_registry import resolve_decode_fn

    uid_requested = "bad_uid_test_123"
    with pytest.raises(KeyError) as exc_info:
        resolve_decode_fn(uid_requested)

    # Error message should mention the UID and registration hint
    assert uid_requested in str(exc_info.value)
    assert "register_decode_fn" in str(exc_info.value)
