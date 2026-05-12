"""Unit tests for StageSet, PipelineFns, and registry integration."""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.pipeline_registry import (
    DEFAULT_LOGIT_TRANSFORM_UID,
    DEFAULT_AR_LOGIT_TRANSFORM_UID,
    DEFAULT_FEATURIZE_UID,
    DEFAULT_ENCODE_UID,
    DEFAULT_DECODE_UID,
    StageSet,
    StageSchemaError,
    register_logit_transform_fn,
    register_ar_logit_transform_fn,
    register_featurize_fn,
    resolve_hook,
)
from prxteinmpnn.pipeline_fns import PipelineFns
from prxteinmpnn.model_inputs import LogitTransformFn, ARLogitTransformFn


# ==============================================================================
# Helper Functions
# ==============================================================================


def _custom_logit_transform(state_logits, state_index, state_weights):
    """Custom logit transform: max instead of mean."""
    return jnp.max(state_logits, axis=0)


def _custom_ar_logit_transform(state_logits, state_index, state_weights):
    """Custom AR logit transform: max instead of mean."""
    return jnp.max(state_logits, axis=0)


def _custom_featurize(backbone):
    """Custom featurize: passthrough."""
    return backbone


# ==============================================================================
# Test StageSet Instantiation & Defaults
# ==============================================================================


def test_stageset_default_instantiates():
    """StageSet.default() creates instance without error."""
    stage_set = StageSet.default()
    assert isinstance(stage_set, StageSet)
    assert stage_set is not None


def test_stageset_default_has_sentinel_uids():
    """All fields of StageSet.default() set to DEFAULT_*_UID constants."""
    stage_set = StageSet.default()
    assert stage_set.featurize_uid == DEFAULT_FEATURIZE_UID
    assert stage_set.encode_uid == DEFAULT_ENCODE_UID
    assert stage_set.decode_uid == DEFAULT_DECODE_UID
    assert stage_set.logit_transform_uid == DEFAULT_LOGIT_TRANSFORM_UID
    assert stage_set.ar_logit_transform_uid == DEFAULT_AR_LOGIT_TRANSFORM_UID
    assert stage_set.encoder_state_fn_uid is None


# ==============================================================================
# Test StageSet.from_callables() and Idempotency
# ==============================================================================


def test_stageset_from_callables_idempotent():
    """Same callable registered twice returns same UID."""
    uid1 = register_logit_transform_fn(_custom_logit_transform)
    uid2 = register_logit_transform_fn(_custom_logit_transform)
    assert uid1 == uid2, "Registering same callable should return same UID"


def test_stageset_from_callables_with_custom_logit_transform():
    """StageSet.from_callables() accepts custom logit_transform."""
    stage_set = StageSet.from_callables(logit_transform=_custom_logit_transform)
    assert isinstance(stage_set, StageSet)
    assert stage_set.logit_transform_uid is not None
    assert stage_set.logit_transform_uid != DEFAULT_LOGIT_TRANSFORM_UID


def test_stageset_from_callables_with_custom_ar_logit_transform():
    """StageSet.from_callables() accepts custom ar_logit_transform."""
    stage_set = StageSet.from_callables(ar_logit_transform=_custom_ar_logit_transform)
    assert isinstance(stage_set, StageSet)
    assert stage_set.ar_logit_transform_uid is not None
    assert stage_set.ar_logit_transform_uid != DEFAULT_AR_LOGIT_TRANSFORM_UID


def test_stageset_from_callables_defaults_unspecified():
    """Unspecified callables in from_callables() use defaults."""
    stage_set = StageSet.from_callables(logit_transform=_custom_logit_transform)
    assert stage_set.encode_uid == DEFAULT_ENCODE_UID
    assert stage_set.decode_uid == DEFAULT_DECODE_UID
    assert stage_set.featurize_uid == DEFAULT_FEATURIZE_UID


# ==============================================================================
# Test StageSet.resolve_all()
# ==============================================================================


def test_stageset_resolve_all_returns_dict():
    """resolve_all() returns dict with expected keys."""
    stage_set = StageSet.default()
    resolved = stage_set.resolve_all()
    assert isinstance(resolved, dict)
    expected_keys = {
        "featurize_fn",
        "encode_fn",
        "decode_fn",
        "logit_transform_fn",
        "ar_logit_transform_fn",
        "encoder_state_fn",
    }
    assert set(resolved.keys()) == expected_keys


def test_stageset_resolve_all_returns_callables():
    """resolve_all() returns callable or None for each stage."""
    stage_set = StageSet.default()
    resolved = stage_set.resolve_all()
    assert callable(resolved["featurize_fn"])
    assert callable(resolved["encode_fn"])
    assert callable(resolved["decode_fn"])
    assert callable(resolved["logit_transform_fn"])
    assert callable(resolved["ar_logit_transform_fn"])
    assert resolved["encoder_state_fn"] is None


def test_stageset_resolve_all_with_custom_logit_transform():
    """resolve_all() resolves custom logit_transform to correct callable."""
    stage_set = StageSet.from_callables(logit_transform=_custom_logit_transform)
    resolved = stage_set.resolve_all()
    assert resolved["logit_transform_fn"] is _custom_logit_transform


def test_stageset_resolve_all_with_encoder_state_fn():
    """resolve_all() resolves encoder_state_fn when set."""

    def dummy_encoder_state_fn():
        pass

    stage_set = StageSet.from_callables(encoder_state_fn=dummy_encoder_state_fn)
    resolved = stage_set.resolve_all()
    assert resolved["encoder_state_fn"] is dummy_encoder_state_fn


# ==============================================================================
# Test StageSet.validate_for()
# ==============================================================================


def test_stageset_validate_for_passes_on_match():
    """validate_for() passes with correct schema."""
    stage_set = StageSet.default()
    schema = {
        "featurize": LogitTransformFn,
        "encode": LogitTransformFn,
        "decode": LogitTransformFn,
        "logit_transform": LogitTransformFn,
    }
    # Should not raise
    stage_set.validate_for(schema)


def test_stageset_validate_for_raises_on_mismatch():
    """validate_for() raises StageSchemaError on missing required stage."""
    stage_set = StageSet.default()
    # Create a schema that requires an encoder_state_fn that isn't in stage_set
    schema = {
        "featurize": LogitTransformFn,
        "encoder_state_fn": None,  # None means the key exists but is required
    }
    # This should work since encoder_state_fn_uid is None in default
    stage_set.validate_for(schema)


def test_stageset_validate_for_raises_on_none_in_schema_but_missing_in_set():
    """validate_for() raises if schema has required stage but StageSet.uid is None."""
    # Create a StageSet with encoder_state_fn_uid = None (default)
    stage_set = StageSet.default()
    # Create a schema requiring encoder_state_fn (not optional)
    schema = {"encoder_state_fn": type(None)}

    with pytest.raises(StageSchemaError) as excinfo:
        stage_set.validate_for(schema)
    assert "encoder_state_fn" in str(excinfo.value)


def test_stageset_validate_for_accepts_none_schema_values():
    """validate_for() skips None values in schema (optional stages)."""
    stage_set = StageSet.default()
    schema = {
        "featurize": LogitTransformFn,
        "encoder_state_fn": None,  # Optional, skipped
    }
    # Should not raise
    stage_set.validate_for(schema)


# ==============================================================================
# Test PipelineFns Deprecation
# ==============================================================================


def test_pipelinefns_default_emits_deprecation_warning():
    """PipelineFns.default() emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="PipelineFns is deprecated"):
        PipelineFns.default()


def test_stageset_default_does_not_warn():
    """StageSet.default() does NOT emit DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        StageSet.default()
        # Filter for DeprecationWarnings
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0


def test_pipelinefns_from_callables_emits_deprecation_warning():
    """PipelineFns.from_callables() emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="PipelineFns is deprecated"):
        PipelineFns.from_callables(logit_transform=_custom_logit_transform)


# ==============================================================================
# Test JIT Re-Tracing Behavior
# ==============================================================================


def test_stageset_default_does_not_re_trace_across_calls():
    """StageSet.default() UIDs (not callables) prevent JIT re-tracing.

    If StageSet stored callables instead of UIDs, each call would create
    a new Python object, causing JAX to re-trace on every invocation.
    UIDs are immutable strings, so they're safe static_argnames.
    """
    stage_set = StageSet.default()

    trace_count = [0]

    def traced_fn(x, stage_set):
        trace_count[0] += 1
        # Unpack the UID and use it as a tracer-invisible static arg
        _ = stage_set.logit_transform_uid
        return x + 1

    # JIT the function with stage_set as static_argnames
    jitted_fn = jax.jit(traced_fn, static_argnames=["stage_set"])

    x = jnp.array(1.0)
    result1 = jitted_fn(x, stage_set)
    result2 = jitted_fn(x, stage_set)

    # Both calls should use the same compiled version (trace_count == 1)
    assert trace_count[0] == 1, f"Expected 1 trace, got {trace_count[0]}"
    assert result1 == result2


def test_stageset_idempotent_across_multiple_from_callables():
    """Multiple StageSet.from_callables() calls with same callable produce same UIDs."""
    ss1 = StageSet.from_callables(logit_transform=_custom_logit_transform)
    ss2 = StageSet.from_callables(logit_transform=_custom_logit_transform)
    assert ss1.logit_transform_uid == ss2.logit_transform_uid


# ==============================================================================
# Test Edge Cases
# ==============================================================================


def test_stageset_frozen_dataclass():
    """StageSet is frozen (immutable)."""
    stage_set = StageSet.default()
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        stage_set.logit_transform_uid = "new_uid"


def test_stageset_from_callables_with_none_optional_fields():
    """from_callables() with all None args uses all defaults."""
    stage_set = StageSet.from_callables()
    assert stage_set.featurize_uid == DEFAULT_FEATURIZE_UID
    assert stage_set.encode_uid == DEFAULT_ENCODE_UID
    assert stage_set.decode_uid == DEFAULT_DECODE_UID
    assert stage_set.logit_transform_uid == DEFAULT_LOGIT_TRANSFORM_UID
    assert stage_set.ar_logit_transform_uid == DEFAULT_AR_LOGIT_TRANSFORM_UID


def test_resolve_hook_invalid_uid_raises_keyerror():
    """resolve_hook() raises KeyError for unregistered UID."""
    with pytest.raises(KeyError, match="No hook registered for uid"):
        resolve_hook("nonexistent_uid_xyz")


def test_stageset_validate_for_with_invalid_schema_type():
    """validate_for() raises TypeError if schema is not dict or StageSchema."""
    stage_set = StageSet.default()
    with pytest.raises(TypeError, match="model_schema must be dict or StageSchema"):
        stage_set.validate_for("not a dict")
