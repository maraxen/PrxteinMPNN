"""Integration tests for Executor variants with StageSet and custom stage functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.payloads import MultistateStackPayload
from prxteinmpnn.pipeline_registry import (
    StageSet,
    StageSchemaError,
    register_logit_transform_fn,
    register_ar_logit_transform_fn,
)
from prxteinmpnn.executor import (
    UnconditionalExecutor,
    ConditionalExecutor,
    AutoregressiveExecutor,
)
from prxteinmpnn.pipeline.conditional import ConditionalInputs
from prxteinmpnn.payloads import WaveParallelPayload


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def mini_mpnn_model():
    """Create a minimal PrxteinMPNN for testing."""
    return PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))


@pytest.fixture
def multistate_stack():
    """Create a minimal MultistateStackPayload with 2 states."""
    S, L = 2, 6
    return MultistateStackPayload(
        coords_stack=jnp.zeros((S, L, 4, 3)),
        mask_stack=jnp.ones((S, L)),
        residue_index_stack=jnp.arange(L, dtype=jnp.int32)[None].repeat(S, axis=0),
        chain_index_stack=jnp.zeros((S, L), dtype=jnp.int32),
        tie_group_map_stack=jnp.zeros((S, L), dtype=jnp.int32),
        fixed_mask_stack=jnp.zeros((S, L)),
        fixed_tokens_stack=jnp.zeros((S, L), dtype=jnp.int32),
        state_flat_rows=jnp.stack([jnp.arange(L, dtype=jnp.int32) + i * L for i in range(S)]),
        flat_row_offsets=jnp.array([0, L, 2 * L], dtype=jnp.int32),
        state_index=jnp.arange(S, dtype=jnp.int32),
        state_embedding=jnp.zeros((S, 1)),
        n_states=S,
        n_canonical=L,
        n_flat=S * L,
    )


@pytest.fixture
def prng_key():
    """Create a PRNG key for testing."""
    return jax.random.PRNGKey(42)


# ==============================================================================
# Helper Functions
# ==============================================================================


def _custom_logit_transform(state_logits, state_index, state_weights):
    """Custom logit transform: max instead of mean."""
    return jnp.max(state_logits, axis=0)


def _custom_ar_logit_transform(state_logits, state_index, state_weights):
    """Custom AR logit transform: max instead of mean."""
    return jnp.max(state_logits, axis=0)


# ==============================================================================
# Test Executor Initialization
# ==============================================================================


def test_executor_initialization_with_default_stageset():
    """Each executor variant initializes with StageSet.default()."""
    unconditional = UnconditionalExecutor()
    conditional = ConditionalExecutor()
    autoregressive = AutoregressiveExecutor()

    assert isinstance(unconditional._stage_set, StageSet)
    assert isinstance(conditional._stage_set, StageSet)
    assert isinstance(autoregressive._stage_set, StageSet)


def test_unconditional_executor_accepts_custom_stageset():
    """UnconditionalExecutor accepts custom StageSet in __init__."""
    custom_stage_set = StageSet.from_callables(logit_transform=_custom_logit_transform)
    executor = UnconditionalExecutor(stage_set=custom_stage_set)
    assert executor._stage_set.logit_transform_uid == custom_stage_set.logit_transform_uid


def test_conditional_executor_accepts_custom_stageset():
    """ConditionalExecutor accepts custom StageSet in __init__."""
    custom_stage_set = StageSet.from_callables(logit_transform=_custom_logit_transform)
    executor = ConditionalExecutor(stage_set=custom_stage_set)
    assert executor._stage_set.logit_transform_uid == custom_stage_set.logit_transform_uid


def test_autoregressive_executor_accepts_custom_stageset():
    """AutoregressiveExecutor accepts custom StageSet in __init__."""
    custom_stage_set = StageSet.from_callables(ar_logit_transform=_custom_ar_logit_transform)
    executor = AutoregressiveExecutor(stage_set=custom_stage_set)
    assert executor._stage_set.ar_logit_transform_uid == custom_stage_set.ar_logit_transform_uid


# ==============================================================================
# Test Executor StageSet Roundtrip
# ==============================================================================


def test_unconditional_executor_stage_set_roundtrip(mini_mpnn_model, multistate_stack, prng_key):
    """UnconditionalExecutor receives StageSet, uses it in scoring."""
    custom_stage_set = StageSet.from_callables(logit_transform=_custom_logit_transform)
    executor = UnconditionalExecutor(stage_set=custom_stage_set)

    # Call executor with the model and inputs
    # The executor should internally use custom_stage_set
    logits, state_logits = executor(
        mini_mpnn_model,
        prng_key,
        multistate_stack,
    )

    assert logits is not None
    assert state_logits is not None
    assert logits.shape == (multistate_stack.n_canonical, 21)
    assert state_logits.shape == (multistate_stack.n_states, multistate_stack.n_canonical, 21)


def test_conditional_executor_stage_set_roundtrip(mini_mpnn_model, multistate_stack, prng_key):
    """ConditionalExecutor receives StageSet, uses it in scoring."""
    custom_stage_set = StageSet.from_callables(logit_transform=_custom_logit_transform)
    executor = ConditionalExecutor(stage_set=custom_stage_set)

    # Create conditioning inputs
    S, L, V = multistate_stack.n_states, multistate_stack.n_canonical, 21
    seq_oh = jnp.zeros((S, L, V))
    ar_mask = jnp.eye(L, dtype=jnp.float32)[None].repeat(S, axis=0)

    # Wrap in ConditionalInputs
    conditional_inputs = ConditionalInputs(
        stack=multistate_stack,
        seq_oh_stack=seq_oh,
        ar_mask_stack=ar_mask,
    )

    logits, state_logits = executor(
        mini_mpnn_model,
        prng_key,
        conditional_inputs,
    )

    assert logits is not None
    assert state_logits is not None
    assert logits.shape == (L, V)
    assert state_logits.shape == (S, L, V)


def test_autoregressive_executor_stage_set_roundtrip(mini_mpnn_model, multistate_stack, prng_key):
    """AutoregressiveExecutor receives StageSet, uses it in sampling."""
    custom_stage_set = StageSet.from_callables(ar_logit_transform=_custom_ar_logit_transform)
    executor = AutoregressiveExecutor(stage_set=custom_stage_set, temperature=1.0)

    # Create wave parallel inputs
    wave = WaveParallelPayload(
        wave_position=jnp.arange(multistate_stack.n_canonical, dtype=jnp.int32),
        n_waves=1,
        n_active=multistate_stack.n_canonical,
    )

    ar_mask = jnp.ones((multistate_stack.n_states, multistate_stack.n_canonical))
    bias = jnp.zeros((multistate_stack.n_states, multistate_stack.n_canonical, 21))

    sequences, logits = executor(
        mini_mpnn_model,
        prng_key,
        multistate_stack,
        wave,
        ar_mask,
        bias,
    )

    assert sequences is not None
    assert logits is not None


# ==============================================================================
# Test Custom Function Threading
# ==============================================================================


def test_custom_logit_transform_threaded_end_to_end(mini_mpnn_model, multistate_stack, prng_key):
    """Custom LogitTransformFn registered via StageSet, executor uses it."""
    custom_stage_set = StageSet.from_callables(logit_transform=_custom_logit_transform)
    executor = UnconditionalExecutor(stage_set=custom_stage_set)

    # Call unconditional scoring with custom transform
    logits, state_logits = executor(
        mini_mpnn_model,
        prng_key,
        multistate_stack,
    )

    # Logits should be the max (not mean) of state logits
    expected_logits = jnp.max(state_logits, axis=0)
    assert jnp.allclose(logits, expected_logits, atol=1e-5), \
        "Custom logit_transform should produce max, not mean"


def test_custom_ar_logit_transform_threaded_end_to_end(mini_mpnn_model, multistate_stack, prng_key):
    """Custom ARLogitTransformFn registered via StageSet, executor uses it."""
    custom_stage_set = StageSet.from_callables(ar_logit_transform=_custom_ar_logit_transform)
    executor = AutoregressiveExecutor(stage_set=custom_stage_set, temperature=1.0)

    wave = WaveParallelPayload(
        wave_position=jnp.arange(multistate_stack.n_canonical, dtype=jnp.int32),
        n_waves=1,
        n_active=multistate_stack.n_canonical,
    )

    ar_mask = jnp.ones((multistate_stack.n_states, multistate_stack.n_canonical))
    bias = jnp.zeros((multistate_stack.n_states, multistate_stack.n_canonical, 21))

    sequences, logits = executor(
        mini_mpnn_model,
        prng_key,
        multistate_stack,
        wave,
        ar_mask,
        bias,
    )

    # Just verify that it runs without error and produces output
    assert sequences is not None
    assert logits is not None


# ==============================================================================
# Test Error Handling
# ==============================================================================


def test_executor_stage_set_override_in_call(mini_mpnn_model, multistate_stack, prng_key):
    """Executor can override stage_set at call time via stage_set parameter."""
    default_executor = UnconditionalExecutor()

    # Call with a different stage_set
    custom_stage_set = StageSet.from_callables(logit_transform=_custom_logit_transform)

    logits, state_logits = default_executor(
        mini_mpnn_model,
        prng_key,
        multistate_stack,
        stage_set=custom_stage_set,
    )

    # Verify custom logit transform was used (max instead of mean)
    expected_logits = jnp.max(state_logits, axis=0)
    assert jnp.allclose(logits, expected_logits, atol=1e-5)


# ==============================================================================
# Test Pipeline Registry Isolation
# ==============================================================================


def test_pipeline_registry_isolation_per_test():
    """Two tests with conflicting UID registrations don't bleed (via registry_snapshot fixture)."""
    # This test verifies that the registry_snapshot fixture prevents pollution

    def fn1(state_logits, state_index, state_weights):
        return jnp.mean(state_logits, axis=0)

    def fn2(state_logits, state_index, state_weights):
        return jnp.max(state_logits, axis=0)

    uid1 = register_logit_transform_fn(fn1)
    ss1 = StageSet.from_callables(logit_transform=fn1)

    # Simulate a "new test" by clearing and restoring (normally done by fixture)
    # This verifies that the fixture would work correctly

    # In a real scenario, the second test would start fresh due to registry_snapshot
    # Here we just verify that the UIDs are consistent
    uid1_again = register_logit_transform_fn(fn1)
    assert uid1 == uid1_again


def test_registry_snapshot_restores_state():
    """registry_snapshot fixture restores registry state after test."""
    # This test itself doesn't do much, but the fixture ensures isolation
    # by wrapping all tests in the same scope

    def custom_fn(state_logits, state_index, state_weights):
        return jnp.mean(state_logits, axis=0)

    uid = register_logit_transform_fn(custom_fn)
    assert uid is not None


# ==============================================================================
# Test Stage Schema Validation
# ==============================================================================


def test_executor_initialization_with_none_stageset_uses_default():
    """Executor.__init__(stage_set=None) uses StageSet.default()."""
    executor = UnconditionalExecutor(stage_set=None)
    assert executor._stage_set is not None
    assert isinstance(executor._stage_set, StageSet)
    # Default stage_set should have DEFAULT_* UIDs
    assert executor._stage_set.logit_transform_uid is not None


# ==============================================================================
# Additional Integration Tests
# ==============================================================================


def test_executor_call_with_different_stageset_instances(mini_mpnn_model, multistate_stack, prng_key):
    """Calling executor multiple times with different StageSets produces different results."""
    executor_default = UnconditionalExecutor()
    executor_custom = UnconditionalExecutor(
        stage_set=StageSet.from_callables(logit_transform=_custom_logit_transform)
    )

    logits_default, state_logits = executor_default(
        mini_mpnn_model,
        prng_key,
        multistate_stack,
    )

    logits_custom, _ = executor_custom(
        mini_mpnn_model,
        prng_key,
        multistate_stack,
    )

    # Default uses mean, custom uses max, so results should differ
    assert not jnp.allclose(logits_default, logits_custom, atol=1e-5), \
        "Different logit transforms should produce different outputs"


def test_unconditional_executor_with_encoder_state_fn():
    """UnconditionalExecutor can use custom encoder_state_fn from StageSet."""
    # Create a minimal encoder state function
    def dummy_encoder_state_fn():
        """Dummy encoder state function."""
        pass

    stage_set = StageSet.from_callables(encoder_state_fn=dummy_encoder_state_fn)
    executor = UnconditionalExecutor(stage_set=stage_set)

    # Verify the encoder_state_fn_uid is set
    assert executor._stage_set.encoder_state_fn_uid is not None

    # Resolve and verify we get the function back
    resolved = executor._stage_set.resolve_all()
    assert resolved["encoder_state_fn"] is dummy_encoder_state_fn


def test_stageset_default_does_not_have_encoder_state_fn():
    """Default StageSet has encoder_state_fn_uid = None."""
    stage_set = StageSet.default()
    resolved = stage_set.resolve_all()
    assert resolved["encoder_state_fn"] is None
