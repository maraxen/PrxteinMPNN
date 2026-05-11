"""Tests for PayloadDispatcher.

Comprehensive test suite for PayloadDispatcher covering:
- Basic unconditional scoring (Task 4.1)
- PRNG key splitting and determinism (Task 4.1)
- Empty list guard (Task 4.1)
- Parity test: single-element dispatcher vs direct call (Task 4.2)
- Conditional scoring (Task 4.3)
- Edge case: mismatched list lengths (Task 4.3)
- Conditional with bias_flat_stack_list (Task 4.3)
"""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.model_inputs import BackboneGeometry, ConditioningFeatures, SamplingInputs
from prxteinmpnn.payloads import MultistateStackPayload, WaveParallelPayload
from prxteinmpnn.run._dispatcher import PayloadDispatcher


@pytest.fixture
def model():
    """Create a PrxteinMPNN instance for testing."""
    key = jax.random.PRNGKey(0)
    return PrxteinMPNN(
        node_features=16,
        edge_features=16,
        hidden_features=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=6,
        key=key,
    )


def _make_sampling_inputs(n_states=2, L=6):
    """Helper to create a SamplingInputs for testing."""
    stack = MultistateStackPayload(
        coords_stack=jnp.zeros((n_states, L, 4, 3)),
        mask_stack=jnp.ones((n_states, L)),
        residue_index_stack=jnp.zeros((n_states, L), dtype=jnp.int32),
        chain_index_stack=jnp.zeros((n_states, L), dtype=jnp.int32),
        tie_group_map_stack=jnp.zeros((n_states, L), dtype=jnp.int32),
        fixed_mask_stack=jnp.zeros((n_states, L)),
        fixed_tokens_stack=jnp.zeros((n_states, L), dtype=jnp.int32),
        state_flat_rows=jnp.zeros((n_states, L), dtype=jnp.int32),
        flat_row_offsets=jnp.arange(n_states, dtype=jnp.int32) * L,
        state_index=jnp.arange(n_states, dtype=jnp.int32),
        state_embedding=jnp.zeros((n_states, 1)),
        n_states=n_states,
        n_canonical=L,
        n_flat=n_states * L,
    )
    backbone = BackboneGeometry(
        coords=jnp.zeros((L, 4, 3)),
        mask=jnp.ones(L),
        residue_index=jnp.zeros(L, dtype=jnp.int32),
        chain_index=jnp.zeros(L, dtype=jnp.int32),
    )
    wave = WaveParallelPayload(
        wave_group_ids=jnp.zeros((L,), dtype=jnp.int32),
        wave_group_positions=jnp.zeros((L,), dtype=jnp.int32),
        wave_group_valid=jnp.ones((1,), dtype=bool),
        wave_position_valid=jnp.ones((L,), dtype=bool),
    )
    cond = ConditioningFeatures(
        fixed_tokens=jnp.zeros(L, dtype=jnp.int32),
        bias=jnp.zeros((L, 21)),
        ar_mask=jnp.eye(L),
    )
    return SamplingInputs(backbone=backbone, state_stack=stack, wave_parallel=wave, conditioning=cond)


def test_dispatcher_score_unconditional_basic(model):
    """Test basic score_unconditional dispatch with 2 structures.

    Task 4.1: Basic unconditional scoring.
    """
    dispatcher = PayloadDispatcher()
    key = jax.random.PRNGKey(0)
    stack_list = [_make_sampling_inputs().state_stack, _make_sampling_inputs().state_stack]

    results = dispatcher.score_unconditional(
        model, key, stack_list,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        inference=True,
    )

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, jax.Array) for r in results)


def test_dispatcher_key_split_determinism(model):
    """Test that identical prng_key produces identical results.

    Task 4.1: PRNG key splitting and determinism.
    """
    dispatcher = PayloadDispatcher()
    key = jax.random.PRNGKey(42)
    stack_list = [_make_sampling_inputs().state_stack]

    results1 = dispatcher.score_unconditional(
        model, key, stack_list,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        inference=True,
    )

    results2 = dispatcher.score_unconditional(
        model, key, stack_list,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        inference=True,
    )

    assert jnp.allclose(results1[0], results2[0], rtol=1e-6, atol=1e-6)


def test_dispatcher_different_keys_differ(model):
    """Test that different prng_keys produce different results when inference=False (stochastic).

    Task 4.1: Verify that different keys lead to different sampling outcomes.
    """
    dispatcher = PayloadDispatcher()
    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(999)
    stack_list = [_make_sampling_inputs().state_stack]

    results1 = dispatcher.score_unconditional(
        model, key1, stack_list,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        inference=False,
    )

    results2 = dispatcher.score_unconditional(
        model, key2, stack_list,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        inference=False,
    )

    # Results should differ because keys are different and model uses stochasticity
    assert not jnp.allclose(results1[0], results2[0], rtol=1e-6, atol=1e-6)


def test_dispatcher_empty_list_returns_empty(model):
    """Test that empty stack_list returns empty list.

    Task 4.1: Empty list guard.
    """
    dispatcher = PayloadDispatcher()
    key = jax.random.PRNGKey(0)

    results = dispatcher.score_unconditional(
        model, key, [],
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        inference=True,
    )

    assert results == []


def test_dispatcher_parity_single_vs_list(model):
    """Test that single-element dispatcher call matches direct model call.

    Task 4.2: Parity test: single-element dispatcher vs direct call.
    Verifies that PayloadDispatcher.score_unconditional with a single
    structure produces the same logits as calling the model method directly.
    """
    dispatcher = PayloadDispatcher()
    key = jax.random.PRNGKey(123)
    stack = _make_sampling_inputs().state_stack

    # Dispatcher result
    dispatcher_results = dispatcher.score_unconditional(
        model, key, [stack],
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        inference=True,
    )

    # Direct model call with manually split key (mimic dispatcher behavior)
    direct_key = jax.random.split(key, 1)[0]
    direct_logits = model.score_unconditional_from_payload(
        direct_key, stack,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        inference=True,
    )

    assert jnp.allclose(dispatcher_results[0], direct_logits, rtol=1e-6, atol=1e-6)


def test_dispatcher_conditional_basic(model):
    """Test basic score_conditional dispatch with 2 structures.

    Task 4.3: Conditional scoring.
    """
    dispatcher = PayloadDispatcher()
    key = jax.random.PRNGKey(0)
    L = 6

    stack_list = [_make_sampling_inputs(n_states=2, L=L).state_stack for _ in range(2)]
    seq_oh_list = [jnp.zeros((2, L, 21)) for _ in range(2)]
    ar_mask_list = [jnp.eye(L) for _ in range(2)]

    results = dispatcher.score_conditional(
        model, key, stack_list,
        seq_oh_stack_list=seq_oh_list,
        ar_mask_stack_list=ar_mask_list,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        inference=True,
    )

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, jax.Array) for r in results)


def test_dispatcher_mismatched_list_lengths_raises(model):
    """Test that mismatched input list lengths raise AssertionError.

    Task 4.3: Edge case: mismatched list lengths.
    """
    dispatcher = PayloadDispatcher()
    key = jax.random.PRNGKey(0)
    L = 6

    stack_list = [_make_sampling_inputs(n_states=2, L=L).state_stack]
    seq_oh_list = [jnp.zeros((2, L, 21)) for _ in range(2)]
    ar_mask_list = [jnp.eye(L)]

    with pytest.raises(AssertionError, match="List lengths must match"):
        dispatcher.score_conditional(
            model, key, stack_list,
            seq_oh_stack_list=seq_oh_list,
            ar_mask_stack_list=ar_mask_list,
            tie_group_map=None,
            multi_state_strategy_idx=0,
            state_weights=None,
            state_mapping=None,
            inference=True,
        )


def test_dispatcher_conditional_with_bias_flat(model):
    """Test score_conditional with bias_flat_stack_list provided.

    Task 4.3: Conditional with bias_flat_stack_list.
    """
    dispatcher = PayloadDispatcher()
    key = jax.random.PRNGKey(0)
    L = 6

    stack_list = [_make_sampling_inputs(n_states=2, L=L).state_stack for _ in range(2)]
    seq_oh_list = [jnp.zeros((2, L, 21)) for _ in range(2)]
    ar_mask_list = [jnp.eye(L) for _ in range(2)]
    bias_flat_list = [jnp.zeros((L,)) for _ in range(2)]

    results = dispatcher.score_conditional(
        model, key, stack_list,
        seq_oh_stack_list=seq_oh_list,
        ar_mask_stack_list=ar_mask_list,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        bias_flat_stack_list=bias_flat_list,
        inference=True,
    )

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, jax.Array) for r in results)


def test_dispatcher_conditional_empty_list_returns_empty(model):
    """Test that empty stack_list returns empty list for conditional scoring."""
    dispatcher = PayloadDispatcher()
    key = jax.random.PRNGKey(0)

    results = dispatcher.score_conditional(
        model, key, [],
        seq_oh_stack_list=[],
        ar_mask_stack_list=[],
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        inference=True,
    )

    assert results == []
