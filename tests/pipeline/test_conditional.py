"""Tests for logit_transform_fn wiring on conditional scoring path."""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.payloads import MultistateStackPayload


def _make_model():
    return PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))


def _make_stack(S=2, L=6):
    return MultistateStackPayload(
        coords_stack=jnp.zeros((S, L, 4, 3)),
        mask_stack=jnp.ones((S, L)),
        residue_index_stack=jnp.arange(L, dtype=jnp.int32)[None].repeat(S, axis=0),
        chain_index_stack=jnp.zeros((S, L), dtype=jnp.int32),
        tie_group_map_stack=jnp.zeros((S, L), dtype=jnp.int32),
        fixed_mask_stack=jnp.zeros((S, L)),
        fixed_tokens_stack=jnp.zeros((S, L), dtype=jnp.int32),
        state_flat_rows=jnp.stack([jnp.arange(L, dtype=jnp.int32) + i * L for i in range(S)]),
        flat_row_offsets=jnp.array([0, L], dtype=jnp.int32),
        state_index=jnp.arange(S, dtype=jnp.int32),
        state_embedding=jnp.zeros((S, 1)),
        n_states=S,
        n_canonical=L,
        n_flat=S * L,
    )


def test_score_conditional_from_payload_accepts_logit_transform_fn():
    S, L, V = 2, 6, 21
    key = jax.random.PRNGKey(1)
    m = _make_model()
    stack = _make_stack(S=S, L=L)
    seq_oh = jnp.zeros((S, L, V))
    ar_mask = jnp.eye(L)[None].repeat(S, axis=0)

    call_count = []
    def counting_transform(state_logits, state_index, state_weights):
        call_count.append(1)
        return jnp.mean(state_logits, axis=0)

    logits = m.score_conditional_state_vmap_exact_from_payload(
        key,
        stack,
        seq_oh_stack=seq_oh,
        ar_mask_stack=ar_mask,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        multi_state_temperature=1.0,
        state_weights=None,
        state_mapping=None,
        logit_transform_fn=counting_transform,
    )
    assert len(call_count) > 0
    assert logits.shape == (L, V)
