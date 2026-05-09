"""Tests for logit_transform_fn wiring on unconditional scoring path."""

import jax
import jax.numpy as jnp

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


def test_score_unconditional_from_payload_accepts_logit_transform_fn():
    """score_unconditional_state_vmap_exact_from_payload accepts logit_transform_fn."""
    key = jax.random.PRNGKey(42)
    m = _make_model()
    stack = _make_stack(S=2, L=6)

    call_count = []
    def counting_transform(state_logits, state_index, state_weights):
        call_count.append(1)
        return jnp.mean(state_logits, axis=0)

    logits = m.score_unconditional_state_vmap_exact_from_payload(
        key,
        stack,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        logit_transform_fn=counting_transform,
    )
    assert len(call_count) > 0, "logit_transform_fn must be called"
    assert logits.shape == (6, 21)  # L=6, V=21


def test_unconditional_pipeline_importable():
    from prxteinmpnn.pipeline.unconditional import UnconditionalPipeline
    assert UnconditionalPipeline is not None


def test_unconditional_pipeline_smoke():
    """UnconditionalPipeline runs and returns logits of correct shape."""
    from prxteinmpnn.pipeline.unconditional import UnconditionalPipeline
    from prxteinmpnn.pipeline_fns import PipelineFns

    S, L, V = 2, 6, 21
    key = jax.random.PRNGKey(7)
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))
    stack = MultistateStackPayload(
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

    fns = PipelineFns.default()
    pipeline = UnconditionalPipeline()
    result = pipeline(m, key, stack, fns=fns)
    logits, state_logits = result
    assert logits.shape == (L, V)
    assert state_logits.shape == (S, L, V)


def test_unconditional_pipeline_matches_direct_call():
    """UnconditionalPipeline output matches direct score_unconditional call."""
    from prxteinmpnn.pipeline.unconditional import UnconditionalPipeline
    from prxteinmpnn.pipeline_fns import PipelineFns

    S, L = 2, 6
    key = jax.random.PRNGKey(11)
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))
    stack = MultistateStackPayload(
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

    fns = PipelineFns.default()
    pipeline = UnconditionalPipeline()
    pipeline_logits, pipeline_state_logits = pipeline(m, key, stack, fns=fns)

    # Direct call with arithmetic mean
    direct_logits = m.score_unconditional_state_vmap_exact_from_payload(
        key,
        stack,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        logit_transform_fn=lambda sl, si, sw: jnp.mean(sl, axis=0),
    )
    assert jnp.allclose(pipeline_logits, direct_logits, atol=1e-5)
