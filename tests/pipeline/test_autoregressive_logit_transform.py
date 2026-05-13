"""Verify ARLogitTransformFn is threaded through the AR scan."""
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.model.ligand_mpnn import PrxteinLigandMPNN
from prxteinmpnn.model_inputs import ARLogitTransformFn
from prxteinmpnn.payloads import MultistateStackPayload, LigandStack
from prxteinmpnn.pipeline_registry import StageSet
import equinox as eqx


def _make_model():
    return eqx.tree_inference(
        PrxteinMPNN(16, 16, 16, 1, 1, 4, key=jax.random.PRNGKey(0)),
        value=True,
    )


def _make_stack(S=2, L=4):
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


def _build_wave_tables(S, L, k_neighbors=4):
    """Build wave tables for S identical states with L canonical positions."""
    import numpy as np
    from prxteinmpnn.utils.wave_parallel import compute_wave_assignments
    from prxteinmpnn.sampling.state_vmap_prep import remap_wave_positions_flat_to_local

    # One canonical position per tie group; S members per group
    tie_flat = np.tile(np.arange(L, dtype=np.int32), S)  # (S*L,)
    flat_offsets = np.array([s * L for s in range(S + 1)], dtype=np.int32)  # S+1 elements
    git = np.full((L, S), -1, dtype=np.int32)
    gvt = np.zeros((L, S), dtype=bool)
    for g in range(L):
        for s in range(S):
            git[g, s] = int(flat_offsets[s] + g)
            gvt[g, s] = True
    ca0 = np.zeros((L, 4, 3), dtype=np.float32)
    w_id, w_pos, w_gv, w_pv = compute_wave_assignments(ca0, tie_flat, git, gvt, k_neighbors=k_neighbors, n_canonical=L)
    w_loc, w_pv2 = remap_wave_positions_flat_to_local(w_pos, w_pv, flat_offsets)
    return (
        jnp.asarray(w_id, dtype=jnp.int32),
        jnp.asarray(w_loc, dtype=jnp.int32),
        jnp.asarray(w_gv),
        jnp.asarray(w_pv2),
    )


def test_ar_logit_transform_fn_changes_output():
    """ARLogitTransformFn must be on the runtime compute path — output must differ."""
    m = _make_model()
    S, L, V = 2, 4, 21
    stack = _make_stack(S=S, L=L)
    w_id, w_loc, w_gv, w_pv = _build_wave_tables(S, L, k_neighbors=4)
    ar_mask = jnp.zeros((S, L, L), dtype=jnp.float32)
    bias = jnp.zeros((S, L, 21), dtype=jnp.float32)
    key = jax.random.PRNGKey(1)

    # Run 1: default path (no transform)
    seqs_default, _ = m.sample_autoregressive_from_payload(
        key, stack, ar_mask, bias, 1.0, 0, None, w_id, w_loc, w_gv, w_pv,
    )

    # Run 2: transform that forces logits strongly toward token 0
    def always_token_zero(state_logits, state_index, state_weights):
        return jnp.zeros(V).at[0].set(1e9)

    seqs_forced, _ = m.sample_autoregressive_from_payload(
        key, stack, ar_mask, bias, 1.0, 0, None, w_id, w_loc, w_gv, w_pv,
        ar_logit_transform_fn=always_token_zero,
    )
    # With 1e9 logit on token 0, sampling must produce all token 0s (one-hot encoded as [1, 0, ...])
    # Extract the first element of one-hot sequences (token 0 indicator)
    token_0_forced = seqs_forced[..., 0]
    assert jnp.all(token_0_forced == 1.0), "Transform must force all-token-0 one-hot sequences"
    # Default path should differ (model produces non-trivial logits)
    assert not jnp.allclose(seqs_default, seqs_forced), "Default and transformed outputs must differ"


def test_ligand_ar_logit_transform_fn_accepted():
    """LigandMPNN explicit-param method must have ar_logit_transform_fn in signature."""
    import inspect
    m = eqx.tree_inference(
        PrxteinLigandMPNN(16, 16, 16, 1, 1, 4, key=jax.random.PRNGKey(0)),
        value=True,
    )
    # Inspect the explicit-param wrapper, not sample_autoregressive_from_payload which is *args/**kwargs
    sig = inspect.signature(m.sample_autoregressive_state_vmap_exact_from_payload)
    assert "ar_logit_transform_fn" in sig.parameters


def test_autoregressive_pipeline_resolves_ar_logit_transform():
    """AutoregressivePipeline must resolve and pass ar_logit_transform_fn — output must change."""
    from prxteinmpnn.pipeline.autoregressive import AutoregressivePipeline, AutoregressiveInputs
    from prxteinmpnn.payloads import WaveParallelPayload

    m = _make_model()
    S, L, V = 2, 4, 21
    stack = _make_stack(S=S, L=L)
    w_id, w_loc, w_gv, w_pv = _build_wave_tables(S, L, k_neighbors=4)
    wave = WaveParallelPayload(
        wave_group_ids=w_id,
        wave_group_positions=w_loc,
        wave_group_valid=w_gv,
        wave_position_valid=w_pv,
    )
    inputs = AutoregressiveInputs(
        stack=stack,
        wave=wave,
        autoregressive_mask_stack=jnp.zeros((S, L, L)),
        bias_stack=jnp.zeros((S, L, V)),
    )

    def always_token_zero(state_logits, state_index, state_weights):
        return jnp.zeros(V).at[0].set(1e9)

    stage_set_default = StageSet.default()
    stage_set_forced = StageSet.from_callables(ar_logit_transform=always_token_zero)
    pipeline = AutoregressivePipeline(temperature=1.0)
    key = jax.random.PRNGKey(0)

    seqs_default, _ = pipeline(m, key, inputs, stage_set=stage_set_default)
    seqs_forced, _ = pipeline(m, key, inputs, stage_set=stage_set_forced)

    # With 1e9 logit on token 0, sampling must produce all token 0s (one-hot encoded as [1, 0, ...])
    token_0_forced = seqs_forced[..., 0]
    assert jnp.all(token_0_forced == 1.0), "Forced transform must produce all-token-0 one-hot sequences"
    # Default path should differ (model produces non-trivial logits)
    assert not jnp.allclose(seqs_default, seqs_forced), "Default and transformed outputs must differ"
