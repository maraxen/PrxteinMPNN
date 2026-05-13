"""Tests for AutoregressivePipeline."""

import jax
import jax.numpy as jnp
import numpy as np

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.payloads import MultistateStackPayload, WaveParallelPayload
from prxteinmpnn.sampling.state_vmap_prep import remap_wave_positions_flat_to_local
from prxteinmpnn.pipeline_registry import StageSet
from prxteinmpnn.utils.wave_parallel import compute_wave_assignments


def _make_stack_and_wave(S=2, L=6):
    """Build a minimal but valid MultistateStackPayload + WaveParallelPayload."""
    flat_row_offsets = np.array([i * L for i in range(S + 1)], dtype=np.int32)

    # Each canonical position maps to one flat position per state.
    tie_flat = np.tile(np.arange(L, dtype=np.int32), S)
    n_groups = L
    git = np.full((n_groups, S), -1, dtype=np.int32)
    gvt = np.zeros((n_groups, S), dtype=bool)
    for g in range(L):
        for s in range(S):
            git[g, s] = int(flat_row_offsets[s] + g)
            gvt[g, s] = True

    ca_coords = np.zeros((L, 4, 3), dtype=np.float32)
    # Perturb CA (index 1) slightly so graph-coloring works on distinct positions.
    for i in range(L):
        ca_coords[i, 1, 0] = float(i)

    w_id, w_pos, w_gv, w_pv = compute_wave_assignments(
        ca_coords, tie_flat, git, gvt, k_neighbors=min(4, L - 1), n_canonical=L
    )
    w_loc, w_pv2 = remap_wave_positions_flat_to_local(w_pos, w_pv, flat_row_offsets)

    stack = MultistateStackPayload(
        coords_stack=jnp.zeros((S, L, 4, 3)),
        mask_stack=jnp.ones((S, L)),
        residue_index_stack=jnp.arange(L, dtype=jnp.int32)[None].repeat(S, axis=0),
        chain_index_stack=jnp.zeros((S, L), dtype=jnp.int32),
        tie_group_map_stack=jnp.zeros((S, L), dtype=jnp.int32),
        fixed_mask_stack=jnp.zeros((S, L)),
        fixed_tokens_stack=jnp.zeros((S, L), dtype=jnp.int32),
        state_flat_rows=jnp.stack([jnp.arange(L, dtype=jnp.int32) + i * L for i in range(S)]),
        flat_row_offsets=jnp.asarray(flat_row_offsets),
        state_index=jnp.arange(S, dtype=jnp.int32),
        state_embedding=jnp.zeros((S, 1)),
        n_states=S,
        n_canonical=L,
        n_flat=S * L,
    )
    wave = WaveParallelPayload(
        wave_group_ids=jnp.asarray(w_id, dtype=jnp.int32),
        wave_group_positions=jnp.asarray(w_loc, dtype=jnp.int32),
        wave_group_valid=jnp.asarray(w_gv, dtype=bool),
        wave_position_valid=jnp.asarray(w_pv2, dtype=bool),
    )
    return stack, wave


def test_autoregressive_pipeline_importable():
    from prxteinmpnn.pipeline.autoregressive import AutoregressivePipeline, AutoregressiveInputs
    assert AutoregressivePipeline is not None


def test_autoregressive_pipeline_smoke():
    """AutoregressivePipeline samples sequences without error."""
    from prxteinmpnn.pipeline.autoregressive import AutoregressivePipeline, AutoregressiveInputs

    S, L = 2, 6
    key = jax.random.PRNGKey(5)
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))
    stack, wave = _make_stack_and_wave(S=S, L=L)

    inputs = AutoregressiveInputs(
        stack=stack,
        wave=wave,
        autoregressive_mask_stack=jnp.zeros((S, L, L)),
        bias_stack=jnp.zeros((S, L, 21)),
    )

    stage_set = StageSet.default()
    pipeline = AutoregressivePipeline(temperature=0.1)
    sequences, logits = pipeline(m, key, inputs, stage_set=stage_set)
    assert sequences.shape[0] == S
