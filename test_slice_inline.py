#!/usr/bin/env python3
"""Inline test for MultistateStackPayload.slice() method."""

import jax.numpy as jnp
from prxteinmpnn.payloads import MultistateStackPayload

s = MultistateStackPayload(
    coords_stack=jnp.zeros((4, 6, 4, 3)),
    mask_stack=jnp.ones((4, 6)),
    residue_index_stack=jnp.zeros((4, 6), dtype=jnp.int32),
    chain_index_stack=jnp.zeros((4, 6), dtype=jnp.int32),
    tie_group_map_stack=jnp.zeros((4, 6), dtype=jnp.int32),
    fixed_mask_stack=jnp.zeros((4, 6)),
    fixed_tokens_stack=jnp.zeros((4, 6), dtype=jnp.int32),
    state_flat_rows=jnp.zeros((4, 6), dtype=jnp.int32),
    flat_row_offsets=jnp.arange(4, dtype=jnp.int32) * 6,
    state_index=jnp.arange(4, dtype=jnp.int32),
    state_embedding=jnp.zeros((4, 1)),
    n_states=4, n_canonical=6, n_flat=24,
)

sliced = s.slice(1, 2)
assert sliced.n_states == 2, f'got {sliced.n_states}'
assert sliced.coords_stack.shape == (2, 6, 4, 3), f'got {sliced.coords_stack.shape}'
assert sliced.flat_row_offsets[0] == 0, f'offsets not rebased: {sliced.flat_row_offsets}'
assert sliced.n_flat == 12, f'n_flat wrong: {sliced.n_flat}'
print('PASS: MultistateStackPayload.slice() works correctly')
