#!/usr/bin/env python3
"""Verification script for MultistateStackPayload.slice()."""

import sys
import jax.numpy as jnp
from prxteinmpnn.payloads import MultistateStackPayload

try:
    # Create a test payload
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

    # Test 1: Basic slice
    sliced = s.slice(1, 2)
    assert sliced.n_states == 2, f'n_states: expected 2, got {sliced.n_states}'

    # Test 2: Shape check
    assert sliced.coords_stack.shape == (2, 6, 4, 3), f'coords_stack shape: expected (2, 6, 4, 3), got {sliced.coords_stack.shape}'

    # Test 3: Offset rebasing
    assert sliced.flat_row_offsets[0] == 0, f'offsets not rebased: {sliced.flat_row_offsets}'

    # Test 4: n_flat recomputation
    assert sliced.n_flat == 12, f'n_flat: expected 12, got {sliced.n_flat}'

    # Test 5: Error handling - negative start
    try:
        s.slice(-1, 1)
        print("FAIL: Should have raised ValueError for negative start")
        sys.exit(1)
    except ValueError:
        pass

    # Test 6: Error handling - count <= 0
    try:
        s.slice(0, 0)
        print("FAIL: Should have raised ValueError for count=0")
        sys.exit(1)
    except ValueError:
        pass

    # Test 7: Error handling - out of range
    try:
        s.slice(3, 2)  # 3 + 2 = 5 > 4
        print("FAIL: Should have raised ValueError for out of range")
        sys.exit(1)
    except ValueError:
        pass

    print('PASS: All inline tests passed')
    sys.exit(0)

except Exception as e:
    print(f'FAIL: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
