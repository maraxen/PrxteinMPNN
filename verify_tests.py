#!/usr/bin/env python3
"""Quick verification that test file is correct."""
import sys
sys.path.insert(0, 'src')

import jax.numpy as jnp
from prxteinmpnn.payloads import MultistateStackPayload

# Test 1: Basic slice
def test_slice_basic():
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
        n_states=4,
        n_canonical=6,
        n_flat=24,
    )
    sliced = s.slice(1, 2)

    assert sliced.n_states == 2, f"Expected n_states=2, got {sliced.n_states}"
    assert sliced.coords_stack.shape == (2, 6, 4, 3), f"Expected shape (2, 6, 4, 3), got {sliced.coords_stack.shape}"
    assert sliced.state_embedding.shape == (2, 1), f"Expected shape (2, 1), got {sliced.state_embedding.shape}"
    assert sliced.n_canonical == 6
    assert sliced.n_flat == 12
    print("PASS: test_slice_basic")

# Test 2: Offset rebase
def test_offsets():
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
        n_states=4,
        n_canonical=6,
        n_flat=24,
    )
    sliced = s.slice(1, 2)
    assert sliced.flat_row_offsets[0] == 0, f"First offset should be 0, got {sliced.flat_row_offsets[0]}"
    assert sliced.flat_row_offsets[1] == 6, f"Second offset should be 6, got {sliced.flat_row_offsets[1]}"
    print("PASS: test_offsets")

# Test 3: Out of range raises
def test_out_of_range():
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
        n_states=4,
        n_canonical=6,
        n_flat=24,
    )
    try:
        s.slice(-1, 1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "slice out of range" in str(e)
    print("PASS: test_out_of_range")

if __name__ == "__main__":
    test_slice_basic()
    test_offsets()
    test_out_of_range()
    print("\nAll manual tests passed!")
