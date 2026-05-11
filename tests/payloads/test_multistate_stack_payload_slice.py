import jax.numpy as jnp
import pytest
from prxteinmpnn.payloads import MultistateStackPayload


def _make_payload(n_states, n_canonical):
    """Helper to create a MultistateStackPayload for testing."""
    return MultistateStackPayload(
        coords_stack=jnp.zeros((n_states, n_canonical, 4, 3)),
        mask_stack=jnp.ones((n_states, n_canonical)),
        residue_index_stack=jnp.zeros((n_states, n_canonical), dtype=jnp.int32),
        chain_index_stack=jnp.zeros((n_states, n_canonical), dtype=jnp.int32),
        tie_group_map_stack=jnp.zeros((n_states, n_canonical), dtype=jnp.int32),
        fixed_mask_stack=jnp.zeros((n_states, n_canonical)),
        fixed_tokens_stack=jnp.zeros((n_states, n_canonical), dtype=jnp.int32),
        state_flat_rows=jnp.zeros((n_states, n_canonical), dtype=jnp.int32),
        flat_row_offsets=jnp.arange(n_states, dtype=jnp.int32) * n_canonical,
        state_index=jnp.arange(n_states, dtype=jnp.int32),
        state_embedding=jnp.zeros((n_states, 1)),
        n_states=n_states,
        n_canonical=n_canonical,
        n_flat=n_states * n_canonical,
    )


def test_slice_basic():
    """Test basic slicing: S=4, L=6, slice [1:3]."""
    s = _make_payload(4, 6)
    sliced = s.slice(1, 2)

    assert sliced.n_states == 2
    assert sliced.coords_stack.shape == (2, 6, 4, 3)
    assert sliced.state_embedding.shape == (2, 1)
    assert sliced.n_canonical == 6
    assert sliced.n_flat == 12


def test_slice_flat_row_offsets_rebased():
    """Test that flat_row_offsets are rebased to start at 0."""
    s = _make_payload(4, 6)
    sliced = s.slice(1, 2)

    # Original offsets: [0, 6, 12, 18]
    # Slice [1:3] should give [6, 12] rebased to [0, 6]
    assert sliced.flat_row_offsets[0] == 0
    assert sliced.flat_row_offsets[1] == 6


def test_slice_n_flat_recomputed():
    """Test that n_flat is correctly recomputed."""
    s = _make_payload(4, 6)
    sliced = s.slice(2, 1)  # slice [2:3]

    assert sliced.n_flat == 1 * 6
    assert sliced.n_canonical == 6


def test_slice_out_of_range_raises():
    """Test that slicing out of range raises ValueError."""
    s = _make_payload(4, 6)

    with pytest.raises(ValueError, match="slice out of range"):
        s.slice(-1, 1)

    with pytest.raises(ValueError, match="slice out of range"):
        s.slice(4, 1)  # start=4 but n_states=4, so out of range

    with pytest.raises(ValueError, match="slice out of range"):
        s.slice(2, 3)  # start=2, count=3, total=5 > 4


def test_slice_full_is_identity():
    """Test that slice(0, n_states) produces an equivalent payload."""
    s = _make_payload(4, 6)
    sliced = s.slice(0, 4)

    # Check that all arrays are element-wise equal
    assert jnp.array_equal(sliced.coords_stack, s.coords_stack)
    assert jnp.array_equal(sliced.mask_stack, s.mask_stack)
    assert jnp.array_equal(sliced.state_index, s.state_index)
    assert sliced.n_states == s.n_states
    assert sliced.n_flat == s.n_flat
