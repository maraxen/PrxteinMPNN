"""Tests for the alignment utilities in `align.py`."""

import chex
import jax.numpy as jnp
import pytest

from aminx.utils.align import (
    align_sequences,
    build_state_position_map,
    needleman_wunsch_alignment,
    smith_waterman,
    smith_waterman_affine,
    smith_waterman_no_gap,
)


@pytest.fixture
def sample_score_matrix():
    """Fixture for a sample score matrix."""
    return jnp.array([[2, -1, 0], [-1, 3, -2], [0, -2, 4]], dtype=jnp.float32)


class TestAlignments(chex.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, sample_score_matrix):
        self.sample_score_matrix = sample_score_matrix

    @chex.variants(with_jit=True, without_jit=True)
    def test_smith_waterman_no_gap(self):
        """Test the Smith-Waterman alignment without gap penalties."""
        masks = (jnp.array([True, True, True]), jnp.array([True, True, True]))
        align_fn = self.variant(smith_waterman_no_gap(unroll_factor=2, batch=False))
        result = align_fn(self.sample_score_matrix, masks, 1.0)
        chex.assert_shape(result, self.sample_score_matrix.shape)
        chex.assert_tree_all_finite(result)
        assert result.sum() > 0, "Sum of alignment trace should be positive."

    @chex.variants(with_jit=True, without_jit=True)
    def test_smith_waterman(self):
        """Test the Smith-Waterman alignment with gap penalties."""
        masks = (jnp.array([True, True, True]), jnp.array([True, True, True]))
        align_fn = self.variant(smith_waterman(unroll_factor=2, ninf=-1e30, batch=False))
        result = align_fn(self.sample_score_matrix, masks, gap=-1.0, temperature=1.0)
        chex.assert_shape(result, self.sample_score_matrix.shape)
        chex.assert_tree_all_finite(result)
        assert result.sum() > 0, "Sum of alignment trace should be positive."

    @chex.variants(with_jit=True, without_jit=True)
    def test_smith_waterman_affine(self):
        """Test the Smith-Waterman alignment with affine gap penalties."""
        masks = (jnp.array([True, True, True]), jnp.array([True, True, True]))
        align_fn = self.variant(smith_waterman_affine(unroll=2, ninf=-1e30, batch=False))
        result = align_fn(
            self.sample_score_matrix,
            masks,
            gap=-1.0,
            open_penalty=-2.0,
            temperature=1.0,
        )
        chex.assert_shape(result, self.sample_score_matrix.shape)
        chex.assert_tree_all_finite(result)
        assert result.sum() > 0, "Sum of alignment trace should be positive."

    @chex.variants(with_jit=True, without_jit=True)
    def test_batch_processing(self):
        """Test batch processing for alignment functions."""
        batch_score_matrices = jnp.stack(
            [self.sample_score_matrix, self.sample_score_matrix],
        )
        masks_a = jnp.array([[True, True, True], [True, True, True]])
        masks_b = jnp.array([[True, True, True], [True, True, True]])
        batch_masks = (masks_a, masks_b)

        align_fn_no_gap = self.variant(
            smith_waterman_no_gap(unroll_factor=2, batch=True),
        )
        align_fn_gap = self.variant(
            smith_waterman(unroll_factor=2, ninf=-1e30, batch=True),
        )
        align_fn_affine = self.variant(
            smith_waterman_affine(unroll=2, ninf=-1e30, batch=True),
        )
        align_fn_nw = self.variant(
            needleman_wunsch_alignment(unroll_factor=2, batch=True),
        )

        result_no_gap = align_fn_no_gap(batch_score_matrices, batch_masks, 1.0)
        result_gap = align_fn_gap(batch_score_matrices, batch_masks, -1.0, 1.0)
        result_affine = align_fn_affine(
            batch_score_matrices, batch_masks, -1.0, -2.0, 1.0,
        )
        result_nw = align_fn_nw(batch_score_matrices, batch_masks, -1.0, 1.0)

        for result in [result_no_gap, result_gap, result_affine, result_nw]:
            chex.assert_shape(result, (2, 3, 3))
            chex.assert_tree_all_finite(result)
            assert jnp.all(
                result.sum(axis=(-1, -2)) > 0,
            ), "All alignment traces should be positive."

    @chex.variants(with_jit=True, without_jit=True)
    def test_needleman_wunsch_alignment(self):
        """Test the Needleman-Wunsch alignment."""
        masks = (jnp.array([True, True, True]), jnp.array([True, True, True]))
        align_fn = self.variant(
            needleman_wunsch_alignment(unroll_factor=2, batch=False),
        )
        result = align_fn(
            self.sample_score_matrix, masks, gap_penalty=-1.0, temperature=1.0,
        )
        chex.assert_shape(result, self.sample_score_matrix.shape)
        chex.assert_tree_all_finite(result)
        assert result.sum() > 0, "Sum of alignment trace should be positive."

    @chex.variants(with_jit=True, without_jit=True)
    def test_align_sequences(self):
        """Test the align_sequences function."""
        align_fn = self.variant(align_sequences)

        # Test with two identical sequences
        seqs = jnp.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=jnp.int32)
        alignment = align_fn(seqs)
        chex.assert_shape(alignment, (1, 4, 2))
        chex.assert_trees_all_close(alignment[0, :, 0], jnp.arange(4))
        chex.assert_trees_all_close(alignment[0, :, 1], jnp.arange(4))
        chex.assert_tree_all_finite(alignment)

        # Test with a gap
        seqs = jnp.array([[0, 1, 2, 3], [0, 1, 4, 3]])
        alignment = align_fn(seqs)
        chex.assert_shape(alignment, (1, 4, 2))
        chex.assert_tree_all_finite(alignment)

        # Test with empty sequences
        seqs = jnp.array([[], []])
        alignment = align_fn(seqs)
        chex.assert_shape(alignment, (0, 0, 2))
        chex.assert_tree_all_finite(alignment)


class TestBuildStatePositionMap:
    """Tests for build_state_position_map (praxia debt #572's alignment builder).

    Host-side, eager-numpy utility -- not jit-traced, so no chex.variants here
    (matches _realign_states_to_reference's caller-side identity/gap contract in
    tests/inference/decode/test_kernel.py::TestRealignStatesToReference).
    """

    # BLOSUM62 column order (see align.py's _AA_SCORE_MATRIX docstring):
    # A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V
    _M, _K, _V, _L, _A, _D = 12, 11, 19, 10, 0, 3

    def test_identical_sequences_are_pure_identity(self):
        """Two identical native sequences map to the identity in both directions."""
        seq = jnp.array([self._M, self._K, self._V, self._L, self._A])
        seqs = jnp.stack([seq, seq])

        result = build_state_position_map(seqs, reference_state_index=0)

        chex.assert_shape(result, (2, 5))
        assert jnp.array_equal(result[0], jnp.arange(5))
        assert jnp.array_equal(result[1], jnp.arange(5))

    def test_single_insertion_recovered_reference_as_seq_a(self):
        """Reference (index 0) vs a state with one inserted residue (D at position 2)."""
        ref = jnp.array([self._M, self._K, self._V, self._L, self._A, -1])
        inserted = jnp.array([self._M, self._K, self._D, self._V, self._L, self._A])
        seqs = jnp.stack([ref, inserted])

        result = build_state_position_map(seqs, reference_state_index=0)

        assert jnp.array_equal(result[0], jnp.arange(6))
        # Reference position 2 (V) is at the inserted state's position 3, not 2.
        expected_row1 = jnp.array([0, 1, 3, 4, 5, -1])
        assert jnp.array_equal(result[1], expected_row1)

    def test_single_state_returns_identity(self):
        """Fewer than 2 states: no alignment possible, returns the identity map."""
        seqs = jnp.array([[self._M, self._K, self._V, self._L, self._A]])

        result = build_state_position_map(seqs, reference_state_index=0)

        assert jnp.array_equal(result[0], jnp.arange(5))

    def test_non_zero_reference_state_index(self):
        """Reference need not be state 0 -- its own row must still be the identity."""
        ref = jnp.array([self._M, self._K, self._V, self._L, self._A, -1])
        inserted = jnp.array([self._M, self._K, self._D, self._V, self._L, self._A])
        seqs = jnp.stack([inserted, ref])  # reference is index 1 this time

        result = build_state_position_map(seqs, reference_state_index=1)

        assert jnp.array_equal(result[1], jnp.arange(6))
        expected_row0 = jnp.array([0, 1, 3, 4, 5, -1])
        assert jnp.array_equal(result[0], expected_row0)
