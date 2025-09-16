"""Tests for the alignment utilities in `align.py`."""

import jax.numpy as jnp
import pytest

from prxteinmpnn.utils.align import (
    align_sequences,
    needleman_wunsch_alignment,
    smith_waterman,
    smith_waterman_affine,
    smith_waterman_no_gap,
)


@pytest.fixture
def sample_score_matrix():
    """Fixture for a sample score matrix."""
    return jnp.array([[2, -1, 0], [-1, 3, -2], [0, -2, 4]], dtype=jnp.float32)


def test_smith_waterman_no_gap(sample_score_matrix):
    """Test the Smith-Waterman alignment without gap penalties."""
    masks = (jnp.array([True, True, True]), jnp.array([True, True, True]))
    align_fn = smith_waterman_no_gap(unroll_factor=2, batch=False)
    result = align_fn(sample_score_matrix, masks, 1.0)
    assert isinstance(result, jnp.ndarray), "Result should be a JAX array."
    assert result.shape == sample_score_matrix.shape, "Result shape should match the score matrix shape."
    assert result.sum() > 0, "Sum of alignment trace should be positive."


def test_smith_waterman(sample_score_matrix):
    """Test the Smith-Waterman alignment with gap penalties."""
    masks = (jnp.array([True, True, True]), jnp.array([True, True, True]))
    align_fn = smith_waterman(unroll_factor=2, ninf=-1e30, batch=False)
    result = align_fn(sample_score_matrix, masks, gap=-1.0, temperature=1.0)
    assert isinstance(result, jnp.ndarray), "Result should be a JAX array."
    assert result.shape == sample_score_matrix.shape, "Result shape should match the score matrix shape."
    assert result.sum() > 0, "Sum of alignment trace should be positive."


def test_smith_waterman_affine(sample_score_matrix):
    """Test the Smith-Waterman alignment with affine gap penalties."""
    masks = (jnp.array([True, True, True]), jnp.array([True, True, True]))
    align_fn = smith_waterman_affine(unroll=2, ninf=-1e30, batch=False)
    result = align_fn(
        sample_score_matrix,
        masks,
        gap=-1.0,
        open_penalty=-2.0,
        temperature=1.0,
    )
    assert isinstance(result, jnp.ndarray), "Result should be a JAX array."
    assert result.shape == sample_score_matrix.shape, "Result shape should match the score matrix shape."
    assert result.sum() > 0, "Sum of alignment trace should be positive."


def test_batch_processing(sample_score_matrix):
    """Test batch processing for alignment functions."""
    batch_score_matrices = jnp.stack([sample_score_matrix, sample_score_matrix])
    masks_a = jnp.array([[True, True, True], [True, True, True]])
    masks_b = jnp.array([[True, True, True], [True, True, True]])
    batch_masks = (masks_a, masks_b)

    align_fn_no_gap = smith_waterman_no_gap(unroll_factor=2, batch=True)
    align_fn_gap = smith_waterman(unroll_factor=2, ninf=-1e30, batch=True)
    align_fn_affine = smith_waterman_affine(unroll=2, ninf=-1e30, batch=True)
    align_fn_nw = needleman_wunsch_alignment(unroll_factor=2, batch=True)

    result_no_gap = align_fn_no_gap(batch_score_matrices, batch_masks, 1.0)
    result_gap = align_fn_gap(batch_score_matrices, batch_masks, -1.0, 1.0)
    result_affine = align_fn_affine(batch_score_matrices, batch_masks, -1.0, -2.0, 1.0)
    result_nw = align_fn_nw(batch_score_matrices, batch_masks, -1.0, 1.0)

    for result in [result_no_gap, result_gap, result_affine, result_nw]:
        assert isinstance(result, jnp.ndarray), "Result should be a JAX array."
        assert result.shape == (2, 3, 3), "Result should have a batch dimension and match matrix shape."
        assert jnp.all(result.sum(axis=(-1, -2)) > 0), "All alignment traces should be positive."


def test_needleman_wunsch_alignment(sample_score_matrix):
    """Test the Needleman-Wunsch alignment."""
    masks = (jnp.array([True, True, True]), jnp.array([True, True, True]))
    align_fn = needleman_wunsch_alignment(unroll_factor=2, batch=False)
    result = align_fn(sample_score_matrix, masks, gap_penalty=-1.0, temperature=1.0)
    assert isinstance(result, jnp.ndarray), "Result should be a JAX array."
    assert result.shape == sample_score_matrix.shape, "Result shape should match the score matrix shape."
    assert result.sum() > 0, "Sum of alignment trace should be positive."


def test_align_sequences():
    """Test the align_sequences function."""
    # Test with two identical sequences
    seqs = jnp.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=jnp.int32)
    alignment = align_sequences(seqs)
    assert alignment.shape == (1, 4, 2)
    assert jnp.all(alignment[0, :, 0] == jnp.arange(4))
    assert jnp.all(alignment[0, :, 1] == jnp.arange(4))

    # Test with a gap
    seqs = jnp.array([[0, 1, 2, 3], [0, 1, 4, 3]])
    alignment = align_sequences(seqs)
    assert alignment.shape == (1, 4, 2)

    # Test with empty sequences
    seqs = jnp.array([[], []])
    alignment = align_sequences(seqs)
    assert alignment.shape == (0, 0, 2)
