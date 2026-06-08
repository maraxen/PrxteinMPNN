"""Tests for prxteinmpnn.host.logit_aggregation pure helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.host.logit_aggregation import (
    aggregate_logits,
    aggregate_pseudo_perplexities,
    compute_pseudo_perplexity,
    pad_to_max,
)


class TestPadToMax:
    """Test suite for pad_to_max function."""

    def test_pad_exact_length(self):
        """Exact-length arrays should be returned unchanged."""
        arr = jnp.arange(12).reshape(3, 4)
        result = pad_to_max(arr, target_len=4, axis=-1, pad_value=0)
        assert result.shape == arr.shape
        np.testing.assert_array_equal(result, arr)

    def test_pad_shorter_array(self):
        """Shorter arrays should be padded to target length."""
        arr = jnp.arange(6).reshape(2, 3)
        result = pad_to_max(arr, target_len=5, axis=-1, pad_value=0)
        assert result.shape == (2, 5)
        np.testing.assert_array_equal(result[:, :3], arr)

    def test_pad_custom_value(self):
        """Custom pad values should be applied correctly."""
        arr = jnp.ones((2, 3))
        result = pad_to_max(arr, target_len=5, axis=-1, pad_value=-1)
        assert result.shape == (2, 5)
        np.testing.assert_array_equal(result[:, :3], jnp.ones((2, 3)))

    def test_pad_negative_axis(self):
        """Negative axis should work correctly."""
        arr = jnp.arange(6).reshape(2, 3)
        result = pad_to_max(arr, target_len=5, axis=-1, pad_value=0)
        result_alt = pad_to_max(arr, target_len=5, axis=1, pad_value=0)
        np.testing.assert_array_equal(result, result_alt)

    def test_pad_axis_0(self):
        """Padding along axis 0 should work correctly."""
        arr = jnp.arange(6).reshape(2, 3)
        result = pad_to_max(arr, target_len=4, axis=0, pad_value=0)
        assert result.shape == (4, 3)


class TestAggregateLogits:
    """Test suite for aggregate_logits function."""

    def test_single_array(self):
        """Single array should be returned unchanged."""
        logits = jnp.ones((2, 4, 2, 2, 10, 21))
        result = aggregate_logits([logits])
        assert result.shape == logits.shape

    def test_same_length(self):
        """Arrays of same length should concatenate along batch axis."""
        logits1 = jnp.ones((2, 4, 2, 2, 10, 21))
        logits2 = jnp.ones((3, 4, 2, 2, 10, 21)) * 2
        result = aggregate_logits([logits1, logits2])
        assert result.shape == (5, 4, 2, 2, 10, 21)

    def test_different_seq_lengths(self):
        """Arrays with different seq_len should be padded to max."""
        logits1 = jnp.ones((2, 4, 2, 2, 8, 21))
        logits2 = jnp.ones((3, 4, 2, 2, 12, 21)) * 2
        result = aggregate_logits([logits1, logits2])
        assert result.shape == (5, 4, 2, 2, 12, 21)

    def test_explicit_max_len(self):
        """Explicit max_len should override computed max."""
        logits1 = jnp.ones((2, 4, 2, 2, 8, 21))
        logits2 = jnp.ones((3, 4, 2, 2, 10, 21))
        result = aggregate_logits([logits1, logits2], max_len=15)
        assert result.shape == (5, 4, 2, 2, 15, 21)

    def test_empty_list(self):
        """Empty list should return empty array."""
        result = aggregate_logits([])
        assert result.shape == (0,)

    def test_many_arrays(self):
        """Aggregating many arrays should work correctly."""
        arrays = [jnp.ones((1, 4, 2, 2, 5 + i, 21)) for i in range(5)]
        result = aggregate_logits(arrays)
        assert result.shape == (5, 4, 2, 2, 9, 21)


class TestComputePseudoPerplexity:
    """Test suite for compute_pseudo_perplexity function."""

    def test_uniform_logits(self):
        """With uniform logits, all sequences have equal probability."""
        batch, samples, noise, temp, seq_len = 2, 3, 2, 2, 5
        logits = jnp.zeros((batch, samples, noise, temp, seq_len, 21))
        sequences = jnp.zeros((batch, samples, noise, temp, seq_len), dtype=jnp.int32)
        result = compute_pseudo_perplexity(logits, sequences)
        assert result.shape == (batch, samples, noise, temp)

    def test_deterministic_sequences(self):
        """Deterministic sequence should have perplexity of 1.0."""
        batch, samples, noise, temp, seq_len = 1, 1, 1, 1, 8
        logits = jnp.ones((batch, samples, noise, temp, seq_len, 21)) * -100.0
        logits = logits.at[:, :, :, :, :, 0].set(100.0)
        sequences = jnp.zeros((batch, samples, noise, temp, seq_len), dtype=jnp.int32)
        result = compute_pseudo_perplexity(logits, sequences)
        assert result.shape == (batch, samples, noise, temp)

    def test_with_mask(self):
        """Masked residues should not contribute to perplexity."""
        batch, samples, noise, temp, seq_len = 2, 1, 1, 1, 5
        logits = jnp.zeros((batch, samples, noise, temp, seq_len, 21))
        sequences = jnp.zeros((batch, samples, noise, temp, seq_len), dtype=jnp.int32)
        mask = jnp.array([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        result = compute_pseudo_perplexity(logits, sequences, mask=mask)
        assert result.shape == (batch, samples, noise, temp)

    def test_no_mask_default(self):
        """Without mask, should use default behavior."""
        batch, samples, noise, temp, seq_len = 2, 1, 1, 1, 5
        logits = jnp.zeros((batch, samples, noise, temp, seq_len, 21))
        sequences = jnp.zeros((batch, samples, noise, temp, seq_len), dtype=jnp.int32)
        result = compute_pseudo_perplexity(logits, sequences, mask=None)
        assert result.shape == (batch, samples, noise, temp)

    def test_output_positive(self):
        """Perplexity should always be positive."""
        batch, samples, noise, temp, seq_len = 3, 2, 2, 2, 10
        logits = jax.random.normal(jax.random.PRNGKey(42), (batch, samples, noise, temp, seq_len, 21))
        sequences = jax.random.randint(jax.random.PRNGKey(43), (batch, samples, noise, temp, seq_len), 0, 21)
        result = compute_pseudo_perplexity(logits, sequences)
        assert jnp.all(result > 0)


class TestAggregatePseudoPerplexities:
    """Test suite for aggregate_pseudo_perplexities function."""

    def test_single(self):
        """Single array should be returned unchanged."""
        perps = jnp.ones((2, 3, 2, 2))
        result = aggregate_pseudo_perplexities([perps])
        assert result.shape == perps.shape

    def test_multiple(self):
        """Multiple arrays should concatenate along batch axis."""
        perps1 = jnp.ones((2, 3, 2, 2))
        perps2 = jnp.ones((3, 3, 2, 2)) * 2
        result = aggregate_pseudo_perplexities([perps1, perps2])
        assert result.shape == (5, 3, 2, 2)

    def test_empty(self):
        """Empty list should return empty array."""
        result = aggregate_pseudo_perplexities([])
        assert result.shape == (0,)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_pipeline(self):
        """Full pipeline should work end to end."""
        logits = jnp.ones((2, 2, 2, 2, 8, 21))
        sequences = jnp.zeros((2, 2, 2, 2, 8), dtype=jnp.int32)

        agg_logits = aggregate_logits([logits])
        perps = compute_pseudo_perplexity(agg_logits, sequences)
        assert perps.shape == (2, 2, 2, 2)

    def test_padding_preserves(self):
        """Padding should preserve original data."""
        original = jnp.arange(20).reshape(4, 5)
        padded = pad_to_max(original, target_len=10, axis=-1, pad_value=0)
        np.testing.assert_array_equal(padded[:, :5], original)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_residue(self):
        """Should handle single residue sequences."""
        logits = jnp.ones((1, 1, 1, 1, 1, 21))
        sequences = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.int32)
        result = compute_pseudo_perplexity(logits, sequences)
        assert result.shape == (1, 1, 1, 1)

    def test_large_batch(self):
        """Should handle large batch sizes."""
        logits = jnp.ones((100, 2, 2, 2, 5, 21))
        sequences = jnp.zeros((100, 2, 2, 2, 5), dtype=jnp.int32)
        result = compute_pseudo_perplexity(logits, sequences)
        assert result.shape == (100, 2, 2, 2)
