"""Tests for STE optimization with tied positions."""

import chex
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.sampling import make_sample_sequences


@pytest.fixture
def rng_key():
    """Create a new random key for testing."""
    return jax.random.key(0)


@pytest.fixture
def model_inputs():
    """Create model inputs from a protein structure."""
    return {
        "structure_coordinates": jnp.ones((10, 37, 3)),
        "mask": jnp.ones((10,)),
        "residue_index": jnp.arange(10),
        "chain_index": jnp.zeros((10,)),
        "sequence": jnp.zeros((10,)),
    }


def test_ste_optimization_with_tied_positions_jit(model_inputs, rng_key):
    """Test that STE optimization produces identical sequences for tied positions."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = jax.jit(
        make_sample_sequences(model, sampling_strategy="straight_through"),
        static_argnames=["num_groups"],
    )
    tie_group_map = jnp.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=jnp.int32)
    seq, logits, order = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        tie_group_map=tie_group_map,
        num_groups=5,
    )

    chex.assert_trees_all_close(seq[0], seq[1])
    chex.assert_trees_all_close(seq[2], seq[3])
    chex.assert_trees_all_close(seq[4], seq[5])
    chex.assert_trees_all_close(seq[6], seq[7])
    chex.assert_trees_all_close(seq[8], seq[9])
    chex.assert_tree_all_finite((seq, logits, order))


def test_ste_optimization_with_tied_positions_no_jit(model_inputs, rng_key):
    """Test that STE optimization produces identical sequences for tied positions."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = make_sample_sequences(model, sampling_strategy="straight_through")
    tie_group_map = jnp.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=jnp.int32)
    seq, logits, order = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        tie_group_map=tie_group_map,
        num_groups=5,
    )

    chex.assert_trees_all_close(seq[0], seq[1])
    chex.assert_trees_all_close(seq[2], seq[3])
    chex.assert_trees_all_close(seq[4], seq[5])
    chex.assert_trees_all_close(seq[6], seq[7])
    chex.assert_trees_all_close(seq[8], seq[9])
    chex.assert_tree_all_finite((seq, logits, order))


def test_ste_optimization_without_tied_positions_jit(model_inputs, rng_key):
    """Test that STE optimization works without tied positions (backward compatibility)."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = jax.jit(
        make_sample_sequences(model, sampling_strategy="straight_through"),
    )
    seq, logits, order = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    chex.assert_shape(seq, (10,))
    chex.assert_tree_all_finite((seq, logits, order))


def test_ste_optimization_without_tied_positions_no_jit(model_inputs, rng_key):
    """Test that STE optimization works without tied positions (backward compatibility)."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = make_sample_sequences(model, sampling_strategy="straight_through")
    seq, logits, order = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    chex.assert_shape(seq, (10,))
    chex.assert_tree_all_finite((seq, logits, order))


def test_ste_tied_logits_remain_identical():
    """Test that logits for tied positions remain identical throughout optimization."""
    # This is more of a unit test for the averaging logic
    n_residues = 6
    num_classes = 21

    # Create tie_group_map: [0, 0, 1, 1, 2, 2] - 3 groups of 2 positions each
    tie_group_map = jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
    num_groups = 3

    # Create logits with some variation
    logits = jax.random.normal(jax.random.key(0), (n_residues, num_classes))

    # Apply the averaging logic (from ste_optimize.py)
    group_one_hot = jax.nn.one_hot(tie_group_map, num_groups, dtype=jnp.float32)
    group_logit_sums = jnp.einsum("ng,na->ga", group_one_hot, logits)
    group_counts = group_one_hot.sum(axis=0)
    group_avg_logits = group_logit_sums / (group_counts[:, None] + 1e-8)
    averaged_logits = jnp.einsum("ng,ga->na", group_one_hot, group_avg_logits)

    # Check that positions in the same group have identical logits
    chex.assert_trees_all_close(averaged_logits[0], averaged_logits[1])
    chex.assert_trees_all_close(averaged_logits[2], averaged_logits[3])
    chex.assert_trees_all_close(averaged_logits[4], averaged_logits[5])

    # Check that different groups have different logits (probabilistic but very likely)
    assert not jnp.allclose(averaged_logits[0], averaged_logits[2])
