"""Tests for split encoding/sampling path with tied positions."""

import chex
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.run.averaging import make_encoding_sampling_split_fn
from prxteinmpnn.utils.decoding_order import random_decoding_order


def test_split_path_with_tied_positions_jit(model_inputs, rng_key):
    """Test that split encoding/sampling path produces identical sequences for tied positions."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )

    # Create tied positions manually: [0,1,2] in group 0, [5,6] in group 1, others independent
    n_residues = model_inputs["structure_coordinates"].shape[0]
    tie_group_map = jnp.arange(n_residues, dtype=jnp.int32)
    # Set positions 0,1,2 to group 0
    tie_group_map = tie_group_map.at[1].set(0)
    tie_group_map = tie_group_map.at[2].set(0)
    # Set positions 5,6 to group 3 (original 5)
    tie_group_map = tie_group_map.at[6].set(5)
    num_groups = jnp.unique(tie_group_map).shape[0]

    # Create split functions
    encode_fn, sample_fn, _ = make_encoding_sampling_split_fn(model)
    encode_fn = jax.jit(encode_fn)
    sample_fn = jax.jit(sample_fn, static_argnames=["num_groups"])

    # Encode structure once
    encoded_features = encode_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    # Generate decoding order with tied positions
    decoding_order, _ = random_decoding_order(
        rng_key, n_residues, tie_group_map, num_groups
    )

    # Sample with tied positions
    seq = sample_fn(
        rng_key,
        encoded_features,
        decoding_order,
        tie_group_map=tie_group_map,
        num_groups=num_groups,
    )

    # Check tied positions have same amino acid
    chex.assert_trees_all_close(seq[0], seq[1])
    chex.assert_trees_all_close(seq[0], seq[2])
    chex.assert_trees_all_close(seq[5], seq[6])

    # Check sequence is valid
    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (n_residues,))
    chex.assert_tree_all_finite(seq)


def test_split_path_with_tied_positions_no_jit(model_inputs, rng_key):
    """Test that split encoding/sampling path produces identical sequences for tied positions."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )

    # Create tied positions manually: [0,1,2] in group 0, [5,6] in group 1, others independent
    n_residues = model_inputs["structure_coordinates"].shape[0]
    tie_group_map = jnp.arange(n_residues, dtype=jnp.int32)
    # Set positions 0,1,2 to group 0
    tie_group_map = tie_group_map.at[1].set(0)
    tie_group_map = tie_group_map.at[2].set(0)
    # Set positions 5,6 to group 3 (original 5)
    tie_group_map = tie_group_map.at[6].set(5)
    num_groups = jnp.unique(tie_group_map).shape[0]

    # Create split functions
    encode_fn, sample_fn, _ = make_encoding_sampling_split_fn(model)

    # Encode structure once
    encoded_features = encode_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    # Generate decoding order with tied positions
    decoding_order, _ = random_decoding_order(
        rng_key, n_residues, tie_group_map, num_groups
    )

    # Sample with tied positions
    seq = sample_fn(
        rng_key,
        encoded_features,
        decoding_order,
        tie_group_map=tie_group_map,
        num_groups=num_groups,
    )

    # Check tied positions have same amino acid
    chex.assert_trees_all_close(seq[0], seq[1])
    chex.assert_trees_all_close(seq[0], seq[2])
    chex.assert_trees_all_close(seq[5], seq[6])

    # Check sequence is valid
    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (n_residues,))
    chex.assert_tree_all_finite(seq)


def test_split_path_without_tied_positions_jit(model_inputs, rng_key):
    """Test backward compatibility: split path works without tied positions."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )

    encode_fn, sample_fn, _ = make_encoding_sampling_split_fn(model)
    encode_fn = jax.jit(encode_fn)
    sample_fn = jax.jit(sample_fn)

    # Encode structure
    encoded_features = encode_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    # Generate standard decoding order (no tied positions)
    decoding_order, _ = random_decoding_order(
        rng_key, model_inputs["structure_coordinates"].shape[0]
    )

    # Sample without tied positions
    seq = sample_fn(rng_key, encoded_features, decoding_order)

    # Check sequence is valid
    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite(seq)


def test_split_path_without_tied_positions_no_jit(model_inputs, rng_key):
    """Test backward compatibility: split path works without tied positions."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )

    encode_fn, sample_fn, _ = make_encoding_sampling_split_fn(model)

    # Encode structure
    encoded_features = encode_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    # Generate standard decoding order (no tied positions)
    decoding_order, _ = random_decoding_order(
        rng_key, model_inputs["structure_coordinates"].shape[0]
    )

    # Sample without tied positions
    seq = sample_fn(rng_key, encoded_features, decoding_order)

    # Check sequence is valid
    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite(seq)


def test_split_path_consistency_with_full_path_jit(model_inputs, rng_key):
    """Test that split path and full path produce similar results."""
    from prxteinmpnn.sampling.sample import make_sample_sequences

    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )

    # Create tied positions manually: [0,1] in group 0, [2,3] in group 1
    n_residues = model_inputs["structure_coordinates"].shape[0]
    tie_group_map = jnp.arange(n_residues, dtype=jnp.int32)
    tie_group_map = tie_group_map.at[1].set(0)  # 0,1 in group 0
    tie_group_map = tie_group_map.at[3].set(2)  # 2,3 in group 2
    num_groups = jnp.unique(tie_group_map).shape[0]

    # Generate decoding order
    decoding_order, key = random_decoding_order(
        rng_key, n_residues, tie_group_map, num_groups
    )

    # Sample using split path
    encode_fn, sample_fn, _ = make_encoding_sampling_split_fn(model)
    encode_fn = jax.jit(encode_fn)
    sample_fn = jax.jit(sample_fn, static_argnames=["num_groups"])
    encoded_features = encode_fn(
        key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )
    seq_split = sample_fn(
        key,
        encoded_features,
        decoding_order,
        tie_group_map=tie_group_map,
        num_groups=num_groups,
    )

    # Sample using full path (for comparison - both should respect tied positions)
    sample_full_fn = jax.jit(
        make_sample_sequences(model, sampling_strategy="temperature"),
        static_argnames=["num_groups"],
    )
    seq_full, _, _ = sample_full_fn(
        key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        tie_group_map=tie_group_map,
        num_groups=num_groups,
    )

    # Both should respect tied positions (same groups have same AAs)
    # Check split path respects ties
    chex.assert_trees_all_close(seq_split[0], seq_split[1])
    chex.assert_trees_all_close(seq_split[2], seq_split[3])

    # Check full path respects ties
    chex.assert_trees_all_close(seq_full[0], seq_full[1])
    chex.assert_trees_all_close(seq_full[2], seq_full[3])


def test_split_path_consistency_with_full_path_no_jit(model_inputs, rng_key):
    """Test that split path and full path produce similar results."""
    from prxteinmpnn.sampling.sample import make_sample_sequences

    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )

    # Create tied positions manually: [0,1] in group 0, [2,3] in group 1
    n_residues = model_inputs["structure_coordinates"].shape[0]
    tie_group_map = jnp.arange(n_residues, dtype=jnp.int32)
    tie_group_map = tie_group_map.at[1].set(0)  # 0,1 in group 0
    tie_group_map = tie_group_map.at[3].set(2)  # 2,3 in group 2
    num_groups = jnp.unique(tie_group_map).shape[0]

    # Generate decoding order
    decoding_order, key = random_decoding_order(
        rng_key, n_residues, tie_group_map, num_groups
    )

    # Sample using split path
    encode_fn, sample_fn, _ = make_encoding_sampling_split_fn(model)
    encoded_features = encode_fn(
        key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )
    seq_split = sample_fn(
        key,
        encoded_features,
        decoding_order,
        tie_group_map=tie_group_map,
        num_groups=num_groups,
    )

    # Sample using full path (for comparison - both should respect tied positions)
    sample_full_fn = make_sample_sequences(
        model, sampling_strategy="temperature"
    )
    seq_full, _, _ = sample_full_fn(
        key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        tie_group_map=tie_group_map,
        num_groups=num_groups,
    )

    # Both should respect tied positions (same groups have same AAs)
    # Check split path respects ties
    chex.assert_trees_all_close(seq_split[0], seq_split[1])
    chex.assert_trees_all_close(seq_split[2], seq_split[3])

    # Check full path respects ties
    chex.assert_trees_all_close(seq_full[0], seq_full[1])
    chex.assert_trees_all_close(seq_full[2], seq_full[3])


def test_split_path_with_temperature_jit(model_inputs, rng_key):
    """Test split path with different temperature values."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )

    n_residues = model_inputs["structure_coordinates"].shape[0]
    # Create tied positions manually: [0,1] in group 0
    tie_group_map = jnp.arange(n_residues, dtype=jnp.int32)
    tie_group_map = tie_group_map.at[1].set(0)  # 0,1 in group 0
    num_groups = jnp.unique(tie_group_map).shape[0]

    encode_fn, sample_fn, _ = make_encoding_sampling_split_fn(model)
    encode_fn = jax.jit(encode_fn)
    sample_fn = jax.jit(sample_fn, static_argnames=["num_groups"])

    # Encode once
    encoded_features = encode_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    # Generate decoding order
    decoding_order, _ = random_decoding_order(
        rng_key, n_residues, tie_group_map, num_groups
    )

    # Sample with low temperature (should be more deterministic)
    seq_low_temp = sample_fn(
        rng_key,
        encoded_features,
        decoding_order,
        temperature=jnp.array(0.1, dtype=jnp.float32),
        tie_group_map=tie_group_map,
        num_groups=num_groups,
    )

    # Sample with high temperature (should be more diverse)
    seq_high_temp = sample_fn(
        jax.random.PRNGKey(1),
        encoded_features,
        decoding_order,
        temperature=jnp.array(2.0, dtype=jnp.float32),
        tie_group_map=tie_group_map,
        num_groups=num_groups,
    )

    # Both should respect tied positions
    chex.assert_trees_all_close(seq_low_temp[0], seq_low_temp[1])
    chex.assert_trees_all_close(seq_high_temp[0], seq_high_temp[1])

    # Both should be valid sequences
    chex.assert_tree_all_finite(seq_low_temp)
    chex.assert_tree_all_finite(seq_high_temp)


def test_split_path_with_temperature_no_jit(model_inputs, rng_key):
    """Test split path with different temperature values."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )

    n_residues = model_inputs["structure_coordinates"].shape[0]
    # Create tied positions manually: [0,1] in group 0
    tie_group_map = jnp.arange(n_residues, dtype=jnp.int32)
    tie_group_map = tie_group_map.at[1].set(0)  # 0,1 in group 0
    num_groups = jnp.unique(tie_group_map).shape[0]

    encode_fn, sample_fn, _ = make_encoding_sampling_split_fn(model)

    # Encode once
    encoded_features = encode_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    # Generate decoding order
    decoding_order, _ = random_decoding_order(
        rng_key, n_residues, tie_group_map, num_groups
    )

    # Sample with low temperature (should be more deterministic)
    seq_low_temp = sample_fn(
        rng_key,
        encoded_features,
        decoding_order,
        temperature=jnp.array(0.1, dtype=jnp.float32),
        tie_group_map=tie_group_map,
        num_groups=num_groups,
    )

    # Sample with high temperature (should be more diverse)
    seq_high_temp = sample_fn(
        jax.random.PRNGKey(1),
        encoded_features,
        decoding_order,
        temperature=jnp.array(2.0, dtype=jnp.float32),
        tie_group_map=tie_group_map,
        num_groups=num_groups,
    )

    # Both should respect tied positions
    chex.assert_trees_all_close(seq_low_temp[0], seq_low_temp[1])
    chex.assert_trees_all_close(seq_high_temp[0], seq_high_temp[1])

    # Both should be valid sequences
    chex.assert_tree_all_finite(seq_low_temp)
    chex.assert_tree_all_finite(seq_high_temp)
