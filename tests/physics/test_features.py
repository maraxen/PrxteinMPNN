"""Tests for physics-based node features."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.physics.features import (
    compute_electrostatic_features_batch,
    compute_electrostatic_node_features,
)
from prxteinmpnn.utils.data_structures import ProteinTuple


def deep_tuple(x):
    """Recursively convert numpy arrays and lists to nested tuples."""
    if isinstance(x, np.ndarray):
        return tuple(deep_tuple(y) for y in x)
    if isinstance(x, list):
        return tuple(deep_tuple(y) for y in x)
    return x


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_compute_electrostatic_node_features_shape(
    pqr_protein_tuple: ProteinTuple, jit_compile
):
    """Test that the computed features have the correct shape."""
    data_dict = pqr_protein_tuple._asdict()
    hashable_dict = {
        k: deep_tuple(v) if isinstance(v, (np.ndarray, list)) else v
        for k, v in data_dict.items()
    }
    hashable_protein_tuple = ProteinTuple(**hashable_dict)

    fn = compute_electrostatic_node_features
    if jit_compile:
        fn = jax.jit(fn, static_argnames="protein")

    features = fn(hashable_protein_tuple)
    n_residues = pqr_protein_tuple.coordinates.shape[0]
    chex.assert_shape(features, (n_residues, 5))
    chex.assert_tree_all_finite(features)


def test_compute_electrostatic_node_features_no_charges(
    pqr_protein_tuple: ProteinTuple,
):
    """Test that a ValueError is raised if protein has no charges."""
    protein_no_charges = pqr_protein_tuple._replace(charges=None)
    with pytest.raises(ValueError, match="must have charges"):
        compute_electrostatic_node_features(protein_no_charges)


def test_compute_electrostatic_node_features_no_full_coordinates(
    pqr_protein_tuple: ProteinTuple,
):
    """Test that a ValueError is raised if protein has no full_coordinates."""
    protein_no_full_coords = pqr_protein_tuple._replace(full_coordinates=None)
    with pytest.raises(ValueError, match="must have full_coordinates"):
        compute_electrostatic_node_features(protein_no_full_coords)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_compute_electrostatic_node_features_jittable(
    pqr_protein_tuple: ProteinTuple, jit_compile
):
    """Test that the feature computation can be JIT compiled."""
    # Convert numpy arrays in the ProteinTuple to nested tuples to make it hashable
    # for JAX's static argument hashing mechanism.
    data_dict = pqr_protein_tuple._asdict()
    hashable_dict = {
        k: deep_tuple(v) if isinstance(v, (np.ndarray, list)) else v
        for k, v in data_dict.items()
    }
    hashable_protein_tuple = ProteinTuple(**hashable_dict)

    fn = compute_electrostatic_node_features
    if jit_compile:
        fn = jax.jit(fn, static_argnames="protein")
    features = fn(hashable_protein_tuple)
    chex.assert_tree_all_finite(features)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_compute_electrostatic_features_batch_shape(
    pqr_protein_tuple: ProteinTuple, jit_compile
):
    """Test that the batched features have the correct shape."""
    data_dict = pqr_protein_tuple._asdict()
    hashable_dict = {
        k: deep_tuple(v) if isinstance(v, (np.ndarray, list)) else v
        for k, v in data_dict.items()
    }
    hashable_protein_tuple = ProteinTuple(**hashable_dict)
    proteins = (hashable_protein_tuple, hashable_protein_tuple)

    fn = compute_electrostatic_features_batch
    if jit_compile:
        fn = jax.jit(fn, static_argnames="proteins")

    features, mask = fn(proteins)
    n_residues = pqr_protein_tuple.coordinates.shape[0]
    chex.assert_shape(features, (2, n_residues, 5))
    chex.assert_shape(mask, (2, n_residues))
    chex.assert_trees_all_close(mask, jnp.ones_like(mask))


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_compute_electrostatic_features_batch_padding(
    pqr_protein_tuple: ProteinTuple, jit_compile
):
    """Test that padding is applied correctly."""
    data_dict = pqr_protein_tuple._asdict()
    hashable_dict = {
        k: deep_tuple(v) if isinstance(v, (np.ndarray, list)) else v
        for k, v in data_dict.items()
    }
    hashable_protein_tuple = ProteinTuple(**hashable_dict)
    proteins = (hashable_protein_tuple,)

    n_residues = pqr_protein_tuple.coordinates.shape[0]
    max_length = n_residues + 10

    fn = compute_electrostatic_features_batch
    if jit_compile:
        fn = jax.jit(fn, static_argnames=["proteins", "max_length"])

    features, mask = fn(proteins, max_length=max_length)
    chex.assert_shape(features, (1, max_length, 5))
    chex.assert_shape(mask, (1, max_length))
    chex.assert_trees_all_close(jnp.sum(mask), n_residues)


def test_compute_electrostatic_features_batch_empty_list():
    """Test that an empty list of proteins raises a ValueError."""
    with pytest.raises(ValueError, match="Must provide at least one protein"):
        compute_electrostatic_features_batch([])


def test_compute_electrostatic_features_batch_max_length_too_small(
    pqr_protein_tuple: ProteinTuple,
):
    """Test that a small max_length raises a ValueError."""
    proteins = [pqr_protein_tuple]
    max_length = pqr_protein_tuple.coordinates.shape[0] - 1
    with pytest.raises(ValueError, match="is less than longest sequence"):
        compute_electrostatic_features_batch(proteins, max_length=max_length)
