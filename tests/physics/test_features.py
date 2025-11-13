"""Tests for physics-based node features."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from prxteinmpnn.physics.features import (
    compute_electrostatic_node_features,
    compute_electrostatic_features_batch,
)
from prxteinmpnn.utils.data_structures import ProteinTuple


def test_compute_electrostatic_node_features_shape(pqr_protein_tuple: ProteinTuple):
    """Test that the computed features have the correct shape."""
    features = compute_electrostatic_node_features(pqr_protein_tuple)
    n_residues = pqr_protein_tuple.coordinates.shape[0]
    assert features.shape == (n_residues, 5)
    assert jnp.all(jnp.isfinite(features))


def test_compute_electrostatic_node_features_no_charges(pqr_protein_tuple: ProteinTuple):
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


def deep_tuple(x):
    """Recursively convert numpy array to nested tuples."""
    if isinstance(x, np.ndarray):
        return tuple(deep_tuple(y) for y in x)
    return x


def test_compute_electrostatic_node_features_jittable(pqr_protein_tuple: ProteinTuple):
    """Test that the feature computation can be JIT compiled."""
    # Convert numpy arrays in the ProteinTuple to nested tuples to make it hashable
    # for JAX's static argument hashing mechanism.
    data_dict = pqr_protein_tuple._asdict()
    hashable_dict = {
        k: deep_tuple(v) if isinstance(v, np.ndarray) else v
        for k, v in data_dict.items()
    }
    hashable_protein_tuple = ProteinTuple(**hashable_dict)

    jitted_fn = jax.jit(
        compute_electrostatic_node_features, static_argnames="protein"
    )
    features = jitted_fn(hashable_protein_tuple)
    assert jnp.all(jnp.isfinite(features))


def test_compute_electrostatic_features_batch_shape(pqr_protein_tuple: ProteinTuple):
    """Test that the batched features have the correct shape."""
    proteins = [pqr_protein_tuple, pqr_protein_tuple]
    features, mask = compute_electrostatic_features_batch(proteins)
    n_residues = pqr_protein_tuple.coordinates.shape[0]
    assert features.shape == (2, n_residues, 5)
    assert mask.shape == (2, n_residues)
    assert jnp.all(mask == 1.0)


def test_compute_electrostatic_features_batch_padding(pqr_protein_tuple: ProteinTuple):
    """Test that padding is applied correctly."""
    proteins = [pqr_protein_tuple]
    max_length = pqr_protein_tuple.coordinates.shape[0] + 10
    features, mask = compute_electrostatic_features_batch(
        proteins, max_length=max_length
    )
    assert features.shape == (1, max_length, 5)
    assert mask.shape == (1, max_length)
    assert jnp.sum(mask) == pqr_protein_tuple.coordinates.shape[0]


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
