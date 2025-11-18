"""Tests for ProteinFeatures with structure_mapping (multi-state mode).

This test suite verifies that the structure_mapping parameter correctly
prevents cross-structure neighbors in multi-state protein design scenarios.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import chex

from prxteinmpnn.model.features import ProteinFeatures

# Import helpers using relative path
import sys
from pathlib import Path

tests_dir = Path(__file__).parent.parent
sys.path.insert(0, str(tests_dir))

from helpers.multistate import (  # noqa: E402
  create_multistate_test_batch,
  create_simple_multistate_protein,
  verify_no_cross_structure_neighbors,
)

class TestFeatures(chex.TestCase):
  
  def setUp(self):
    """Create a ProteinFeatures module for testing."""
    key = jax.random.key(42)
    self.features_module = ProteinFeatures(
      node_features=128,
      edge_features=128,
      k_neighbors=30,
      key=key,
    )


  def test_features_structure_mapping_prevents_cross_neighbors(self):
    """Verify structure_mapping prevents cross-structure neighbors in multi-state mode.

    This is the core correctness test for multi-state design. Without proper
    masking, residues from different conformational states can become neighbors
    based on spatial proximity, causing information leakage during encoding.
    """
    # Create two structures close together (y-offset 0.5 Å)
    protein = create_simple_multistate_protein(key=jax.random.key(0))

    # Extract required inputs
    prng_key = jax.random.key(1)

    # Call features WITH structure_mapping
    mask = jnp.asarray(protein.atom_mask, dtype=jnp.float32)

    edge_features, neighbor_indices, _, _ = self.features_module(
      prng_key,
      jnp.asarray(protein.coordinates),
      mask,
      jnp.asarray(protein.residue_index),
      jnp.asarray(protein.chain_index),
      backbone_noise=None,
      structure_mapping=jnp.asarray(protein.mapping),
    )

    chex.assert_tree_all_finite((edge_features, neighbor_indices))

    # Verify neighbor indices respect structure boundaries
    is_valid, error_msg = verify_no_cross_structure_neighbors(
      neighbor_indices,
      protein.mapping,
    )

    assert is_valid, f"Cross-structure neighbors found:\n{error_msg}"

    # Additional checks
    assert edge_features.shape[0] == protein.coordinates.shape[0]
    assert neighbor_indices.shape[0] == protein.coordinates.shape[0]


  def test_features_without_structure_mapping_allows_cross_neighbors(self):
    """Verify that WITHOUT structure_mapping, cross-structure neighbors CAN occur.

    This test documents the problem being solved: when structures are spatially
    close and no structure_mapping is provided, residues from different structures
    become neighbors, which is incorrect for multi-state design.
    """
    # Create two structures close together
    protein = create_simple_multistate_protein(key=jax.random.key(0))

    prng_key = jax.random.key(1)
    mask = jnp.asarray(protein.atom_mask, dtype=jnp.float32)

    # Call features WITHOUT structure_mapping
    edge_features, neighbor_indices, _, _ = self.features_module(
      prng_key,
      jnp.asarray(protein.coordinates),
      mask,
      jnp.asarray(protein.residue_index),
      jnp.asarray(protein.chain_index),
      backbone_noise=None,
      structure_mapping=None,  # No masking
    )
    chex.assert_tree_all_finite((edge_features, neighbor_indices))

    # Check if cross-structure neighbors exist
    is_valid, _ = verify_no_cross_structure_neighbors(
      neighbor_indices,
      protein.mapping,
    )

    # We expect violations when structure_mapping is NOT provided
    # (This is the bug the feature fixes)
    assert not is_valid, (
      "Expected cross-structure neighbors without structure_mapping. "
      "If this fails, structures may be too far apart in test data."
    )


  def test_features_structure_mapping_backward_compatible(self):
    """Verify single-structure behavior is unchanged by structure_mapping.

    Tests that structure_mapping with all same values (single structure)
    produces identical results to structure_mapping=None.
    """
    # Create a single structure
    protein = create_multistate_test_batch(
      n_structures=1,
      n_residues_each=10,
      key=jax.random.key(0),
    )

    prng_key = jax.random.key(1)

    # Call with structure_mapping=None
    features_none, neighbors_none, _, _ = self.features_module(
      prng_key,
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      backbone_noise=None,
      structure_mapping=None,
    )
    chex.assert_tree_all_finite((features_none, neighbors_none))

    # Call with structure_mapping=all zeros (single structure)
    structure_mapping_single = jnp.zeros(protein.coordinates.shape[0], dtype=jnp.int32)
    features_mapped, neighbors_mapped, _, _ = self.features_module(
      prng_key,
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      backbone_noise=None,
      structure_mapping=structure_mapping_single,
    )
    chex.assert_tree_all_finite((features_mapped, neighbors_mapped))

    # Neighbor indices should be identical
    assert jnp.array_equal(
      neighbors_none,
      neighbors_mapped,
    ), "Single-structure results differ with/without structure_mapping"


  def test_features_structure_mapping_shape_validation(self):
    """Test behavior with invalid structure_mapping shapes.

    Verifies that mismatched shapes are handled gracefully.
    """
    protein = create_simple_multistate_protein(key=jax.random.key(0))
    prng_key = jax.random.key(1)

    # Mismatched length - too short
    wrong_mapping = jnp.array([0, 0, 1], dtype=jnp.int32)  # Only 3 elements, need 6

    # This should either raise an error or handle gracefully
    # JAX typically broadcasts or errors on shape mismatch
    with pytest.raises((ValueError, IndexError, TypeError)):
      self.features_module(
        prng_key,
        protein.coordinates,
        protein.atom_mask,
        protein.residue_index,
        protein.chain_index,
        backbone_noise=None,
        structure_mapping=wrong_mapping,
      )

  @chex.variants(with_jit=True, without_jit=True, with_device=True)
  def test_features_structure_mapping_jit_compatible(self):
    """Verify structure_mapping works under JAX JIT compilation.

    Tests that structure_mapping doesn't introduce Python control flow
    that would break JIT tracing.
    """
    protein = create_simple_multistate_protein(key=jax.random.key(0))

    @self.variant
    def jitted_features(prng_key, coords, mask, res_idx, chain_idx, noise, mapping):
      return self.features_module(
        prng_key,
        coords,
        mask,
        res_idx,
        chain_idx,
        noise,
        mapping,
      )

    prng_key = jax.random.key(1)

    # Call with structure_mapping
    edge_features_jit, neighbors_jit, _, _ = jitted_features(
      prng_key,
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      None,
      protein.mapping,
    )
    chex.assert_tree_all_finite((edge_features_jit, neighbors_jit))

    # Call without JIT for comparison
    edge_features_nojit, neighbors_nojit, _, _ = self.features_module(
      prng_key,
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      None,
      protein.mapping,
    )
    chex.assert_tree_all_finite((edge_features_nojit, neighbors_nojit))

    # Results should be identical (allow small numerical differences)
    assert jnp.allclose(edge_features_jit, edge_features_nojit, atol=1e-5)
    assert jnp.array_equal(neighbors_jit, neighbors_nojit)


  def test_features_structure_mapping_multiple_structures(self):
    """Test structure_mapping with more than 2 structures.

    Verifies that neighbor isolation works with arbitrary numbers of structures.
    """
    # Create 4 structures with 40 residues each (enough for k=30)
    protein = create_multistate_test_batch(
      n_structures=4,
      n_residues_each=40,
      spatial_offset=0.3,  # Very close structures
      key=jax.random.key(0),
    )

    prng_key = jax.random.key(1)

    edge_features, neighbor_indices, _, _ = self.features_module(
      prng_key,
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      backbone_noise=None,
      structure_mapping=protein.mapping,
    )
    chex.assert_tree_all_finite((edge_features, neighbor_indices))

    # Verify no cross-structure neighbors
    is_valid, error_msg = verify_no_cross_structure_neighbors(
      neighbor_indices,
      protein.mapping,
    )

    assert is_valid, f"Cross-structure neighbors found with 4 structures:\n{error_msg}"

    # Verify expected mapping structure (40 residues per structure, 4 structures)
    expected_mapping = jnp.concatenate([
      jnp.full(40, 0, dtype=jnp.int32),
      jnp.full(40, 1, dtype=jnp.int32),
      jnp.full(40, 2, dtype=jnp.int32),
      jnp.full(40, 3, dtype=jnp.int32),
    ])
    assert jnp.array_equal(protein.mapping, expected_mapping)


  def test_features_structure_mapping_with_backbone_noise(self):
    """Test that structure_mapping works correctly with backbone noise.

    Verifies that the structure masking is applied correctly even when
    coordinates are perturbed by noise.
    """
    protein = create_simple_multistate_protein(key=jax.random.key(0))
    prng_key = jax.random.key(1)

    # Apply significant backbone noise
    backbone_noise = jnp.array(1.0, dtype=jnp.float32)

    edge_features, neighbor_indices, _, _ = self.features_module(
      prng_key,
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      backbone_noise=backbone_noise,
      structure_mapping=protein.mapping,
    )
    chex.assert_tree_all_finite((edge_features, neighbor_indices))

    # Even with noise, structure boundaries should be respected
    is_valid, error_msg = verify_no_cross_structure_neighbors(
      neighbor_indices,
      protein.mapping,
    )

    assert is_valid, f"Cross-structure neighbors found with noise:\n{error_msg}"


  def test_features_structure_mapping_edge_feature_shapes(self):
    """Verify edge features have correct shapes with structure_mapping.

    Tests that structure_mapping doesn't affect output tensor dimensions.
    """
    protein = create_simple_multistate_protein(key=jax.random.key(0))
    prng_key = jax.random.key(1)

    edge_features, neighbor_indices, _, _ = self.features_module(
      prng_key,
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      backbone_noise=None,
      structure_mapping=protein.mapping,
    )

    n_residues = protein.coordinates.shape[0]
    chex.assert_tree_all_finite((edge_features, neighbor_indices))

    # Check shapes
    chex.assert_shape(edge_features, (n_residues, 30, 128))
    chex.assert_shape(neighbor_indices, (n_residues, 30))

    # Check that edge features are finite
    assert jnp.all(jnp.isfinite(edge_features)), "Edge features contain NaN or inf"
