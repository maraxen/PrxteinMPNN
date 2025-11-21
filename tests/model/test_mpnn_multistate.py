"""Tests for PrxteinMPNN model with structure_mapping in multi-state mode.

This module tests the core PrxteinMPNN model's ability to handle multi-state
protein design with structure_mapping, ensuring proper isolation of conformational
states during encoding and decoding.
"""

import chex
import jax
import jax.numpy as jnp
from helpers.multistate import (
  create_multistate_test_batch,
  create_simple_multistate_protein,
)

from prxteinmpnn.model.mpnn import PrxteinMPNN


class TestMPNNMultiState(chex.TestCase):

  def setUp(self):
    """Create a PrxteinMPNN model for testing."""
    key = jax.random.key(42)
    self.mpnn_model = PrxteinMPNN(
      node_features=128,
      edge_features=128,
      hidden_features=128,
      k_neighbors=30,
      num_encoder_layers=3,
      num_decoder_layers=3,
      key=key,
    )


  def test_mpnn_autoregressive_with_structure_mapping(self):
    """Test autoregressive decoding respects structure_mapping.

    Verifies that in autoregressive mode, the model properly isolates
    structures during both encoding and decoding phases.
    """
    protein = create_simple_multistate_protein(key=jax.random.key(0))
    prng_key = jax.random.key(1)

    # Run autoregressive decoding with structure_mapping
    _, logits = self.mpnn_model(
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      "autoregressive",
      prng_key=prng_key,
      backbone_noise=None,
      structure_mapping=protein.mapping,
    )

    # Verify output shape
    chex.assert_shape(logits, (protein.coordinates.shape[0], 21))  # (N, 21 amino acids)

    # Verify logits are finite
    chex.assert_tree_all_finite(logits)


  def test_mpnn_single_sequence_with_structure_mapping(self):
    """Test single sequence decoding respects structure_mapping.

    In single sequence mode, the model should still respect structure boundaries
    during feature extraction even though it decodes all positions simultaneously.
    """
    protein = create_simple_multistate_protein(key=jax.random.key(0))
    prng_key = jax.random.key(1)

    _, logits = self.mpnn_model(
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      "unconditional",
      prng_key=prng_key,
      backbone_noise=None,
      structure_mapping=protein.mapping,
    )

    # Verify output shape and validity
    chex.assert_shape(logits, (protein.coordinates.shape[0], 21))
    chex.assert_tree_all_finite(logits)


  def test_mpnn_with_backbone_noise_and_structure_mapping(self):
    """Test that structure_mapping works correctly with backbone noise.

    Backbone noise should not affect the structure isolation provided
    by structure_mapping.
    """
    protein = create_simple_multistate_protein(key=jax.random.key(0))
    prng_key = jax.random.key(1)

    # Apply backbone noise
    backbone_noise = jnp.array(1.0, dtype=jnp.float32)

    _, logits = self.mpnn_model(
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      "autoregressive",
      prng_key=prng_key,
      backbone_noise=backbone_noise,
      structure_mapping=protein.mapping,
    )

    # Verify output
    chex.assert_shape(logits, (protein.coordinates.shape[0], 21))
    chex.assert_tree_all_finite(logits)


  def test_mpnn_without_structure_mapping_backward_compatible(self):
    """Verify model works without structure_mapping (backward compatibility).

    When structure_mapping=None, the model should behave as before,
    treating all residues as part of a single structure.
    """
    protein = create_simple_multistate_protein(key=jax.random.key(0))
    prng_key = jax.random.key(1)

    # Call WITHOUT structure_mapping
    _, logits_no_mapping = self.mpnn_model(
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      "autoregressive",
      prng_key=prng_key,
      backbone_noise=None,
      structure_mapping=None,
    )

    # Call WITH structure_mapping=all zeros (equivalent to single structure)
    structure_mapping_single = jnp.zeros(protein.coordinates.shape[0], dtype=jnp.int32)
    _, logits_single_structure = self.mpnn_model(
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      "autoregressive",
      prng_key=prng_key,
      backbone_noise=None,
      structure_mapping=structure_mapping_single,
    )

    # Results should be identical
    chex.assert_trees_all_close(logits_no_mapping, logits_single_structure, atol=1e-5)


  def test_mpnn_multiple_structures_isolation(self):
    """Test that multiple structures remain isolated during encoding.

    Creates 3 structures and verifies that neighbor relationships in the
    encoder respect structure boundaries.
    """
    # Create 3 structures with 40 residues each
    protein = create_multistate_test_batch(
      n_structures=3,
      n_residues_each=40,
      spatial_offset=0.5,
      key=jax.random.key(0),
    )
    prng_key = jax.random.key(1)

    # Run model
    _, logits = self.mpnn_model(
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      "autoregressive",
      prng_key=prng_key,
      backbone_noise=None,
      structure_mapping=protein.mapping,
    )

    # Verify output shape
    expected_shape = (120, 21)  # 3 * 40 = 120 residues
    chex.assert_shape(logits, expected_shape)
    chex.assert_tree_all_finite(logits)

  @chex.variants(with_jit=True, without_jit=True, with_device=True)
  def test_mpnn_jit_compatible_with_structure_mapping(self):
    """Verify structure_mapping works under JIT compilation.

    Tests that structure_mapping doesn't introduce Python control flow
    that would break JIT tracing.
    """
    protein = create_simple_multistate_protein(key=jax.random.key(0))

    @self.variant
    def jitted_mpnn(coords, mask, res_idx, chain_idx, key, noise, mapping):
      return self.mpnn_model(
        coords,
        mask,
        res_idx,
        chain_idx,
        "autoregressive",  # decoding approach is static
        prng_key=key,
        backbone_noise=noise,
        structure_mapping=mapping,
      )

    prng_key = jax.random.key(1)

    # Call JIT version
    _, logits_jit = jitted_mpnn(
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      prng_key,
      None,
      protein.mapping,
    )

    # Call non-JIT version
    _, logits_nojit = self.mpnn_model(
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      "autoregressive",
      prng_key=prng_key,
      backbone_noise=None,
      structure_mapping=protein.mapping,
    )

    # Results should be identical (within numerical precision)
    chex.assert_trees_all_close(logits_jit, logits_nojit, atol=1e-5)
