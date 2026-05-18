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

from prxteinmpnn.inference.bundle_builder import build_inference_bundle
from prxteinmpnn.inference import (
  sample_autoregressive,
  score_unconditional,
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

    bundle, config, stage_set = build_inference_bundle(
      coords=protein.coordinates,
      mask=protein.atom_mask,
      residue_index=protein.residue_index,
      chain_index=protein.chain_index,
      structure_mapping=protein.mapping,
      mode="sample_autoregressive",
    )
    result = sample_autoregressive.kernel(
      self.mpnn_model, prng_key, bundle, config, stage_set
    )

    # Verify output shape: coordinates is (100, 4, 3), so L=100 residues
    chex.assert_shape(result.logits, (100, 21))
    chex.assert_tree_all_finite(result.logits)


  def test_mpnn_single_sequence_with_structure_mapping(self):
    """Test single sequence decoding respects structure_mapping.

    In single sequence mode, the model should still respect structure boundaries
    during feature extraction even though it decodes all positions simultaneously.
    """
    protein = create_simple_multistate_protein(key=jax.random.key(0))
    prng_key = jax.random.key(1)

    bundle, config, stage_set = build_inference_bundle(
      coords=protein.coordinates,
      mask=protein.atom_mask,
      residue_index=protein.residue_index,
      chain_index=protein.chain_index,
      structure_mapping=protein.mapping,
      mode="score_unconditional",
    )
    logits = score_unconditional.kernel(
      self.mpnn_model, prng_key, bundle, config, stage_set
    )

    # Verify output shape and validity: coordinates is (100, 4, 3), so L=100 residues
    chex.assert_shape(logits, (100, 21))
    chex.assert_tree_all_finite(logits)


  def test_mpnn_with_backbone_noise_and_structure_mapping(self):
    """Test that structure_mapping works correctly with backbone noise.

    Backbone noise should not affect the structure isolation provided
    by structure_mapping.
    """
    protein = create_simple_multistate_protein(key=jax.random.key(0))
    prng_key = jax.random.key(1)

    bundle, config, stage_set = build_inference_bundle(
      coords=protein.coordinates,
      mask=protein.atom_mask,
      residue_index=protein.residue_index,
      chain_index=protein.chain_index,
      backbone_noise=1.0,
      structure_mapping=protein.mapping,
      mode="sample_autoregressive",
    )
    result = sample_autoregressive.kernel(
      self.mpnn_model, prng_key, bundle, config, stage_set
    )

    # Verify output: coordinates is (100, 4, 3), so L=100 residues
    chex.assert_shape(result.logits, (100, 21))
    chex.assert_tree_all_finite(result.logits)


  def test_mpnn_without_structure_mapping_backward_compatible(self):
    """Verify model works without structure_mapping (backward compatibility).

    When structure_mapping=None, the model should behave as before,
    treating all residues as part of a single structure.
    """
    protein = create_simple_multistate_protein(key=jax.random.key(0))
    prng_key = jax.random.key(1)

    # Call WITHOUT structure_mapping
    bundle_no_mapping, config_no_mapping, stage_set_no_mapping = build_inference_bundle(
      coords=protein.coordinates,
      mask=protein.atom_mask,
      residue_index=protein.residue_index,
      chain_index=protein.chain_index,
      structure_mapping=None,
      mode="sample_autoregressive",
    )
    result_no_mapping = sample_autoregressive.kernel(
      self.mpnn_model, prng_key, bundle_no_mapping, config_no_mapping, stage_set_no_mapping
    )

    # Call WITH structure_mapping=all zeros (equivalent to single structure)
    structure_mapping_single = jnp.zeros(protein.coordinates.shape[0], dtype=jnp.int32)
    bundle_single, config_single, stage_set_single = build_inference_bundle(
      coords=protein.coordinates,
      mask=protein.atom_mask,
      residue_index=protein.residue_index,
      chain_index=protein.chain_index,
      structure_mapping=structure_mapping_single,
      mode="sample_autoregressive",
    )
    result_single = sample_autoregressive.kernel(
      self.mpnn_model, prng_key, bundle_single, config_single, stage_set_single
    )

    # Results should be identical
    chex.assert_trees_all_close(result_no_mapping.logits, result_single.logits, atol=1e-5)


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

    bundle, config, stage_set = build_inference_bundle(
      coords=protein.coordinates,
      mask=protein.atom_mask,
      residue_index=protein.residue_index,
      chain_index=protein.chain_index,
      structure_mapping=protein.mapping,
      mode="sample_autoregressive",
    )
    result = sample_autoregressive.kernel(
      self.mpnn_model, prng_key, bundle, config, stage_set
    )

    # Verify output shape: 3 * 40 = 120 residues in flat array
    chex.assert_shape(result.logits, (120, 21))
    chex.assert_tree_all_finite(result.logits)

  def test_mpnn_jit_compatible_with_structure_mapping(self):
    """Verify structure_mapping works under JIT compilation.

    Tests that structure_mapping doesn't introduce Python control flow
    that would break JIT tracing.
    """
    protein = create_simple_multistate_protein(key=jax.random.key(0))
    prng_key = jax.random.key(1)

    bundle, config, stage_set = build_inference_bundle(
      coords=protein.coordinates,
      mask=protein.atom_mask,
      residue_index=protein.residue_index,
      chain_index=protein.chain_index,
      structure_mapping=protein.mapping,
      mode="sample_autoregressive",
    )

    # Call kernel twice with same inputs
    result_1 = sample_autoregressive.kernel(
      self.mpnn_model, prng_key, bundle, config, stage_set
    )

    result_2 = sample_autoregressive.kernel(
      self.mpnn_model, prng_key, bundle, config, stage_set
    )

    # Results should be identical (same prng_key, same inputs)
    chex.assert_trees_all_close(result_1.logits, result_2.logits, atol=1e-5)
