"""Integration tests for complete physics pipeline."""

import jax.numpy as jnp
import pytest

from prxteinmpnn.physics import (
    compute_coulomb_forces_at_backbone,
    project_forces_onto_backbone,
)


def test_complete_physics_pipeline(
    backbone_positions_multi_residue, backbone_charges_multi_residue, simple_positions, simple_charges,
):
    """Test complete pipeline from positions to projected features."""
    # Compute forces
    forces = compute_coulomb_forces_at_backbone(
        backbone_positions_multi_residue,
        simple_positions,
        backbone_charges_multi_residue,
        simple_charges,
    )
    
    # Project onto backbone
    features = project_forces_onto_backbone(
        forces,
        backbone_positions_multi_residue,
        aggregation="mean",
    )
    
    # Check output shape and validity
    assert features.shape == (3, 5)  # 3 residues, 5 features
    assert jnp.all(jnp.isfinite(features))
    
    # Features should have reasonable magnitudes (not too large)
    assert jnp.all(jnp.abs(features) < 1000.0)


def test_physics_features_are_rotation_invariant(
    backbone_positions_single_residue, backbone_charges_single_residue, simple_positions, simple_charges,
):
    """Test that complete pipeline produces rotation-invariant features."""
    from scipy.spatial.transform import Rotation
    
    # Original
    forces_orig = compute_coulomb_forces_at_backbone(
        backbone_positions_single_residue,
        simple_positions,
        backbone_charges_single_residue,
        simple_charges,
    )
    features_orig = project_forces_onto_backbone(
        forces_orig, backbone_positions_single_residue
    )
    
    # Rotated
    R = Rotation.random().as_matrix()
    bb_rotated = jnp.dot(backbone_positions_single_residue, R.T)
    pos_rotated = jnp.dot(simple_positions, R.T)

    forces_rot = compute_coulomb_forces_at_backbone(bb_rotated, pos_rotated, backbone_charges_single_residue, simple_charges)
    features_rot = project_forces_onto_backbone(forces_rot, bb_rotated)
    
    # Features should be identical (rotation invariant)
    assert jnp.allclose(features_orig, features_rot, rtol=1e-4, atol=1e-4)


def test_physics_features_scale_with_charges(
    backbone_positions_single_residue, backbone_charges_single_residue, simple_positions, simple_charges
):
    """Test that features scale linearly with charge magnitude."""
    # Compute with 1x charges
    forces_1x = compute_coulomb_forces_at_backbone(
        backbone_positions_single_residue,
        simple_positions,
        backbone_charges_single_residue,
        simple_charges,
    )
    features_1x = project_forces_onto_backbone(
        forces_1x, backbone_positions_single_residue
    )
    
    # Compute with 2x charges
    forces_2x = compute_coulomb_forces_at_backbone(
        backbone_positions_single_residue,
        simple_positions,
        backbone_charges_single_residue,
        simple_charges * 2.0,
    )
    features_2x = project_forces_onto_backbone(
        forces_2x, backbone_positions_single_residue
    )
    
    # Features should scale linearly
    assert jnp.allclose(features_2x, features_1x * 2.0, rtol=1e-5)


def test_zero_charges_produce_zero_features(
    backbone_positions_single_residue, simple_positions
):
    """Test that zero charges produce zero physics features."""
    zero_charges = jnp.zeros_like(simple_positions[:, 0])
    zero_bb_charges = jnp.zeros((1, 5))  # 5 backbone atoms
    
    forces = compute_coulomb_forces_at_backbone(
        backbone_positions_single_residue,
        simple_positions,
        zero_bb_charges,
        zero_charges,
    )
    features = project_forces_onto_backbone(
        forces, backbone_positions_single_residue
    )
    
    # All features should be zero
    assert jnp.allclose(features, 0.0, atol=1e-10)