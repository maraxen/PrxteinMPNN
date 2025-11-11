"""Tests for electrostatic calculations."""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.physics.electrostatics import (
    compute_coulomb_forces,
    compute_coulomb_forces_at_backbone,
    compute_pairwise_displacements,
)


def test_pairwise_displacements_shape(simple_positions):
    """Test that pairwise displacements have correct shape."""
    pos1 = simple_positions[:2]  # 2 atoms
    pos2 = simple_positions  # 4 atoms
    
    displacements, distances = compute_pairwise_displacements(pos1, pos2)
    
    assert displacements.shape == (2, 4, 3)
    assert distances.shape == (2, 4)


def test_pairwise_displacements_symmetry(simple_positions):
    """Test that distances are symmetric."""
    displacements, distances = compute_pairwise_displacements(
        simple_positions, simple_positions
    )
    
    # Distance matrix should be symmetric
    assert jnp.allclose(distances, distances.T)


def test_pairwise_displacements_diagonal_zero(simple_positions):
    """Test that diagonal distances (self-distances) are zero."""
    _, distances = compute_pairwise_displacements(simple_positions, simple_positions)
    
    diagonal = jnp.diag(distances)
    assert jnp.allclose(diagonal, 0.0, atol=1e-6)


def test_pairwise_displacements_known_distance():
    """Test displacement calculation against known values."""
    pos1 = jnp.array([[0.0, 0.0, 0.0]])
    pos2 = jnp.array([[3.0, 4.0, 0.0]])
    
    displacements, distances = compute_pairwise_displacements(pos1, pos2)
    
    # Distance should be 5.0 (3-4-5 triangle)
    assert jnp.allclose(distances[0, 0], 5.0)
    
    # Displacement: jax_md returns displacement from first to second arg
    # So displacement_fn(pos_i, pos_j) = pos_i - pos_j (based on implementation)
    # We call it with (pos_i, pos_j) so we get pos_i - pos_j
    assert jnp.allclose(jnp.abs(displacements[0, 0]), jnp.array([3.0, 4.0, 0.0]))


def test_coulomb_forces_opposite_charges_attract():
    """Test that opposite charges produce attractive forces."""
    # Two point charges: +1 at origin, -1 at (5, 0, 0)
    positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    charges = jnp.array([1.0, -1.0])
    
    displacements, distances = compute_pairwise_displacements(positions, positions)
    forces = compute_coulomb_forces(displacements, distances, charges, exclude_self=True)
    
    # Force at position 0 should point toward position 1 (positive x)
    assert forces[0, 0] > 0  # Force in +x direction
    assert jnp.allclose(forces[0, 1:], 0.0, atol=1e-6)  # No y or z component


def test_coulomb_forces_same_charges_repel():
    """Test that same-sign charges produce repulsive forces."""
    # Two positive charges
    positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    charges = jnp.array([1.0, 1.0])
    
    displacements, distances = compute_pairwise_displacements(positions, positions)
    forces = compute_coulomb_forces(displacements, distances, charges, exclude_self=True)
    
    # Force at position 0 should point away from position 1 (negative x)
    assert forces[0, 0] < 0  # Force in -x direction


def test_coulomb_forces_magnitude_scales_with_charge(simple_positions):
    """Test that force magnitude scales linearly with charge."""
    charges_1x = jnp.array([1.0, -1.0, 0.5, -0.5])
    charges_2x = charges_1x * 2.0
    
    displacements, distances = compute_pairwise_displacements(
        simple_positions, simple_positions
    )
    
    forces_1x = compute_coulomb_forces(displacements, distances, charges_1x)
    forces_2x = compute_coulomb_forces(displacements, distances, charges_2x)
    
    # Forces should scale linearly with charge
    assert jnp.allclose(forces_2x, forces_1x * 2.0, rtol=1e-5)


def test_coulomb_forces_magnitude_scales_with_charge(simple_positions):
    """Test that force magnitude scales linearly with charge."""
    charges_1x = jnp.array([1.0, -1.0, 0.5, -0.5])
    charges_2x = charges_1x * 2.0
    
    displacements, distances = compute_pairwise_displacements(
        simple_positions, simple_positions
    )
    
    forces_1x = compute_coulomb_forces(displacements, distances, charges_1x, exclude_self=True)
    forces_2x = compute_coulomb_forces(displacements, distances, charges_2x, exclude_self=True)
    
    # Forces should scale linearly with charge
    assert jnp.allclose(forces_2x, forces_1x * 2.0, rtol=1e-5)


def test_coulomb_forces_at_backbone_shape(
    backbone_positions_single_residue, simple_positions, simple_charges
):
    """Test that backbone forces have correct shape."""
    forces = compute_coulomb_forces_at_backbone(
        backbone_positions_single_residue,
        simple_positions,
        simple_charges,
    )
    
    assert forces.shape == (1, 5, 3)  # 1 residue, 5 atoms (N,CA,C,O,CB), 3D


def test_coulomb_forces_at_backbone_multi_residue(
    backbone_positions_multi_residue, simple_positions, simple_charges
):
    """Test backbone forces for multiple residues."""
    forces = compute_coulomb_forces_at_backbone(
        backbone_positions_multi_residue,
        simple_positions,
        simple_charges,
    )
    
    assert forces.shape == (3, 5, 3)  # 3 residues
    assert jnp.all(jnp.isfinite(forces))




def test_coulomb_forces_is_jittable(simple_positions, simple_charges):
    """Test that Coulomb force calculation can be JIT compiled."""
    displacements, distances = compute_pairwise_displacements(
        simple_positions, simple_positions
    )
    
    jitted_fn = jax.jit(compute_coulomb_forces)
    forces = jitted_fn(displacements, distances, simple_charges)
    
    assert jnp.all(jnp.isfinite(forces))


def test_coulomb_forces_is_vmappable(simple_positions, simple_charges):
    """Test that Coulomb forces can be vmapped over batches."""
    # Create batch of 3 charge distributions
    batch_charges = jnp.stack([simple_charges, simple_charges * 2, simple_charges * 0.5])
    
    displacements, distances = compute_pairwise_displacements(
        simple_positions, simple_positions
    )
    
    # Vmap over charge distributions
    vmapped_fn = jax.vmap(
        lambda charges: compute_coulomb_forces(displacements, distances, charges)
    )
    
    forces_batch = vmapped_fn(batch_charges)
    
    assert forces_batch.shape == (3, 4, 3)  # 3 batches, 4 atoms, 3D


def test_coulomb_forces_is_differentiable(simple_positions, simple_charges):
    """Test that Coulomb forces are differentiable w.r.t. positions."""
    def force_magnitude(positions):
        displacements, distances = compute_pairwise_displacements(positions, positions)
        forces = compute_coulomb_forces(displacements, distances, simple_charges)
        return jnp.sum(jnp.linalg.norm(forces, axis=-1))
    
    grad_fn = jax.grad(force_magnitude)
    grads = grad_fn(simple_positions)
    
    assert grads.shape == simple_positions.shape
    assert jnp.all(jnp.isfinite(grads))


def test_coulomb_forces_zero_for_neutral():
    """Test that neutral charges produce zero net force."""
    positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    charges = jnp.array([0.0, 0.0])
    
    displacements, distances = compute_pairwise_displacements(positions, positions)
    forces = compute_coulomb_forces(displacements, distances, charges, exclude_self=True)
    
    assert jnp.allclose(forces, 0.0, atol=1e-10)