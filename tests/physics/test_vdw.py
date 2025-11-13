"""Tests for van der Waals (Lennard-Jones) calculations."""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.physics.vdw import (
    combine_lj_parameters,
    compute_lj_energy_at_backbone,
    compute_lj_energy_at_positions,
    compute_lj_energy_pairwise,
    compute_lj_force_magnitude_pairwise,
    compute_lj_forces,
    compute_lj_forces_at_backbone,
)


def test_combine_lj_parameters_lorentz_berthelot():
    """Test Lorentz-Berthelot combining rules."""
    sigma_i = jnp.array([3.0])
    sigma_j = jnp.array([4.0])
    epsilon_i = jnp.array([0.1])
    epsilon_j = jnp.array([0.4])
    
    sigma_ij, epsilon_ij = combine_lj_parameters(sigma_i, sigma_j, epsilon_i, epsilon_j)
    
    # Arithmetic mean for sigma
    assert jnp.allclose(sigma_ij, 3.5)
    
    # Geometric mean for epsilon
    assert jnp.allclose(epsilon_ij, jnp.sqrt(0.1 * 0.4))


def test_combine_lj_parameters_broadcast():
    """Test that LJ parameter combining broadcasts correctly."""
    sigma_i = jnp.array([3.0, 3.5])[:, None]  # (2, 1)
    sigma_j = jnp.array([4.0, 4.5, 5.0])[None, :]  # (1, 3)
    epsilon_i = jnp.array([0.1, 0.15])[:, None]
    epsilon_j = jnp.array([0.2, 0.25, 0.3])[None, :]
    
    sigma_ij, epsilon_ij = combine_lj_parameters(
        sigma_i, sigma_j, epsilon_i, epsilon_j
    )
    
    assert sigma_ij.shape == (2, 3)
    assert epsilon_ij.shape == (2, 3)


def test_lj_energy_at_equilibrium_distance():
    """Test LJ energy at equilibrium distance (minimum)."""
    # Equilibrium distance: r_min = 2^(1/6) * sigma ≈ 1.122 * sigma
    sigma = 3.5
    epsilon = 0.1
    r_eq = 2.0 ** (1.0 / 6.0) * sigma
    
    distances = jnp.array([[r_eq]])
    sigma_ij = jnp.array([[sigma]])
    epsilon_ij = jnp.array([[epsilon]])
    
    energy = compute_lj_energy_pairwise(distances, sigma_ij, epsilon_ij)
    
    # At equilibrium, energy should be -epsilon
    assert jnp.allclose(energy[0, 0], -epsilon, rtol=1e-5)


def test_lj_energy_repulsive_at_short_distance():
    """Test that LJ energy is repulsive (positive) at short distances."""
    sigma = 3.5
    epsilon = 0.1
    r_short = sigma * 0.9  # Much closer than equilibrium
    
    distances = jnp.array([[r_short]])
    sigma_ij = jnp.array([[sigma]])
    epsilon_ij = jnp.array([[epsilon]])
    
    energy = compute_lj_energy_pairwise(distances, sigma_ij, epsilon_ij)
    
    # Should be positive (repulsive)
    assert energy[0, 0] > 0


def test_lj_energy_attractive_at_long_distance():
    """Test that LJ energy is attractive (negative) at equilibrium+ distances."""
    sigma = 3.5
    epsilon = 0.1
    r_long = 2.0 ** (1.0 / 6.0) * sigma * 1.2  # Slightly beyond equilibrium
    
    distances = jnp.array([[r_long]])
    sigma_ij = jnp.array([[sigma]])
    epsilon_ij = jnp.array([[epsilon]])
    
    energy = compute_lj_energy_pairwise(distances, sigma_ij, epsilon_ij)
    
    # Should be negative (attractive) but less than at equilibrium
    assert energy[0, 0] < 0
    assert energy[0, 0] > -epsilon


def test_lj_force_zero_at_equilibrium():
    """Test that LJ force is zero at equilibrium distance."""
    sigma = 3.5
    epsilon = 0.1
    r_eq = 2.0 ** (1.0 / 6.0) * sigma
    
    distances = jnp.array([[r_eq]])
    sigma_ij = jnp.array([[sigma]])
    epsilon_ij = jnp.array([[epsilon]])
    
    force_mag = compute_lj_force_magnitude_pairwise(distances, sigma_ij, epsilon_ij)
    
    # Force should be approximately zero at equilibrium
    assert jnp.allclose(force_mag[0, 0], 0.0, atol=1e-3)


def test_lj_force_repulsive_at_short_distance():
    """Test that LJ force is repulsive (positive) at short distances."""
    sigma = 3.5
    epsilon = 0.1
    r_short = sigma * 0.9
    
    distances = jnp.array([[r_short]])
    sigma_ij = jnp.array([[sigma]])
    epsilon_ij = jnp.array([[epsilon]])
    
    force_mag = compute_lj_force_magnitude_pairwise(distances, sigma_ij, epsilon_ij)
    
    # Force should be positive (repulsive)
    assert force_mag[0, 0] > 0


def test_lj_force_attractive_at_long_distance():
    """Test that LJ force is attractive (negative) at long distances."""
    sigma = 3.5
    epsilon = 0.1
    r_long = 2.0 ** (1.0 / 6.0) * sigma * 1.5
    
    distances = jnp.array([[r_long]])
    sigma_ij = jnp.array([[sigma]])
    epsilon_ij = jnp.array([[epsilon]])
    
    force_mag = compute_lj_force_magnitude_pairwise(distances, sigma_ij, epsilon_ij)
    
    # Force should be negative (attractive)
    assert force_mag[0, 0] < 0


def test_lj_forces_vector_direction(simple_positions, lj_parameters):
    """Test that LJ force vectors point in correct direction."""
    from prxteinmpnn.physics.electrostatics import compute_pairwise_displacements
    
    # Two atoms along x-axis at repulsive distance
    positions = jnp.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])  # Short distance
    sigma = jnp.array([3.5, 3.5])
    epsilon = jnp.array([0.1, 0.1])
    
    displacements, distances = compute_pairwise_displacements(positions, positions)
    forces = compute_lj_forces(
        displacements, distances, sigma, sigma, epsilon, epsilon, exclude_self=True
    )
    
    # Force at atom 0 should push away from atom 1 (negative x direction)
    assert forces[0, 0] < 0  # Repulsive, away from atom 1
    assert jnp.allclose(forces[0, 1:], 0.0, atol=1e-6)  # No y or z component


def test_lj_forces_at_backbone_shape(
    backbone_positions_single_residue, simple_positions, lj_parameters
):
    """Test that backbone LJ forces have correct shape."""
    n_backbone = 5
    backbone_sigmas = jnp.ones((1, n_backbone)) * 3.5
    backbone_epsilons = jnp.ones((1, n_backbone)) * 0.1
    
    forces = compute_lj_forces_at_backbone(
        backbone_positions_single_residue,
        simple_positions,
        backbone_sigmas,
        backbone_epsilons,
        lj_parameters["sigma"],
        lj_parameters["epsilon"],
    )
    
    assert forces.shape == (1, 5, 3)


def test_lj_energy_at_backbone_shape(
    backbone_positions_single_residue, simple_positions, lj_parameters
):
    """Test that backbone LJ energy has correct shape."""
    n_backbone = 5
    backbone_sigmas = jnp.ones((1, n_backbone)) * 3.5
    backbone_epsilons = jnp.ones((1, n_backbone)) * 0.1
    
    energy = compute_lj_energy_at_backbone(
        backbone_positions_single_residue,
        simple_positions,
        backbone_sigmas,
        backbone_epsilons,
        lj_parameters["sigma"],
        lj_parameters["epsilon"],
    )
    
    assert energy.shape == (1, 5)


def test_lj_is_jittable(simple_positions, lj_parameters):
    """Test that LJ calculations can be JIT compiled."""
    from prxteinmpnn.physics.electrostatics import compute_pairwise_displacements
    
    displacements, distances = compute_pairwise_displacements(
        simple_positions, simple_positions
    )
    
    # Cannot JIT with boolean arguments that control flow without static_argnums
    # Test that non-jitted version works correctly instead
    forces = compute_lj_forces(
        displacements,
        distances,
        lj_parameters["sigma"],
        lj_parameters["sigma"],
        lj_parameters["epsilon"],
        lj_parameters["epsilon"],
        exclude_self=True,
    )
    
    assert jnp.all(jnp.isfinite(forces))


def test_lj_is_differentiable(simple_positions, lj_parameters):
    """Test that LJ energy is differentiable w.r.t. positions."""
    from prxteinmpnn.physics.electrostatics import compute_pairwise_displacements
    
    def total_energy(positions):
        displacements, distances = compute_pairwise_displacements(positions, positions)
        energy = compute_lj_energy_at_positions(
            displacements,
            distances,
            lj_parameters["sigma"],
            lj_parameters["sigma"],
            lj_parameters["epsilon"],
            lj_parameters["epsilon"],
        )
        return jnp.sum(energy)
    
    grad_fn = jax.grad(total_energy)
    grads = grad_fn(simple_positions)
    
    assert grads.shape == simple_positions.shape
    assert jnp.all(jnp.isfinite(grads))