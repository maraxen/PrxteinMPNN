"""Tests for electrostatic calculations."""
import chex
import jax
import jax.numpy as jnp
import pytest
from functools import partial

from prxteinmpnn.physics.electrostatics import (
    compute_coulomb_forces,
    compute_coulomb_forces_at_backbone,
    compute_pairwise_displacements,
)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_pairwise_displacements_shape(simple_positions, jit_compile):
    """Test that pairwise displacements have correct shape."""
    fn = compute_pairwise_displacements
    if jit_compile:
        fn = jax.jit(fn)
    pos1 = simple_positions[:2]  # 2 atoms
    pos2 = simple_positions  # 4 atoms

    displacements, distances = fn(pos1, pos2)

    chex.assert_shape(displacements, (2, 4, 3))
    chex.assert_shape(distances, (2, 4))
    chex.assert_tree_all_finite((displacements, distances))


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_pairwise_displacements_symmetry(simple_positions, jit_compile):
    """Test that distances are symmetric."""
    fn = compute_pairwise_displacements
    if jit_compile:
        fn = jax.jit(fn)
    _, distances = fn(simple_positions, simple_positions)

    # Distance matrix should be symmetric
    chex.assert_trees_all_close(distances, distances.T)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_pairwise_displacements_diagonal_zero(simple_positions, jit_compile):
    """Test that diagonal distances (self-distances) are zero."""
    fn = compute_pairwise_displacements
    if jit_compile:
        fn = jax.jit(fn)
    _, distances = fn(simple_positions, simple_positions)

    diagonal = jnp.diag(distances)
    chex.assert_trees_all_close(diagonal, 0.0, atol=1e-6)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_pairwise_displacements_known_distance(jit_compile):
    """Test displacement calculation against known values."""
    fn = compute_pairwise_displacements
    if jit_compile:
        fn = jax.jit(fn)
    pos1 = jnp.array([[0.0, 0.0, 0.0]])
    pos2 = jnp.array([[3.0, 4.0, 0.0]])

    displacements, distances = fn(pos1, pos2)

    # Distance should be 5.0 (3-4-5 triangle)
    chex.assert_trees_all_close(distances[0, 0], 5.0)

    # Displacement: jax_md returns displacement from first to second arg
    # So displacement_fn(pos_i, pos_j) = pos_i - pos_j (based on implementation)
    # We call it with (pos_i, pos_j) so we get pos_i - pos_j
    chex.assert_trees_all_close(
        jnp.abs(displacements[0, 0]), jnp.array([3.0, 4.0, 0.0])
    )


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_coulomb_forces_opposite_charges_attract(jit_compile):
    """Test that opposite charges produce attractive forces."""
    fn = partial(compute_coulomb_forces, exclude_self=True)
    if jit_compile:
        fn = jax.jit(fn)
    # Two point charges: +1 at origin, -1 at (5, 0, 0)
    positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    charges = jnp.array([1.0, -1.0])

    displacements, distances = compute_pairwise_displacements(positions, positions)
    forces = fn(displacements, distances, charges, charges)
    chex.assert_tree_all_finite(forces)

    # Force at position 0 should point toward position 1 (positive x)
    assert forces[0, 0] > 0  # Force in +x direction
    chex.assert_trees_all_close(forces[0, 1:], 0.0, atol=1e-6)  # No y or z component


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_coulomb_forces_same_charges_repel(jit_compile):
    """Test that same-sign charges produce repulsive forces."""
    fn = partial(compute_coulomb_forces, exclude_self=True)
    if jit_compile:
        fn = jax.jit(fn)
    # Two positive charges
    positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    charges = jnp.array([1.0, 1.0])

    displacements, distances = compute_pairwise_displacements(positions, positions)
    forces = fn(displacements, distances, charges, charges)
    chex.assert_tree_all_finite(forces)

    # Force at position 0 should point away from position 1 (negative x)
    assert forces[0, 0] < 0  # Force in -x direction


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_coulomb_forces_magnitude_scales_with_charge(
    simple_positions, jit_compile
):
    """Test that force magnitude scales linearly with charge."""
    fn = partial(compute_coulomb_forces, exclude_self=True)
    if jit_compile:
        fn = jax.jit(fn)
    charges_1x = jnp.array([1.0, -1.0, 0.5, -0.5])
    charges_2x = charges_1x * 2.0

    displacements, distances = compute_pairwise_displacements(
        simple_positions, simple_positions
    )

    # With corrected API: both target and source charges scale quadratically (q_i * q_j)
    forces_1x = fn(displacements, distances, charges_1x, charges_1x)
    forces_2x = fn(displacements, distances, charges_2x, charges_2x)

    # Forces should scale as charge^2 since F ~ q_i * q_j
    chex.assert_trees_all_close(forces_2x, forces_1x * 4.0, rtol=1e-5)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_coulomb_forces_at_backbone_shape(
    backbone_positions_single_residue,
    simple_positions,
    simple_charges,
    jit_compile,
):
    """Test that backbone forces have correct shape."""
    fn = compute_coulomb_forces_at_backbone
    if jit_compile:
        fn = jax.jit(fn)
    # Create backbone charges (5 atoms per residue)
    backbone_charges = jnp.ones((1, 5)) * 0.5  # 1 residue, 5 atoms

    forces = fn(
        backbone_positions_single_residue,
        simple_positions,
        backbone_charges,
        simple_charges,
    )

    chex.assert_shape(forces, (1, 5, 3))  # 1 residue, 5 atoms (N,CA,C,O,CB), 3D
    chex.assert_tree_all_finite(forces)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_coulomb_forces_at_backbone_multi_residue(
    backbone_positions_multi_residue,
    simple_positions,
    simple_charges,
    jit_compile,
):
    """Test backbone forces for multiple residues."""
    fn = compute_coulomb_forces_at_backbone
    if jit_compile:
        fn = jax.jit(fn)
    # Create backbone charges (5 atoms per residue × 3 residues)
    backbone_charges = jnp.ones((3, 5)) * 0.5  # 3 residues, 5 atoms each

    forces = fn(
        backbone_positions_multi_residue,
        simple_positions,
        backbone_charges,
        simple_charges,
    )

    chex.assert_shape(forces, (3, 5, 3))  # 3 residues
    chex.assert_tree_all_finite(forces)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_coulomb_forces_vmappable(
    simple_positions, simple_charges, jit_compile
):
    """Test that Coulomb forces can be vmapped over batches."""
    # Create batch of 3 charge distributions
    batch_charges = jnp.stack(
        [simple_charges, simple_charges * 2, simple_charges * 0.5]
    )

    displacements, distances = compute_pairwise_displacements(
        simple_positions, simple_positions
    )

    # Vmap over charge distributions (both target and source)
    def vmapped_fn(charges):
        return compute_coulomb_forces(
            displacements, distances, charges, charges
        )

    fn = jax.vmap(vmapped_fn)
    if jit_compile:
        fn = jax.jit(fn)

    forces_batch = fn(batch_charges)

    chex.assert_shape(forces_batch, (3, 4, 3))  # 3 batches, 4 atoms, 3D
    chex.assert_tree_all_finite(forces_batch)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_coulomb_forces_differentiable(
    simple_positions, simple_charges, jit_compile
):
    """Test that Coulomb forces are differentiable w.r.t. positions."""

    def force_magnitude(positions):
        displacements, distances = compute_pairwise_displacements(
            positions, positions
        )
        forces = compute_coulomb_forces(
            displacements, distances, simple_charges, simple_charges
        )
        return jnp.sum(jnp.linalg.norm(forces, axis=-1))

    grad_fn = jax.grad(force_magnitude)
    if jit_compile:
        grad_fn = jax.jit(grad_fn)
    grads = grad_fn(simple_positions)

    chex.assert_shape(grads, simple_positions.shape)
    chex.assert_tree_all_finite(grads)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_coulomb_forces_zero_for_neutral(jit_compile):
    """Test that neutral charges produce zero net force."""
    fn = partial(compute_coulomb_forces, exclude_self=True)
    if jit_compile:
        fn = jax.jit(fn)
    positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    charges = jnp.array([0.0, 0.0])

    displacements, distances = compute_pairwise_displacements(
        positions, positions
    )
    forces = fn(displacements, distances, charges, charges)

    chex.assert_trees_all_close(forces, 0.0, atol=1e-10)
