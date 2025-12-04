"""Tests for force projection onto backbone geometry."""
import chex
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.physics.projections import (
    compute_backbone_frame,
    project_forces_onto_backbone,
    project_forces_onto_backbone_per_atom,
)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_backbone_frame_shape(backbone_positions_single_residue, jit_compile):
    """Test that backbone frame has correct shape."""
    fn = compute_backbone_frame
    if jit_compile:
        fn = jax.jit(fn)
    forward, backward, sidechain, normal = fn(
        backbone_positions_single_residue,
    )

    chex.assert_shape(forward, (1, 3))
    chex.assert_shape(backward, (1, 3))
    chex.assert_shape(sidechain, (1, 3))
    chex.assert_shape(normal, (1, 3))


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_backbone_frame_unit_vectors(backbone_positions_single_residue, jit_compile):
    """Test that backbone frame vectors are unit vectors."""
    fn = compute_backbone_frame
    if jit_compile:
        fn = jax.jit(fn)
    forward, backward, sidechain, normal = fn(
        backbone_positions_single_residue,
    )

    chex.assert_trees_all_close(jnp.linalg.norm(forward, axis=-1), 1.0, rtol=1e-5)
    chex.assert_trees_all_close(jnp.linalg.norm(backward, axis=-1), 1.0, rtol=1e-5)
    chex.assert_trees_all_close(jnp.linalg.norm(sidechain, axis=-1), 1.0, rtol=1e-5)
    chex.assert_trees_all_close(jnp.linalg.norm(normal, axis=-1), 1.0, rtol=1e-5)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_backbone_frame_orthogonality(jit_compile):
    """Test that normal is perpendicular to forward and backward."""
    fn = compute_backbone_frame
    if jit_compile:
        fn = jax.jit(fn)
    # Create idealized backbone in xy-plane
    positions = jnp.array(
        [
            [
                [0.0, 0.0, 0.0],  # N
                [1.0, 0.0, 0.0],  # CA
                [1.5, 1.0, 0.0],  # C
                [1.5, 2.0, 0.0],  # O
                [1.0, 0.0, 1.0],  # CB (along z)
            ],
        ],
    )

    forward, backward, _, normal = fn(positions)

    # Normal should be perpendicular to forward and backward
    chex.assert_trees_all_close(
        jnp.sum(normal * forward, axis=-1), 0.0, atol=1e-5,
    )
    chex.assert_trees_all_close(
        jnp.sum(normal * backward, axis=-1), 0.0, atol=1e-5,
    )


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_backbone_frame_forward_direction(jit_compile):
    """Test that forward vector points from CA to C."""
    fn = compute_backbone_frame
    if jit_compile:
        fn = jax.jit(fn)
    # Simple linear backbone along x-axis
    positions = jnp.array(
        [
            [
                [0.0, 0.0, 0.0],  # N
                [1.0, 0.0, 0.0],  # CA
                [2.0, 0.0, 0.0],  # C
                [2.0, 1.0, 0.0],  # O
                [1.0, 0.0, 1.0],  # CB
            ],
        ],
    )

    forward, _, _, _ = fn(positions)

    # Forward should point along +x
    chex.assert_trees_all_close(forward, jnp.array([[1.0, 0.0, 0.0]]), atol=1e-5)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_backbone_frame_backward_direction(jit_compile):
    """Test that backward vector points from CA to N."""
    fn = compute_backbone_frame
    if jit_compile:
        fn = jax.jit(fn)
    positions = jnp.array(
        [
            [
                [0.0, 0.0, 0.0],  # N
                [1.0, 0.0, 0.0],  # CA
                [2.0, 0.0, 0.0],  # C
                [2.0, 1.0, 0.0],  # O
                [1.0, 0.0, 1.0],  # CB
            ],
        ],
    )

    _, backward, _, _ = fn(positions)

    # Backward should point along -x
    chex.assert_trees_all_close(backward, jnp.array([[-1.0, 0.0, 0.0]]), atol=1e-5)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_backbone_frame_sidechain_direction(jit_compile):
    """Test that sidechain vector points from CA to CB."""
    fn = compute_backbone_frame
    if jit_compile:
        fn = jax.jit(fn)
    positions = jnp.array(
        [
            [
                [0.0, 0.0, 0.0],  # N
                [1.0, 0.0, 0.0],  # CA
                [2.0, 0.0, 0.0],  # C
                [2.0, 1.0, 0.0],  # O
                [1.0, 0.0, 1.0],  # CB (along +z from CA)
            ],
        ],
    )

    _, _, sidechain, _ = fn(positions)

    # Sidechain should point along +z
    chex.assert_trees_all_close(sidechain, jnp.array([[0.0, 0.0, 1.0]]), atol=1e-5)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_project_forces_shape(backbone_positions_single_residue, jit_compile):
    """Test that projected forces have correct shape."""
    fn = project_forces_onto_backbone
    if jit_compile:
        fn = jax.jit(fn)
    forces = jnp.ones((1, 5, 3))  # Force at each backbone atom

    projections = fn(forces, backbone_positions_single_residue)

    chex.assert_shape(projections, (1, 5))  # 5 scalar features per residue


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_project_forces_all_features_present(
    backbone_positions_single_residue, jit_compile,
):
    """Test that all 5 projection features are computed."""
    fn = project_forces_onto_backbone
    if jit_compile:
        fn = jax.jit(fn)
    forces = jnp.ones((1, 5, 3))

    projections = fn(forces, backbone_positions_single_residue)

    # Should have [f_forward, f_backward, f_sidechain, f_out_of_plane, f_magnitude]
    assert projections.shape[1] == 5


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_project_forces_magnitude_matches_norm(
    backbone_positions_single_residue, jit_compile,
):
    """Test that magnitude feature matches force norm."""
    fn = project_forces_onto_backbone
    if jit_compile:
        fn = jax.jit(fn, static_argnames="aggregation")
    # Create known force
    force_vector = jnp.array([1.0, 2.0, 3.0])
    forces = jnp.tile(force_vector, (1, 5, 1))  # Same force at all atoms

    projections = fn(
        forces, backbone_positions_single_residue, aggregation="mean",
    )

    # Last feature should be magnitude
    expected_magnitude = jnp.linalg.norm(force_vector)
    chex.assert_trees_all_close(projections[0, 4], expected_magnitude, rtol=1e-5)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_project_forces_aligned_with_forward(jit_compile):
    """Test projection when force is aligned with forward direction."""
    fn = project_forces_onto_backbone
    if jit_compile:
        fn = jax.jit(fn, static_argnames="aggregation")
    # Backbone along x-axis
    positions = jnp.array(
        [
            [
                [0.0, 0.0, 0.0],  # N
                [1.0, 0.0, 0.0],  # CA
                [2.0, 0.0, 0.0],  # C
                [2.0, 1.0, 0.0],  # O
                [1.0, 0.0, 1.0],  # CB
            ],
        ],
    )

    # Force pointing along +x (forward direction) for all 5 atoms
    forces = jnp.ones((1, 5, 3))
    forces = forces.at[:, :, :].set(jnp.array([1.0, 0.0, 0.0]))

    projections = fn(forces, positions, aggregation="mean")
    # f_forward should be ~1.0
    # f_backward should be ~-1.0 (opposite direction)
    # f_sidechain should be ~0.0
    # f_out_of_plane should be ~0.0
    assert projections[0, 0] > 0.9  # f_forward
    assert projections[0, 1] < -0.9  # f_backward
    chex.assert_trees_all_close(projections[0, 2], 0.0, atol=0.1)  # f_sidechain
    chex.assert_trees_all_close(
        projections[0, 3], 0.0, atol=0.1,
    )  # f_out_of_plane


def rotation_x(angle_deg: float) -> jnp.ndarray:
    """Rotation matrix around X-axis."""
    theta = jnp.radians(angle_deg)
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_y(angle_deg: float) -> jnp.ndarray:
    """Rotation matrix around Y-axis."""
    theta = jnp.radians(angle_deg)
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_z(angle_deg: float) -> jnp.ndarray:
    """Rotation matrix around Z-axis."""
    theta = jnp.radians(angle_deg)
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@pytest.mark.parametrize(
    "rotation_fn,angle",
    [
        (rotation_x, 90),
        (rotation_x, 180),
        (rotation_x, 270),
        (rotation_y, 90),
        (rotation_y, 180),
        (rotation_y, 270),
        (rotation_z, 90),
        (rotation_z, 180),
        (rotation_z, 270),
    ],
)
@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_project_forces_rotation_invariance(rotation_fn, angle, jit_compile):
    """Test that projections are rotation invariant using exact rotations."""
    fn = project_forces_onto_backbone
    if jit_compile:
        fn = jax.jit(fn)
    positions = jnp.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
        ],
    )

    forces = jnp.ones((1, 5, 3))

    # Compute projections for original
    proj_original = fn(forces, positions)

    # Apply rotation
    R = rotation_fn(angle)
    positions_rotated = jnp.einsum("bij,jk->bik", positions, R)
    forces_rotated = jnp.einsum("bij,jk->bik", forces, R)

    # Compute projections for rotated
    proj_rotated = fn(forces_rotated, positions_rotated)

    # With exact rotations, tolerance can be very tight
    chex.assert_trees_all_close(proj_original, proj_rotated, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_project_forces_per_atom_shape(
    backbone_positions_single_residue, jit_compile,
):
    """Test that per-atom projections have correct shape."""
    fn = project_forces_onto_backbone_per_atom
    if jit_compile:
        fn = jax.jit(fn)
    forces = jnp.ones((1, 5, 3))

    projections = fn(forces, backbone_positions_single_residue)

    # Should have 25 features (5 atoms × 5 projections)
    chex.assert_shape(projections, (1, 25))


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_project_forces_aggregation_methods(
    backbone_positions_single_residue, jit_compile,
):
    """Test different aggregation methods."""
    fn = project_forces_onto_backbone
    if jit_compile:
        fn = jax.jit(fn, static_argnames="aggregation")
    forces = jnp.ones((1, 5, 3))

    proj_mean = fn(
        forces, backbone_positions_single_residue, aggregation="mean",
    )
    proj_sum = fn(
        forces, backbone_positions_single_residue, aggregation="sum",
    )

    # Sum should be 5x mean (5 atoms)
    chex.assert_trees_all_close(proj_sum, proj_mean * 5.0, rtol=1e-5)


def test_project_forces_invalid_aggregation(backbone_positions_single_residue):
    """Test that invalid aggregation method raises error."""
    forces = jnp.ones((1, 5, 3))

    with pytest.raises(ValueError, match="Unknown aggregation method"):
        project_forces_onto_backbone(
            forces, backbone_positions_single_residue, aggregation="invalid",
        )


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_project_forces_is_vmappable(backbone_positions_multi_residue, jit_compile):
    """Test that projection can be vmapped over batches."""
    # Create batch of force vectors
    batch_forces = jnp.ones((3, 3, 5, 3))  # 3 batches, 3 residues, 5 atoms

    # Vmap over batch dimension
    vmapped_fn = jax.vmap(
        lambda forces: project_forces_onto_backbone(
            forces, backbone_positions_multi_residue,
        ),
    )
    if jit_compile:
        vmapped_fn = jax.jit(vmapped_fn)

    projections_batch = vmapped_fn(batch_forces)

    chex.assert_shape(
        projections_batch, (3, 3, 5),
    )  # 3 batches, 3 residues, 5 features
    chex.assert_tree_all_finite(projections_batch)
