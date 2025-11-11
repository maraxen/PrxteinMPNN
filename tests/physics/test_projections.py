"""Tests for force projection onto backbone geometry."""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.physics.projections import (
    compute_backbone_frame,
    project_forces_onto_backbone,
    project_forces_onto_backbone_per_atom,
)


def test_backbone_frame_shape(backbone_positions_single_residue):
    """Test that backbone frame has correct shape."""
    forward, backward, sidechain, normal = compute_backbone_frame(
        backbone_positions_single_residue
    )
    
    assert forward.shape == (1, 3)
    assert backward.shape == (1, 3)
    assert sidechain.shape == (1, 3)
    assert normal.shape == (1, 3)


def test_backbone_frame_unit_vectors(backbone_positions_single_residue):
    """Test that backbone frame vectors are unit vectors."""
    forward, backward, sidechain, normal = compute_backbone_frame(
        backbone_positions_single_residue
    )
    
    assert jnp.allclose(jnp.linalg.norm(forward, axis=-1), 1.0, rtol=1e-5)
    assert jnp.allclose(jnp.linalg.norm(backward, axis=-1), 1.0, rtol=1e-5)
    assert jnp.allclose(jnp.linalg.norm(sidechain, axis=-1), 1.0, rtol=1e-5)
    assert jnp.allclose(jnp.linalg.norm(normal, axis=-1), 1.0, rtol=1e-5)


def test_backbone_frame_orthogonality():
    """Test that normal is perpendicular to forward and backward."""
    # Create idealized backbone in xy-plane
    positions = jnp.array([[
        [0.0, 0.0, 0.0],      # N
        [1.0, 0.0, 0.0],      # CA
        [1.5, 1.0, 0.0],      # C
        [1.5, 2.0, 0.0],      # O
        [1.0, 0.0, 1.0],      # CB (along z)
    ]])
    
    forward, backward, sidechain, normal = compute_backbone_frame(positions)
    
    # Normal should be perpendicular to forward and backward
    assert jnp.allclose(jnp.sum(normal * forward, axis=-1), 0.0, atol=1e-5)
    assert jnp.allclose(jnp.sum(normal * backward, axis=-1), 0.0, atol=1e-5)


def test_backbone_frame_forward_direction():
    """Test that forward vector points from CA to C."""
    # Simple linear backbone along x-axis
    positions = jnp.array([[
        [0.0, 0.0, 0.0],      # N
        [1.0, 0.0, 0.0],      # CA
        [2.0, 0.0, 0.0],      # C
        [2.0, 1.0, 0.0],      # O
        [1.0, 0.0, 1.0],      # CB
    ]])
    
    forward, _, _, _ = compute_backbone_frame(positions)
    
    # Forward should point along +x
    assert jnp.allclose(forward[0], jnp.array([1.0, 0.0, 0.0]), atol=1e-5)


def test_backbone_frame_backward_direction():
    """Test that backward vector points from CA to N."""
    positions = jnp.array([[
        [0.0, 0.0, 0.0],      # N
        [1.0, 0.0, 0.0],      # CA
        [2.0, 0.0, 0.0],      # C
        [2.0, 1.0, 0.0],      # O
        [1.0, 0.0, 1.0],      # CB
    ]])
    
    _, backward, _, _ = compute_backbone_frame(positions)
    
    # Backward should point along -x
    assert jnp.allclose(backward[0], jnp.array([-1.0, 0.0, 0.0]), atol=1e-5)


def test_backbone_frame_sidechain_direction():
    """Test that sidechain vector points from CA to CB."""
    positions = jnp.array([[
        [0.0, 0.0, 0.0],      # N
        [1.0, 0.0, 0.0],      # CA
        [2.0, 0.0, 0.0],      # C
        [2.0, 1.0, 0.0],      # O
        [1.0, 0.0, 1.0],      # CB (along +z from CA)
    ]])
    
    _, _, sidechain, _ = compute_backbone_frame(positions)
    
    # Sidechain should point along +z
    assert jnp.allclose(sidechain[0], jnp.array([0.0, 0.0, 1.0]), atol=1e-5)


def test_project_forces_shape(backbone_positions_single_residue):
    """Test that projected forces have correct shape."""
    forces = jnp.ones((1, 5, 3))  # Force at each backbone atom
    
    projections = project_forces_onto_backbone(
        forces, backbone_positions_single_residue
    )
    
    assert projections.shape == (1, 5)  # 5 scalar features per residue


def test_project_forces_all_features_present(backbone_positions_single_residue):
    """Test that all 5 projection features are computed."""
    forces = jnp.ones((1, 5, 3))
    
    projections = project_forces_onto_backbone(
        forces, backbone_positions_single_residue
    )
    
    # Should have [f_forward, f_backward, f_sidechain, f_out_of_plane, f_magnitude]
    assert projections.shape[1] == 5


def test_project_forces_magnitude_matches_norm(backbone_positions_single_residue):
    """Test that magnitude feature matches force norm."""
    # Create known force
    force_vector = jnp.array([1.0, 2.0, 3.0])
    forces = jnp.tile(force_vector, (1, 5, 1))  # Same force at all atoms
    
    projections = project_forces_onto_backbone(
        forces, backbone_positions_single_residue, aggregation="mean"
    )
    
    # Last feature should be magnitude
    expected_magnitude = jnp.linalg.norm(force_vector)
    assert jnp.allclose(projections[0, 4], expected_magnitude, rtol=1e-5)


def test_project_forces_aligned_with_forward():
    """Test projection when force is aligned with forward direction."""
    # Backbone along x-axis
    positions = jnp.array([[
        [0.0, 0.0, 0.0],      # N
        [1.0, 0.0, 0.0],      # CA
        [2.0, 0.0, 0.0],      # C
        [2.0, 1.0, 0.0],      # O
        [1.0, 0.0, 1.0],      # CB
    ]])

    # Force pointing along +x (forward direction) for all 5 atoms
    forces = jnp.ones((1, 5, 3))
    forces = forces.at[:, :, :].set(jnp.array([1.0, 0.0, 0.0]))

    projections = project_forces_onto_backbone(forces, positions, aggregation="mean")    # f_forward should be ~1.0
    # f_backward should be ~-1.0 (opposite direction)
    # f_sidechain should be ~0.0
    # f_out_of_plane should be ~0.0
    assert projections[0, 0] > 0.9  # f_forward
    assert projections[0, 1] < -0.9  # f_backward
    assert jnp.allclose(projections[0, 2], 0.0, atol=0.1)  # f_sidechain
    assert jnp.allclose(projections[0, 3], 0.0, atol=0.1)  # f_out_of_plane


def rotation_x(angle_deg: float) -> jnp.ndarray:
    """Rotation matrix around X-axis."""
    theta = jnp.radians(angle_deg)
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ])


def rotation_y(angle_deg: float) -> jnp.ndarray:
    """Rotation matrix around Y-axis."""
    theta = jnp.radians(angle_deg)
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([
        [ c,  0,  s],
        [ 0,  1,  0],
        [-s,  0,  c]
    ])


def rotation_z(angle_deg: float) -> jnp.ndarray:
    """Rotation matrix around Z-axis."""
    theta = jnp.radians(angle_deg)
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([
        [c, -s,  0],
        [s,  c,  0],
        [0,  0,  1]
    ])


def random_rotation_matrix_jax(key: jnp.ndarray) -> jnp.ndarray:
    """Generate uniformly distributed random rotation matrix using JAX.
    
    Args:
        key: JAX random key from jax.random.PRNGKey()
        
    Returns:
        Random rotation matrix (3, 3)
    """
    # Generate random quaternion uniformly on the 4-sphere
    key1, key2 = jax.random.split(key)
    
    u1 = jax.random.uniform(key1, minval=0, maxval=1)
    u2 = jax.random.uniform(key2, shape=(2,), minval=0, maxval=2*jnp.pi)
    
    # Convert to quaternion (Shoemake's method)
    q0 = jnp.sqrt(1 - u1) * jnp.sin(u2[0])
    q1 = jnp.sqrt(1 - u1) * jnp.cos(u2[0])
    q2 = jnp.sqrt(u1) * jnp.sin(u2[1])
    q3 = jnp.sqrt(u1) * jnp.cos(u2[1])
    
    q = jnp.array([q0, q1, q2, q3])
    
    # Normalize and convert quaternion to rotation matrix
    q = q / jnp.linalg.norm(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    R = jnp.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R


@pytest.mark.parametrize("rotation_fn,angle", [
    (rotation_x, 90),
    (rotation_x, 180),
    (rotation_x, 270),
    (rotation_y, 90),
    (rotation_y, 180),
    (rotation_y, 270),
    (rotation_z, 90),
    (rotation_z, 180),
    (rotation_z, 270),
])
def test_project_forces_rotation_invariance(rotation_fn, angle):
    """Test that projections are rotation invariant using exact rotations.
    
    This replaces the original skipped test. It uses exact 90°/180°/270° rotations
    which have perfect floating-point representations, eliminating numerical errors
    from the rotation itself.
    
    Benefits:
    - No scipy dependency
    - Numerically stable (exact rotations)
    - Comprehensive coverage (9 different rotations)
    - Fast execution
    """
    positions = jnp.array([[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ]])
    
    forces = jnp.ones((1, 5, 3))
    
    # Compute projections for original
    proj_original = project_forces_onto_backbone(forces, positions)
    
    # Apply rotation
    R = rotation_fn(angle)
    positions_rotated = jnp.einsum('bij,jk->bik', positions, R)
    forces_rotated = jnp.einsum('bij,jk->bik', forces, R)
    
    # Compute projections for rotated
    proj_rotated = project_forces_onto_backbone(forces_rotated, positions_rotated)
    
    # With exact rotations, tolerance can be very tight
    assert jnp.allclose(proj_original, proj_rotated, rtol=1e-6, atol=1e-7), \
        f"Rotation invariance failed for {rotation_fn.__name__}({angle}°)"

            
def test_rotation_matrices_are_valid():
    """Verify that our rotation generation produces valid rotation matrices."""
    
    for angle in [0, 90, 180, 270]:
        for rot_fn in [rotation_x, rotation_y, rotation_z]:
            R = rot_fn(angle)
            
            # Check orthogonality: R @ R.T = I
            assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-6)
            
            # Check determinant = 1
            assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-6)
    
    # Test random rotations
    for seed in range(10):
        key = jax.random.PRNGKey(seed)
        R = random_rotation_matrix_jax(key)
        
        # Check orthogonality
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-5)
        
        # Check determinant = 1
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-5)


def test_project_forces_per_atom_shape(backbone_positions_single_residue):
    """Test that per-atom projections have correct shape."""
    forces = jnp.ones((1, 5, 3))
    
    projections = project_forces_onto_backbone_per_atom(
        forces, backbone_positions_single_residue
    )
    
    # Should have 25 features (5 atoms × 5 projections)
    assert projections.shape == (1, 25)


def test_project_forces_aggregation_methods(backbone_positions_single_residue):
    """Test different aggregation methods."""
    forces = jnp.ones((1, 5, 3))
    
    proj_mean = project_forces_onto_backbone(
        forces, backbone_positions_single_residue, aggregation="mean"
    )
    proj_sum = project_forces_onto_backbone(
        forces, backbone_positions_single_residue, aggregation="sum"
    )
    
    # Sum should be 5x mean (5 atoms)
    assert jnp.allclose(proj_sum, proj_mean * 5.0, rtol=1e-5)


def test_project_forces_invalid_aggregation(backbone_positions_single_residue):
    """Test that invalid aggregation method raises error."""
    forces = jnp.ones((1, 5, 3))
    
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        project_forces_onto_backbone(
            forces, backbone_positions_single_residue, aggregation="invalid"
        )


def test_project_forces_is_jittable(backbone_positions_single_residue):
    """Test that projection can be JIT compiled."""
    forces = jnp.ones((1, 5, 3))
    
    jitted_fn = jax.jit(project_forces_onto_backbone)
    projections = jitted_fn(forces, backbone_positions_single_residue)
    
    assert jnp.all(jnp.isfinite(projections))


def test_project_forces_is_vmappable(backbone_positions_multi_residue):
    """Test that projection can be vmapped over batches."""
    # Create batch of force vectors
    batch_forces = jnp.ones((3, 3, 5, 3))  # 3 batches, 3 residues, 5 atoms
    
    # Vmap over batch dimension
    vmapped_fn = jax.vmap(
        lambda forces: project_forces_onto_backbone(
            forces, backbone_positions_multi_residue
        )
    )
    
    projections_batch = vmapped_fn(batch_forces)
    
    assert projections_batch.shape == (3, 3, 5)  # 3 batches, 3 residues, 5 features