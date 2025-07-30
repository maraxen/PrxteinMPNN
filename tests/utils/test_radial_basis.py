"""Tests for radial basis function utilities."""

import chex
import jax
import jax.numpy as jnp
from prxteinmpnn.utils.radial_basis import compute_radial_basis, RADIAL_BASES


def test_compute_radial_basis_shape():
    """Test the output shape of the radial basis function computation.

    Raises:
        AssertionError: If the output shape is incorrect.
    """
    L, K = 10, 8  # Num residues, num neighbors
    backbone_coords = jnp.zeros((L, 5, 3))
    key = jax.random.PRNGKey(0)
    neighbor_indices = jax.random.randint(key, (L, K), 0, L)

    rbf_output = compute_radial_basis(backbone_coords, neighbor_indices)

    # Expected shape is (L, K, num_pairs * num_bases)
    # num_pairs is 25, num_bases is 16
    expected_shape = (L, K, 25 * RADIAL_BASES)
    chex.assert_shape(rbf_output, expected_shape)
    chex.assert_type(rbf_output, backbone_coords.dtype)


def test_compute_radial_basis_values():
    """Test the output values of the RBF for a simple case.

    Raises:
        AssertionError: If the output values are incorrect.
    """
    L, K = 2, 1
    # Set all atom coords to zero. All distances will be zero.
    backbone_coords = jnp.zeros((L, 5, 3))
    # Each residue is its own neighbor
    neighbor_indices = jnp.array([[0], [1]])

    rbf_output = compute_radial_basis(backbone_coords, neighbor_indices)
    distance = jnp.sqrt(1e-6)  # From the implementation

    # For a distance of ~0, the exponent is -(centers^2 / sigma^2)
    from prxteinmpnn.utils.radial_basis import RBF_CENTERS, RBF_SIGMA

    expected_rbf_values = jnp.exp(-((distance - RBF_CENTERS) ** 2) / RBF_SIGMA**2)

    # All 25 atom pairs have the same distance (0), so all RBFs should be the same.
    # We check the RBF for the first residue, first neighbor, first atom pair.
    chex.assert_trees_all_close(
        rbf_output[0, 0, :RADIAL_BASES], expected_rbf_values, atol=1e-5
    )
    # And for the last atom pair
    chex.assert_trees_all_close(
        rbf_output[0, 0, -RADIAL_BASES:], expected_rbf_values, atol=1e-5
    )