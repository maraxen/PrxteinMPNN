"""Test suite for the spatial module."""
import jax.numpy as jnp
from prxteinmpnn.psa.spatial import cartesian_to_cylindrical

def test_cartesian_to_cylindrical():
    cartesian_coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    cylindrical_coords = cartesian_to_cylindrical(cartesian_coords)
    expected_coords = jnp.array(
        [[1.0, 0.0, 0.0], [1.0, jnp.pi / 2, 0.0], [jnp.sqrt(2.0), jnp.pi / 4, 1.0]]
    )
    assert jnp.allclose(cylindrical_coords, expected_coords)
