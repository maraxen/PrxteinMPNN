"""Tests for concatenation utilities."""

import chex
import jax
import jax.numpy as jnp

from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes


def test_concatenate_neighbor_nodes():
    """Test the concatenation of node features with neighbor edge features.

    Raises:
        AssertionError: If the output does not match the expected value.

    """
    L, K, C_V, C_E = 5, 4, 3, 2
    key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, 3)

    node_features = jax.random.normal(key1, (L, C_V))
    edge_features = jax.random.normal(key2, (L, K, C_E))
    neighbor_indices = jax.random.randint(key3, (L, K), 0, L)

    result = concatenate_neighbor_nodes(node_features, edge_features, neighbor_indices)

    chex.assert_shape(result, (L, K, C_V + C_E))
    chex.assert_type(result, node_features.dtype)

    # Check values
    gathered_nodes = node_features[neighbor_indices]
    expected = jnp.concatenate([edge_features, gathered_nodes], axis=-1)
    chex.assert_trees_all_close(result, expected)
