"""Tests for the masked_attention module."""
import chex
import jax.numpy as jnp
from prxteinmpnn.model.masked_attention import mask_attention


def test_mask_attention():
    """Test the mask_attention function."""
    message = jnp.ones((10, 5, 128))
    attention_mask = jnp.array([[1, 1, 0, 0, 0] for _ in range(10)])
    masked_message = mask_attention(message, attention_mask)
    chex.assert_shape(masked_message, (10, 5, 128))
    chex.assert_trees_all_close(masked_message[:, 2:, :], jnp.zeros((10, 3, 128)))
