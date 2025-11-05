"""Unit tests for the Straight-Through Estimator (STE) module."""

import jax
import jax.numpy as jnp
import chex

from prxteinmpnn.utils import ste


def test_straight_through_estimator():
    """Test the straight_through_estimator function for correct shape and values."""
    logits = jnp.array([[0.1, 0.9], [0.8, 0.2]])
    result = ste.straight_through_estimator(logits)
    expected_one_hot = jnp.array([[0.0, 1.0], [1.0, 0.0]])

    # Check that the output is close to the one-hot encoding of the argmax
    chex.assert_trees_all_close(
        jnp.argmax(result, axis=-1), jnp.argmax(expected_one_hot, axis=-1)
    )
    chex.assert_shape(result, logits.shape)


def test_ste_loss():
    """Test the ste_loss function for correctness."""
    logits_to_optimize = jnp.array([[0.1, 0.9], [0.8, 0.2]])
    target_logits = jnp.array([[0.2, 0.8], [0.7, 0.3]])
    mask = jnp.array([True, True])

    loss = ste.ste_loss(logits_to_optimize, target_logits, mask)

    # Manually compute the expected loss
    seq_one_hot = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    target_log_probs = jax.nn.log_softmax(target_logits)
    expected_loss_per_position = -(seq_one_hot * target_log_probs).sum(axis=-1)
    expected_loss = (expected_loss_per_position * mask).sum() / mask.sum()

    chex.assert_trees_all_close(loss, expected_loss)


def test_ste_loss_with_mask():
    """Test the ste_loss function with a mask to ignore certain positions."""
    logits_to_optimize = jnp.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
    target_logits = jnp.array([[0.2, 0.8], [0.7, 0.3], [0.6, 0.4]])
    mask = jnp.array([True, False, True])

    loss = ste.ste_loss(logits_to_optimize, target_logits, mask)

    # Manually compute the expected loss for the unmasked positions
    seq_one_hot = ste.straight_through_estimator(logits_to_optimize)
    target_log_probs = jax.nn.log_softmax(target_logits)
    expected_loss_per_position = -(seq_one_hot * target_log_probs).sum(axis=-1)
    expected_loss = (expected_loss_per_position * mask).sum() / mask.sum()

    chex.assert_trees_all_close(loss, expected_loss, atol=1e-6)


def test_ste_gradient_flow():
    """Test that gradients flow through the ste_loss function."""
    logits_to_optimize = jnp.array([[0.1, 0.9], [0.8, 0.2]])
    target_logits = jnp.array([[0.2, 0.8], [0.7, 0.3]])
    mask = jnp.array([True, True])

    grad_fn = jax.grad(ste.ste_loss, argnums=0)
    grads = grad_fn(logits_to_optimize, target_logits, mask)

    # Check that gradients are not all zero
    assert not jnp.all(grads == 0)
    chex.assert_shape(grads, logits_to_optimize.shape)
