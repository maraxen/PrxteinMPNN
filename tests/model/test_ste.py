"""Tests for the ste module."""
import chex
import jax
import jax.numpy as jnp
from prxteinmpnn.model.ste import straight_through_estimator, ste_loss


def test_straight_through_estimator():
    """Test the straight_through_estimator function."""
    logits = jnp.array([[0.1, 0.9], [0.8, 0.2]])
    one_hot = straight_through_estimator(logits)
    chex.assert_shape(one_hot, (2, 2))
    chex.assert_trees_all_close(jnp.argmax(one_hot, axis=-1), jnp.array([1, 0]))


def test_ste_loss():
    """Test the ste_loss function."""
    logits_to_optimize = jnp.array([[0.1, 0.9], [0.8, 0.2]])
    target_logits = jnp.array([[0.2, 0.8], [0.7, 0.3]])
    mask = jnp.array([True, True])
    loss = ste_loss(logits_to_optimize, target_logits, mask)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()


def test_ste_loss_grad():
    """Test the gradient of the ste_loss function."""
    logits_to_optimize = jnp.array([[0.1, 0.9], [0.8, 0.2]])
    target_logits = jnp.array([[0.2, 0.8], [0.7, 0.3]])
    mask = jnp.array([True, True])
    grad_fn = jax.grad(ste_loss)
    grads = grad_fn(logits_to_optimize, target_logits, mask)
    chex.assert_shape(grads, (2, 2))
    assert not jnp.allclose(grads, 0.0)
