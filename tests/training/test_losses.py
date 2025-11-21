"""Tests for loss functions."""
import chex
import jax
import jax.numpy as jnp

from prxteinmpnn.training.losses import (
    cross_entropy_loss,
    perplexity,
    sequence_recovery_accuracy,
)


def test_cross_entropy_loss_basic(mock_logits, mock_targets, mock_mask, apply_jit):
    """Test basic cross-entropy loss computation."""
    loss_fn = apply_jit(cross_entropy_loss)
    loss = loss_fn(mock_logits, mock_targets, mock_mask)
    chex.assert_shape(loss, ())
    chex.assert_tree_all_finite(loss)
    assert loss >= 0.0
def test_cross_entropy_loss_with_label_smoothing(
    mock_logits, mock_targets, mock_mask, apply_jit,
):
    """Test cross-entropy loss with label smoothing."""
    loss_fn = apply_jit(cross_entropy_loss, static_argnames=["label_smoothing"])
    loss_no_smoothing = loss_fn(
        mock_logits, mock_targets, mock_mask, label_smoothing=0.0,
    )
    loss_with_smoothing = loss_fn(
        mock_logits, mock_targets, mock_mask, label_smoothing=0.1,
    )
    assert not jnp.allclose(loss_no_smoothing, loss_with_smoothing)
def test_cross_entropy_loss_with_mask(apply_jit):
    """Test that mask correctly zeros out invalid positions."""
    loss_fn = apply_jit(cross_entropy_loss)
    logits = jnp.ones((10, 21))
    targets = jnp.arange(10) % 21
    mask_full = jnp.ones(10)
    mask_half = jnp.concatenate([jnp.ones(5), jnp.zeros(5)])
    loss_full = loss_fn(logits, targets, mask_full)
    loss_half = loss_fn(logits, targets, mask_half)
    assert jnp.allclose(loss_full, loss_half)
    chex.assert_tree_all_finite(loss_half)
def test_cross_entropy_loss_perfect_prediction(apply_jit):
    """Test loss with perfect predictions."""
    loss_fn = apply_jit(cross_entropy_loss)
    targets = jnp.array([0, 1, 2, 3, 4])
    logits = jnp.zeros((5, 21))
    logits = logits.at[jnp.arange(5), targets].set(100.0)
    mask = jnp.ones(5)
    loss = loss_fn(logits, targets, mask)
    assert loss < 0.01
def test_sequence_recovery_accuracy(mock_logits, mock_targets, mock_mask, apply_jit):
    """Test sequence recovery accuracy computation."""
    acc_fn = apply_jit(sequence_recovery_accuracy)
    accuracy = acc_fn(mock_logits, mock_targets, mock_mask)
    chex.assert_shape(accuracy, ())
    chex.assert_tree_all_finite(accuracy)
    assert 0.0 <= accuracy <= 1.0
def test_sequence_recovery_accuracy_perfect(apply_jit):
    """Test accuracy with perfect predictions."""
    acc_fn = apply_jit(sequence_recovery_accuracy)
    targets = jnp.array([0, 1, 2, 3, 4])
    logits = jnp.zeros((5, 21))
    logits = logits.at[jnp.arange(5), targets].set(10.0)
    mask = jnp.ones(5)
    accuracy = acc_fn(logits, targets, mask)
    assert accuracy == 1.0
def test_sequence_recovery_accuracy_zero(apply_jit):
    """Test accuracy with completely wrong predictions."""
    acc_fn = apply_jit(sequence_recovery_accuracy)
    targets = jnp.array([0, 1, 2, 3, 4])
    logits = jnp.zeros((5, 21))
    wrong_targets = (targets + 1) % 21
    logits = logits.at[jnp.arange(5), wrong_targets].set(10.0)
    mask = jnp.ones(5)
    accuracy = acc_fn(logits, targets, mask)
    assert accuracy == 0.0
def test_sequence_recovery_accuracy_with_mask(apply_jit):
    """Test accuracy computation with partial masking."""
    acc_fn = apply_jit(sequence_recovery_accuracy)
    targets = jnp.array([0, 1, 2, 3, 4])
    logits = jnp.zeros((5, 21))
    logits = logits.at[jnp.arange(5), targets].set(10.0)
    mask = jnp.array([1, 1, 1, 0, 0])
    accuracy = acc_fn(logits, targets, mask)
    assert accuracy == 1.0
def test_perplexity(mock_logits, mock_targets, mock_mask, apply_jit):
    """Test perplexity computation."""
    ppl_fn = apply_jit(perplexity)
    ppl = ppl_fn(mock_logits, mock_targets, mock_mask)
    chex.assert_shape(ppl, ())
    chex.assert_tree_all_finite(ppl)
    assert ppl >= 1.0
def test_perplexity_perfect_prediction(apply_jit):
    """Test perplexity with perfect predictions."""
    ppl_fn = apply_jit(perplexity)
    targets = jnp.array([0, 1, 2, 3, 4])
    logits = jnp.zeros((5, 21))
    logits = logits.at[jnp.arange(5), targets].set(100.0)
    mask = jnp.ones(5)
    ppl = ppl_fn(logits, targets, mask)
    assert ppl < 1.1
def test_perplexity_random_prediction(apply_jit):
    """Test perplexity with random (uniform) predictions."""
    ppl_fn = apply_jit(perplexity)
    targets = jnp.array([0, 1, 2, 3, 4])
    logits = jnp.zeros((5, 21))
    mask = jnp.ones(5)
    ppl = ppl_fn(logits, targets, mask)
    assert 15 < ppl < 25
def test_loss_functions_are_differentiable(mock_logits, mock_targets, mock_mask):
    """Test that loss functions are differentiable."""
    def loss_fn(logits):
        return cross_entropy_loss(logits, mock_targets, mock_mask)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(mock_logits)
    chex.assert_shape(grads, mock_logits.shape)
    chex.assert_tree_all_finite(grads)
