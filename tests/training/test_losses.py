"""Tests for loss functions."""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.training.losses import (
    cross_entropy_loss,
    perplexity,
    sequence_recovery_accuracy,
)


def test_cross_entropy_loss_basic(mock_logits, mock_targets, mock_mask):
    """Test basic cross-entropy loss computation."""
    loss = cross_entropy_loss(mock_logits, mock_targets, mock_mask)
    
    assert loss.shape == ()  # Scalar
    assert jnp.isfinite(loss)
    assert loss >= 0.0  # Loss should be non-negative


def test_cross_entropy_loss_with_label_smoothing(mock_logits, mock_targets, mock_mask):
    """Test cross-entropy loss with label smoothing."""
    loss_no_smoothing = cross_entropy_loss(
        mock_logits, mock_targets, mock_mask, label_smoothing=0.0
    )
    loss_with_smoothing = cross_entropy_loss(
        mock_logits, mock_targets, mock_mask, label_smoothing=0.1
    )
    
    # With smoothing, loss should be different (usually slightly higher)
    assert loss_no_smoothing != loss_with_smoothing


def test_cross_entropy_loss_with_mask():
    """Test that mask correctly zeros out invalid positions."""
    logits = jnp.ones((10, 21))
    targets = jnp.arange(10) % 21
    mask_full = jnp.ones(10)
    mask_half = jnp.concatenate([jnp.ones(5), jnp.zeros(5)])
    
    loss_full = cross_entropy_loss(logits, targets, mask_full)
    loss_half = cross_entropy_loss(logits, targets, mask_half)
    
    # Losses should be different due to masking
    assert loss_full != loss_half
    assert jnp.isfinite(loss_half)


def test_cross_entropy_loss_perfect_prediction():
    """Test loss with perfect predictions."""
    # Create logits that strongly predict the correct class
    targets = jnp.array([0, 1, 2, 3, 4])
    logits = jnp.zeros((5, 21))
    logits = logits.at[jnp.arange(5), targets].set(100.0)  # Very high for correct class
    mask = jnp.ones(5)
    
    loss = cross_entropy_loss(logits, targets, mask)
    
    # Loss should be very close to 0
    assert loss < 0.01


def test_sequence_recovery_accuracy(mock_logits, mock_targets, mock_mask):
    """Test sequence recovery accuracy computation."""
    accuracy = sequence_recovery_accuracy(mock_logits, mock_targets, mock_mask)
    
    assert accuracy.shape == ()  # Scalar
    assert 0.0 <= accuracy <= 1.0  # Should be in [0, 1]


def test_sequence_recovery_accuracy_perfect():
    """Test accuracy with perfect predictions."""
    targets = jnp.array([0, 1, 2, 3, 4])
    logits = jnp.zeros((5, 21))
    logits = logits.at[jnp.arange(5), targets].set(10.0)
    mask = jnp.ones(5)
    
    accuracy = sequence_recovery_accuracy(logits, targets, mask)
    
    assert accuracy == 1.0


def test_sequence_recovery_accuracy_zero():
    """Test accuracy with completely wrong predictions."""
    targets = jnp.array([0, 1, 2, 3, 4])
    logits = jnp.zeros((5, 21))
    # Predict wrong classes (add 1 to each target)
    wrong_targets = (targets + 1) % 21
    logits = logits.at[jnp.arange(5), wrong_targets].set(10.0)
    mask = jnp.ones(5)
    
    accuracy = sequence_recovery_accuracy(logits, targets, mask)
    
    assert accuracy == 0.0


def test_sequence_recovery_accuracy_with_mask():
    """Test accuracy computation with partial masking."""
    targets = jnp.array([0, 1, 2, 3, 4])
    logits = jnp.zeros((5, 21))
    logits = logits.at[jnp.arange(5), targets].set(10.0)
    
    # Mask out last 2 positions
    mask = jnp.array([1, 1, 1, 0, 0])
    
    accuracy = sequence_recovery_accuracy(logits, targets, mask)
    
    # Should be 100% for the 3 unmasked positions
    assert accuracy == 1.0


def test_perplexity(mock_logits, mock_targets, mock_mask):
    """Test perplexity computation."""
    ppl = perplexity(mock_logits, mock_targets, mock_mask)
    
    assert ppl.shape == ()  # Scalar
    assert ppl >= 1.0  # Perplexity should be >= 1


def test_perplexity_perfect_prediction():
    """Test perplexity with perfect predictions."""
    targets = jnp.array([0, 1, 2, 3, 4])
    logits = jnp.zeros((5, 21))
    logits = logits.at[jnp.arange(5), targets].set(100.0)
    mask = jnp.ones(5)
    
    ppl = perplexity(logits, targets, mask)
    
    # Perplexity should be very close to 1 for perfect predictions
    assert ppl < 1.1


def test_perplexity_random_prediction():
    """Test perplexity with random (uniform) predictions."""
    targets = jnp.array([0, 1, 2, 3, 4])
    logits = jnp.zeros((5, 21))  # Uniform distribution
    mask = jnp.ones(5)
    
    ppl = perplexity(logits, targets, mask)
    
    # For uniform distribution over 21 classes, perplexity should be close to 21
    assert 15 < ppl < 25


def test_loss_functions_are_jittable(mock_logits, mock_targets, mock_mask):
    """Test that loss functions can be JIT compiled."""
    jitted_ce = jax.jit(cross_entropy_loss)
    jitted_acc = jax.jit(sequence_recovery_accuracy)
    jitted_ppl = jax.jit(perplexity)
    
    # Should not raise
    loss = jitted_ce(mock_logits, mock_targets, mock_mask)
    acc = jitted_acc(mock_logits, mock_targets, mock_mask)
    ppl = jitted_ppl(mock_logits, mock_targets, mock_mask)
    
    assert jnp.isfinite(loss)
    assert jnp.isfinite(acc)
    assert jnp.isfinite(ppl)


def test_loss_functions_are_differentiable(mock_logits, mock_targets, mock_mask):
    """Test that loss functions are differentiable."""
    def loss_fn(logits):
        return cross_entropy_loss(logits, mock_targets, mock_mask)
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(mock_logits)
    
    assert grads.shape == mock_logits.shape
    assert jnp.all(jnp.isfinite(grads))