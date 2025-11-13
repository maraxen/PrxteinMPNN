"""Tests for training metrics."""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.training.metrics import (
    EvaluationMetrics,
    TrainingMetrics,
    compute_grad_norm,
)


def test_training_metrics_creation():
    """Test creating TrainingMetrics."""
    metrics = TrainingMetrics(
        loss=jnp.array(1.5),
        accuracy=jnp.array(0.75),
        perplexity=jnp.array(4.5),
        learning_rate=1e-4,
        grad_norm=jnp.array(0.5),
    )
    
    assert metrics.loss == 1.5
    assert metrics.accuracy == 0.75
    assert metrics.perplexity == 4.5
    assert metrics.learning_rate == 1e-4
    assert metrics.grad_norm == 0.5


def test_training_metrics_to_dict():
    """Test converting TrainingMetrics to dictionary."""
    metrics = TrainingMetrics(
        loss=jnp.array(1.5),
        accuracy=jnp.array(0.75),
        perplexity=jnp.array(4.5),
        learning_rate=1e-4,
        grad_norm=jnp.array(0.5),
    )
    
    metrics_dict = metrics.to_dict()
    
    assert isinstance(metrics_dict, dict)
    assert "loss" in metrics_dict
    assert "accuracy" in metrics_dict
    assert "perplexity" in metrics_dict
    assert "learning_rate" in metrics_dict
    assert "grad_norm" in metrics_dict
    
    # All values should be Python floats, not JAX arrays
    assert isinstance(metrics_dict["loss"], float)
    assert isinstance(metrics_dict["accuracy"], float)


def test_training_metrics_without_grad_norm():
    """Test TrainingMetrics without gradient norm."""
    metrics = TrainingMetrics(
        loss=jnp.array(1.5),
        accuracy=jnp.array(0.75),
        perplexity=jnp.array(4.5),
        learning_rate=1e-4,
        grad_norm=None,
    )
    
    metrics_dict = metrics.to_dict()
    assert "grad_norm" not in metrics_dict


def test_evaluation_metrics_creation():
    """Test creating EvaluationMetrics."""
    metrics = EvaluationMetrics(
        val_loss=jnp.array(1.2),
        val_accuracy=jnp.array(0.85),
        val_perplexity=jnp.array(3.5),
    )
    
    assert metrics.val_loss == 1.2
    assert metrics.val_accuracy == 0.85
    assert metrics.val_perplexity == 3.5


def test_evaluation_metrics_to_dict():
    """Test converting EvaluationMetrics to dictionary."""
    metrics = EvaluationMetrics(
        val_loss=jnp.array(1.2),
        val_accuracy=jnp.array(0.85),
        val_perplexity=jnp.array(3.5),
    )
    
    metrics_dict = metrics.to_dict()
    
    assert isinstance(metrics_dict, dict)
    assert "val_loss" in metrics_dict
    assert "val_accuracy" in metrics_dict
    assert "val_perplexity" in metrics_dict


def test_compute_grad_norm_simple():
    """Test gradient norm computation with simple gradients."""
    # Create a simple gradient PyTree
    grads = {
        "layer1": jnp.ones((10, 10)),
        "layer2": jnp.ones((5, 5)),
    }
    
    grad_norm = compute_grad_norm(grads)
    
    # Expected: sqrt(10*10*1^2 + 5*5*1^2) = sqrt(125) ≈ 11.18
    expected = jnp.sqrt(100.0 + 25.0)
    assert jnp.allclose(grad_norm, expected)


def test_compute_grad_norm_nested():
    """Test gradient norm with nested PyTree structure."""
    grads = {
        "encoder": {
            "layer1": jnp.ones((5, 5)),
            "layer2": jnp.ones((3, 3)),
        },
        "decoder": {
            "layer1": jnp.ones((4, 4)),
        },
    }
    
    grad_norm = compute_grad_norm(grads)
    
    # Expected: sqrt(5*5 + 3*3 + 4*4) = sqrt(50)
    expected = jnp.sqrt(25.0 + 9.0 + 16.0)
    assert jnp.allclose(grad_norm, expected)


def test_compute_grad_norm_with_zeros():
    """Test gradient norm when all gradients are zero."""
    grads = {
        "layer1": jnp.zeros((10, 10)),
        "layer2": jnp.zeros((5, 5)),
    }
    
    grad_norm = compute_grad_norm(grads)
    
    assert grad_norm == 0.0


def test_compute_grad_norm_is_jittable():
    """Test that gradient norm computation can be JIT compiled."""
    grads = {
        "layer1": jnp.ones((5, 5)),
        "layer2": jnp.ones((3, 3)),
    }
    
    jitted_fn = jax.jit(compute_grad_norm)
    grad_norm = jitted_fn(grads)
    
    assert jnp.isfinite(grad_norm)
    assert grad_norm > 0