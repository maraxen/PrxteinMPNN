"""Tests for training metrics."""
import chex
import jax.numpy as jnp

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
class TestGradNorm(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_grad_norm_simple(self):
        """Test gradient norm computation with simple gradients."""
        grads = {
            "layer1": jnp.ones((10, 10)),
            "layer2": jnp.ones((5, 5)),
        }
        grad_norm_fn = self.variant(compute_grad_norm)
        grad_norm = grad_norm_fn(grads)
        expected = jnp.sqrt(100.0 + 25.0)
        chex.assert_trees_all_close(grad_norm, expected)
        chex.assert_tree_all_finite(grad_norm)
    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_grad_norm_nested(self):
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
        grad_norm_fn = self.variant(compute_grad_norm)
        grad_norm = grad_norm_fn(grads)
        expected = jnp.sqrt(25.0 + 9.0 + 16.0)
        chex.assert_trees_all_close(grad_norm, expected)
        chex.assert_tree_all_finite(grad_norm)
    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_grad_norm_with_zeros(self):
        """Test gradient norm when all gradients are zero."""
        grads = {
            "layer1": jnp.zeros((10, 10)),
            "layer2": jnp.zeros((5, 5)),
        }
        grad_norm_fn = self.variant(compute_grad_norm)
        grad_norm = grad_norm_fn(grads)
        assert grad_norm == 0.0
