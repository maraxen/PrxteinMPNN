"""Training and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class TrainingMetrics:
  """Container for training metrics."""

  loss: jax.Array
  accuracy: jax.Array
  perplexity: jax.Array
  learning_rate: float
  grad_norm: jax.Array | None = None

  def to_dict(self) -> dict[str, float]:
    """Convert metrics to dictionary for logging."""
    metrics_dict = {
      "loss": float(self.loss),
      "accuracy": float(self.accuracy),
      "perplexity": float(self.perplexity),
      "learning_rate": self.learning_rate,
    }
    if self.grad_norm is not None:
      metrics_dict["grad_norm"] = float(self.grad_norm)
    return metrics_dict


@dataclass
class EvaluationMetrics:
  """Container for evaluation metrics."""

  val_loss: jax.Array
  val_accuracy: jax.Array
  val_perplexity: jax.Array

  def to_dict(self) -> dict[str, float]:
    """Convert metrics to dictionary for logging."""
    return {
      "val_loss": float(self.val_loss),
      "val_accuracy": float(self.val_accuracy),
      "val_perplexity": float(self.val_perplexity),
    }


def compute_grad_norm(grads: dict) -> jax.Array:
  """Compute global gradient norm across all parameters.

  Args:
      grads: PyTree of gradients

  Returns:
      Global gradient norm

  """
  leaves = jax.tree_util.tree_leaves(grads)
  return jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves))
