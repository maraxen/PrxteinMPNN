"""Training and evaluation metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class TrainingMetrics:
  """Container for training metrics."""

  loss: jax.Array
  accuracy: jax.Array
  perplexity: jax.Array
  learning_rate: float
  grad_norm: jax.Array | None = None

  def to_dict(self) -> dict[str, float | None]:
    """Convert metrics to a dictionary of Python floats."""
    metrics_dict = {
        "loss": float(jax.device_get(self.loss)),
        "accuracy": float(jax.device_get(self.accuracy)),
        "perplexity": float(jax.device_get(self.perplexity)),
        "learning_rate": float(self.learning_rate),
    }
    if self.grad_norm is not None:
        metrics_dict["grad_norm"] = float(jax.device_get(self.grad_norm))
    return metrics_dict


@dataclass
class EvaluationMetrics:
  """Container for evaluation metrics."""

  val_loss: jax.Array
  val_accuracy: jax.Array
  val_perplexity: jax.Array

  def to_dict(self) -> dict[str, float]:
    """Convert metrics to a dictionary of Python floats."""
    return {
        "val_loss": float(jax.device_get(self.val_loss)),
        "val_accuracy": float(jax.device_get(self.val_accuracy)),
        "val_perplexity": float(jax.device_get(self.val_perplexity)),
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
