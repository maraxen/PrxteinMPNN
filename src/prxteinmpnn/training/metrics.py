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


@dataclass
class EvaluationMetrics:
  """Container for evaluation metrics."""

  val_loss: jax.Array
  val_accuracy: jax.Array
  val_perplexity: jax.Array


def compute_grad_norm(grads: dict) -> jax.Array:
  """Compute global gradient norm across all parameters.

  Args:
      grads: PyTree of gradients

  Returns:
      Global gradient norm

  """
  leaves = jax.tree_util.tree_leaves(grads)
  return jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves))
