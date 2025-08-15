"""Straight-Through Estimator (STE) for JAX.

prxteinmpnn.sampling.ste

Note: Only use this for discrete optimization problems where you want to allow gradients
to pass through the argmax operation. Useful for tasks like protein sequence
optimization when a model outputs logits for amino acid sequences.

Unclear if the optimized sequences will be valid proteins, so this is a heuristic
approach to allow gradient-based optimization on discrete outputs to assess how well
other samplers are navigating model landscapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import (
    AtomMask,
    CEELoss,
    Logits,
  )

DEFAULT_EPS = 1e-8


def straight_through_estimator(logits: Logits) -> Logits:
  """Implement the straight-through estimator (STE).

  Allow gradients to pass through the discrete argmax operation.
  """
  probs = jax.nn.softmax(logits, axis=-1)
  one_hot = jax.nn.one_hot(jnp.argmax(probs, axis=-1), num_classes=probs.shape[-1])
  return jax.lax.stop_gradient(one_hot - probs) + probs


def ste_loss(
  logits_to_optimize: Logits,
  target_logits: Logits,
  mask: AtomMask,
  eps: float = DEFAULT_EPS,
) -> CEELoss:
  """Calculate cross-entropy between one-hot sequence (from STE) and target distribution.

  Args:
      logits_to_optimize: Logits to optimize, shape (sequence_length, num_classes).
      target_logits: Target logits for the sequence, shape (sequence_length, num_classes).
        These are the logits from the model that we want to match, such as MPNN model's
        unconditional logits.
      mask: Boolean mask indicating valid positions in the sequence.
        Used to ignore padding or invalid positions.
      eps: Small value to avoid division by zero.

  Returns:
      Loss value as a scalar.

  Example:
      >>> logits_to_optimize = jnp.array([[0.1, 0.9], [0.8, 0.2]])
      >>> target_logits = jnp.array([[0.2, 0.8], [0.7, 0.3]])
      >>> mask = jnp.array([True, True])
      >>> loss = ste_loss(logits_to_optimize, target_logits, mask)

  """
  seq_one_hot = straight_through_estimator(logits_to_optimize)
  target_log_probs = jax.nn.log_softmax(target_logits)
  loss_per_position = -(seq_one_hot * target_log_probs).sum(axis=-1)
  return (loss_per_position * mask).sum() / (mask.sum() + eps)
