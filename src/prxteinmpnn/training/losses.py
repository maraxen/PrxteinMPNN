"""Loss functions for training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import Logits, ProteinSequence


def cross_entropy_loss(
  logits: Logits,  # (N, 21)
  targets: ProteinSequence,  # (N,) integer sequence
  mask: jnp.ndarray,  # (N,) binary mask
  label_smoothing: float = 0.0,
) -> jax.Array:
  """Compute masked cross-entropy loss with optional label smoothing.

  Args:
      logits: Model predictions of shape (N, 21)
      targets: Ground truth sequence of shape (N,) with integer labels [0-20]
      mask: Binary mask of shape (N,) indicating valid positions
      label_smoothing: Label smoothing factor in [0, 1]

  Returns:
      Scalar loss value

  Example:
      >>> logits = jnp.ones((10, 21))
      >>> targets = jnp.arange(10) % 21
      >>> mask = jnp.ones(10)
      >>> loss = cross_entropy_loss(logits, targets, mask)

  """
  num_classes = logits.shape[-1]

  # One-hot encode targets
  targets_onehot = jax.nn.one_hot(targets, num_classes)  # (N, 21)

  # Apply label smoothing
  if label_smoothing > 0:
    targets_smooth = (1.0 - label_smoothing) * targets_onehot + label_smoothing / num_classes
  else:
    targets_smooth = targets_onehot

  # Compute log probabilities
  log_probs = jax.nn.log_softmax(logits, axis=-1)  # (N, 21)

  # Compute cross-entropy
  loss_per_position = -jnp.sum(targets_smooth * log_probs, axis=-1)  # (N,)

  # Apply mask and average
  masked_loss = loss_per_position * mask
  total_loss = jnp.sum(masked_loss)
  num_valid = jnp.sum(mask)

  return total_loss / jnp.maximum(num_valid, 1.0)


def sequence_recovery_accuracy(
  logits: Logits,  # (N, 21)
  targets: ProteinSequence,  # (N,)
  mask: jnp.ndarray,  # (N,)
) -> jax.Array:
  """Compute sequence recovery accuracy (percentage of correct predictions).

  Args:
      logits: Model predictions of shape (N, 21)
      targets: Ground truth sequence of shape (N,)
      mask: Binary mask of shape (N,)

  Returns:
      Accuracy as a scalar in [0, 1]

  """
  predictions = jnp.argmax(logits, axis=-1)  # (N,)
  correct = (predictions == targets).astype(jnp.float32)  # (N,)
  masked_correct = correct * mask

  return jnp.sum(masked_correct) / jnp.maximum(jnp.sum(mask), 1.0)


def perplexity(
  logits: Logits,
  targets: ProteinSequence,
  mask: jnp.ndarray,
) -> jax.Array:
  """Compute perplexity (exp of cross-entropy loss).

  Lower perplexity indicates better model confidence.

  Args:
      logits: Model predictions of shape (N, 21)
      targets: Ground truth sequence of shape (N,)
      mask: Binary mask of shape (N,)

  Returns:
      Perplexity as a scalar

  """
  ce_loss = cross_entropy_loss(logits, targets, mask, label_smoothing=0.0)
  return jnp.exp(ce_loss)
