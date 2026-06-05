"""Logit and score aggregation utilities for sampling output post-processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from jaxtyping import Array


def compute_pseudo_perplexity(
  sampled_logits: Array,
  sampled_sequences: Array,
  mask: Array | None = None,
) -> Array:
  """Compute pseudo-perplexity (negative log likelihood per residue).

  Pseudo-perplexity measures sequence likelihood by computing the average
  negative log-likelihood across masked residues. It provides a score
  for evaluating sampled sequence quality.

  Args:
    sampled_logits: Logits array with shape [batch, samples, noise, temp, seq_len, 21]
    sampled_sequences: One-hot encoded or integer sequences [batch, samples, noise, temp, seq_len]
    mask: Optional mask with shape [batch, seq_len]. If None, all residues are used.

  Returns:
    pseudo_perplexity: Array with shape [batch, samples, noise, temp] containing
      the exponentiated average negative log-likelihood per sequence.

  """
  one_hot_sequences = jax.nn.one_hot(sampled_sequences, num_classes=21)
  log_probs = jax.nn.log_softmax(sampled_logits, axis=-1)

  # Sum over sequence length (last 2 dims: one-hot @ log_softmax)
  # nll shape: [batch, samples, noise, temp]
  nll = -jnp.sum(one_hot_sequences * log_probs, axis=(-1, -2))

  # If mask not provided, assume all residues are valid
  if mask is None:
    # Assume the batch dimension is valid
    mask = jnp.ones((one_hot_sequences.shape[0], one_hot_sequences.shape[-2]), dtype=jnp.float32)

  # mask is [batch, seq_len]; sum over seq_len gives [batch]
  mask_sum = jnp.sum(mask, axis=-1)  # [batch]

  # Normalize by masked sum and broadcast to [batch, 1, 1, 1] for division
  pseudo_perplexity = jnp.exp(nll / mask_sum[:, None, None, None])

  return pseudo_perplexity


def pad_to_max(
  arr: jax.Array,
  target_len: int,
  axis: int = -1,
  pad_value: int = 0,
) -> jax.Array:
  """Pad the specified dimension of a JAX array to target_len.

  This utility is used to align logits and sequence arrays to the maximum
  sequence length before concatenation.

  Args:
    arr: JAX array to pad.
    target_len: Target length for the specified axis.
    axis: Axis along which to pad (default: -1, the last axis).
    pad_value: Value to use for padding (default: 0).

  Returns:
    Padded JAX array with shape [..., target_len, ...] along the specified axis.

  """
  diff = target_len - arr.shape[axis]
  if diff == 0:
    return arr
  padding_config = [(0, 0)] * arr.ndim
  # Handle negative axis
  axis = axis % arr.ndim
  padding_config[axis] = (0, diff)
  return jnp.pad(arr, padding_config, constant_values=pad_value)


def aggregate_logits(
  all_logits: list[jax.Array],
  max_len: int | None = None,
) -> jax.Array:
  """Aggregate logits from multiple batches by padding and concatenating.

  Args:
    all_logits: List of logit arrays, each with shape [batch, samples, noise, temp, seq_len, 21].
    max_len: Maximum sequence length. If None, computed from all_logits.

  Returns:
    Concatenated logits array with shape [total_batch, samples, noise, temp, seq_len, 21].

  """
  if not all_logits:
    return jnp.array([], dtype=jnp.float32)

  if max_len is None:
    max_len = max(arr.shape[-2] for arr in all_logits)

  # Pad logits along seq_len dimension (axis=-2)
  all_logits_padded = [pad_to_max(logits, max_len, axis=-2, pad_value=0) for logits in all_logits]

  return jnp.concatenate(all_logits_padded, axis=0)


def aggregate_pseudo_perplexities(
  all_pseudo_perplexities: list[jax.Array],
) -> jax.Array:
  """Aggregate pseudo-perplexity scores from multiple batches.

  Args:
    all_pseudo_perplexities: List of pseudo-perplexity arrays.

  Returns:
    Concatenated pseudo-perplexity array.

  """
  if not all_pseudo_perplexities:
    return jnp.array([], dtype=jnp.float32)

  return jnp.concatenate(all_pseudo_perplexities, axis=0)
