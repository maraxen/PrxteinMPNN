"""Multi-state protein design utilities for tied positions.

This module provides functions for combining logits across tied positions
representing the same residue in different conformational states, with support
for state-specific weighting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
  from jaxtyping import ArrayLike

  from prxteinmpnn.utils.types import Logits


def arithmetic_mean_logits(
  logits: Logits,
  group_mask: ArrayLike,
  state_weights: jnp.ndarray | None = None,
  state_mapping: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """Average logits across positions in a tie group using log-sum-exp.

  This implements numerically stable weighted logit averaging for tied positions.
  Given logits of shape (N, 21) and a boolean mask indicating which
  positions belong to the current group, returns averaged logits of shape (1, 21).

  If state_weights and state_mapping are provided, it computes a weighted average:
  L_combined = log(sum(w_i * exp(L_i)) / sum(w_i))
             = log_sum_exp(L_i + log(w_i)) - log(sum(w_i))

  Args:
    logits: Logits array of shape (N, 21).
    group_mask: Boolean mask of shape (N,) indicating group membership.
    state_weights: Optional weights for each state, shape (num_states,).
    state_mapping: Optional mapping of each residue to its state index, shape (N,).

  Returns:
    Averaged logits of shape (1, 21).

  """
  group_mask = jnp.asarray(group_mask)

  if state_weights is not None and state_mapping is not None:
    # Get weights for each residue in the group
    w = state_weights[state_mapping]
    # log(w) with safety for zero weights
    log_w = jnp.log(jnp.where(w > 0, w, 1e-9))
    weighted_logits = logits + log_w[:, None]

    max_logits = jnp.max(
      weighted_logits,
      where=group_mask[:, None],
      initial=-1e9,
      axis=0,
      keepdims=True,
    )
    shifted_logits = weighted_logits - max_logits
    exp_logits = jnp.exp(shifted_logits)
    masked_exp_logits = jnp.where(group_mask[:, None], exp_logits, 0.0)
    sum_exp_logits = jnp.sum(masked_exp_logits, axis=0, keepdims=True)

    # Normalize by sum of weights in group
    sum_w = jnp.sum(jnp.where(group_mask, w, 0.0))
    return jnp.log(sum_exp_logits / jnp.where(sum_w > 0, sum_w, 1.0)) + max_logits

  # Default uniform case
  max_logits = jnp.max(
    logits,
    where=group_mask[:, None],
    initial=-1e9,
    axis=0,
    keepdims=True,
  )

  shifted_logits = logits - max_logits
  exp_logits = jnp.exp(shifted_logits)

  masked_exp_logits = jnp.where(group_mask[:, None], exp_logits, 0.0)
  sum_exp_logits = jnp.sum(masked_exp_logits, axis=0, keepdims=True)

  num_in_group = jnp.sum(group_mask)
  avg_exp_logits = sum_exp_logits / num_in_group
  return jnp.log(avg_exp_logits) + max_logits


def geometric_mean_logits(
  logits: Logits,
  group_mask: ArrayLike,
  temperature: float,
  state_weights: jnp.ndarray | None = None,
  state_mapping: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """Combine logits using geometric mean in probability space.

  This computes the weighted geometric mean of probabilities across states:
  log P_geom = sum(w_i * logits_i / temp) / sum(w_i)

  Args:
    logits: Logits array of shape (N, 21).
    group_mask: Boolean mask of shape (N,) for group positions.
    temperature: Sampling temperature for scaling.
    state_weights: Optional weights for each state, shape (num_states,).
    state_mapping: Optional mapping of each residue to its state index, shape (N,).

  Returns:
    Combined logits of shape (1, 21).
  """
  group_mask = jnp.asarray(group_mask)

  if state_weights is not None and state_mapping is not None:
    w = state_weights[state_mapping]
    weighted_logits = logits * w[:, None]
    masked_weighted_logits = jnp.where(group_mask[:, None], weighted_logits, 0.0)
    sum_weighted_logits = jnp.sum(masked_weighted_logits, axis=0, keepdims=True)
    sum_w = jnp.sum(jnp.where(group_mask, w, 0.0))
    return sum_weighted_logits / (temperature * jnp.where(sum_w > 0, sum_w, 1.0))

  # Default uniform case
  masked_logits = jnp.where(group_mask[:, None], logits, 0.0)
  sum_logits = jnp.sum(masked_logits, axis=0, keepdims=True)
  num_in_group = jnp.sum(group_mask)

  return sum_logits / (temperature * num_in_group)


def product_of_probabilities_logits(
  logits: Logits,
  group_mask: ArrayLike,
  state_weights: jnp.ndarray | None = None,
  state_mapping: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """Combine states by multiplying probabilities (summing log-probabilities).

  log P_combined = sum(w_i * logits_i)

  Args:
    logits: Logits array of shape (N, 21).
    group_mask: Boolean mask of shape (N,) for group positions.
    state_weights: Optional weights for each state, shape (num_states,).
    state_mapping: Optional mapping of each residue to its state index, shape (N,).

  Returns:
    Sum of weighted logits of shape (1, 21).
  """
  group_mask = jnp.asarray(group_mask)

  if state_weights is not None and state_mapping is not None:
    w = state_weights[state_mapping]
    weighted_logits = logits * w[:, None]
    masked_weighted_logits = jnp.where(group_mask[:, None], weighted_logits, 0.0)
    return jnp.sum(masked_weighted_logits, axis=0, keepdims=True)

  # Default uniform case
  masked_logits = jnp.where(group_mask[:, None], logits, 0.0)
  return jnp.sum(masked_logits, axis=0, keepdims=True)
