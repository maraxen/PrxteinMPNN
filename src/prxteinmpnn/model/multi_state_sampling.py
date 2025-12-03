"""Multi-state protein design utilities for tied positions.

This module provides functions for combining logits across tied positions
representing the same residue in different conformational states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import Logits


def arithmetic_mean_logits(
  logits: Logits,
  group_mask: jnp.ndarray,
) -> jnp.ndarray:
  """Average logits across positions in a tie group using log-sum-exp.

  This implements numerically stable logit averaging for tied positions.
  Given logits of shape (N, 21) and a boolean mask indicating which
  positions belong to the current group, returns averaged logits of shape (1, 21).

  This is the standard approach for combining predictions across multiple states,
  treating each state's prediction equally and finding a consensus.

  Args:
    logits: Logits array of shape (N, 21).
    group_mask: Boolean mask of shape (N,) indicating group membership.

  Returns:
    Averaged logits of shape (1, 21).

  Example:
    >>> logits = jnp.array([[0.1, 0.9], [0.3, 0.7]])
    >>> group_mask = jnp.array([True, True])
    >>> avg_logits = arithmetic_mean_logits(logits, group_mask)

  """
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
  group_mask: jnp.ndarray,
  temperature: float,
) -> jnp.ndarray:
  """Combine logits using geometric mean in probability space.

  This computes the geometric mean of probabilities across states:
  P_geom = (P_1 * P_2 * ... * P_N)^(1/N)

  In log space with temperature scaling:
  log P_geom = (sum(logits) / temperature) / N

  Args:
    logits: Logits array of shape (N, 21).
    group_mask: Boolean mask of shape (N,) for group positions.
    temperature: Sampling temperature for scaling.

  Returns:
    Combined logits of shape (1, 21).

  Example:
    >>> logits = jnp.array([
    ...     [10.0, -5.0, 0.0],  # State 1
    ...     [8.0, -3.0, 0.0],   # State 2
    ... ])
    >>> group_mask = jnp.array([True, True])
    >>> temp = 1.0
    >>> geom_logits = geometric_mean_logits(logits, group_mask, temp)
    >>> # geom_logits = (10.0 + 8.0) / (1.0 * 2) = 9.0 for AA0

  """
  masked_logits = jnp.where(group_mask[:, None], logits, 0.0)
  sum_logits = jnp.sum(masked_logits, axis=0, keepdims=True)
  num_in_group = jnp.sum(group_mask)
  
  return sum_logits / (temperature * num_in_group)


def product_of_probabilities_logits(
  logits: Logits,
  group_mask: jnp.ndarray,
) -> jnp.ndarray:
  """Combine states by multiplying probabilities (summing log-probabilities).

  This finds amino acids with high probability in ALL states by computing:
  P_combined(aa) = P_state1(aa) * P_state2(aa) * ... * P_stateN(aa)

  In log space: log P_combined = log P_state1 + log P_state2 + ... + log P_stateN

  Since logits are unnormalized log-probabilities, we can approximate this by
  simply summing the logits across states.

  Args:
    logits: Logits array of shape (N, 21).
    group_mask: Boolean mask of shape (N,) for group positions.

  Returns:
    Sum of logits of shape (1, 21), representing log of probability product.

  Example:
    >>> logits = jnp.array([
    ...     [10.0, -5.0, 0.0],  # State 1: P(AA0) high, P(AA1) low
    ...     [8.0, -3.0, 0.0],   # State 2: P(AA0) high, P(AA1) low
    ... ])
    >>> group_mask = jnp.array([True, True])
    >>> product_logits = product_of_probabilities_logits(logits, group_mask)
    >>> # product_logits ≈ [18.0, -8.0, 0.0]
    >>> # AA0 has even higher combined probability, AA1 has even lower

  """
  masked_logits = jnp.where(group_mask[:, None], logits, 0.0)

  return jnp.sum(masked_logits, axis=0, keepdims=True)
