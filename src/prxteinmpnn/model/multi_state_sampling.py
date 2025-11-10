"""Multi-state protein design utilities for tied positions.

This module provides functions for sampling sequences that are compatible with
multiple conformational states of the same protein. Instead of averaging logits
(which compromises between states), we find amino acids that maximize the
MINIMUM probability across all states, ensuring the sequence works well in ALL
conformations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import Logits


def min_over_group_logits(
  logits: Logits,
  group_mask: jnp.ndarray,
) -> jnp.ndarray:
  """Find amino acids that work well in ALL tied positions (multi-state design).

  Instead of averaging logits (which creates a compromise), this function finds
  amino acids that have good probability in ALL states. This is appropriate when
  tied positions represent the same residue in different conformational states.

  Strategy: For each amino acid, take the MINIMUM logit across all tied positions.
  This ensures the selected amino acid works well in the WORST case (least favorable
  state), making it robust across all conformations.

  Args:
    logits: Logits array of shape (N, 21) where N includes all positions.
    group_mask: Boolean mask of shape (N,) indicating which positions are in the
                current tie group.

  Returns:
    Min logits of shape (1, 21) representing the worst-case logit for each amino
    acid across all states.

  Example:
    >>> # Two states with different preferences
    >>> logits = jnp.array([
    ...     [10.0, -5.0, -5.0],  # State 1: strong preference for AA 0
    ...     [-5.0, 8.0, -5.0],   # State 2: strong preference for AA 1
    ... ])
    >>> group_mask = jnp.array([True, True])
    >>> min_logits = min_over_group_logits(logits, group_mask)
    >>> # min_logits ≈ [-5.0, -5.0, -5.0]
    >>> # Both AA 0 and AA 1 have poor worst-case performance
    >>> # Need to find amino acid that works reasonably well in BOTH states

  """
  # For positions not in group, use very large value so they don't affect min
  masked_logits = jnp.where(
    group_mask[:, None],
    logits,
    1e9,  # Large value that won't be selected as minimum
  )

  # Take minimum across all positions for each amino acid
  return jnp.min(masked_logits, axis=0, keepdims=True)


def max_min_over_group_logits(
  logits: Logits,
  group_mask: jnp.ndarray,
  alpha: float = 0.5,
) -> jnp.ndarray:
  """Hybrid approach: balance between worst-case and average-case performance.

  This combines min-over-states (robustness) with mean-over-states (optimality)
  using a weighted combination. Higher alpha favors robustness, lower alpha
  favors average performance.

  Args:
    logits: Logits array of shape (N, 21).
    group_mask: Boolean mask of shape (N,) for group positions.
    alpha: Weight for min component (0=pure average, 1=pure min). Default 0.5.

  Returns:
    Combined logits of shape (1, 21).

  Example:
    >>> logits = jnp.array([
    ...     [10.0, -5.0, -5.0],  # State 1
    ...     [-5.0, 8.0, -5.0],   # State 2
    ... ])
    >>> group_mask = jnp.array([True, True])
    >>> # Pure min (alpha=1.0): favors robust amino acids
    >>> robust_logits = max_min_over_group_logits(logits, group_mask, alpha=1.0)
    >>> # Pure mean (alpha=0.0): favors average performance
    >>> avg_logits = max_min_over_group_logits(logits, group_mask, alpha=0.0)
    >>> # Balanced (alpha=0.5): compromise
    >>> balanced_logits = max_min_over_group_logits(logits, group_mask, alpha=0.5)

  """
  # Compute min logits (worst-case)
  min_logits = min_over_group_logits(logits, group_mask)

  # Compute mean logits (average-case) using standard averaging
  masked_logits = jnp.where(group_mask[:, None], logits, 0.0)
  sum_logits = jnp.sum(masked_logits, axis=0, keepdims=True)
  num_in_group = jnp.sum(group_mask)
  mean_logits = sum_logits / num_in_group

  # Weighted combination
  return alpha * min_logits + (1.0 - alpha) * mean_logits


def softmin_over_group_logits(
  logits: Logits,
  group_mask: jnp.ndarray,
  temperature: float = 1.0,
) -> jnp.ndarray:
  """Soft minimum using log-sum-exp trick for numerical stability.

  This is a differentiable approximation to the hard minimum. As temperature
  approaches 0, this converges to the hard minimum. As temperature increases,
  this approaches a weighted average.

  The formula is: softmin(x) = -temperature * log(sum(exp(-x/temperature)))

  Args:
    logits: Logits array of shape (N, 21).
    group_mask: Boolean mask of shape (N,) for group positions.
    temperature: Temperature for soft minimum (lower = closer to hard min).

  Returns:
    Soft minimum logits of shape (1, 21).

  Example:
    >>> logits = jnp.array([[10.0, -5.0], [8.0, -3.0]])
    >>> group_mask = jnp.array([True, True])
    >>> # Low temperature: close to hard min
    >>> hard_min = softmin_over_group_logits(logits, group_mask, temperature=0.1)
    >>> # High temperature: closer to average
    >>> soft_avg = softmin_over_group_logits(logits, group_mask, temperature=10.0)

  """
  # Mask out non-group positions
  masked_logits = jnp.where(
    group_mask[:, None],
    logits,
    1e9,  # Large value for non-group positions
  )

  # Compute soft minimum using log-sum-exp
  # softmin(x) = -log(sum(exp(-x/T))) * T
  #           = -logsumexp(-x/T) * T
  neg_scaled_logits = -masked_logits / temperature
  log_sum_exp = jnp.max(neg_scaled_logits, axis=0, keepdims=True) + jnp.log(
    jnp.sum(
      jnp.exp(neg_scaled_logits - jnp.max(neg_scaled_logits, axis=0, keepdims=True)),
      axis=0,
      keepdims=True,
    ),
  )
  return -log_sum_exp * temperature


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
  # Mask out non-group positions (use 0 so they don't affect sum)
  masked_logits = jnp.where(group_mask[:, None], logits, 0.0)

  # Sum logits across all states
  return jnp.sum(masked_logits, axis=0, keepdims=True)
