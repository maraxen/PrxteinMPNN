"""Cross-cutting helpers shared by :class:`~prxteinmpnn.model.mpnn.PrxteinMPNN` call surfaces.

Phase **5d** extraction: multistate logits fusion + tie-group index tables live here so
``mpnn.py`` stays thinner without changing public ``PrxteinMPNN._.*`` delegates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from prxteinmpnn.model.multistate_sampling import (
  arithmetic_mean_logits,
  geometric_mean_logits,
  product_of_probabilities_logits,
)

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import GroupMask, Logits, TieGroupMap


def create_group_index_table(
  tie_group_map: jnp.ndarray,
  max_group_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Create a table of indices belonging to each group."""
  num_residues = tie_group_map.shape[0]
  mask_matrix = jnp.where(
    tie_group_map[None, :] == jnp.arange(num_residues)[:, None],
    jnp.arange(num_residues)[None, :],
    -1,
  )

  def sort_row(row: jnp.ndarray) -> jnp.ndarray:
    is_valid = row >= 0
    return jnp.sort(jnp.where(is_valid, row, num_residues + 1))

  sorted_indices = jax.vmap(sort_row)(mask_matrix)
  group_indices = sorted_indices[:, :max_group_size]
  valid_mask = group_indices < num_residues

  return group_indices, valid_mask


def combine_logits_multistate(
  logits: Logits,
  group_mask: GroupMask,
  strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
  temperature: float = 1.0,
  state_weights: jnp.ndarray | None = None,
  state_mapping: jnp.ndarray | None = None,
) -> Logits:
  """Combine logits across tied positions using different multi-state strategies."""
  if strategy == "arithmetic_mean":
    return arithmetic_mean_logits(logits, group_mask, state_weights, state_mapping)
  if strategy == "geometric_mean":
    return geometric_mean_logits(logits, group_mask, temperature, state_weights, state_mapping)
  if strategy == "product":
    return product_of_probabilities_logits(logits, group_mask, state_weights, state_mapping)
  msg = f"Unknown multi-state strategy: {strategy}"
  raise ValueError(msg)


def apply_multistate_to_all_logits(
  logits: Logits,
  tie_group_map: TieGroupMap,
  strategy_idx: jnp.ndarray,
  temperature: float = 1.0,
  state_weights: jnp.ndarray | None = None,
  state_mapping: jnp.ndarray | None = None,
) -> Logits:
  """Apply multi-state combination strategies across ALL groups in parallel."""
  num_total = tie_group_map.shape[0]

  def apply_arithmetic(ll: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    if state_weights is not None and state_mapping is not None:
      w = state_weights[state_mapping]
      log_w = jnp.log(jnp.where(w > 0, w, 1e-9))
      weighted_l = ll + log_w

      max_per_group = jax.ops.segment_max(weighted_l, g, num_segments=num_total)
      l_shifted = weighted_l - max_per_group[g]
      exp_l = jnp.exp(l_shifted)
      sum_exp = jax.ops.segment_sum(exp_l, g, num_segments=num_total)
      sum_w = jax.ops.segment_sum(w, g, num_segments=num_total)
      log_avg = jnp.log(sum_exp / jnp.where(sum_w > 0, sum_w, 1.0))
      return (log_avg + max_per_group)[g]

    max_per_group = jax.ops.segment_max(ll, g, num_segments=num_total)
    l_shifted = ll - max_per_group[g]
    exp_l = jnp.exp(l_shifted)
    sum_exp = jax.ops.segment_sum(exp_l, g, num_segments=num_total)
    count = jax.ops.segment_sum(jnp.ones_like(g, dtype=jnp.float32), g, num_segments=num_total)
    log_avg = jnp.log(sum_exp / jnp.where(count > 0, count, 1.0))
    return (log_avg + max_per_group)[g]

  def apply_geometric(ll: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    if state_weights is not None and state_mapping is not None:
      w = state_weights[state_mapping]
      sum_wl = jax.ops.segment_sum(ll * w, g, num_segments=num_total)
      sum_w = jax.ops.segment_sum(w, g, num_segments=num_total)
      avg_l = sum_wl / (jnp.where(sum_w > 0, sum_w, 1.0) * temperature)
      return avg_l[g]

    sum_l = jax.ops.segment_sum(ll, g, num_segments=num_total)
    count = jax.ops.segment_sum(jnp.ones_like(g, dtype=jnp.float32), g, num_segments=num_total)
    avg_l = sum_l / (jnp.where(count > 0, count, 1.0) * temperature)
    return avg_l[g]

  def apply_product(ll: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    if state_weights is not None and state_mapping is not None:
      w = state_weights[state_mapping]
      return jax.ops.segment_sum(ll * w, g, num_segments=num_total)[g]
    return jax.ops.segment_sum(ll, g, num_segments=num_total)[g]

  def switch_strategy(ll: jnp.ndarray, g: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.switch(
      idx,
      [
        lambda x: apply_arithmetic(x[0], x[1]),
        lambda x: apply_geometric(x[0], x[1]),
        lambda x: apply_product(x[0], x[1]),
      ],
      (ll, g),
    )

  return jax.vmap(switch_strategy, in_axes=(1, None, None), out_axes=1)(
    logits,
    tie_group_map,
    strategy_idx,
  )


def combine_logits_multistate_idx(
  logits: Logits,
  group_mask: GroupMask,
  strategy_idx: jnp.ndarray,
  temperature: float = 1.0,
  state_weights: jnp.ndarray | None = None,
  state_mapping: jnp.ndarray | None = None,
) -> Logits:
  """Combine logits using strategy index (JAX-traceable ``lax.switch`` wrapper)."""

  def arithmetic_mean_fn(_: tuple) -> jnp.ndarray:
    return arithmetic_mean_logits(logits, group_mask, state_weights, state_mapping)

  def geometric_mean_fn(_: tuple) -> jnp.ndarray:
    return geometric_mean_logits(logits, group_mask, temperature, state_weights, state_mapping)

  def product_fn(_: tuple) -> jnp.ndarray:
    return product_of_probabilities_logits(logits, group_mask, state_weights, state_mapping)

  branches = [arithmetic_mean_fn, geometric_mean_fn, product_fn]
  return jax.lax.switch(strategy_idx, branches, ())
