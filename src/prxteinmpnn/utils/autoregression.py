"""Autoregression utilities."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from collections.abc import Sequence

  from prxteinmpnn.run.batch_prep import ProteinMPNNInput
  from prxteinmpnn.run.specs import RunSpecification

  from .types import AutoRegressiveMask, DecodingOrder


def get_decoding_step_map(
  tie_group_map: jnp.ndarray,
  group_decoding_order: jnp.ndarray,
  num_groups: int | None = None,
) -> jnp.ndarray:
  """Map each residue to its decoding step index based on group order.

  Args:
    tie_group_map: (N,) array of group ids in range [0, num_groups-1].
    group_decoding_order: (M,) array, permutation of group ids [0, ..., M-1].
    num_groups: Number of unique groups. If not provided, inferred from
        group_decoding_order length.

  Returns:
    decoding_step_map: (N,) array, decoding step for each residue.

  """
  if num_groups is None:
    num_groups = len(group_decoding_order)
  group_to_step = (
    jnp.zeros(num_groups, dtype=jnp.int32)
    .at[group_decoding_order]
    .set(jnp.arange(len(group_decoding_order)))
  )
  return group_to_step[tie_group_map]


def make_autoregressive_mask(decoding_step_map: jnp.ndarray) -> jnp.ndarray:
  """Create an (N, N) AR mask for group-based decoding.

  Positions at step i can attend to all positions at steps <= i.
  This allows positions in the same group (same step) to attend to each other.

  Args:
    decoding_step_map: (N,) array, decoding step for each residue.

  Returns:
    mask: (N, N) boolean array where mask[i,j]=True means position i
          can attend to position j.

  """
  steps_i = decoding_step_map[:, None]
  steps_j = decoding_step_map[None, :]
  return steps_i >= steps_j


def resolve_tie_groups(
  spec: RunSpecification,
  combined_input: ProteinMPNNInput,
  structure_mappings: Sequence[dict] | None = None,
) -> jnp.ndarray:
  """Resolve tie groups for tied_positions modes.

  Args:
      spec: RunSpecification with tied_positions.
      combined_input: ProteinMPNNInput (must have chain_id, residue_index, etc.).
      structure_mappings: Optional, for 'auto' mode.

  Returns:
      tie_group_map: jnp.ndarray of shape (n,) with group ids.

  """
  n = combined_input.chain_id.shape[0]
  tie_group_map = jnp.arange(n, dtype=jnp.int32)
  tied_positions = spec.tied_positions

  if tied_positions is None:
    return tie_group_map

  if tied_positions == "direct":
    # All inputs must be same length
    num_inputs = getattr(combined_input, "num_inputs", 1)
    ll = n // num_inputs
    if n % ll != 0:
      msg = "Inputs must be same length for 'direct' mode."
      raise ValueError(msg)
    k = n // ll
    return jnp.tile(jnp.arange(ll, dtype=jnp.int32), k)

  if tied_positions == "auto":
    if structure_mappings is None:
      msg = "structure_mappings required for 'auto' mode."
      raise ValueError(msg)
    for seq_pos, struct_pos_list in enumerate(structure_mappings):
      if len(struct_pos_list) > 1:
        group_id = n + seq_pos
        tie_group_map = tie_group_map.at[jnp.array(struct_pos_list)].set(group_id)
    _, tie_group_map = jnp.unique(tie_group_map, return_inverse=True)
    return tie_group_map

  def _collect_group_indices(
    groups: Sequence[tuple[int, int]],
    chain_ids: jnp.ndarray,
    residue_indices: jnp.ndarray,
  ) -> list[tuple[int, list[int]]]:
    group_map = defaultdict(list)
    for group_idx, group in enumerate(groups):
      if isinstance(group[0], (list, tuple)):
        for tup in group:
          group_map[group_idx].append(tup)
      else:
        group_map[group_idx].append(group)
    group_indices = []
    for group_idx, tuples in group_map.items():
      indices = []
      for chain_idx, res_idx in tuples:
        mask = (chain_ids == chain_idx) & (residue_indices == res_idx)
        idx = jnp.where(mask)[0]
        if idx.size > 0:
          indices.append(idx[0])
      group_indices.append((group_idx, indices))
    return group_indices

  group_indices = _collect_group_indices(
    tied_positions,
    combined_input.chain_id,
    combined_input.residue_index,
  )
  for group_idx, indices in group_indices:
    if indices:
      group_id = n + group_idx
      tie_group_map = tie_group_map.at[jnp.array(indices)].set(group_id)
  _, tie_group_map = jnp.unique(tie_group_map, return_inverse=True)
  return tie_group_map


@partial(jax.jit, static_argnames=("num_groups",))
def generate_ar_mask(
  decoding_order: DecodingOrder,
  chain_idx: jnp.ndarray | None = None,
  tie_group_map: jnp.ndarray | None = None,
  num_groups: int | None = None,
) -> AutoRegressiveMask:
  """Get the autoregressive mask for the given decoding order.

  When tie_group_map is provided, positions in the same group can attend to each
  other (within the same decoding step), enabling tied sampling.

  Args:
      decoding_order: The order in which atoms are decoded.
      chain_idx: Optional chain indices. If provided, atoms can only attend to
          atoms in the same chain that come before them in the decoding order.
      tie_group_map: Optional (N,) array mapping each position to a group ID.
          When provided, positions in the same group are decoded simultaneously
          and can attend to each other.
      num_groups: Number of unique groups in tie_group_map. Required if
          tie_group_map is provided. Should equal tie_group_map.max() + 1
          when groups are normalized to [0, 1, ..., num_groups-1].

  Returns:
      An autoregressive mask (N, N) where mask[i,j]=1 means position i can
      attend to position j during decoding.

  Example:
      >>> # Standard AR mask
      >>> order = jnp.array([0, 1, 2])
      >>> mask = generate_ar_mask(order)
      >>> # mask = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
      >>>
      >>> # With tied positions
      >>> tie_map = jnp.array([0, 1, 0])  # Positions 0 and 2 are tied
      >>> mask = generate_ar_mask(order, tie_group_map=tie_map, num_groups=2)
      >>> # Positions 0 and 2 can attend to each other

  """
  if tie_group_map is None:
    # Original implementation: standard autoregressive mask
    row_indices = decoding_order[:, None]
    col_indices = decoding_order[None, :]
    ar_mask = (row_indices >= col_indices).astype(int)
  else:
    if num_groups is None:
      msg = "num_groups must be provided when tie_group_map is not None"
      raise ValueError(msg)

    # Tied positions: use vectorized operations to find group decoding order
    # Create mask: (num_groups, N) where mask[g, i] = True if position i is in group g
    group_mask = tie_group_map[decoding_order][None, :] == jnp.arange(num_groups)[:, None]
    # Find first occurrence of each group in decoding order
    group_first_occurrence = jnp.argmax(group_mask, axis=1)
    # Sort groups by their first appearance to get group decoding order
    group_decoding_order = jnp.argsort(group_first_occurrence)

    # Generate step map and AR mask
    decoding_step_map = get_decoding_step_map(tie_group_map, group_decoding_order, num_groups)
    ar_mask = make_autoregressive_mask(decoding_step_map).astype(int)

  # If chain indices are provided, mask out cross-chain attention
  if chain_idx is not None:
    same_chain = (chain_idx[:, None] == chain_idx[None, :]).astype(int)
    ar_mask = ar_mask * same_chain

  return ar_mask
