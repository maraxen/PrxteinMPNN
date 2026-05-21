"""Autoregression utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from collections.abc import Sequence

  from prxteinmpnn.run.specs import RunSpecification
  from prxteinmpnn.types.arrays import AutoRegressiveMask, DecodingOrder
  from prxteinmpnn.utils.data_structures import Protein


def get_decoding_step_map(
  tie_group_map: jnp.ndarray,
  group_decoding_order: jnp.ndarray,
  num_groups: int | None = None,
) -> jnp.ndarray:
  """Map each residue to its decoding step index based on group order."""
  # Use N as a safe upper bound for num_groups if not provided
  N = tie_group_map.shape[0]
  M = group_decoding_order.shape[0]

  group_to_step = (
    jnp.zeros(N, dtype=jnp.int32)
    .at[group_decoding_order].set(jnp.arange(M))
  )
  return group_to_step[tie_group_map]


def make_autoregressive_mask(decoding_step_map: jnp.ndarray) -> jnp.ndarray:
  """Create an (N, N) AR mask for group-based decoding."""
  steps_i = decoding_step_map[:, None]
  steps_j = decoding_step_map[None, :]
  return steps_i >= steps_j


def resolve_tie_groups(
  spec: RunSpecification,
  combined_protein: Protein,
  structure_mappings: Sequence[dict] | None = None,
  num_structures: int | None = None,
) -> jnp.ndarray:
  """Resolve tie groups for tied_positions modes.

  Args:
      spec: RunSpecification with tied_positions.
      combined_protein: Protein dataclass with batch_dim=1 (1, seq_len, ...).
      structure_mappings: Optional, for 'auto' mode.
      num_structures: Optional, number of structures concatenated (for 'direct' mode).

  Returns:
      tie_group_map: jnp.ndarray of shape (n,) with group ids.

  """
  chain_ids = combined_protein.chain_index[0]
  residue_indices = combined_protein.residue_index[0]
  n = chain_ids.shape[0]

  tie_group_map = jnp.arange(n, dtype=jnp.int32)
  tied_positions = spec.tied_positions

  if tied_positions is None:
    return tie_group_map

  if tied_positions == "direct":
    if num_structures is None:
      if combined_protein.mapping is not None:
        structure_indices = combined_protein.mapping[0]  # Remove batch dim
        num_inputs = int(jnp.max(structure_indices)) + 1
      else:
        msg = (
          "Cannot determine number of structures for 'direct' mode. "
          "The concatenated protein should have a 'mapping' field with structure indices, "
          "or pass num_structures explicitly."
        )
        raise ValueError(msg)
    else:
      num_inputs = num_structures

    ll = n // num_inputs
    if n % ll != 0:
      msg = (
        f"Inputs must be same length for 'direct' mode. "
        f"Total length {n} is not divisible by {num_inputs} structures."
      )
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
    chain_id_arr: jnp.ndarray,
    residue_idx_arr: jnp.ndarray,
  ) -> list[tuple[int, list[int]]]:
    group_map = defaultdict(list)
    for group_idx, group in enumerate(groups):
      if isinstance(group[0], (list | tuple)):
        for tup in group:
          group_map[group_idx].append(tup)
      else:
        group_map[group_idx].append(group)
    group_indices = []
    for group_idx, tuples in group_map.items():
      indices = []
      for chain_idx, res_idx in tuples:
        mask = (chain_id_arr == chain_idx) & (residue_idx_arr == res_idx)
        idx = jnp.where(mask)[0]
        if idx.size > 0:
          indices.append(idx[0])
      group_indices.append((group_idx, indices))
    return group_indices

  group_indices = _collect_group_indices(
    tied_positions,
    chain_ids,
    residue_indices,
  )
  for group_idx, indices in group_indices:
    if indices:
      group_id = n + group_idx
      tie_group_map = tie_group_map.at[jnp.array(indices)].set(group_id)
  _, tie_group_map = jnp.unique(tie_group_map, return_inverse=True)
  return tie_group_map


@jax.jit
def generate_ar_mask(
  decoding_order: DecodingOrder,
  chain_idx: jnp.ndarray | None = None,
  tie_group_map: jnp.ndarray | None = None,
  num_groups: int | None = None,
) -> AutoRegressiveMask:
  """Get the autoregressive mask for the given decoding order."""
  N = decoding_order.shape[0]

  if tie_group_map is None:
    row_indices = decoding_order[:, None]
    col_indices = decoding_order[None, :]
    ar_mask = (row_indices >= col_indices).astype(jnp.int32)
  else:
    # Use N as the static size for range-based ops
    # group_mask: (N, N)
    group_mask = tie_group_map[decoding_order][None, :] == jnp.arange(N)[:, None]

    # Identify which groups are actually present
    group_present = jnp.any(group_mask, axis=1)

    # group_first_occurrence: (N,)
    group_first_occurrence = jnp.argmax(group_mask, axis=1)

    # Sort groups by their first occurrence in the decoding order
    # We only care about present groups
    group_decoding_order = jnp.argsort(jnp.where(group_present, group_first_occurrence, N + 1))

    # If num_groups is provided, we can use it to mask the decoding steps
    # but for now, we just use the full order found.
    # The decoding_step_map will only be indexed by tie_group_map.

    decoding_step_map = get_decoding_step_map(tie_group_map, group_decoding_order)
    ar_mask = make_autoregressive_mask(decoding_step_map).astype(jnp.int32)

  if chain_idx is not None:
    same_chain = (chain_idx[:, None] == chain_idx[None, :]).astype(jnp.int32)
    ar_mask = ar_mask * same_chain

  return ar_mask
