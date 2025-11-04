"""Autoregression utilities."""

from __future__ import annotations

from collections import defaultdict
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
) -> jnp.ndarray:
  """Map each residue to its decoding step index based on group order.

  Args:
    tie_group_map: (N,) array of group ids.
    group_decoding_order: (M,) array, permutation of unique group ids.

  Returns:
    decoding_step_map: (N,) array, decoding step for each residue.

  """
  max_gid = jnp.ceil(tie_group_map.max()) + 1
  group_to_step = (
    jnp.zeros(max_gid, dtype=jnp.int32)
    .at[group_decoding_order]
    .set(jnp.arange(len(group_decoding_order)))
  )
  return group_to_step[tie_group_map]


def make_autoregressive_mask(decoding_step_map: jnp.ndarray) -> jnp.ndarray:
  """Create an (N, N) AR mask for group-based decoding.

  Args:
    decoding_step_map: (N,) array, decoding step for each residue.

  Returns:
    mask: (N, N) boolean array.

  """
  steps_i = decoding_step_map[:, None]
  steps_j = decoding_step_map[None, :]
  is_self = jnp.eye(len(decoding_step_map), dtype=bool)
  return (steps_i > steps_j) | is_self


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


@jax.jit
def generate_ar_mask(
  decoding_order: DecodingOrder,
  chain_idx: jnp.ndarray | None = None,
) -> AutoRegressiveMask:
  """Get the autoregressive mask for the given decoding order.

  Args:
      decoding_order: The order in which atoms are decoded.
      chain_idx: Optional chain indices. If provided, atoms can only attend to
          atoms in the same chain that come before them in the decoding order.

  Returns:
      An atom mask where each atom can only attend to itself and previous atoms
      (and only atoms in the same chain if chain_idx is provided).

  """
  row_indices = decoding_order[:, None]
  col_indices = decoding_order[None, :]
  ar_mask = (row_indices >= col_indices).astype(int)

  # If chain indices are provided, mask out cross-chain attention
  if chain_idx is not None:
    same_chain = (chain_idx[:, None] == chain_idx[None, :]).astype(int)
    ar_mask = ar_mask * same_chain

  return ar_mask
