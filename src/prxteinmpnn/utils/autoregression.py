"""Autoregression utilities."""

from collections.abc import Sequence

import jax.numpy as jnp

from prxteinmpnn.run.specs import RunSpecification


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
  combined_input,
  structure_mappings: Sequence[dict] | None = None,
) -> jnp.ndarray:
  """Resolve tie groups for tied_positions modes.

  Args:
    spec: RunSpecification with tied_positions.
    combined_input: ProteinMPNNInput (must have chain_id, residue_index, etc.).
    structure_mappings: Optional, for 'auto' mode.

  Returns:
    tie_group_map: jnp.ndarray of shape (N,) with group ids.

  """
  N = combined_input.chain_id.shape[0]
  tie_group_map = jnp.arange(N, dtype=jnp.int32)
  if spec.tied_positions is None:
    return tie_group_map
  if spec.tied_positions == "direct":
    # All inputs must be same length
    L = N // getattr(combined_input, "num_inputs", 1)
    if N % L != 0:
      raise ValueError("Inputs must be same length for 'direct' mode.")
    K = N // L
    tie_group_map = jnp.tile(jnp.arange(L, dtype=jnp.int32), K)
    return tie_group_map
  if spec.tied_positions == "auto":
    if structure_mappings is None:
      raise ValueError("structure_mappings required for 'auto' mode.")
    for seq_pos, struct_pos_list in enumerate(structure_mappings):
      if len(struct_pos_list) > 1:
        group_id = N + seq_pos
        tie_group_map = tie_group_map.at[jnp.array(struct_pos_list)].set(group_id)
    _, tie_group_map = jnp.unique(tie_group_map, return_inverse=True)
    return tie_group_map
  # Explicit list of (chain_idx, res_idx) tuples
  # Group tuples by group index (assume input is list[list[tuple]])
  from collections import defaultdict

  group_map = defaultdict(list)
  for group_idx, group in enumerate(spec.tied_positions):
    if isinstance(group[0], (list, tuple)):
      for tup in group:
        group_map[group_idx].append(tup)
    else:
      group_map[group_idx].append(group)
  for group_idx, tuples in group_map.items():
    indices = []
    for chain_idx, res_idx in tuples:
      mask = (combined_input.chain_id == chain_idx) & (combined_input.residue_index == res_idx)
      idx = jnp.where(mask)[0]
      if idx.size > 0:
        indices.append(idx[0])
    if indices:
      group_id = N + group_idx
      tie_group_map = tie_group_map.at[jnp.array(indices)].set(group_id)
  _, tie_group_map = jnp.unique(tie_group_map, return_inverse=True)
  return tie_group_map


"""Utilities for autoregression.

prxteinmpnn.utils.autoregression
"""

import jax

from .types import AutoRegressiveMask, DecodingOrder


@jax.jit
def generate_ar_mask(decoding_order: DecodingOrder) -> AutoRegressiveMask:
  """Get the autoregressive mask for the given decoding order.

  Args:
    decoding_order: The order in which atoms are decoded.

  Returns:
    An atom mask where each atom can only attend to itself and previous atoms.

  """
  row_indices = decoding_order[:, None]
  col_indices = decoding_order[None, :]
  return (row_indices >= col_indices).astype(int)
