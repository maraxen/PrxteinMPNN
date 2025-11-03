"""Utilities for autoregression.

prxteinmpnn.utils.autoregression
"""
from typing import Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.run.specs import RunSpecification
from prxteinmpnn.utils.data_structures import Protein

from .types import AutoRegressiveMask, DecodingOrder


def resolve_tie_groups(
    spec: RunSpecification,
    combined_input: Protein,
    structure_mappings: List[Dict[int, List[int]]] = None,
) -> jnp.ndarray:
    """Resolve tied_positions into a concrete map that downstream functions can use."""
    n = combined_input.coordinates.shape[0]
    tie_group_map = jnp.arange(n)

    if spec.tied_positions is None:
        return tie_group_map

    elif spec.tied_positions == "direct":
        if len(spec.inputs) < 2:
            raise ValueError("Direct tie mode requires at least 2 inputs.")
        l = n // len(spec.inputs)
        if n % len(spec.inputs) != 0:
            raise ValueError("All inputs must have the same length for direct tie mode.")
        tie_group_map = jnp.tile(jnp.arange(l), len(spec.inputs))

    elif spec.tied_positions == "auto":
        if structure_mappings is None:
            raise ValueError("Structure mappings must be provided for auto tie mode.")

        for seq_pos, struct_pos_list in structure_mappings.items():
            if len(struct_pos_list) > 1:
                group_id = n + seq_pos
                tie_group_map = tie_group_map.at[jnp.array(struct_pos_list)].set(group_id)
        _, tie_group_map = jnp.unique(tie_group_map, return_inverse=True)

    elif isinstance(spec.tied_positions, Sequence):
        groups: Dict[int, List[Tuple[int, int]]] = {}
        for i, group in enumerate(spec.tied_positions):
            groups[i] = list(group)

        for group_idx, group in groups.items():
            global_indices = []
            for chain_idx, res_idx in group:
                indices = jnp.where(
                    (combined_input.chain_index == chain_idx)
                    & (combined_input.residue_index == res_idx)
                )[0]
                if indices.size > 0:
                    global_indices.append(indices[0])
            if len(global_indices) > 1:
                group_id = n + group_idx
                tie_group_map = tie_group_map.at[jnp.array(global_indices)].set(group_id)
        _, tie_group_map = jnp.unique(tie_group_map, return_inverse=True)

    return tie_group_map


def get_decoding_step_map(
    tie_group_map: jnp.ndarray, group_decoding_order: jnp.ndarray
) -> jnp.ndarray:
    """Get the decoding step for each residue."""
    group_to_step = jnp.zeros(
        tie_group_map.max() + 1, dtype=jnp.int32
    ).at[group_decoding_order].set(jnp.arange(len(group_decoding_order)))
    decoding_step_map = group_to_step[tie_group_map]
    return decoding_step_map


@jax.jit
def make_autoregressive_mask(decoding_step_map: jnp.ndarray) -> jnp.ndarray:
    """Make the autoregressive mask based on the decoding step map."""
    n = decoding_step_map.shape[0]
    steps_i = decoding_step_map[:, None]
    steps_j = decoding_step_map[None, :]
    is_self = jnp.eye(n, dtype=jnp.bool_)
    mask = (steps_i > steps_j) | is_self
    return mask
