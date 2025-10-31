def prepare_inter_mode_batch(inputs: Sequence[ProteinMPNNInput], spec: RunSpecification):
"""Batch preprocessing for inter pass mode in PrxteinMPNN."""

import jax.numpy as jnp
from collections.abc import Sequence
from typing import NamedTuple
from prxteinmpnn.run.specs import RunSpecification

class InterModeMap(NamedTuple):
    """Mapping from new chain IDs to (input_idx, original_chain_id)."""
    chain_map: dict[int, tuple[int, int]]

class ProteinMPNNInput(NamedTuple):
    """Minimal ProteinMPNNInput for inter mode batch prep (replace with actual import)."""
    X: jnp.ndarray
    S: jnp.ndarray
    mask: jnp.ndarray
    chain_id: jnp.ndarray
    residue_index: jnp.ndarray
    structure_to_sequence_mappings: list[dict]

def prepare_inter_mode_batch(
    inputs: Sequence[ProteinMPNNInput], spec: RunSpecification
) -> tuple[ProteinMPNNInput, InterModeMap]:
    """Combine a batch of ProteinMPNNInput into a single input for inter mode.

    Args:
        inputs: List of ProteinMPNNInput objects.
        spec: RunSpecification with pass_mode and tied_positions.

    Returns:
        combined_input: ProteinMPNNInput
        inter_mode_map: InterModeMap

    Raises:
        ValueError: If pass_mode is not 'inter'.
    """
    if spec.pass_mode != "inter":
        msg = "prepare_inter_mode_batch is only valid for pass_mode='inter'."
        raise ValueError(msg)
    xs, ss, masks, chain_ids, residue_indices, mappings = [], [], [], [], [], []
    chain_map = {}
    global_chain_counter = 0
    for i, inp in enumerate(inputs):
        unique_chains = jnp.unique(inp.chain_id)
        chain_id_map = {}
        for orig_chain in unique_chains:
            chain_id_map[orig_chain.item()] = global_chain_counter
            chain_map[global_chain_counter] = (i, orig_chain.item())
            global_chain_counter += 1
        # Remap chain_id
        remapped_chain_id = jnp.array([chain_id_map[c.item()] for c in inp.chain_id])
        xs.append(inp.X)
        ss.append(inp.S)
        masks.append(inp.mask)
        chain_ids.append(remapped_chain_id)
        residue_indices.append(inp.residue_index)
        # Adjust mappings if needed (placeholder, depends on actual mapping structure)
        mappings.extend(inp.structure_to_sequence_mappings)
    combined_input = ProteinMPNNInput(
        X=jnp.concatenate(xs, axis=0),
        S=jnp.concatenate(ss, axis=0),
        mask=jnp.concatenate(masks, axis=0),
        chain_id=jnp.concatenate(chain_ids, axis=0),
        residue_index=jnp.concatenate(residue_indices, axis=0),
        structure_to_sequence_mappings=mappings,
    )
    return combined_input, InterModeMap(chain_map=chain_map)
