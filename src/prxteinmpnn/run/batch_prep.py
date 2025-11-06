"""Batch preprocessing for inter pass mode in PrxteinMPNN."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp

if TYPE_CHECKING:
  from prxteinmpnn.run.specs import RunSpecification
  from prxteinmpnn.utils.data_structures import Protein


class InterModeMap(NamedTuple):
  """Mapping from new chain IDs to (input_idx, original_chain_id)."""

  chain_map: dict[int, tuple[int, int]]


class ProteinMPNNInput(NamedTuple):
  """Minimal ProteinMPNNInput for inter mode batch prep."""

  X: jnp.ndarray
  S: jnp.ndarray
  mask: jnp.ndarray
  chain_id: jnp.ndarray
  residue_index: jnp.ndarray
  structure_to_sequence_mappings: list[dict]


def prepare_inter_mode_batch_from_protein(
  batched_protein: Protein,
  spec: RunSpecification,
) -> tuple[ProteinMPNNInput, InterModeMap]:
  """Combine a batched Protein into a single input for inter mode.

  Args:
    batched_protein: Protein dataclass with batch dimension (batch_size, seq_len, ...).
    spec: RunSpecification with pass_mode and tied_positions.

  Returns:
    Tuple of (combined_input, inter_mode_map).

  Raises:
    ValueError: If pass_mode is not 'inter'.

  Example:
    >>> protein_batch = ...  # Protein with shape (2, 91, ...)
    >>> spec = RunSpecification(pass_mode="inter")
    >>> combined, mapping = prepare_inter_mode_batch_from_protein(protein_batch, spec)

  """
  if spec.pass_mode != "inter":
    msg = "prepare_inter_mode_batch_from_protein is only valid for pass_mode='inter'."
    raise ValueError(msg)

  batch_size = batched_protein.coordinates.shape[0]
  xs, ss, masks, chain_ids, residue_indices = [], [], [], [], []
  chain_map = {}
  global_chain_counter = 0

  for i in range(batch_size):
    # Extract i-th structure from batch
    coords_i = batched_protein.coordinates[i]
    mask_i = batched_protein.mask[i]
    chain_idx_i = batched_protein.chain_index[i]
    residue_idx_i = batched_protein.residue_index[i]

    # Remap chain IDs to avoid conflicts
    unique_chains = jnp.unique(chain_idx_i)
    chain_id_map = {}
    for orig_chain in unique_chains:
      chain_id_map[orig_chain.item()] = global_chain_counter
      chain_map[global_chain_counter] = (i, orig_chain.item())
      global_chain_counter += 1

    remapped_chain_id = jnp.array([chain_id_map[c.item()] for c in chain_idx_i])

    xs.append(coords_i)
    ss.append(jnp.zeros(mask_i.shape[0], dtype=jnp.int32))  # Placeholder sequence
    masks.append(mask_i)
    chain_ids.append(remapped_chain_id)
    residue_indices.append(residue_idx_i)

  combined_input = ProteinMPNNInput(
    X=jnp.concatenate(xs, axis=0),
    S=jnp.concatenate(ss, axis=0),
    mask=jnp.concatenate(masks, axis=0),
    chain_id=jnp.concatenate(chain_ids, axis=0),
    residue_index=jnp.concatenate(residue_indices, axis=0),
    structure_to_sequence_mappings=[],
  )
  return combined_input, InterModeMap(chain_map=chain_map)


def prepare_inter_mode_batch(
  inputs: Sequence[ProteinMPNNInput],
  spec: RunSpecification,
) -> tuple[ProteinMPNNInput, InterModeMap]:
  """Combine a batch of ProteinMPNNInput into a single input for inter mode.

  Args:
    inputs: List of ProteinMPNNInput objects.
    spec: RunSpecification with pass_mode and tied_positions.

  Returns:
    Tuple of (combined_input, inter_mode_map).

  Raises:
    ValueError: If pass_mode is not 'inter'.

  Example:
    >>> inputs = [input1, input2]
    >>> spec = RunSpecification(pass_mode="inter")
    >>> combined, mapping = prepare_inter_mode_batch(inputs, spec)

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
