"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np

from prxteinmpnn.utils.data_structures import Protein

if TYPE_CHECKING:
  from io import StringIO

  from grain.python import IterDataset

  from prxteinmpnn.run.specs import RunSpecification, Specs
  from prxteinmpnn.utils.types import (
    ModelParameters,
  )


from prxteinmpnn.io import loaders
from prxteinmpnn.mpnn import get_mpnn_model

InterModeMap = Dict[int, Tuple[int, int]]


def prepare_inter_mode_batch(
    inputs: List[Protein], spec: RunSpecification
) -> (Protein, InterModeMap):
    """Combine multiple inputs into a single logical input for "inter" pass mode."""
    if spec.pass_mode != "inter":
        raise ValueError('prepare_inter_mode_batch is only for "inter" pass mode.')

    all_coords, all_aatypes, all_one_hots, all_masks, all_res_indices, all_chain_indices = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    inter_mode_map: InterModeMap = {}
    global_chain_counter = 0

    for i, protein_input in enumerate(inputs):
        all_coords.append(protein_input.coordinates)
        all_aatypes.append(protein_input.aatype)
        all_one_hots.append(protein_input.one_hot_sequence)
        all_masks.append(protein_input.mask)
        all_res_indices.append(protein_input.residue_index)

        # Chain Re-indexing
        unique_chains = np.unique(protein_input.chain_index)
        chain_map = {int(c): global_chain_counter + j for j, c in enumerate(unique_chains)}

        new_chain_index = np.vectorize(chain_map.get)(protein_input.chain_index)
        all_chain_indices.append(new_chain_index)

        for original_chain_id in unique_chains:
            new_chain_id = chain_map[int(original_chain_id)]
            inter_mode_map[new_chain_id] = (i, int(original_chain_id))

        global_chain_counter += len(unique_chains)

    # All optional fields must be present in all inputs to be included.
    has_dihedrals = all(inp.dihedrals is not None for inp in inputs)
    has_full_coords = all(inp.full_coordinates is not None for inp in inputs)
    has_full_atom_mask = all(inp.full_atom_mask is not None for inp in inputs)

    combined_protein = Protein(
        coordinates=jnp.concatenate(all_coords, axis=0),
        aatype=jnp.concatenate(all_aatypes, axis=0),
        one_hot_sequence=jnp.concatenate(all_one_hots, axis=0),
        mask=jnp.concatenate(all_masks, axis=0),
        residue_index=jnp.concatenate(all_res_indices, axis=0),
        chain_index=jnp.concatenate(all_chain_indices, axis=0),
        dihedrals=jnp.concatenate([inp.dihedrals for inp in inputs], axis=0) if has_dihedrals else None,
        full_coordinates=jnp.concatenate([inp.full_coordinates for inp in inputs], axis=0) if has_full_coords else None,
        full_atom_mask=jnp.concatenate([inp.full_atom_mask for inp in inputs], axis=0) if has_full_atom_mask else None,
    )

    return combined_protein, inter_mode_map


def _loader_inputs(inputs: Sequence[str | StringIO] | str | StringIO) -> Sequence[str | StringIO]:
  return (inputs,) if not isinstance(inputs, Sequence) else inputs


def prep_protein_stream_and_model(spec: Specs) -> tuple[IterDataset, ModelParameters]:
  """Prepare the protein data stream and model parameters.

  Args:
      spec: A RunSpecification object containing configuration options.

  Returns:
      A tuple containing the protein data iterator and model parameters.

  """
  parse_kwargs = {
    "chain_id": spec.chain_id,
    "model": spec.model,
    "altloc": spec.altloc,
    "topology": spec.topology,
  }
  protein_iterator = loaders.create_protein_dataset(
    _loader_inputs(spec.inputs),
    batch_size=spec.batch_size,
    foldcomp_database=spec.foldcomp_database,
    parse_kwargs=parse_kwargs,
  )
  model_parameters = get_mpnn_model(
    model_version=spec.model_version,
    model_weights=spec.model_weights,
  )
  return protein_iterator, model_parameters
