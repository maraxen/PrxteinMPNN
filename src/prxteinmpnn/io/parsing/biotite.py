"""Parsing utilities for Biotite."""

import logging
import pathlib
from collections.abc import Sequence
from typing import Any

import hydride
import numpy as np
from biotite import structure
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure import io as structure_io

from prxteinmpnn.io.parsing.structures import ProcessedStructure
from prxteinmpnn.io.parsing.utils import (
  _process_chain_id,
  _validate_atom_array_type,
  processed_structure_to_protein_tuples,
)
from prxteinmpnn.utils.data_structures import (
  EstatInfo,
  ProteinStream,
)

logger = logging.getLogger(__name__)


def _remove_solvent_from_structure(
  atom_array: AtomArray | AtomArrayStack,
) -> AtomArray | AtomArrayStack:
  """Remove solvent atoms from the structure."""
  solvent_mask = structure.filter_solvent(atom_array)
  if np.any(solvent_mask):
    n_solvent = np.sum(solvent_mask)
    logger.info("Removing %d solvent atoms", n_solvent)

    if isinstance(atom_array, AtomArrayStack):
      atom_array = atom_array[:, ~solvent_mask]
    else:
      atom_array = atom_array[~solvent_mask]

    logger.debug("Structure after solvent removal: %d atoms", atom_array.array_length())
  return atom_array


def _add_hydrogens_to_structure(
  atom_array: AtomArray | AtomArrayStack,
) -> AtomArray | AtomArrayStack:
  """Add hydrogens to the structure using Hydride."""
  # Check if hydrogens exist
  has_hydrogens = (atom_array.element == "H").any()
  if has_hydrogens:
    logger.debug("Structure already has hydrogens.")
    return atom_array

  logger.info("No hydrogens found. Adding hydrogens using Hydride.")

  if isinstance(atom_array, AtomArrayStack):
    logger.warning(
      "Hydride integration with AtomArrayStack is experimental. "
      "Processing frame 0 only for now or iterating?",
    )
    # Hydride returns a single AtomArray.
    # For now, let's just return the stack as is if we can't easily protonate it,
    # or warn.
    # Ideally we should iterate and protonate each frame, but topology might change?
    # Let's just support AtomArray for now for hydride.
    return atom_array

  # Hydride requires a BondList. If not present, we must infer it.
  if not atom_array.bonds:
    logger.info("No BondList found. Inferring bonds via residue names.")
    try:
      atom_array.bonds = structure.connect_via_residue_names(atom_array)  # type: ignore[unresolved-attribute]
    except Exception as e:  # noqa: BLE001
      logger.warning("Failed to connect via residue names: %s. Falling back to distances.", e)
      atom_array.bonds = structure.connect_via_distances(atom_array)  # type: ignore[unresolved-attribute]

  # Hydride also requires a 'charge' annotation, even if 0.
  if "charge" not in atom_array.get_annotation_categories():
    logger.info("No charge annotation found. Adding zero charges for Hydride compatibility.")
    atom_array.set_annotation("charge", np.zeros(atom_array.array_length(), dtype=int))

  atom_array, _ = hydride.add_hydrogen(atom_array)
  logger.info("Hydrogens added. New atom count: %d", atom_array.array_length())

  return atom_array


def load_structure_with_hydride(
  source: str | pathlib.Path,
  model: int | None = None,
  altloc: str | None = None,
  topology: str | pathlib.Path | None = None,
  *,
  add_hydrogens: bool = True,
  remove_solvent: bool = True,
  **kwargs: Any,  # noqa: ANN401
) -> AtomArray | AtomArrayStack:
  """Load a structure and optionally add hydrogens using Hydride.

  Args:
      source: Path to structure file
      model: Model number to load (for multi-model files)
      altloc: Alternate location identifier
      topology: Optional topology file
      add_hydrogens: Whether to add hydrogens using hydride
      remove_solvent: Whether to remove solvent molecules (water, ions)
      **kwargs: Additional arguments for structure loading

  Returns:
      AtomArray or AtomArrayStack with processed structure

  """
  altloc = altloc if altloc is not None else "first"
  topology_array = None

  if topology is not None:
    topology_array = structure_io.load_structure(topology)

  if pathlib.Path(source).suffix.lower() in [".xtc"]:
    atom_array = structure_io.load_structure(
      source,
      template=topology_array,
      **kwargs,
    )
  else:
    atom_array = structure_io.load_structure(
      source,
      model=model,
      altloc=altloc,
      template=topology_array,
      **kwargs,
    )

  logger.info(
    "Loading structure from %s (add_hydrogens=%s, remove_solvent=%s)",
    source,
    add_hydrogens,
    remove_solvent,
  )
  _validate_atom_array_type(atom_array)

  if remove_solvent:
    atom_array = _remove_solvent_from_structure(atom_array)

  if add_hydrogens:
    atom_array = _add_hydrogens_to_structure(atom_array)

  return atom_array


def _parse_biotite(
  source: str | pathlib.Path,
  model: int | None,
  altloc: str | None,
  chain_id: Sequence[str] | str | None,
  topology: str | pathlib.Path | None = None,
  estat_info: EstatInfo | None = None,
  *,
  extract_dihedrals: bool = False,
  **kwargs: Any,  # noqa: ANN401
) -> ProteinStream:
  """Parse standard structure files using biotite."""
  logger.info(
    "Starting Biotite parsing for source: %s (model: %s, altloc: %s)",
    source,
    model,
    altloc,
  )
  try:
    # Load structure with hydride (adds H if missing)
    atom_array = load_structure_with_hydride(
      source,
      model=model,
      altloc=altloc,
      topology=topology,
      add_hydrogens=True,
      **kwargs,
    )

    if chain_id is not None:
      atom_array, _ = _process_chain_id(atom_array, chain_id)

    # Create ProcessedStructure
    charges = None
    radii = None
    epsilons = None

    if estat_info:
      charges = estat_info.charges
      radii = estat_info.radii
      epsilons = estat_info.epsilons

    processed = ProcessedStructure(
      atom_array=atom_array,
      r_indices=atom_array.res_id,
      chain_ids=np.zeros(atom_array.array_length(), dtype=np.int32),  # Placeholder
      charges=charges,
      radii=radii,
      epsilons=epsilons,
    )

    yield from processed_structure_to_protein_tuples(
      processed,
      source_name=str(source),
      extract_dihedrals=extract_dihedrals,
    )

  except Exception as e:
    msg = f"Failed to parse structure from source: {source}. {type(e).__name__}: {e}"
    logger.exception(msg)
    raise RuntimeError(msg) from e
