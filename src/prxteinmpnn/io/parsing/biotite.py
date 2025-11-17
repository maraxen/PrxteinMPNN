"""Parsing utilities for Biotite."""

import logging
import pathlib
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, cast

import numpy as np
from biotite import structure
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure import io as structure_io

from prxteinmpnn.utils.data_structures import (
  EstatInfo,
  ProteinStream,
  ProteinTuple,
  TrajectoryStaticFeatures,
)
from prxteinmpnn.utils.residue_constants import (
  atom_order,
)

from .coords import process_coordinates
from .mappings import atom_names_to_index, residue_names_to_aatype

logger = logging.getLogger(__name__)


def _check_atom_array_length(atom_array: AtomArray | AtomArrayStack) -> None:
  """Check if the AtomArray has a valid length."""
  length = atom_array.array_length()
  logger.debug("Checking AtomArray length: %d", length)
  if length == 0:
    msg = "AtomArray is empty."
    logger.error(msg)
    raise ValueError(msg)


def _get_chain_index(
  atom_array: AtomArray | AtomArrayStack,
) -> np.ndarray:
  """Get the chain index from the AtomArray."""
  if atom_array.chain_id is None:
    logger.debug("Chain ID not available, returning zeros for chain index.")
    return np.zeros(atom_array.array_length(), dtype=np.int32)

  if atom_array.chain_id.dtype != np.int32:
    logger.debug("Converting string chain IDs to integer indices (A=0, B=1, ...).")
    return np.asarray(
      np.char.encode(atom_array.chain_id.astype("U1")).view(np.uint8) - ord("A"),
      dtype=np.int32,
    )

  logger.debug("Using existing integer chain IDs.")
  return np.asarray(atom_array.chain_id, dtype=np.int32)


def _process_chain_id(
  atom_array: AtomArray | AtomArrayStack,
  chain_id: Sequence[str] | str | None = None,
) -> tuple[AtomArray | AtomArrayStack, np.ndarray]:
  """Process the chain_id of the AtomArray."""
  if chain_id is None:
    logger.debug("No chain_id specified. Using all available chains.")
    chain_index = _get_chain_index(atom_array)
    return atom_array, chain_index

  logger.info("Processing structure with specified chain_id(s): %s", chain_id)

  if isinstance(chain_id, str):
    chain_id = [chain_id]

  if not isinstance(chain_id, Sequence):
    msg = f"Expected chain_id to be a string or a sequence of strings, but got {type(chain_id)}."
    logger.error(msg)
    raise TypeError(msg)

  if atom_array.chain_id is None:
    msg = "Chain ID is not available in the structure, but chain_id was specified."
    logger.error(msg)
    raise ValueError(msg)

  chain_mask = np.isin(atom_array.chain_id, chain_id)

  if not np.any(chain_mask):
    logger.warning("No atoms found for specified chain(s) %s.", chain_id)

  if isinstance(atom_array, AtomArrayStack):
    atom_array = cast("AtomArray | AtomArrayStack", atom_array[:, chain_mask])
  else:
    atom_array = cast("AtomArray | AtomArrayStack", atom_array[chain_mask])

  chain_index = _get_chain_index(atom_array)
  logger.debug("Filtered AtomArray to %d atoms for specified chains.", atom_array.array_length())
  return (
    atom_array,
    chain_index,
  )


def _extract_biotite_static_features(
  atom_array: AtomArray | AtomArrayStack,
  atom_map: dict[str, int] | None = None,
  chain_id: Sequence[str] | str | None = None,
) -> tuple[TrajectoryStaticFeatures, AtomArray | AtomArrayStack]:
  """Extract static features from a Biotite AtomArray."""
  logger.info("Extracting static features using Biotite.")
  if atom_map is None:
    atom_map = atom_order

  atom_array, chain_index = _process_chain_id(atom_array, chain_id)
  _check_atom_array_length(atom_array)
  num_residues_all_atoms = structure.get_residue_count(atom_array)

  residue_indices, residue_names = structure.get_residues(atom_array)
  logger.debug("Found %d residues in the processed AtomArray.", num_residues_all_atoms)

  residue_indices = np.asarray(residue_indices, dtype=np.int32)
  chain_index = chain_index[structure.get_residue_starts(atom_array)]
  residue_inv_indices = structure.get_residue_positions(
    atom_array,
    np.arange(atom_array.array_length()),
  )

  atom_names = atom_array.atom_name

  if atom_names is None:
    msg = "Atom names are not available in the structure."
    logger.error(msg)
    raise ValueError(msg)

  atom37_indices = atom_names_to_index(np.array(atom_names, dtype="U5"))

  atom_mask = atom37_indices != -1

  # This is the mask for 37 atoms for ALL residues (before filtering)
  atom_mask_37 = np.zeros((num_residues_all_atoms, 37), dtype=bool)

  res_indices_flat = np.asarray(residue_inv_indices)[atom_mask]
  atom_indices_flat = atom37_indices[atom_mask]

  atom_mask_37[res_indices_flat, atom_indices_flat] = 1

  aatype = residue_names_to_aatype(residue_names)
  nitrogen_mask = atom_mask_37[:, atom_map["N"]] == 1

  # Filter to residues that have an N atom (required for backbone trace)
  aatype = aatype[nitrogen_mask]
  atom_mask_37 = atom_mask_37[nitrogen_mask]
  residue_indices = residue_indices[nitrogen_mask]
  chain_index = chain_index[nitrogen_mask]

  num_residues = aatype.shape[0]
  logger.info("Filtered AtomArray to %d valid residues (those containing an N atom).", num_residues)

  valid_residue_mask = nitrogen_mask[np.asarray(residue_inv_indices)]
  atom_mask &= valid_residue_mask

  return TrajectoryStaticFeatures(
    aatype=aatype,
    static_atom_mask_37=atom_mask_37,
    residue_indices=residue_indices,
    chain_index=chain_index,
    valid_atom_mask=atom_mask,
    nitrogen_mask=nitrogen_mask,
    num_residues=num_residues,
  ), atom_array


def atom_array_dihedrals(
  atom_array: AtomArray | AtomArrayStack,
) -> np.ndarray | None:
  """Compute backbone dihedral angles (phi, psi, omega) for the given AtomArray."""
  logger.debug("Computing backbone dihedral angles using Biotite.")
  phi, psi, omega = structure.dihedral_backbone(atom_array)
  phi = np.asarray(phi)
  psi = np.asarray(psi)
  omega = np.asarray(omega)
  if (
    phi is None
    or psi is None
    or omega is None
    or np.all(np.isnan(phi))
    or np.all(np.isnan(psi))
    or np.all(np.isnan(omega))
  ):
    logger.warning("Dihedral calculation resulted in all NaN values or None.")
    return None

  dihedrals = np.stack([phi, psi, omega], axis=-1)

  clean_dihedrals = dihedrals[~np.any(np.isnan(dihedrals), axis=-1)]
  logger.debug("Calculated %d valid dihedral sets.", clean_dihedrals.shape[0])

  return clean_dihedrals


def _validate_atom_array_type(atom_array: Any) -> None:  # noqa: ANN401
  """Validate that the atom array is of the expected type."""
  logger.debug("Validating atom array type.")
  if not isinstance(atom_array, (AtomArray | AtomArrayStack)):
    msg = f"Expected AtomArray or AtomArrayStack, but got {type(atom_array)}."
    logger.error(msg)
    raise TypeError(msg)


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
  frame_count = 0
  try:
    altloc = altloc if altloc is not None else "first"
    dihedrals = None
    topology_array = None
    if topology is not None:
      topology_array = structure_io.load_structure(
        topology,
      )
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
    logger.debug("Structure loaded successfully using Biotite.")
    _validate_atom_array_type(atom_array)

    if isinstance(atom_array, (AtomArray | AtomArrayStack)):
      static_features, atom_array = _extract_biotite_static_features(atom_array, chain_id=chain_id)
      num_frames = atom_array.stack_depth() if isinstance(atom_array, AtomArrayStack) else 1
      frame_count = 0

      if isinstance(atom_array, AtomArrayStack):
        for frame in atom_array:
          frame_count += 1
          if extract_dihedrals:
            dihedrals = atom_array_dihedrals(frame)

          coords = np.asarray(frame.coord)
          coords_37 = process_coordinates(
            coords,
            static_features.num_residues,
            static_features.static_atom_mask_37,
            static_features.valid_atom_mask,
          )
          logger.debug("Yielding frame %d of %d from Biotite stack.", frame_count, num_frames)
          yield ProteinTuple(
            coordinates=coords_37,
            aatype=static_features.aatype,
            atom_mask=static_features.static_atom_mask_37,
            residue_index=static_features.residue_indices,
            chain_index=static_features.chain_index,
            dihedrals=dihedrals,
            source=str(source),
            full_coordinates=coords,
            **asdict(estat_info) if estat_info is not None else {},
          )

      elif isinstance(atom_array, AtomArray):
        frame_count += 1
        if extract_dihedrals:
          dihedrals = atom_array_dihedrals(atom_array)

        coords = np.asarray(atom_array.coord)
        coords_37 = process_coordinates(
          coords,
          static_features.num_residues,
          static_features.static_atom_mask_37,
          static_features.valid_atom_mask,
        )
        logger.debug("Yielding single frame from Biotite AtomArray.")
        yield ProteinTuple(
          coordinates=coords_37,
          aatype=static_features.aatype,
          atom_mask=static_features.static_atom_mask_37,
          residue_index=static_features.residue_indices,
          chain_index=static_features.chain_index,
          dihedrals=dihedrals,
          source=str(source),
          full_coordinates=coords,
          **asdict(estat_info) if estat_info is not None else {},
        )

    logger.info("Finished Biotite parsing. Yielded %d frames.", frame_count)

  except Exception as e:
    msg = f"Failed to parse structure from source: {source}. {type(e).__name__}: {e}"
    logger.exception(msg)
    yield ProteinTuple(
      coordinates=np.empty((0, 37, 3), dtype=np.float32),
      aatype=np.empty((0,), dtype=np.int32),
      atom_mask=np.empty((0, 37), dtype=bool),
      residue_index=np.empty((0,), dtype=np.int32),
      chain_index=np.empty((0,), dtype=np.int32),
    )
