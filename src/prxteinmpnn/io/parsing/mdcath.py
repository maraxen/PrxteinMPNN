"""Utilities for processing structure and trajectory files.

prxteinmpnn.io.parsing

This module has been refactored to contain only synchronous parsing logic,
making it suitable for use in parallel worker processes managed by Grain.
All async operations and direct I/O handling have been moved to the
`sources.py` and `operations.py` modules.
"""

import logging
import pathlib
import warnings
from collections.abc import Iterator, Sequence
from io import StringIO
from typing import cast

import h5py
import numpy as np
from biotite import structure
from biotite.structure import AtomArray, filter_solvent

from prxteinmpnn.io.parsing.structures import ProcessedStructure
from prxteinmpnn.utils import residue_constants as rc
from prxteinmpnn.utils.data_structures import TrajectoryStaticFeatures

from .coords import process_coordinates
from .mappings import residue_names_to_aatype

logger = logging.getLogger(__name__)


def parse_mdcath_to_processed_structure(  # noqa: PLR0915, C901
  source: str | StringIO | pathlib.Path,
  chain_id: Sequence[str] | str | None,
) -> Iterator[ProcessedStructure]:
  """Parse mdCATH HDF5 files."""
  logger.info("Starting mdCATH HDF5 parsing for source: %s", source)
  try:
    with h5py.File(source, "r") as f:
      domain_id = cast("str", next(iter(f.keys())))
      domain_group = cast("h5py.Group", f[domain_id])
      logger.info("Parsing domain %s from mdCATH HDF5 file.", domain_id)

      if chain_id is not None:
        msg = "Chain selection is not supported for mdCATH files. Ignoring chain_id parameter."
        logger.warning(msg)
        warnings.warn(msg, stacklevel=2)

      first_temp_key = next(iter(domain_group.keys()))
      first_replica_key = next(iter(cast("h5py.Group", domain_group[first_temp_key]).keys()))
      dssp_sample = cast(
        "h5py.Dataset",
        cast("h5py.Group", domain_group[first_temp_key])[first_replica_key],
      )["dssp"]
      num_residues_from_dssp = dssp_sample.shape[1]
      logger.debug("Initial residue count from DSSP dataset: %d", num_residues_from_dssp)

      aatype: np.ndarray
      try:
        # Convert 3-letter residue names to aatype indices, as in other workflows
        resnames = cast("h5py.Dataset", domain_group["resname"])[:].astype("U3")
        aatype = residue_names_to_aatype(resnames)
        if aatype.shape[0] != num_residues_from_dssp:
          msg = (
            f"Shape of 'resname' ({aatype.shape[0]}) does not match "
            f"num_residues ({num_residues_from_dssp}) derived from 'dssp'. "
            "Using 'resid' for aatype, but investigate discrepancy."
          )
          logger.warning(msg)
          warnings.warn(msg, stacklevel=2)
      except KeyError:
        msg = (
          " 'resid' dataset not found at domain_group level. "
          "Using a generic aatype (all Alanine). Please confirm 'resid' location."
        )
        logger.warning(msg)
        warnings.warn(msg, stacklevel=2)
        aatype = np.zeros(num_residues_from_dssp, dtype=np.int32)

      num_residues = aatype.shape[0]
      logger.info("Final residue count used: %d", num_residues)

      residue_indices = np.arange(num_residues)
      chain_index = np.zeros(num_residues, dtype=np.int32)

      atom_mask_37 = np.zeros((num_residues, 37), dtype=bool)
      resnames = cast("h5py.Dataset", domain_group["resname"])[:].astype("U3")
      three_to_one = {
          name: name for name in np.unique(resnames)
      }
      for i, resname in enumerate(resnames):
          if resname in three_to_one:
              for atom_name in rc.restype_name_to_atom14_names[three_to_one[resname]]:
                  if atom_name in rc.atom_order:
                      atom_mask_37[i, rc.atom_order[atom_name]] = True

      sample_coords_shape = cast(
        "h5py.Dataset",
        cast("h5py.Group", domain_group[first_temp_key])[first_replica_key],
      )["coords"].shape
      num_full_atoms = sample_coords_shape[1]
      logger.info("Number of full atoms from sample coords: %d", num_full_atoms)
      valid_atom_mask = np.ones(num_full_atoms, dtype=bool)

      # Reshape to be in 37, using the number of atoms per residue identity

      static_features = TrajectoryStaticFeatures(
        aatype=aatype,
        static_atom_mask_37=atom_mask_37,
        residue_indices=residue_indices,
        chain_index=chain_index,
        valid_atom_mask=valid_atom_mask,
        nitrogen_mask=np.ones(num_residues, dtype=bool),
        num_residues=num_residues,
      )

      frame_count = 0
      for temp_key in domain_group:
        temp_group = cast("h5py.Group", domain_group[temp_key])
        if cast("str", temp_key).isdigit():
          logger.debug("Processing temperature group: %s", temp_key)
          for replica_key in temp_group:
            replica_group = cast("h5py.Group", temp_group[replica_key])

            coords_dataset = cast("h5py.Dataset", replica_group["coords"])
            logger.debug(
              "Processing replica %s with %d frames.",
              replica_key,
              coords_dataset.shape[0],
            )

            for frame_index in range(coords_dataset.shape[0]):
              frame_count += 1
              frame_coords_full = coords_dataset[frame_index]

              process_coordinates(
                frame_coords_full,
                static_features.num_residues,
                static_features.static_atom_mask_37,
                static_features.valid_atom_mask,
              )

              # Construct AtomArrayStack for this frame?
              # Or just yield ProcessedStructure for each frame?
              # ProcessedStructure expects AtomArray or AtomArrayStack.
              # Here we are iterating frames, so we yield one by one.

              # We need to construct an AtomArray from the coordinates and static features.
              # This is a bit involved as we need to map back from 37-atom representation or full atoms?
              # frame_coords_full is (num_full_atoms, 3).
              # We have valid_atom_mask which is all ones.

              # We can construct an AtomArray with all atoms.
              # We need atom names and residue names.

              # Reconstructing AtomArray from raw arrays is tedious but necessary for ProcessedStructure.
              # However, for MDCATH, we might just want to wrap the data we have?
              # But ProcessedStructure requires AtomArray.

              # Build AtomArray from coordinates
              num_atoms = frame_coords_full.shape[0]
              atom_array = AtomArray(num_atoms)
              atom_array.coord = frame_coords_full

              # Populate basic atom information
              # We need to expand residue-level info to atom-level
              # This is simplified - ideally we'd reconstruct full atom details
              atom_array.res_id = np.repeat(static_features.residue_indices, num_atoms // static_features.num_residues)
              atom_array.res_name = np.repeat(resnames, num_atoms // static_features.num_residues)
              atom_array.chain_id = np.repeat(["A"], num_atoms)  # Simplified

              # Apply solvent removal
              solvent_mask = filter_solvent(atom_array)
              if np.any(solvent_mask):
                  n_solvent = np.sum(solvent_mask)
                  logger.info("Removing %d solvent atoms from MDCATH frame", n_solvent)
                  atom_array = atom_array[~solvent_mask]

              # Add hydrogens if missing
              has_hydrogens = (atom_array.element == "H").any() if hasattr(atom_array, "element") else False
              if not has_hydrogens:
                  logger.info("Adding hydrogens to MDCATH AtomArray")
                  # Infer bonds
                  if not atom_array.bonds:
                      try:
                          atom_array.bonds = structure.connect_via_residue_names(atom_array)
                      except Exception as e:
                          logger.warning("Failed to infer bonds: %s", e)
                          atom_array.bonds = structure.connect_via_distances(atom_array)

                  # Add charge annotation
                  if "charge" not in atom_array.get_annotation_categories():
                      atom_array.set_annotation("charge", np.zeros(atom_array.array_length(), dtype=int))

                  try:
                      import hydride
                      atom_array, _ = hydride.add_hydrogen(atom_array)
                      logger.info("Hydrogens added to MDCATH structure")
                  except Exception as e:
                      logger.warning("Failed to add hydrogens: %s", e)

              yield ProcessedStructure(
                  atom_array=atom_array,
                  r_indices=atom_array.res_id,
                  chain_ids=np.zeros(atom_array.array_length(), dtype=np.int32),
              )
      logger.info("Finished mdCATH HDF5 parsing. Yielded %d frames.", frame_count)

  except Exception as e:
    msg = f"Failed to parse mdCATH HDF5 structure from source: {source}. {type(e).__name__}: {e}"
    logger.exception(msg)
    warnings.warn(msg, stacklevel=2)
