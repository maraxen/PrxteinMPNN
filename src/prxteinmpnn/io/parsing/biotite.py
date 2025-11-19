"""Parsing utilities for Biotite."""

import logging
import pathlib
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import hydride
from biotite import structure
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure import io as structure_io

from prxteinmpnn.utils.data_structures import (
  EstatInfo,
  ProteinStream,
  ProteinTuple,
  TrajectoryStaticFeatures,
)
from prxteinmpnn.io.parsing.structures import ProcessedStructure
from prxteinmpnn.utils.residue_constants import (
  atom_order,
)

from .coords import process_coordinates
from .mappings import atom_names_to_index, residue_names_to_aatype

logger = logging.getLogger(__name__)


def load_structure_with_hydride(
  source: str | pathlib.Path,
  model: int | None = None,
  altloc: str | None = None,
  topology: str | pathlib.Path | None = None,
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
          **kwargs
      )
  else:
      atom_array = structure_io.load_structure(
          source,
          model=model,
          altloc=altloc,
          template=topology_array,
          **kwargs
      )

  logger.info(
      "Loading structure from %s (add_hydrogens=%s, remove_solvent=%s)", 
      source, add_hydrogens, remove_solvent
  )
  _validate_atom_array_type(atom_array)

  # Remove solvent before adding hydrogens
  if remove_solvent:
      solvent_mask = structure.filter_solvent(atom_array)
      if np.any(solvent_mask):
          n_solvent = np.sum(solvent_mask)
          logger.info("Removing %d solvent atoms", n_solvent)
          
          if isinstance(atom_array, AtomArrayStack):
              atom_array = atom_array[:, ~solvent_mask]
          else:
              atom_array = atom_array[~solvent_mask]
          
          logger.debug("Structure after solvent removal: %d atoms", atom_array.array_length())

  if add_hydrogens:
      # Check if hydrogens exist
      has_hydrogens = (atom_array.element == "H").any()
      if has_hydrogens:
          logger.debug("Structure already has hydrogens.")
          return atom_array
          
      logger.info("No hydrogens found. Adding hydrogens using Hydride.")
      
      if isinstance(atom_array, AtomArrayStack):
             logger.warning("Hydride integration with AtomArrayStack is experimental. Processing frame 0 only for now or iterating?")
             # Hydride returns a single AtomArray. 
             # For now, let's just return the stack as is if we can't easily protonate it, 
             # or warn.
             # Ideally we should iterate and protonate each frame, but topology might change?
             # Let's just support AtomArray for now for hydride.
             pass
      else:
        # Hydride requires a BondList. If not present, we must infer it.
        if not atom_array.bonds:
            logger.info("No BondList found. Inferring bonds via residue names.")
            try:
                atom_array.bonds = structure.connect_via_residue_names(atom_array)
            except Exception as e:
                logger.warning("Failed to connect via residue names: %s. Falling back to distances.", e)
                atom_array.bonds = structure.connect_via_distances(atom_array)
        
        # Hydride also requires a 'charge' annotation, even if 0.
        if "charge" not in atom_array.get_annotation_categories():
            logger.info("No charge annotation found. Adding zero charges for Hydride compatibility.")
            atom_array.set_annotation("charge", np.zeros(atom_array.array_length(), dtype=int))

        atom_array, _ = hydride.add_hydrogen(atom_array)
        logger.info("Hydrogens added. New atom count: %d", atom_array.array_length())

  return atom_array


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


def processed_structure_to_protein_tuples(
  processed_structure: ProcessedStructure,
  source_name: str,
  extract_dihedrals: bool = False,
  populate_physics: bool = True,
  force_field_name: str = "amber14-all",
) -> ProteinStream:
  """Convert a ProcessedStructure into a stream of ProteinTuples.
  
  Args:
      processed_structure: The ProcessedStructure to convert
      source_name: Name of the source file
      extract_dihedrals: Whether to extract dihedral angles
      populate_physics: Whether to populate physics parameters if missing
      force_field_name: Force field to use for parameter population
  """
  atom_array = processed_structure.atom_array
  
  # Extract static features
  # We assume the AtomArray in ProcessedStructure is what we want to process.
  # So we pass chain_id=None to _extract_biotite_static_features.
  
  static_features, atom_array = _extract_biotite_static_features(atom_array, chain_id=None)
  
  # Populate physics parameters if not already present and requested
  charges = processed_structure.charges
  radii = processed_structure.radii
  sigmas = processed_structure.sigmas
  epsilons = processed_structure.epsilons
  
  if populate_physics and (charges is None or sigmas is None or epsilons is None):
      logger.info("Populating missing physics parameters from force field")
      from prxteinmpnn.io.parsing.physics_utils import populate_physics_parameters
      
      charges_ff, sigmas_ff, epsilons_ff = populate_physics_parameters(
          atom_array,
          force_field_name=force_field_name
      )
      
      # Use force field values for missing parameters
      if charges is None:
          charges = charges_ff
      if sigmas is None:
          sigmas = sigmas_ff
      if epsilons is None:
          epsilons = epsilons_ff
      
      # Radii: use van der Waals radii if not present
      if radii is None:
          from prxteinmpnn.io.parsing.physics_utils import _get_default_parameters
          _, _, _ = _get_default_parameters(atom_array)  # Just for consistency
          # Simple element-based radii
          element_radii = {
              "H": 1.20, "C": 1.70, "N": 1.55,
              "O": 1.52, "S": 1.80, "P": 1.80,
          }
          radii = np.array([
              element_radii.get(elem, 1.70)
              for elem in atom_array.element
          ], dtype=np.float32)
  
  num_frames = atom_array.stack_depth() if isinstance(atom_array, AtomArrayStack) else 1
  frame_count = 0

  if isinstance(atom_array, AtomArrayStack):
    for frame in atom_array:
      frame_count += 1
      dihedrals = None
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
        source=str(source_name),
        full_coordinates=coords,
        charges=charges,
        radii=radii,
        epsilons=epsilons,
        sigmas=sigmas,
      )

  elif isinstance(atom_array, AtomArray):
    frame_count += 1
    dihedrals = None
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
      source=str(source_name),
      full_coordinates=coords,
      charges=charges,
      radii=radii,
      epsilons=epsilons,
      sigmas=sigmas,
    )


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
        **kwargs
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
        chain_ids=np.zeros(atom_array.array_length(), dtype=np.int32), # Placeholder
        charges=charges,
        radii=radii,
        epsilons=epsilons,
    )
    
    yield from processed_structure_to_protein_tuples(
        processed,
        source_name=str(source),
        extract_dihedrals=extract_dihedrals
    )

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
