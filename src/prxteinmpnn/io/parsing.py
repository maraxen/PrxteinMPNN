"""Utilities for processing structure and trajectory files.

prxteinmpnn.io.parsing

This module has been refactored to contain only synchronous parsing logic,
making it suitable for use in parallel worker processes managed by Grain.
All async operations and direct I/O handling have been moved to the
`sources.py` and `operations.py` modules.
"""

import logging
import pathlib
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from io import StringIO
from typing import Any, cast

import h5py
import mdtraj as md
import numpy as np
from biotite import structure
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure import io as structure_io

from prxteinmpnn.utils.data_structures import ProteinStream, ProteinTuple, TrajectoryStaticFeatures
from prxteinmpnn.utils.residue_constants import (
  atom_order,
  residue_atoms,
  resname_to_idx,
  restype_order,
  restype_order_with_x,
  unk_restype_index,
)

logger = logging.getLogger(__name__)


MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"
_AF_TO_MPNN_PERM = np.array(
  [MPNN_ALPHABET.index(k) for k in AF_ALPHABET],
)

_MPNN_TO_AF_PERM = np.array(
  [AF_ALPHABET.index(k) for k in MPNN_ALPHABET],
)


def af_to_mpnn(sequence: np.ndarray) -> np.ndarray:
  """Convert a sequence of integer indices from AlphaFold's to ProteinMPNN's alphabet order."""
  logger.debug("Converting sequence indices from AlphaFold to ProteinMPNN alphabet.")
  return _AF_TO_MPNN_PERM[sequence]


def mpnn_to_af(sequence: np.ndarray) -> np.ndarray:
  """Convert a sequence of integer indices from ProteinMPNN's to AlphaFold's alphabet order."""
  logger.debug("Converting sequence indices from ProteinMPNN to AlphaFold alphabet.")
  return _MPNN_TO_AF_PERM[sequence]


def _check_if_file_empty(file_path: str) -> bool:
  """Check if the file is empty."""
  logger.debug("Checking if file path %s is empty.", file_path)
  path = pathlib.Path(file_path)
  suffix = path.suffix.lower()
  try:
    with path.open() as f:
      if suffix in {".h5", ".hdf5"}:
        is_empty = not f.readable()
        if is_empty:
          logger.warning("HDF5 file path %s is not readable.", file_path)
        return is_empty

      # For text files
      is_empty = f.readable() and f.read().strip() == ""
      if is_empty:
        logger.warning("Text file path %s is readable but content is empty.", file_path)
      return is_empty
  except FileNotFoundError:
    logger.warning("File not found: %s", file_path)
    return True
  except Exception as e:
    logger.exception("Error checking if file %s is empty.", file_path, exc_info=e)
    return True


def extend_coordinate(
  atom_a: np.ndarray,
  atom_b: np.ndarray,
  atom_c: np.ndarray,
  bond_length: float,
  bond_angle: float,
  dihedral_angle: float,
) -> np.ndarray:
  """Compute fourth atom (D) position given three atoms (A, B, C) and internal coordinates."""
  logger.debug("Computing extended coordinate (D) from A, B, C.")

  def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)

  bc = normalize(atom_b - atom_c)
  normal = normalize(np.cross(atom_b - atom_a, bc))
  term1 = bond_length * np.cos(bond_angle) * bc
  term2 = bond_length * np.sin(bond_angle) * np.cos(dihedral_angle) * np.cross(normal, bc)
  term3 = bond_length * np.sin(bond_angle) * np.sin(dihedral_angle) * -normal
  return atom_c + term1 + term2 + term3


def compute_cb_precise(
  n_coord: np.ndarray,
  ca_coord: np.ndarray,
  c_coord: np.ndarray,
) -> np.ndarray:
  """Compute the C-beta atom position from backbone N, CA, and C coordinates."""
  logger.debug("Computing C-beta coordinate from N, CA, C backbone atoms.")
  return extend_coordinate(
    c_coord,
    n_coord,
    ca_coord,
    bond_length=1.522,
    bond_angle=1.927,
    dihedral_angle=-2.143,
  )


def string_key_to_index(
  string_keys: np.ndarray,
  key_map: Mapping[str, int],
  unk_index: int | None = None,
) -> np.ndarray:
  """Convert string keys to integer indices based on a mapping."""
  logger.debug("Converting %d string keys to integer indices.", len(string_keys))
  if unk_index is None:
    unk_index = len(key_map)

  sorted_keys = np.array(sorted(key_map.keys()))
  sorted_values = np.array([key_map[k] for k in sorted_keys])
  indices = np.searchsorted(sorted_keys, string_keys)
  indices = np.clip(indices, 0, len(sorted_keys) - 1)

  found_keys = sorted_keys[indices]
  is_known = found_keys == string_keys

  num_unknown = np.sum(~is_known)
  if num_unknown > 0:
    logger.debug("%d unknown keys encountered and mapped to index %d.", num_unknown, unk_index)

  return np.where(is_known, sorted_values[indices], unk_index)


def string_to_protein_sequence(
  sequence: str,
  aa_map: dict | None = None,
  unk_index: int | None = None,
) -> np.ndarray:
  """Convert a string sequence to a ProteinSequence."""
  logger.debug("Converting protein sequence string of length %d to indices.", len(sequence))
  if unk_index is None:
    unk_index = unk_restype_index

  if aa_map is None:
    aa_map = restype_order
    return af_to_mpnn(
      string_key_to_index(np.array(list(sequence), dtype="U3"), aa_map, unk_index),
    )
  return string_key_to_index(np.array(list(sequence), dtype="U3"), aa_map, unk_index)


def protein_sequence_to_string(
  sequence: np.ndarray,
  aa_map: dict | None = None,
) -> str:
  """Convert a ProteinSequence to a string."""
  logger.debug("Converting protein sequence indices of length %d to string.", len(sequence))
  if aa_map is None:
    aa_map = {i: aa for aa, i in restype_order_with_x.items()}

  af_seq = mpnn_to_af(sequence)

  return "".join([aa_map.get(int(aa), "X") for aa in af_seq])


def residue_names_to_aatype(
  residue_names: np.ndarray,
  aa_map: dict | None = None,
) -> np.ndarray:
  """Convert 3-letter residue names to amino acid type indices."""
  logger.debug("Converting %d 3-letter residue names to aatype indices.", len(residue_names))
  if aa_map is None:
    aa_map = resname_to_idx

  aa_indices = string_key_to_index(residue_names, aa_map, unk_restype_index)
  aa_indices = af_to_mpnn(aa_indices)
  return np.asarray(aa_indices, dtype=np.int8)


def atom_names_to_index(
  atom_names: np.ndarray,
  atom_map: dict | None = None,
) -> np.ndarray:
  """Convert atom names to atom type indices."""
  logger.debug("Converting %d atom names to atom type indices.", len(atom_names))
  if atom_map is None:
    atom_map = atom_order

  atom_indices = string_key_to_index(atom_names, atom_map, -1)
  return np.asarray(atom_indices)


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


def mdtraj_dihedrals(
  traj: md.Trajectory,
  num_residues: int,
  nitrogen_mask: np.ndarray,
) -> np.ndarray | None:
  """Compute backbone dihedral angles (phi, psi, omega) for the given md.Trajectory chunk."""
  logger.debug("Computing backbone dihedral angles using MDTraj.")
  phi_indices, phi_angles = md.compute_phi(traj)
  psi_indices, psi_angles = md.compute_psi(traj)
  omega_indices, omega_angles = md.compute_omega(traj)

  dihedrals = np.full((num_residues, 3), np.nan, dtype=np.float64)
  if phi_indices.size > 0:
    dihedrals[phi_indices[:, 1], 0] = phi_angles[0]
  if psi_indices.size > 0:
    dihedrals[psi_indices[:, 1], 1] = psi_angles[0]
  if omega_indices.size > 0:
    dihedrals[omega_indices[:, 0], 2] = omega_angles[0]

  final_dihedrals = dihedrals[nitrogen_mask]
  logger.debug("MDTraj calculated dihedrals for %d residues.", final_dihedrals.shape[0])

  return final_dihedrals


def process_coordinates(
  coordinates: np.ndarray,
  num_residues: int,
  atom_37_indices: np.ndarray,
  valid_atom_mask: np.ndarray,
) -> np.ndarray:
  """Process an AtomArray to create a Protein inputs."""
  logger.debug("Processing coordinates into (N_res, 37, 3) format.")
  coords_37 = np.zeros((num_residues, 37, 3), dtype=np.float32)
  coords_37[atom_37_indices] = np.asarray(
    coordinates,
  )[valid_atom_mask]
  return coords_37


def _select_chain_mdtraj(
  traj: md.Trajectory,
  chain_id: Sequence[str] | str | None = None,
) -> md.Trajectory:
  """Select specific chains from an md.Trajectory."""
  if traj.top is None:
    msg = "Trajectory does not have a topology."
    logger.error(msg)
    raise ValueError(msg)

  if chain_id is not None:
    if isinstance(chain_id, str):
      chain_id = [chain_id]

    logger.info("Selecting chain(s) %s in MDTraj topology.", chain_id)
    selection = " or ".join(f"chainid {cid}" for cid in chain_id)
    atom_indices = traj.top.select(selection)

    if atom_indices.size == 0:
      msg = f"No atoms found for chain(s) {chain_id}."
      logger.warning(msg)
      # Retain the original warning call behavior
      raise ValueError(msg)

    traj = traj.atom_slice(atom_indices)
    logger.debug("Sliced MDTraj trajectory to %d atoms.", traj.n_atoms)

  return traj


def _extract_mdtraj_static_features(
  traj_chunk: md.Trajectory,
  atom_map: dict[str, int] | None = None,
) -> TrajectoryStaticFeatures:
  """Extract frame-invariant (static) features from a trajectory chunk's topology."""
  logger.info("Extracting static features using MDTraj topology.")
  if traj_chunk.top is None:
    msg = "Trajectory does not have a topology."
    logger.error(msg)
    raise ValueError(msg)
  if atom_map is None:
    atom_map = atom_order

  topology = traj_chunk.top
  num_residues_all = topology.n_residues
  if num_residues_all == 0:
    msg = "Trajectory has no residues after filtering."
    logger.error(msg)
    raise ValueError(msg)
  logger.debug("MDTraj topology contains %d residues.", num_residues_all)

  # Pre-compute all static topology-derived information
  atom_names = np.array([a.name for a in topology.atoms])
  atom37_indices = atom_names_to_index(atom_names.astype("U5"))
  residue_inv_indices = np.array([a.residue.index for a in topology.atoms])
  valid_atom_mask = atom37_indices != -1
  res_indices_flat = residue_inv_indices[valid_atom_mask]
  atom_indices_flat = atom37_indices[valid_atom_mask]

  residue_names = np.array([r.name for r in topology.residues])
  aatype = residue_names_to_aatype(residue_names)
  residue_indices = np.array([r.resSeq for r in topology.residues], dtype=np.int32)

  chain_ids_per_res = [r.chain.index for r in topology.residues]
  unique_chain_ids = sorted(set(chain_ids_per_res))
  chain_map = {cid: i for i, cid in enumerate(unique_chain_ids)}
  chain_index = np.array([chain_map[cid] for cid in chain_ids_per_res], dtype=np.int32)
  static_atom_mask_37 = np.zeros((num_residues_all, 37), dtype=bool)
  static_atom_mask_37[res_indices_flat, atom_indices_flat] = True
  nitrogen_mask = static_atom_mask_37[:, atom_map["N"]]

  if not np.any(nitrogen_mask):
    msg = "No residues with backbone nitrogen atoms found."
    logger.warning(msg)
    # Retain original warning/error behavior
    raise ValueError(msg)

  num_residues = np.sum(nitrogen_mask)
  logger.info("Found %d valid residues (with N atom) for feature extraction.", num_residues)

  return TrajectoryStaticFeatures(
    aatype=aatype[nitrogen_mask],
    static_atom_mask_37=static_atom_mask_37[nitrogen_mask],
    residue_indices=residue_indices[nitrogen_mask],
    chain_index=chain_index[nitrogen_mask],
    valid_atom_mask=valid_atom_mask,
    nitrogen_mask=nitrogen_mask,
    num_residues=num_residues,
  )


def _determine_h5_structure(source: str | StringIO | pathlib.Path) -> str:
  """Determine the structure of an HDF5 file."""
  logger.debug("Attempting to determine HDF5 file structure for source: %s", source)
  try:
    with h5py.File(source, "r") as f:
      keys = list(f.keys())
      if "coordinates" in f:
        logger.info("HDF5 structure determined as 'mdtraj' (found 'coordinates' key).")
        return "mdtraj"
      if len(keys) == 1:
        logger.info(
          "HDF5 structure determined as 'mdcath' (found single top-level key: %s).",
          keys[0],
        )
        return "mdcath"
      msg = f"Could not determine HDF5 structure: Unrecognized format. Top-level keys: {keys}"
      logger.warning(msg)
      warnings.warn(msg, stacklevel=2)  # Retain original warning behavior
      return "unknown"
  except Exception as e:
    msg = f"Could not determine HDF5 structure: {type(e).__name__}: {e}"
    logger.exception(msg)
    warnings.warn(msg, stacklevel=2)  # Retain original warning behavior
    return "unknown"


def _parse_mdtraj_hdf5(
  source: str | StringIO | pathlib.Path,
  chain_id: Sequence[str] | str | None,
  *,
  extract_dihedrals: bool = False,
) -> ProteinStream:
  """Parse HDF5 structure files directly using mdtraj."""
  logger.info("Starting MDTraj HDF5 parsing for source: %s", source)
  try:
    dihedrals = None
    first_frame = md.load_frame(str(source), 0)
    logger.debug("Loaded first frame to determine topology.")

    first_frame = _select_chain_mdtraj(first_frame, chain_id=chain_id)

    static_features = _extract_mdtraj_static_features(
      first_frame,
    )
    logger.info(
      "Successfully extracted static features for %d residues.",
      static_features.num_residues,
    )

    traj_iterator = md.iterload(str(source))
    frame_count = 0
    for traj_chunk in traj_iterator:
      logger.debug("Processing MDTraj chunk with %d frames.", traj_chunk.n_frames)
      for frame in traj_chunk:
        frame_count += 1
        coords = frame.xyz

        if extract_dihedrals:
          dihedrals = mdtraj_dihedrals(
            frame,
            static_features.num_residues,
            static_features.nitrogen_mask,
          )

        coords_37 = process_coordinates(
          coords[0],  # MDTraj xyz is (n_frames, n_atoms, 3)
          static_features.num_residues,
          static_features.static_atom_mask_37,
          static_features.valid_atom_mask,
        )
        logger.debug("Yielding frame %d from source %s.", frame_count, source)
        yield ProteinTuple(
          coordinates=coords_37,
          aatype=static_features.aatype,
          atom_mask=static_features.static_atom_mask_37,
          residue_index=static_features.residue_indices,
          chain_index=static_features.chain_index,
          dihedrals=dihedrals,
          source=str(source),
          full_coordinates=coords[0],
        )
    logger.info("Finished MDTraj HDF5 parsing. Yielded %d frames.", frame_count)

  except Exception as e:
    msg = f"Failed to parse HDF5 structure from source: {source}. {type(e).__name__}: {e}"
    logger.exception(msg)
    raise RuntimeError(msg) from e


def _parse_mdcath_hdf5(
  source: str | StringIO | pathlib.Path,
  chain_id: Sequence[str] | str | None,
  *,
  extract_dihedrals: bool = False,  # noqa: ARG001
) -> ProteinStream:
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
      atom_mask_37[:, 0:5] = True  # Set CA, CB, N, C, O atoms to be present

      sample_coords_shape = cast(
        "h5py.Dataset",
        cast("h5py.Group", domain_group[first_temp_key])[first_replica_key],
      )["coords"].shape
      num_full_atoms = sample_coords_shape[1]
      logger.info("Number of full atoms from sample coords: %d", num_full_atoms)

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

              coords_37 = process_coordinates(
                frame_coords_full,
                static_features.num_residues,
                static_features.static_atom_mask_37,
                static_features.valid_atom_mask,
              )

              yield ProteinTuple(
                coordinates=coords_37,
                aatype=static_features.aatype,
                atom_mask=static_features.static_atom_mask_37,
                residue_index=static_features.residue_indices,
                chain_index=static_features.chain_index,
                dihedrals=None,
                source=str(source),
                full_coordinates=frame_coords_full,
              )
      logger.info("Finished mdCATH HDF5 parsing. Yielded %d frames.", frame_count)

  except Exception as e:
    msg = f"Failed to parse mdCATH HDF5 structure from source: {source}. {type(e).__name__}: {e}"
    logger.exception(msg)
    warnings.warn(msg, stacklevel=2)


def _validate_atom_array_type(atom_array: Any) -> None:
  """Validate that the atom array is of the expected type."""
  logger.debug("Validating atom array type.")
  if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
    msg = f"Expected AtomArray or AtomArrayStack, but got {type(atom_array)}."
    logger.error(msg)
    raise TypeError(msg)


def _parse_biotite(
  source: str | StringIO | pathlib.Path,
  model: int | None,
  altloc: str | None,
  chain_id: Sequence[str] | str | None,
  *,
  extract_dihedrals: bool = False,
  **kwargs: Any,
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

    atom_array = structure_io.load_structure(
      source,
      model=model,
      altloc=altloc,
      **kwargs,
    )
    logger.debug("Structure loaded successfully using Biotite.")
    _validate_atom_array_type(atom_array)

    if isinstance(atom_array, (AtomArray, AtomArrayStack)):
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
        )

    logger.info("Finished Biotite parsing. Yielded %d frames.", frame_count)

  except Exception as e:
    msg = f"Failed to parse structure from source: {source}. {type(e).__name__}: {e}"
    logger.exception(msg)
    raise RuntimeError(msg) from e


def parse_input(
  source: str | StringIO | pathlib.Path,
  *,
  model: int | None = None,
  altloc: str | None = None,
  chain_id: Sequence[str] | str | None = None,
  extract_dihedrals: bool = False,
  **kwargs: Any,
) -> ProteinStream:
  """Parse a structure file or string into a generator of ProteinTuples.

  This is a synchronous, CPU-bound function intended to be run in a worker process.
  It dispatches to the correct low-level parser based on file type.

  Args:
      source: A file path (str or Path) or a file-like object (StringIO).
      model: The model number to load. If None, all models are loaded.
      altloc: The alternate location identifier to use.
      chain_id: Specific chain(s) to parse from the structure.
      extract_dihedrals: Whether to compute and include backbone dihedral angles.
      **kwargs: Additional keyword arguments to pass to the structure loader.

  Yields:
      A tuple containing a `ProteinTuple` and the source identifier string.

  """
  logger.info("Starting input parsing for source: %s", source)
  temp_path = None

  if isinstance(source, StringIO):
    logger.debug("Source is StringIO. Creating temporary PDB file.")
    with tempfile.NamedTemporaryFile(
      mode="w",
      delete=False,
      suffix=".pdb",
    ) as tmp:
      tmp.write(source.read())
      temp_path = pathlib.Path(tmp.name)
      logger.info("Content written to temporary file: %s", temp_path)
      source = temp_path

  try:
    if isinstance(source, (str, pathlib.Path)):
      path = pathlib.Path(source)
      if path.suffix.lower() in {".h5", ".hdf5"}:
        h5_structure = _determine_h5_structure(path)

        if h5_structure == "mdcath":
          logger.info("Dispatching to mdCATH HDF5 parser.")
          yield from _parse_mdcath_hdf5(path, chain_id, extract_dihedrals=extract_dihedrals)
        elif h5_structure == "mdtraj":
          logger.info("Dispatching to MDTraj HDF5 parser.")
          yield from _parse_mdtraj_hdf5(path, chain_id, extract_dihedrals=extract_dihedrals)
        else:
          logger.warning("Unknown HDF5 structure, returning early.")
        return

    logger.info(
      "Dispatching to general Biotite parser for file type: %s",
      pathlib.Path(source).suffix,
    )
    yield from _parse_biotite(
      source,
      model,
      altloc=altloc,
      chain_id=chain_id,
      extract_dihedrals=extract_dihedrals,
      **kwargs,
    )

  finally:
    if temp_path is not None:
      # Clean up temporary file created from StringIO
      try:
        temp_path.unlink()
        logger.debug("Cleaned up temporary file: %s", temp_path)
      except OSError as e:
        logger.warning("Could not delete temporary file %s: %s", temp_path, e)
