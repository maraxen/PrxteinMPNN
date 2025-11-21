"""Parsing utilities for MDTraj."""

import logging
import pathlib
from collections.abc import Iterator, Sequence
from io import StringIO

import mdtraj as md
import numpy as np
from biotite import structure
from biotite.structure import AtomArray, AtomArrayStack, filter_solvent

from prxteinmpnn.io.parsing.structures import ProcessedStructure
from prxteinmpnn.utils.data_structures import TrajectoryStaticFeatures
from prxteinmpnn.utils.residue_constants import (
  atom_order,
)

from .mappings import atom_names_to_index, residue_names_to_aatype

logger = logging.getLogger(__name__)

ALPHABET_SIZE = 26


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
    chain_indices = [c.index for c in traj.top.chains if c.chain_id in chain_id]
    selection = " or ".join(f"chainid {idx}" for idx in chain_indices)
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
  atom37_indices = atom_names_to_index(atom_names)
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


def _mdtraj_to_atom_array(
  traj: md.Trajectory,
) -> AtomArray | AtomArrayStack:
  """Convert an mdtraj trajectory to a biotite AtomArray or AtomArrayStack."""
  # Create AtomArray
  if traj.n_frames > 1:
    atom_array = AtomArrayStack(traj.n_frames, traj.n_atoms)
    atom_array.coord = traj.xyz * 10  # Convert nm to Angstrom
  else:
    atom_array = AtomArray(traj.n_atoms)
    atom_array.coord = traj.xyz[0] * 10  # Convert nm to Angstrom

  # Topology
  top = traj.top

  # We need to map mdtraj topology to biotite arrays
  # This can be slow for large systems, but necessary for standardization.

  # Residue IDs
  res_ids = np.array([a.residue.resSeq for a in top.atoms], dtype=int)
  atom_array.res_id = res_ids

  # Residue Names
  res_names = np.array([a.residue.name for a in top.atoms], dtype="U3")
  atom_array.res_name = res_names

  # Atom Names
  atom_names = np.array([a.name for a in top.atoms], dtype="U6")
  atom_array.atom_name = atom_names

  # Chain IDs
  # MDTraj chain indices are 0-based. Biotite expects strings usually, or we can use A, B, C...
  # Let's map 0->A, 1->B etc.
  chain_indices = np.array([a.residue.chain.index for a in top.atoms], dtype=int)

  # Handle > 26 chains?
  # For now simple mapping.
  def chain_idx_to_id(idx: int) -> str:
    if idx < ALPHABET_SIZE:
      return chr(ord("A") + idx)
    return str(idx)  # Fallback

  chain_ids = np.array([chain_idx_to_id(i) for i in chain_indices], dtype="U3")
  atom_array.chain_id = chain_ids

  # Elements
  elements = np.array([a.element.symbol for a in top.atoms], dtype="U2")
  atom_array.element = elements

  return atom_array


def _add_hydrogens_if_needed(atom_array: AtomArray) -> AtomArray:
  """Add hydrogens to AtomArray if missing."""
  has_hydrogens = (atom_array.element == "H").any()
  if not has_hydrogens:
    logger.info("Adding hydrogens to MDTraj AtomArray")
    # Infer bonds for hydride
    if not atom_array.bonds:
      try:
        atom_array.bonds = structure.connect_via_residue_names(atom_array)
      except Exception as e:  # noqa: BLE001
        logger.warning("Failed to infer bonds: %s", e)
        atom_array.bonds = structure.connect_via_distances(atom_array)

    # Add charge annotation
    if "charge" not in atom_array.get_annotation_categories():
      atom_array.set_annotation("charge", np.zeros(atom_array.array_length(), dtype=int))

    try:
      import hydride  # noqa: PLC0415

      atom_array, _ = hydride.add_hydrogen(atom_array)
      logger.info("Hydrogens added to MDTraj structure")
    except Exception as e:  # noqa: BLE001
      logger.warning("Failed to add hydrogens: %s", e)
  return atom_array


def _process_mdtraj_chunk(
  traj_chunk: md.Trajectory,
  chain_id: Sequence[str] | str | None,
) -> ProcessedStructure:
  """Process a single MDTraj chunk."""
  logger.debug("Processing MDTraj chunk with %d frames.", traj_chunk.n_frames)

  # Apply chain selection if needed
  if chain_id is not None:
    traj_chunk = _select_chain_mdtraj(traj_chunk, chain_id=chain_id)

  # Convert to AtomArray
  atom_array = _mdtraj_to_atom_array(traj_chunk)

  # Apply solvent removal if needed
  solvent_mask = filter_solvent(atom_array)
  if np.any(solvent_mask):
    n_solvent = np.sum(solvent_mask)
    logger.info("Removing %d solvent atoms from MDTraj chunk", n_solvent)
    if isinstance(atom_array, AtomArrayStack):
      atom_array = atom_array[:, ~solvent_mask]
    else:
      atom_array = atom_array[~solvent_mask]

  # Add hydrogens if missing
  if isinstance(atom_array, AtomArray):  # Only for single frames
    atom_array = _add_hydrogens_if_needed(atom_array)

  return ProcessedStructure(
    atom_array=atom_array,
    r_indices=atom_array.res_id,
    chain_ids=np.zeros(
      atom_array.array_length(), dtype=np.int32,
    ),  # Placeholder, will be recomputed
  )


def parse_mdtraj_to_processed_structure(
  source: str | StringIO | pathlib.Path,
  chain_id: Sequence[str] | str | None,
  *,
  extract_dihedrals: bool = False,  # noqa: ARG001
  topology: str | pathlib.Path | None = None,
) -> Iterator[ProcessedStructure]:
  """Parse HDF5 structure files directly using mdtraj."""
  logger.info("Starting MDTraj HDF5 parsing for source: %s", source)
  try:
    if not topology:
      first_frame = md.load_frame(str(source), 0)
    else:
      first_frame = md.load_frame(str(source), 0, top=str(topology))
    logger.debug("Loaded first frame to determine topology.")

    # We don't need to extract static features here anymore,
    # as ProcessedStructure will be processed downstream.
    # But we DO need to handle chain selection here to reduce data size.

    # Note: _select_chain_mdtraj returns a new trajectory with sliced topology.
    _ = _select_chain_mdtraj(first_frame, chain_id=chain_id)

    # Re-derive selection indices
    if chain_id is not None and isinstance(chain_id, str):
      chain_id = [chain_id]

    traj_iterator = md.iterload(str(source))
    frame_count = 0

    for traj_chunk in traj_iterator:
      processed_chunk = _process_mdtraj_chunk(traj_chunk, chain_id)
      if isinstance(processed_chunk.atom_array, AtomArrayStack):
        frame_count += processed_chunk.atom_array.stack_depth()
      else:
        frame_count += 1
      yield processed_chunk

    logger.info("Finished MDTraj HDF5 parsing. Yielded %d frames.", frame_count)

  except Exception as e:
    msg = f"Failed to parse HDF5 structure from source: {source}. {type(e).__name__}: {e}"
    logger.exception(msg)
    raise RuntimeError(msg) from e
