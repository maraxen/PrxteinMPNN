"""Parsing utilities for MDTraj."""

import logging
import pathlib
from collections.abc import Sequence
from io import StringIO

import mdtraj as md
import numpy as np

from prxteinmpnn.utils.data_structures import ProteinStream, ProteinTuple, TrajectoryStaticFeatures
from prxteinmpnn.utils.residue_constants import (
  atom_order,
)

from .coordinates import process_coordinates
from .mappings import atom_names_to_index, residue_names_to_aatype

logger = logging.getLogger(__name__)


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


def _parse_mdtraj_hdf5(
  source: str | StringIO | pathlib.Path,
  chain_id: Sequence[str] | str | None,
  *,
  extract_dihedrals: bool = False,
  topology: str | pathlib.Path | None = None,
) -> ProteinStream:
  """Parse HDF5 structure files directly using mdtraj."""
  logger.info("Starting MDTraj HDF5 parsing for source: %s", source)
  try:
    dihedrals = None
    if not topology:
      first_frame = md.load_frame(str(source), 0)
    else:
      first_frame = md.load_frame(str(source), 0, top=str(topology))
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
