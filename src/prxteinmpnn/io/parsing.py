"""Utilities for processing structure and trajectory files.

prxteinmpnn.io.parsing

This module has been refactored to contain only synchronous parsing logic,
making it suitable for use in parallel worker processes managed by Grain.
All async operations and direct I/O handling have been moved to the
`sources.py` and `operations.py` modules.
"""

import pathlib
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from io import StringIO
from typing import Any, cast

import mdtraj as md
import numpy as np
from biotite import structure
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure import io as structure_io

from prxteinmpnn.utils.data_structures import ProteinStream, ProteinTuple, TrajectoryStaticFeatures
from prxteinmpnn.utils.residue_constants import (
  atom_order,
  resname_to_idx,
  restype_order,
  restype_order_with_x,
  unk_restype_index,
)

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
  return _AF_TO_MPNN_PERM[sequence]


def mpnn_to_af(sequence: np.ndarray) -> np.ndarray:
  """Convert a sequence of integer indices from ProteinMPNN's to AlphaFold's alphabet order."""
  return _MPNN_TO_AF_PERM[sequence]


def _check_if_file_empty(file_path: str) -> bool:
  """Check if the file is empty."""
  path = pathlib.Path(file_path)
  suffix = path.suffix.lower()
  try:
    with path.open() as f:
      if suffix not in {".h5", ".hdf5"}:
        return f.readable() and f.read().strip() == ""
      return not f.readable()
  except FileNotFoundError:
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
  if unk_index is None:
    unk_index = len(key_map)

  sorted_keys = np.array(sorted(key_map.keys()))
  sorted_values = np.array([key_map[k] for k in sorted_keys])
  indices = np.searchsorted(sorted_keys, string_keys)
  indices = np.clip(indices, 0, len(sorted_keys) - 1)

  found_keys = sorted_keys[indices]
  is_known = found_keys == string_keys

  return np.where(is_known, sorted_values[indices], unk_index)


def string_to_protein_sequence(
  sequence: str,
  aa_map: dict | None = None,
  unk_index: int | None = None,
) -> np.ndarray:
  """Convert a string sequence to a ProteinSequence."""
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
  if aa_map is None:
    aa_map = {i: aa for aa, i in restype_order_with_x.items()}

  af_seq = mpnn_to_af(sequence)

  return "".join([aa_map.get(int(aa), "X") for aa in af_seq])


def residue_names_to_aatype(
  residue_names: np.ndarray,
  aa_map: dict | None = None,
) -> np.ndarray:
  """Convert 3-letter residue names to amino acid type indices."""
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
  if atom_map is None:
    atom_map = atom_order

  atom_indices = string_key_to_index(atom_names, atom_map, -1)
  return np.asarray(atom_indices)


def _check_atom_array_length(atom_array: AtomArray | AtomArrayStack) -> None:
  """Check if the AtomArray has a valid length."""
  if atom_array.array_length() == 0:
    msg = "AtomArray is empty."
    raise ValueError(msg)


def _get_chain_index(
  atom_array: AtomArray | AtomArrayStack,
) -> np.ndarray:
  """Get the chain index from the AtomArray."""
  if atom_array.chain_id is None:
    return np.zeros(atom_array.array_length(), dtype=np.int32)

  if atom_array.chain_id.dtype != np.int32:
    return np.asarray(
      np.char.encode(atom_array.chain_id.astype("U1")).view(np.uint8) - ord("A"),
      dtype=np.int32,
    )

  return np.asarray(atom_array.chain_id, dtype=np.int32)


def _process_chain_id(
  atom_array: AtomArray | AtomArrayStack,
  chain_id: Sequence[str] | str | None = None,
) -> tuple[AtomArray | AtomArrayStack, np.ndarray]:
  """Process the chain_id of the AtomArray."""
  if chain_id is None:
    chain_index = _get_chain_index(atom_array)
    return atom_array, chain_index

  if isinstance(chain_id, str):
    chain_id = [chain_id]

  if not isinstance(chain_id, Sequence):
    msg = f"Expected chain_id to be a string or a sequence of strings, but got {type(chain_id)}."
    raise TypeError(msg)

  if atom_array.chain_id is None:
    msg = "Chain ID is not available in the structure."
    raise ValueError(msg)
  chain_mask = np.isin(atom_array.chain_id, chain_id)
  if isinstance(atom_array, AtomArrayStack):
    atom_array = cast("AtomArray | AtomArrayStack", atom_array[:, chain_mask])
  else:
    atom_array = cast("AtomArray | AtomArrayStack", atom_array[chain_mask])
  chain_index = _get_chain_index(atom_array)
  return (
    atom_array,
    chain_index,
  )


def _extract_biotite_static_features(
  atom_array: AtomArray | AtomArrayStack,
  atom_map: dict[str, int] | None = None,
  chain_id: Sequence[str] | str | None = None,
) -> tuple[TrajectoryStaticFeatures, AtomArray | AtomArrayStack]:
  if atom_map is None:
    atom_map = atom_order

  atom_array, chain_index = _process_chain_id(atom_array, chain_id)
  _check_atom_array_length(atom_array)
  num_residues = structure.get_residue_count(atom_array)
  residue_indices, residue_names = structure.get_residues(atom_array)
  residue_indices = np.asarray(residue_indices, dtype=np.int32)
  chain_index = chain_index[structure.get_residue_starts(atom_array)]
  residue_inv_indices = structure.get_residue_positions(
    atom_array,
    np.arange(atom_array.array_length()),
  )

  atom_names = atom_array.atom_name

  if atom_names is None:
    msg = "Atom names are not available in the structure."
    raise ValueError(msg)

  atom37_indices = atom_names_to_index(np.array(atom_names, dtype="U5"))

  atom_mask = atom37_indices != -1

  atom_mask_37 = np.zeros((num_residues, 37), dtype=bool)

  res_indices_flat = np.asarray(residue_inv_indices)[atom_mask]
  atom_indices_flat = atom37_indices[atom_mask]

  atom_mask_37[res_indices_flat, atom_indices_flat] = 1

  aatype = residue_names_to_aatype(residue_names)
  nitrogen_mask = atom_mask_37[:, atom_map["N"]] == 1
  aatype = aatype[nitrogen_mask]
  atom_mask_37 = atom_mask_37[nitrogen_mask]
  residue_indices = residue_indices[nitrogen_mask]
  chain_index = chain_index[nitrogen_mask]

  return TrajectoryStaticFeatures(
    aatype=residue_names_to_aatype(residue_names),
    static_atom_mask_37=atom_mask_37,
    residue_indices=residue_indices,
    chain_index=chain_index,
    res_indices_flat=res_indices_flat,
    atom_indices_flat=atom_indices_flat,
    valid_atom_mask=atom_mask,
    nitrogen_mask=nitrogen_mask,
    num_residues=num_residues,
  ), atom_array


def atom_array_dihedrals(
  atom_array: AtomArray | AtomArrayStack,
) -> np.ndarray | None:
  """Compute backbone dihedral angles (phi, psi, omega) for the given AtomArray."""
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
    return None
  dihedrals = np.stack([phi, psi, omega], axis=-1)
  return dihedrals[~np.any(np.isnan(dihedrals), axis=-1)]


def mdtraj_dihedrals(
  traj: md.Trajectory,
  num_residues: int,
  nitrogen_mask: np.ndarray,
) -> np.ndarray | None:
  """Compute backbone dihedral angles (phi, psi, omega) for the given md.Trajectory chunk."""
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

  return dihedrals[nitrogen_mask]


def process_coordinates(
  coordinates: np.ndarray,
  num_residues: int,
  res_indices_flat: np.ndarray,
  atom_indices_flat: np.ndarray,
  valid_atom_mask: np.ndarray,
) -> np.ndarray:
  """Process an AtomArray to create a Protein inputs."""
  coords_37 = np.zeros((num_residues, 37, 3), dtype=np.float32)

  coords_37[res_indices_flat, atom_indices_flat] = np.asarray(
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
    raise ValueError(msg)
  if chain_id is not None:
    if isinstance(chain_id, str):
      chain_id = [chain_id]
    selection = " or ".join(f"chainid {cid}" for cid in chain_id)
    atom_indices = traj.top.select(selection)
    if atom_indices.size == 0:
      msg = f"No atoms found for chain(s) {chain_id}."
      warnings.warn(msg, stacklevel=2)
      raise ValueError(msg)

    traj = traj.atom_slice(atom_indices)

  return traj


def _extract_mdtraj_static_features(
  traj_chunk: md.Trajectory,
  atom_map: dict[str, int] | None = None,
) -> TrajectoryStaticFeatures:
  """Extract frame-invariant (static) features from a trajectory chunk's topology."""
  if traj_chunk.top is None:
    msg = "Trajectory does not have a topology."
    raise ValueError(msg)
  if atom_map is None:
    atom_map = atom_order

  topology = traj_chunk.top
  if topology is None:
    msg = "Trajectory does not have a topology."
    raise ValueError(msg)
  num_residues = topology.n_residues
  if num_residues == 0:
    msg = "Trajectory has no residues after filtering."
    raise ValueError(msg)

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
  static_atom_mask_37 = np.zeros((num_residues, 37), dtype=bool)
  static_atom_mask_37[res_indices_flat, atom_indices_flat] = True
  nitrogen_mask = static_atom_mask_37[:, atom_map["N"]]

  if not np.any(nitrogen_mask):
    msg = "No residues with backbone nitrogen atoms found."
    warnings.warn(msg, stacklevel=2)
    raise ValueError(msg)

  return TrajectoryStaticFeatures(
    aatype=aatype[nitrogen_mask],
    static_atom_mask_37=static_atom_mask_37[nitrogen_mask],
    residue_indices=residue_indices[nitrogen_mask],
    chain_index=chain_index[nitrogen_mask],
    res_indices_flat=res_indices_flat,
    atom_indices_flat=atom_indices_flat,
    valid_atom_mask=valid_atom_mask,
    nitrogen_mask=nitrogen_mask,
    num_residues=num_residues,
  )


def _parse_hdf5(
  source: str | StringIO | pathlib.Path,
  chain_id: Sequence[str] | str | None,
  *,
  extract_dihedrals: bool = False,
) -> ProteinStream:
  """Parse HDF5 structure files directly using mdtraj."""
  try:
    dihedrals = None
    first_frame = md.load_frame(str(source), 0)
    first_frame = _select_chain_mdtraj(first_frame, chain_id=chain_id)
    static_features = _extract_mdtraj_static_features(
      first_frame,
    )
    traj_iterator = md.iterload(str(source))
    for traj_chunk in traj_iterator:
      for frame in traj_chunk:
        coords = frame.xyz
        if extract_dihedrals:
          dihedrals = mdtraj_dihedrals(
            frame,
            static_features["num_residues"],
            static_features["nitrogen_mask"],
          )

        coords_37 = process_coordinates(
          coords[0],  # MDTraj xyz is (n_frames, n_atoms, 3)
          static_features["num_residues"],
          static_features["res_indices_flat"],
          static_features["atom_indices_flat"],
          static_features["valid_atom_mask"],
        )
        yield ProteinTuple(
          coordinates=coords_37,
          aatype=static_features["aatype"],
          atom_mask=static_features["static_atom_mask_37"],
          residue_index=static_features["residue_indices"],
          chain_index=static_features["chain_index"],
          dihedrals=dihedrals,
          source=str(source),
        )
  except Exception as e:
    msg = f"Failed to parse HDF5 structure from source: {e}"
    warnings.warn(msg, stacklevel=2)
    raise RuntimeError(msg) from e


def _validate_atom_array_type(atom_array: Any) -> None:
  """Validate that the atom array is of the expected type."""
  if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
    msg = f"Expected AtomArray or AtomArrayStack, but got {type(atom_array)}."
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
  try:
    altloc = altloc if altloc is not None else "first"
    dihedrals = None
    atom_array = structure_io.load_structure(
      source,
      model=model,
      altloc=altloc,
      **kwargs,
    )
    _validate_atom_array_type(atom_array)

    if isinstance(atom_array, (AtomArray, AtomArrayStack)):
      static_features, atom_array = _extract_biotite_static_features(atom_array, chain_id=chain_id)

      if isinstance(atom_array, AtomArrayStack):
        for frame in atom_array:
          if extract_dihedrals:
            dihedrals = atom_array_dihedrals(frame)
          coords = np.asarray(frame.coord)
          coords_37 = process_coordinates(
            coords,
            static_features["num_residues"],
            static_features["res_indices_flat"],
            static_features["atom_indices_flat"],
            static_features["valid_atom_mask"],
          )
          yield ProteinTuple(
            coordinates=coords_37,
            aatype=static_features["aatype"],
            atom_mask=static_features["static_atom_mask_37"],
            residue_index=static_features["residue_indices"],
            chain_index=static_features["chain_index"],
            dihedrals=dihedrals,
            source=str(source),
          )
      elif isinstance(atom_array, AtomArray):
        if extract_dihedrals:
          dihedrals = atom_array_dihedrals(atom_array)
        coords = np.asarray(atom_array.coord)
        coords_37 = process_coordinates(
          coords,
          static_features["num_residues"],
          static_features["res_indices_flat"],
          static_features["atom_indices_flat"],
          static_features["valid_atom_mask"],
        )
        yield ProteinTuple(
          coordinates=coords_37,
          aatype=static_features["aatype"],
          atom_mask=static_features["static_atom_mask_37"],
          residue_index=static_features["residue_indices"],
          chain_index=static_features["chain_index"],
          dihedrals=dihedrals,
          source=str(source),
        )

  except Exception as e:
    msg = f"Failed to parse structure from source: {e}"
    warnings.warn(msg, stacklevel=2)
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
  if isinstance(source, StringIO):
    with tempfile.NamedTemporaryFile(
      mode="w",
      delete=False,
      suffix=".pdb",
    ) as tmp:  # TODO: suffix based on format
      tmp.write(source.read())
      source = pathlib.Path(tmp.name)

  if isinstance(source, (str, pathlib.Path)):
    path = pathlib.Path(source)
    if path.suffix.lower() in {".h5", ".hdf5"}:
      yield from _parse_hdf5(path, chain_id, extract_dihedrals=extract_dihedrals)
      return

  yield from _parse_biotite(
    source,
    model,
    altloc=altloc,
    chain_id=chain_id,
    extract_dihedrals=extract_dihedrals,
    **kwargs,
  )
