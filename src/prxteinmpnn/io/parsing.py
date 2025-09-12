"""Utilities for processing structure and trajectory files.

prxteinmpnn.io.parsing

This uses numpy to allow for multiprocessing and avoid conflicts with Jax.
"""

import io
import pathlib
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from io import StringIO
from typing import Any, cast

import numpy as np
from biotite import structure
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure import io as structure_io
from jax import vmap

from prxteinmpnn.utils.data_structures import ProteinStream, ProteinTuple
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
  try:
    with path.open() as f:
      return f.readable() and f.read().strip() == ""
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
  """Compute the position of a fourth atom (D) given three atoms (A, B, C) and internal coordinates.

  Given coordinates for atoms A, B, and C, and the desired bond length, bond angle, and dihedral
  angle, compute the coordinates of atom D such that:
    - |C-D| = bond_length
    - angle(B, C, D) = bond_angle
    - dihedral(A, B, C, D) = dihedral_angle

  Args:
    atom_a: Coordinates of atom A, shape (3,).
    atom_b: Coordinates of atom B, shape (3,).
    atom_c: Coordinates of atom C, shape (3,).
    bond_length: Desired bond length between C and D.
    bond_angle: Desired bond angle (in radians) at atom C.
    dihedral_angle: Desired dihedral angle (in radians) for atoms A-B-C-D.

  Returns:
    Coordinates of atom D, shape (3,).

  Example:
    >>> d = extend_coordinate(a, b, c, 1.5, 2.0, 3.14)
    >>> d.shape
    (3,)

  """

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
  """Compute the C-beta atom position from backbone N, CA, and C coordinates.

  Does so precisely using trigonometric relationships based on the backbone geometry.

  Specifically, the position of the C-beta atom is determined by:

  - The bond length between the alpha carbon and the C-beta atom.
  - The bond angle between the nitrogen, alpha carbon, and C-beta atoms.
  - The dihedral angle involving the nitrogen, alpha carbon, and C-beta atoms.


  Unlike the compute_c_beta function, this function does not use a linear combination of bond
  vectors with approximate fixed coefficients. This is more accurate and flexible for different
  configurations of the protein backbone, but more computationally intensive.

  It is used in preparation of the atomic coordinates for the model input.
  It is not used in the model itself, but rather in the preprocessing of the input data
  to ensure that the C-beta atom is correctly placed based on the backbone structure.

  Uses standard geometry for C-beta placement:
    - N-CA-CB bond length: 1.522 Å
    - N-CA-CB bond angle: 1.927 radians
    - C-N-CA-CB dihedral angle: -2.143 radians

  Args:
    n_coord: Coordinates of the N atom, shape (3,).
    ca_coord: Coordinates of the CA atom, shape (3,).
    c_coord: Coordinates of the C atom, shape (3,).

  Returns:
    Coordinates of the C-beta atom, shape (3,).

  Example:
    >>> cb = compute_cb_precise(n, ca, c)
    >>> cb.shape
    (3,)

  """
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
  """Convert string keys to integer indices based on a mapping.

  Efficient vectorized implementation to convert a 1D array of string keys
  to a 1D array of integer indices using a provided mapping. If a key is not found in the mapping,
  it is replaced with a specified unknown index.

  Args:
    string_keys: A 1D array of string keys.
    key_map: A dictionary mapping string keys to integer indices.
    unk_index: The index to use for unknown keys not found in the mapping. If None, uses the
      length of the key_map as the unknown index.

  Returns:
    A 1D array of integer indices corresponding to the string keys.

  """
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
  """Convert a string sequence to a ProteinSequence.

  Args:
    sequence: A string containing the protein sequence.
    aa_map: A dictionary mapping amino acid names to integer indices. If None, uses the default
      `restype_order` mapping.
    unk_index: The index to use for unknown amino acids not found in the mapping. If None, uses
      `unk_restype_index`.

  Returns:
    A ProteinSequence containing the amino acid type indices corresponding to the input string.

  """
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
  """Convert a ProteinSequence to a string.

  Args:
    sequence: A ProteinSequence containing amino acid type indices.
    aa_map: A dictionary mapping amino acid type indices to their corresponding names. If None,
      uses the default `restype_order` mapping.

  Returns:
    A string representation of the protein sequence.

  """
  if aa_map is None:
    aa_map = {i: aa for aa, i in restype_order_with_x.items()}

  af_seq = mpnn_to_af(sequence)

  return "".join([aa_map.get(int(aa), "X") for aa in af_seq])


def residue_names_to_aatype(
  residue_names: np.ndarray,
  aa_map: dict | None = None,
) -> np.ndarray:
  """Convert 3-letter residue names to amino acid type indices.

  Args:
    residue_names: A 1D array of residue names (strings).
    aa_map: A dictionary mapping residue names to integer indices. If None, uses the default
      `resname_to_idx` mapping.

  Returns:
    A 1D array of amino acid type indices corresponding to the residue names.

  """
  if aa_map is None:
    aa_map = resname_to_idx

  aa_indices = string_key_to_index(residue_names, aa_map, unk_restype_index)
  aa_indices = af_to_mpnn(aa_indices)
  return np.asarray(aa_indices, dtype=np.int8)


def atom_names_to_index(
  atom_names: np.ndarray,
  atom_map: dict | None = None,
) -> np.ndarray:
  """Convert atom names to atom type indices.

  Args:
    atom_names: A 1D array of atom names (strings).
    atom_map: A dictionary mapping atom names to integer indices. If None, uses the default
      `atomname_to_idx` mapping.

  Returns:
    A 1D array of atom type indices corresponding to the atom names.

  """
  if atom_map is None:
    atom_map = atom_order

  atom_indices = string_key_to_index(atom_names, atom_map, -1)
  return np.asarray(atom_indices)


def _check_atom_array_length(atom_array: AtomArray) -> None:
  """Check if the AtomArray has a valid length.

  Args:
    atom_array: The AtomArray to check.

  Raises:
    ValueError: If the AtomArray is empty.

  """
  if atom_array.array_length() == 0:
    msg = "AtomArray is empty."
    raise ValueError(msg)


def _get_chain_index(
  atom_array: AtomArray,
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
  atom_array: AtomArray,
  chain_id: Sequence[str] | str | None = None,
) -> tuple[AtomArray, np.ndarray]:
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

  indices_to_include = np.isin(atom_array.chain_id, chain_id)
  atom_array = cast("AtomArray", atom_array[indices_to_include])
  chain_index = _get_chain_index(atom_array)
  return atom_array, chain_index


def _fill_in_cb_coordinates(
  coords_37: np.ndarray,
  residue_names: np.ndarray,
  atom_map: dict[str, int] | None = None,
) -> np.ndarray:
  """Fill in the CB coordinates for residues that have them.

  Args:
    coords_37: A 2D array of shape (N, 37, 3) containing the coordinates of the atoms.
    residue_names: A 1D array of residue names corresponding to the coordinates.
    atom_map: A dictionary mapping residue names to their atom indices. If None, uses the default
      `atom_order` mapping.

  Returns:
    A 2D array of shape (N, 37, 3) with the C-beta coordinates filled in for residues that have
      them.
    For glycine residues, the C-beta coordinates are computed precisely based on the N, CA, and C
      atoms.
    For other residues, the original C-beta coordinates are retained if they exist.

    NOTE: This is not part of the pipeline, as despite this happening in the original code, it is
      bypassed during feature extraction.

  """
  if atom_map is None:
    atom_map = atom_order
  is_glycine = np.array([name == "GLY" for name in residue_names])

  n_coords = coords_37[:, atom_map["N"], :]
  ca_coords = coords_37[:, atom_map["CA"], :]
  c_coords = coords_37[:, atom_map["C"], :]

  precise_cbs = vmap(compute_cb_precise)(n_coords, ca_coords, c_coords)

  original_cbs = coords_37[:, atom_map["CB"], :]

  updated_cbs = np.where(is_glycine[:, None], precise_cbs, original_cbs)
  coords_37[:, atom_map["CB"], :] = updated_cbs
  return coords_37


def process_atom_array(
  atom_array: AtomArray,
  atom_map: dict[str, int] | None = None,
  chain_id: Sequence[str] | str | None = None,
) -> ProteinTuple:
  """Process an AtomArray to create a Protein inputs."""
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

  coords_37 = np.zeros((num_residues, 37, 3), dtype=np.float32)
  atom_mask_37 = np.zeros((num_residues, 37), dtype=np.bool)

  res_indices_flat = np.asarray(residue_inv_indices)[atom_mask]
  atom_indices_flat = atom37_indices[atom_mask]

  coords_37[res_indices_flat, atom_indices_flat] = np.asarray(atom_array.coord)[atom_mask]
  atom_mask_37[res_indices_flat, atom_indices_flat] = 1

  aatype = residue_names_to_aatype(residue_names)
  nitrogen_mask = atom_mask_37[:, atom_map["N"]] == 1
  coords_37 = coords_37[nitrogen_mask]
  aatype = aatype[nitrogen_mask]
  atom_mask_37 = atom_mask_37[nitrogen_mask]
  residue_indices = residue_indices[nitrogen_mask]
  chain_index = chain_index[nitrogen_mask]
  phi, psi, omega = structure.dihedral_backbone(atom_array)
  dihedrals = np.stack([phi, psi, omega], axis=-1) if phi is not None else None
  return ProteinTuple(
    coordinates=coords_37,
    aatype=aatype,
    atom_mask=atom_mask_37,
    residue_index=residue_indices,
    chain_index=chain_index,
    dihedrals=dihedrals,
  )


async def parse_input(
  source: str | StringIO | pathlib.Path,
  *,
  model: int | None = None,
  altloc: str | None = None,
  chain_id: Sequence[str] | str | None = None,
  **kwargs: Any,  # noqa: ANN401
) -> ProteinStream:
  """Parse a structure file or string into a list of Protein objects.

  This is a synchronous, CPU-bound function intended to be run in an executor
  to avoid blocking the main event loop.

  Args:
      source: A file path (str) or a file-like object (StringIO) containing
              the structure data.
      model: The model number to load. If None, all models are loaded.
      altloc: The alternate location identifier to use.
      chain_id: Specific chain(s) to parse from the structure.
      **kwargs: Additional keyword arguments to pass to the structure loader.

  Returns:
      A ProteinEnsemble containing one or more parsed ProteinStructure objects.

  """
  if isinstance(source, io.StringIO):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pdb") as tmp:
      tmp.write(source.read())
      source = pathlib.Path(tmp.name)

  try:
    atom_array_or_stack = structure_io.load_structure(
      source,
      model=model,
      altloc=altloc,
      **kwargs,
    )

    if isinstance(atom_array_or_stack, AtomArrayStack):
      for frame in atom_array_or_stack:
        yield (process_atom_array(frame, chain_id=chain_id), str(source))
    elif isinstance(atom_array_or_stack, AtomArray):
      yield (process_atom_array(atom_array_or_stack, chain_id=chain_id), str(source))

  except Exception as e:
    msg = f"Failed to parse structure from source: {e}"
    warnings.warn(msg, stacklevel=2)
    raise RuntimeError(msg) from e
