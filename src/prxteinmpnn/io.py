"""Utilities for processing PDB files."""

import pathlib
from collections.abc import Iterator, Mapping

import jax.numpy as jnp
import numpy as np
from biotite import structure
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure import io as structure_io

from prxteinmpnn.utils.data_structures import ProteinStructure
from prxteinmpnn.utils.residue_constants import atom_order, resname_to_idx, unk_restype_index
from prxteinmpnn.utils.types import (
  Sequence,
)


def _check_if_file_empty(file_path: str) -> bool:
  """Check if the file is empty."""
  path = pathlib.Path(file_path)
  try:
    with path.open() as f:
      return f.readable() and f.read().strip() == ""
  except FileNotFoundError:
    return True


def string_key_to_index(
  string_keys: np.ndarray,
  key_map: Mapping[str, int],
  unk_index: int | None = None,
) -> jnp.ndarray:
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

  return jnp.where(is_known, sorted_values[indices], unk_index)


def residue_names_to_aatype(
  residue_names: np.ndarray,
  aa_map: dict | None = None,
) -> Sequence:
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
  return jnp.asarray(aa_indices, dtype=jnp.int8)


def atom_names_to_index(
  atom_names: np.ndarray,
  atom_map: dict | None = None,
) -> Sequence:
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
  return jnp.asarray(atom_indices, dtype=jnp.int8)


def process_atom_array(
  atom_array: AtomArray,
  chain_id: str | None = None,
) -> ProteinStructure:
  """Process an AtomArray to create a ProteinStructure."""
  if atom_array.array_length() == 0:
    msg = (
      f"No atoms found in the structure for chain '{chain_id}'."
      if chain_id
      else "No atoms found in the structure."
    )
    raise ValueError(msg)

  num_residues = structure.get_residue_count(atom_array)
  residue_indices, residue_names = structure.get_residues(atom_array)
  residue_indices = jnp.asarray(residue_indices, dtype=jnp.int32)
  residue_inv_indices = structure.get_residue_positions(
    atom_array,
    jnp.arange(atom_array.array_length()),
  )

  if atom_array.chain_id is None:
    msg = "Chain ID is not available in the structure."
    raise ValueError(msg)

  atom_names = atom_array.atom_name

  if atom_names is None:
    msg = "Atom names are not available in the structure."
    raise ValueError(msg)

  atom37_indices = atom_names_to_index(np.array(atom_names, dtype="U5"))

  atom_mask = atom37_indices != -1

  coords_37 = jnp.zeros((num_residues, 37, 3), dtype=jnp.float32)
  atom_mask_37 = jnp.zeros((num_residues, 37), dtype=jnp.float32)
  bfactors_37 = jnp.zeros((num_residues, 37), dtype=jnp.float32)

  res_indices_flat = jnp.asarray(residue_inv_indices)[atom_mask]
  atom_indices_flat = atom37_indices[atom_mask]

  coords_37 = coords_37.at[res_indices_flat, atom_indices_flat].set(
    jnp.asarray(atom_array.coord)[atom_mask],
  )
  atom_mask_37 = atom_mask_37.at[res_indices_flat, atom_indices_flat].set(1.0)
  bfactors_37 = bfactors_37.at[res_indices_flat, atom_indices_flat].set(
    jnp.asarray(atom_array.b_factor)[atom_mask],
  )

  aatype = residue_names_to_aatype(residue_names)

  return ProteinStructure(
    coordinates=coords_37,
    aatype=aatype,
    atom_mask=atom_mask_37,
    residue_index=residue_indices,
    b_factors=bfactors_37,
  )


def from_structure_file(
  file_path: str,
  model: int = 1,
  chain_id: str | None = None,
) -> ProteinStructure:
  """Construct a Protein object from a structure file (PDB, PDBx/mmCIF).

  This implementation uses biotite for robust parsing and JAX for efficient
  vectorized processing to create a dense, fixed-size representation for
  each residue (37 atoms).

  WARNING: All non-standard residue types will be converted into UNK. All
    atoms not in the canonical 37-atom set will be ignored.

  Args:
    file_path: The path to the structure file.
    model: The model number to load from the structure file. Defaults to 1.
    chain_id: If specified, only this chain is parsed. If None, the entire
      structure is parsed.

  Returns:
    A new `ProteinStructure` parsed from the file contents.

  """
  if _check_if_file_empty(file_path):
    msg = (
      f"The file '{file_path}' is empty or does not exist. Please provide a valid structure file."
    )
    raise FileNotFoundError(msg)
  atom_array = structure_io.load_structure(file_path, model=model, extra_fields=["b_factor"])

  if chain_id is not None and not isinstance(chain_id, str):
    msg = f"Expected chain_id to be a string, but got {type(chain_id)}."
    raise TypeError(msg)
  if atom_array.chain_id is None:
    msg = "Chain ID is not available in the structure."
    raise ValueError(msg)
  if not isinstance(atom_array, AtomArray):
    msg = f"Expected a single structure, but got {type(atom_array)}."
    raise TypeError(msg)
  atom_array = atom_array[atom_array.chain_id == chain_id] if chain_id else atom_array
  if not isinstance(atom_array, AtomArray):
    msg = f"Unexpected transformation to {type(atom_array)}."
    raise TypeError(msg)
  return process_atom_array(atom_array, chain_id=chain_id)


def from_trajectory(
  trajectory_file: str,
  topology_file: str | None = None,
  chain_id: str | None = None,
) -> Iterator["ProteinStructure"]:
  """Construct ProteinStructure objects from a trajectory file.

  This function reads a trajectory and yields a ProteinStructure for each frame.

  Args:
    trajectory_file: Path to the trajectory file (e.g., DCD, XTC, multi-model PDB).
    topology_file: Path to the topology file (e.g., PDB, PSF), required for
                    coordinate-only trajectory formats.
    chain_id: If specified, only atoms from this chain will be included.

  Returns:
    An iterator that yields a ProteinStructure for each frame in the trajectory.

  """
  if _check_if_file_empty(trajectory_file):
    msg = (
      f"The file '{trajectory_file}' is empty or does not exist. "
      "Please provide a valid trajectory file."
    )
    raise FileNotFoundError(msg)

  atom_stack = structure_io.load_structure(
    trajectory_file,
    template=topology_file,
    extra_fields=["b_factor"],
  )

  if not isinstance(atom_stack, AtomArrayStack):
    msg = (
      f"Expected a trajectory (AtomArrayStack), but loaded a single "
      f"frame ({type(atom_stack)}). Use a different function for single structures."
    )
    raise TypeError(msg)

  if atom_stack.stack_depth() == 0:
    msg = (
      "Trajectory file is empty or could not be read. "
      "Ensure the file exists and is in a supported format."
    )
    raise ValueError(msg)

  if chain_id:
    mask = atom_stack[0].chain_id == chain_id
    atom_stack = atom_stack[:, mask]

  if not isinstance(atom_stack, AtomArrayStack):
    msg = f"Unexpected transformation to {type(atom_stack)}."
    raise TypeError(msg)

  return (process_atom_array(frame) for frame in atom_stack)
