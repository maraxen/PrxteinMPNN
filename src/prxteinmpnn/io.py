"""Utilities for processing structure and trajectory files."""

import pathlib
from collections.abc import Iterator, Mapping, Sequence
from io import StringIO
from typing import cast

import jax.numpy as jnp
import numpy as np
from biotite import structure
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure import io as structure_io
from biotite.structure.io.pdb import PDBFile
from jax import vmap

from prxteinmpnn.utils.aa_convert import af_to_mpnn
from prxteinmpnn.utils.coordinates import compute_cb_precise
from prxteinmpnn.utils.data_structures import ModelInputs, ProteinStructure
from prxteinmpnn.utils.residue_constants import atom_order, resname_to_idx, unk_restype_index
from prxteinmpnn.utils.types import AtomChainIndex, ChainIndex, InputBias, ProteinSequence


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
) -> ProteinSequence:
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
  return jnp.asarray(aa_indices, dtype=jnp.int8)


def atom_names_to_index(
  atom_names: np.ndarray,
  atom_map: dict | None = None,
) -> ProteinSequence:
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
) -> AtomChainIndex:
  """Get the chain index from the AtomArray."""
  if atom_array.chain_id is None:
    return jnp.zeros(atom_array.array_length(), dtype=jnp.int32)

  if atom_array.chain_id.dtype != jnp.int32:
    return jnp.asarray(
      np.char.encode(atom_array.chain_id.astype("U1")).view(np.uint8) - ord("A"),
      dtype=jnp.int32,
    )

  return jnp.asarray(atom_array.chain_id, dtype=jnp.int32)


def _process_chain_id(
  atom_array: AtomArray,
  chain_id: Sequence[str] | str | None = None,
) -> tuple[AtomArray, AtomChainIndex]:
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
  coords_37: jnp.ndarray,
  residue_names: np.ndarray,
  atom_map: dict[str, int] | None = None,
) -> jnp.ndarray:
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
  is_glycine = jnp.array([name == "GLY" for name in residue_names])

  n_coords = coords_37[:, atom_map["N"], :]
  ca_coords = coords_37[:, atom_map["CA"], :]
  c_coords = coords_37[:, atom_map["C"], :]

  precise_cbs = vmap(compute_cb_precise)(n_coords, ca_coords, c_coords)

  original_cbs = coords_37[:, atom_map["CB"], :]

  updated_cbs = jnp.where(is_glycine[:, None], precise_cbs, original_cbs)

  return coords_37.at[:, atom_map["CB"], :].set(updated_cbs)


def process_atom_array(
  atom_array: AtomArray,
  atom_map: dict[str, int] | None = None,
  chain_id: Sequence[str] | str | None = None,
) -> ProteinStructure:
  """Process an AtomArray to create a ProteinStructure."""
  if atom_map is None:
    atom_map = atom_order
  atom_array, chain_index = _process_chain_id(atom_array, chain_id)
  _check_atom_array_length(atom_array)
  num_residues = structure.get_residue_count(atom_array)
  residue_indices, residue_names = structure.get_residues(atom_array)
  residue_indices = jnp.asarray(residue_indices, dtype=jnp.int32)
  chain_index: ChainIndex = chain_index[structure.get_residue_starts(atom_array)]
  residue_inv_indices = structure.get_residue_positions(
    atom_array,
    jnp.arange(atom_array.array_length()),
  )

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
  nitrogen_mask = atom_mask_37[:, atom_map["N"]] == 1
  coords_37 = coords_37[nitrogen_mask]
  aatype = aatype[nitrogen_mask]
  atom_mask_37 = atom_mask_37[nitrogen_mask]
  residue_indices = residue_indices[nitrogen_mask]
  chain_index = chain_index[nitrogen_mask]
  bfactors_37 = bfactors_37[nitrogen_mask]

  return ProteinStructure(
    coordinates=coords_37,
    aatype=aatype,
    atom_mask=atom_mask_37,
    residue_index=residue_indices,
    chain_index=chain_index,
    b_factors=bfactors_37,
  )


def from_structure_file(
  file_path: str,
  model: int = 1,
  chain_id: str | Sequence[str] | None = None,
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

  if not isinstance(atom_array, AtomArray):
    msg = f"Unexpected transformation to {type(atom_array)}."
    raise TypeError(msg)
  return process_atom_array(atom_array, chain_id=chain_id)


def from_trajectory(
  trajectory_file: str,
  topology_file: str | None = None,
  chain_id: str | Sequence[str] | None = None,
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
    msg = f"Unexpected transformation to {type(atom_stack)}."
    raise TypeError(msg)

  return (process_atom_array(frame, chain_id=chain_id) for frame in atom_stack)


def from_string(
  pdb_string: str,
  model: int = 1,
  chain_id: str | Sequence[str] | None = None,
) -> ProteinStructure:
  """Construct a ProteinStructure from a PDB string.

  Args:
    pdb_string: The PDB formatted string.
    model: The model number to load from the structure string. Defaults to 1.
    chain_id: If specified, only this chain is parsed. If None, the entire
      structure is parsed.

  Returns:
    A new `ProteinStructure` parsed from the PDB string.

  """
  if not pdb_string.strip():
    msg = "AtomArray is empty."
    raise ValueError(msg)

  pdb_file = PDBFile.read(StringIO(pdb_string))
  atom_array = pdb_file.get_structure(
    model=model,
    extra_fields=["b_factor"],
  )

  if isinstance(atom_array, AtomArrayStack) and atom_array.stack_depth() > 0:
    atom_array = atom_array[0]
  elif isinstance(atom_array, AtomArrayStack) and atom_array.stack_depth() == 0:
    msg = "No models found in the provided PDB string."
    raise ValueError(msg)
  if not isinstance(atom_array, AtomArray):
    msg = f"Unexpected transformation to {type(atom_array)}."
    raise TypeError(msg)

  return process_atom_array(atom_array, chain_id=chain_id)


def protein_structure_to_model_inputs(
  protein_structure: ProteinStructure,
  bias: InputBias | None = None,
) -> ModelInputs:
  """Convert a ProteinStructure to model inputs.

  Args:
    protein_structure: A ProteinStructure object containing the structure data.
    bias: An optional InputBias jnp.ndarray with shape (num_residues, 20) containing
    bias information. This will shift output probabilities for each residue. Default
    to zero.

  Returns:
    A dictionary containing the model inputs derived from the ProteinStructure.

  """
  mask = protein_structure.atom_mask[:, 1]
  return ModelInputs(
    structure_coordinates=protein_structure.coordinates,
    sequence=protein_structure.aatype,
    mask=mask,
    residue_index=protein_structure.residue_index,
    chain_index=protein_structure.chain_index,
    lengths=jnp.array([len(protein_structure.aatype)], dtype=jnp.int32),
    bias=jnp.zeros((len(protein_structure.aatype), 20), dtype=jnp.float32)
    if bias is None
    else bias,
  )
