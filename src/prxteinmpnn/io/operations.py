"""Data operations for processing protein structures within a Grain pipeline."""

import warnings
from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp  # Keep for tree_map util
import numpy as np       # Using NumPy for CPU-based data loading

from prxteinmpnn.physics.features import compute_electrostatic_node_features
from prxteinmpnn.utils.data_structures import Protein, ProteinTuple

_MAX_TRIES = 5

def truncate_protein(
  protein: ProteinTuple,
  max_length: int | None,
  strategy: str = "none",
) -> ProteinTuple:
  """Truncate a protein to a maximum length."""
  if max_length is None or strategy == "none":
    return protein

  length = protein.coordinates.shape[0]
  if length <= max_length:
    return protein

  if strategy == "center_crop":
    start = (length - max_length) // 2
  elif strategy == "random_crop":
    start = np.random.default_rng().integers(0, length - max_length + 1)
  else:
    msg = f"Unknown truncation strategy: {strategy}"
    raise ValueError(msg)

  end = start + max_length

  def slice_array(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
      return None
    if hasattr(arr, "shape") and arr.shape[0] == length:
      return arr[start:end]
    return arr

  return protein._replace(
    coordinates=slice_array(protein.coordinates),
    aatype=slice_array(protein.aatype),
    atom_mask=slice_array(protein.atom_mask),
    residue_index=slice_array(protein.residue_index),
    chain_index=slice_array(protein.chain_index),
    full_coordinates=slice_array(protein.full_coordinates),
    dihedrals=slice_array(protein.dihedrals),
    mapping=slice_array(protein.mapping),
    charges=slice_array(protein.charges),
    radii=slice_array(protein.radii),
    sigmas=slice_array(protein.sigmas),
    epsilons=slice_array(protein.epsilons),
    estat_backbone_mask=slice_array(protein.estat_backbone_mask),
    estat_resid=slice_array(protein.estat_resid),
    estat_chain_index=slice_array(protein.estat_chain_index),
    physics_features=slice_array(protein.physics_features),
  )


def concatenate_proteins_for_inter_mode(elements: Sequence[ProteinTuple]) -> Protein:
  """Concatenate proteins for inter-chain mode using NumPy."""
  if not elements:
    msg = "Cannot concatenate an empty list of proteins."
    raise ValueError(msg)

  tries = 0
  while not all(isinstance(p, ProteinTuple) for p in elements):
    if any(isinstance(p, Sequence) for p in elements):
      elements = [p[0] if isinstance(p, Sequence) else p for p in elements]
      tries += 1
    if tries > _MAX_TRIES:
      raise ValueError("Too many nested sequences.")

  # Convert to Protein objects using NumPy method
  proteins = [Protein.from_tuple_numpy(p) for p in elements]

  structure_indices = []
  for i, protein in enumerate(proteins):
    length = protein.coordinates.shape[0]
    structure_indices.append(np.full(length, i, dtype=np.int32))

  structure_mapping = np.concatenate(structure_indices, axis=0)
  
  remapped_chain_ids = []
  chain_offset = 0

  for protein in proteins:
    original_chains = protein.chain_index
    remapped_chains = original_chains + chain_offset
    remapped_chain_ids.append(remapped_chains)
    chain_offset = int(np.max(remapped_chains)) + 1

  chain_ids = np.concatenate(remapped_chain_ids, axis=0)
  
  concatenated = jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis=0) if x[0] is not None else None, *proteins)
  concatenated = concatenated.replace(chain_index=chain_ids, mapping=structure_mapping)
  return jax.tree_util.tree_map(lambda x: x[None, ...] if x is not None else None, concatenated)


def _validate_and_flatten_elements(elements, override=False):
  if override:
    return list(elements)
  if not elements:
    raise ValueError("Cannot collate empty list.")
  
  tries = 0
  while not all(isinstance(p, ProteinTuple) for p in elements):
    if any(isinstance(p, Sequence) for p in elements):
      elements = [p[0] if isinstance(p, Sequence) else p for p in elements]
      tries += 1
    if tries > _MAX_TRIES:
      raise ValueError("Too many nested sequences.")
  return list(elements)


def _apply_electrostatics_if_needed(elements, *, use_electrostatics, estat_noise, estat_noise_mode):
  if not use_electrostatics:
    return elements

  noise_val = estat_noise[0] if isinstance(estat_noise, Sequence) else estat_noise
  
  new_elements = []
  for p in elements:
    feat = compute_electrostatic_node_features(
      p, noise_scale=noise_val, noise_mode=estat_noise_mode,
    )
    if hasattr(feat, "device"): 
        feat = np.array(feat) # Convert JAX/Torch tensors to numpy
        
    new_elements.append(p._replace(physics_features=feat))
  return new_elements


def _pad_protein(protein: Protein, max_len: int) -> Protein:
  """Pad a single Protein using NumPy."""
  pad_len = max_len - protein.coordinates.shape[0]
  protein_len = protein.coordinates.shape[0]
  full_coords_len = (
    protein.full_coordinates.shape[0] if protein.full_coordinates is not None else None
  )
  full_coords_pad_len = max_len - full_coords_len if full_coords_len is not None else 0

  def pad_fn(x):
    if x is None: 
        return None
    x = np.asarray(x)
    if x.ndim == 0:
        return x

    if full_coords_len is not None and x.shape[0] == full_coords_len:
      return np.pad(x, ((0, full_coords_pad_len),) + ((0, 0),) * (x.ndim - 1))

    if x.shape[0] == protein_len:
      return np.pad(x, ((0, pad_len),) + ((0, 0),) * (x.ndim - 1))

    return x

  return jax.tree_util.tree_map(pad_fn, protein)


def _stack_padded_proteins(padded_proteins: list[Protein]) -> Protein:
  """Stack using NumPy."""
  def stack_fn(*arrays):
    non_none = [a for a in arrays if a is not None]
    if not non_none:
      return None
    non_none = [np.asarray(a) for a in non_none]
    
    first = non_none[0]
    # For scalars or 0-dim arrays, stack works fine in numpy
    return np.stack(non_none, axis=0)

  return jax.tree_util.tree_map(stack_fn, *padded_proteins)


def pad_and_collate_proteins(
  elements: Sequence[ProteinTuple],
  *,
  use_electrostatics: bool = False,
  use_vdw: bool = False, 
  estat_noise: Sequence[float] | float | None = None,
  estat_noise_mode: str = "direct",
  vdw_noise: Sequence[float] | float | None = None, 
  vdw_noise_mode: str = "direct", 
  max_length: int | None = None,
  override: bool = False,
) -> Protein:
  """Batch and pad a list of ProteinTuples using NumPy backend."""
  
  elements = _validate_and_flatten_elements(elements, override=override)
  elements = _apply_electrostatics_if_needed(
    elements,
    use_electrostatics=use_electrostatics,
    estat_noise=estat_noise,
    estat_noise_mode=estat_noise_mode,
  )
  
  # IMPORTANT: Use from_tuple_numpy to avoid early JAX conversion
  proteins = [Protein.from_tuple_numpy(p) for p in elements]

  pad_len = max_length if max_length is not None else max(p.coordinates.shape[0] for p in proteins)

  padded_proteins = [_pad_protein(p, pad_len) for p in proteins]
  return _stack_padded_proteins(padded_proteins)