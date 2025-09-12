"""Utilities for batching, padding, and aligning protein sequences."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from prxteinmpnn.utils.align import smith_waterman_affine
from prxteinmpnn.utils.data_structures import Protein, ProteinEnsemble
from prxteinmpnn.utils.types import ProteinSequence

# Constants for array dimensionality checks
_1D_ARRAY = 1
_2D_ARRAY = 2
_3D_ARRAY = 3

# BLOSUM62 scoring matrix with an added column/row for gaps
_AA_SCORE_MATRIX = jnp.array(
  [  # A,   R,   N,   D, C,   Q,   E,   G,   H,   I,   L,   K,   M,   F,   P,  S,  T,W, Y, V, Gap
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -4],  # A
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -4],  # R
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, -4],  # N
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, -4],  # D
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -4],  # C
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, -4],  # Q
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, -4],  # E
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -4],  # G
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, -4],  # H
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -4],  # I
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4],  # L
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, -4],  # K
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -4],  # M
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -4],  # F
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -4],  # P
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, -4],  # S
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -4],  # T
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4],  # W
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -4],  # Y
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -4],  # V
    [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1],  # Gap
  ],
  dtype=jnp.float32,
)


def _pad_protein_to_length(protein: Protein, new_length: int, mapping: jax.Array) -> Protein:
  """Pad a Protein object to a new length based on an alignment mapping.

  Args:
    protein (Protein): The Protein object to pad.
    new_length (int): The target length for padding.
    mapping (jax.Array): An array where `mapping[i]` gives the index in the original
      protein for the i-th position in the padded sequence. -1 indicates a gap.

  Returns:
    Protein: A new Protein object padded to the specified length.

  """
  padded_coords = jnp.zeros((new_length, 37, 3), dtype=protein.coordinates.dtype)
  padded_aatype = jnp.full((new_length,), -1, dtype=protein.aatype.dtype)
  padded_atom_mask = jnp.zeros((new_length, 37), dtype=protein.atom_mask.dtype)
  padded_residue_index = jnp.zeros((new_length,), dtype=protein.residue_index.dtype)
  padded_chain_index = jnp.zeros((new_length,), dtype=protein.chain_index.dtype)

  valid_indices = mapping != -1
  original_indices = mapping[valid_indices]
  original_indices = jnp.clip(original_indices, 0, protein.coordinates.shape[0] - 1)

  padded_coords = padded_coords.at[valid_indices].set(protein.coordinates[original_indices])
  padded_aatype = padded_aatype.at[valid_indices].set(protein.aatype[original_indices])
  padded_atom_mask = padded_atom_mask.at[valid_indices].set(protein.atom_mask[original_indices])
  padded_residue_index = padded_residue_index.at[valid_indices].set(
    protein.residue_index[original_indices],
  )
  padded_chain_index = padded_chain_index.at[valid_indices].set(
    protein.chain_index[original_indices],
  )
  # Regenerate one-hot from padded aatype, handling -1 for gaps
  padded_one_hot = jax.nn.one_hot(
    jnp.where(padded_aatype == -1, 20, padded_aatype),
    num_classes=21,
  )
  padded_one_hot = padded_one_hot * (padded_aatype[:, None] != -1)

  return Protein(
    coordinates=padded_coords,
    aatype=padded_aatype,
    atom_mask=padded_atom_mask,
    residue_index=padded_residue_index,
    chain_index=padded_chain_index,
    dihedrals=None,
    one_hot_sequence=padded_one_hot,
  )


def batch_and_pad_sequences(sequences: Sequence[ProteinSequence]) -> tuple[jax.Array, jax.Array]:
  """Batch and pad pre-tokenized protein sequences for JAX operations.

  Args:
    sequences (Sequence[ProteinSequence]): A list or tuple of pre-tokenized protein sequences.
      Each sequence should be a JAX array of integer amino acid indices.

  Returns:
    tuple[jax.Array, jax.Array]: A tuple containing the padded tokenized sequences
      and a corresponding boolean mask.

  Raises:
    ValueError: If the input sequence list is empty.

  """
  if not sequences:
    msg = "Cannot process an empty list of sequences."
    raise ValueError(msg)

  # Treat scalars as length 1, otherwise use the array's length.
  def get_len(arr: jax.Array) -> int:
    return arr.shape[0] if arr.ndim > 0 else 1

  max_len = max((get_len(s) for s in sequences), default=0)

  # Handle case where all inputs are empty strings.
  if max_len == 0:
    return jnp.empty((len(sequences), 0), dtype=jnp.int8), jnp.empty(
      (len(sequences), 0),
      dtype=jnp.bool_,
    )

  def _pad(arr: jax.Array, val: int | bool) -> jax.Array:  # noqa: FBT001
    """Pad an array, correctly handling scalars by reshaping them first."""
    if arr.ndim == 0:
      arr = arr.reshape(1)  # Reshape scalar to a 1-element array

    padding_needed = max_len - arr.shape[0]
    return jnp.pad(arr, (0, padding_needed), "constant", constant_values=val)

  batched_tokens = jnp.stack([_pad(s, -1) for s in sequences])

  padded_masks = []
  for s in sequences:
    current_len = get_len(s)
    mask = jnp.ones(current_len, dtype=jnp.bool_)
    padding_needed = max_len - current_len
    padded_mask = jnp.pad(mask, (0, padding_needed), "constant", constant_values=False)
    padded_masks.append(padded_mask)
  batched_masks = jnp.stack(padded_masks)

  return batched_tokens, batched_masks


def _get_sw_score_matrix(seq_a_token: jax.Array, seq_b_token: jax.Array) -> jax.Array:
  """Construct a scoring matrix for two tokenized sequences based on BLOSUM62.

  Args:
    seq_a_token (jax.Array): The first tokenized sequence.
    seq_b_token (jax.Array): The second tokenized sequence.

  Returns:
    jax.Array: A 2D scoring matrix.

  """
  seq_a = jnp.clip(seq_a_token, 0, 20)
  seq_b = jnp.clip(seq_b_token, 0, 20)
  return _AA_SCORE_MATRIX[seq_a[:, None], seq_b[None, :]]


def _align_sequences_jax(
  tokenized_seqs: list[jax.Array],
  gap_open: float,
  gap_extend: float,
  temp: float,
) -> tuple[int, int, list[jax.Array | None]]:
  """Perform pairwise alignment of all sequences to a reference and computes MSA info."""
  if not tokenized_seqs:
    return -1, 0, []

  # Find reference sequence (longest non-empty)
  non_empty_indices = [i for i, s in enumerate(tokenized_seqs) if s.size > 0]
  if not non_empty_indices:
    return -1, 0, [None] * len(tokenized_seqs)

  ref_idx = max(non_empty_indices, key=lambda i: tokenized_seqs[i].shape[0])
  ref_seq = tokenized_seqs[ref_idx]
  len_ref = ref_seq.shape[0]

  other_indices = [i for i in non_empty_indices if i != ref_idx]
  if not other_indices:
    msa_length = len_ref
    return ref_idx, msa_length, [None] * len(tokenized_seqs)

  other_seqs = [tokenized_seqs[i] for i in other_indices]
  other_seq_lengths = jnp.array([s.shape[0] for s in other_seqs])
  max_len_other = other_seq_lengths.max()

  padded_other_seqs = jnp.stack(
    [jnp.pad(s, (0, max_len_other - s.shape[0]), constant_values=0) for s in other_seqs],
  )

  ref_seq_safe = jnp.clip(ref_seq, 0, 20)
  padded_other_seqs_safe = jnp.clip(padded_other_seqs, 0, 20)

  batched_score_matrices = _AA_SCORE_MATRIX[
    ref_seq_safe[None, :, None],
    padded_other_seqs_safe[:, None, :],
  ]

  batched_lengths = jnp.stack(
    [
      jnp.full_like(other_seq_lengths, len_ref),
      other_seq_lengths,
    ],
    axis=1,
  )

  sw_aligner_batched = smith_waterman_affine(batch=True)
  tracebacks = sw_aligner_batched(
    batched_score_matrices,
    batched_lengths,
    gap_extend,
    gap_open,
    temp,
  )

  grads: list[None | jax.Array] = [None] * len(tokenized_seqs)
  for i, original_idx in enumerate(other_indices):
    original_len = other_seqs[i].shape[0]
    grads[original_idx] = tracebacks[i, :, :original_len]

  max_insertions = max((s.shape[0] - len_ref for s in other_seqs), default=0)
  msa_length = len_ref + max(0, max_insertions)

  return ref_idx, msa_length, grads


def perform_star_alignment(
  proteins: Sequence[Protein],
  gap_open: float = -10.0,
  gap_extend: float = -1.0,
  temp: float = 1.0,
) -> list[Protein]:
  """Perform star alignment on proteins and returns padded Protein objects.

  This implementation uses a simplified approach where the final MSA length is
  determined by the reference sequence plus the largest insertion found in any
  pairwise alignment. A heuristic based on maximum marginal probabilities is used
  to map sequences to the final MSA.

  Args:
    proteins (Sequence[Protein]): A list of Protein objects to align.
    gap_open (float): The penalty for opening a gap.
    gap_extend (float): The penalty for extending a gap.
    temp (float): The temperature for soft-maximum in the SW algorithm.

  Returns:
    list[Protein]: A list of new, padded Protein objects representing the alignment.

  """
  if not proteins:
    return []

  tokenized_seqs = [p.aatype for p in proteins]
  ref_idx, msa_length, grads = _align_sequences_jax(tokenized_seqs, gap_open, gap_extend, temp)

  if ref_idx == -1 or msa_length == 0:
    return [_pad_protein_to_length(p, 0, jnp.array([])) for p in proteins]

  aligned_proteins = []
  for i, protein in enumerate(proteins):
    mapping = jnp.full(msa_length, -1, dtype=jnp.int32)
    traceback = grads[i]
    if i == ref_idx:
      mapping = mapping.at[: protein.aatype.shape[0]].set(jnp.arange(protein.aatype.shape[0]))
    elif traceback is not None:
      aligned_ref_indices = jnp.argmax(traceback, axis=0)
      current_indices = jnp.arange(aligned_ref_indices.shape[0])
      unique_ref_indices, first_occurrence_indices = jnp.unique(
        aligned_ref_indices,
        return_index=True,
      )
      unique_current_indices = current_indices[first_occurrence_indices]
      mapping = mapping.at[unique_ref_indices].set(unique_current_indices)

    aligned_proteins.append(_pad_protein_to_length(protein, msa_length, mapping))

  return aligned_proteins


def _generate_cross_protein_mapping(
  proteins: Sequence[Protein],
  gap_open: float = -1.0,
  gap_extend: float = -0.1,
  temp: float = 0.1,
) -> jax.Array:
  """Generate cross-protein position mapping using sequence alignment.

  Creates a mapping array for cross-protein position comparisons using
  Smith-Waterman alignment. Only upper triangle is computed for memory efficiency.

  Args:
    proteins (Sequence[Protein]): List of protein objects to align.
    gap_open (float): Gap opening penalty for alignment.
    gap_extend (float): Gap extension penalty for alignment.
    temp (float): Temperature parameter for soft alignment.

  Returns:
    jax.Array: Upper triangle mapping array of shape (num_pairs, max_length, 2)
      where num_pairs = n*(n-1)/2 for n proteins. Each entry contains
      [pos_in_protein_i, pos_in_protein_j] or [-1, -1] for unaligned positions.

  """
  if len(proteins) < 2:  # noqa: PLR2004
    return jnp.array([]).reshape(0, 0, 2)

  n_proteins = len(proteins)
  max_len = max(p.aatype.shape[0] for p in proteins)
  n_pairs = (n_proteins * (n_proteins - 1)) // 2

  # Initialize mapping array
  mapping = jnp.full((n_pairs, max_len, 2), -1, dtype=jnp.int32)

  pair_idx = 0
  for i in range(n_proteins):
    for j in range(i + 1, n_proteins):
      seq_i = jnp.clip(proteins[i].aatype, 0, 20)
      seq_j = jnp.clip(proteins[j].aatype, 0, 20)

      # Generate scoring matrix
      score_matrix = _AA_SCORE_MATRIX[seq_i[:, None], seq_j[None, :]]
      lengths = jnp.array([seq_i.shape[0], seq_j.shape[0]])

      # Perform alignment
      sw_aligner = smith_waterman_affine(batch=False)
      try:
        traceback = sw_aligner(score_matrix, lengths, gap_extend, gap_open, temp)

        # Extract aligned positions - use both dimensions of traceback
        # traceback shape is (len_i, len_j) where high values indicate alignment
        threshold = jnp.max(traceback) * 0.1  # Use 10% of max as threshold

        # Find positions where alignment score is above threshold
        align_mask = traceback > threshold
        i_indices, j_indices = jnp.where(align_mask, size=max_len, fill_value=-1)

        # Filter out the fill values and create mapping
        valid_mask = (i_indices != -1) & (j_indices != -1)
        valid_i = i_indices[valid_mask]
        valid_j = j_indices[valid_mask]

        # Ensure we don't exceed max_len
        n_valid = min(len(valid_i), max_len)
        if n_valid > 0:
          pair_mapping = jnp.stack([valid_i[:n_valid], valid_j[:n_valid]], axis=1)
          mapping = mapping.at[pair_idx, :n_valid].set(pair_mapping)

      except Exception:  # noqa: BLE001
        # If alignment fails, try simple diagonal mapping for similar length sequences
        len_i, len_j = seq_i.shape[0], seq_j.shape[0]
        min_len = min(len_i, len_j, max_len)
        max_length_diff = 10
        if abs(len_i - len_j) <= max_length_diff:  # If sequences are similar in length
          diagonal_mapping = jnp.stack(
            [
              jnp.arange(min_len),
              jnp.arange(min_len),
            ],
            axis=1,
          )
          mapping = mapping.at[pair_idx, :min_len].set(diagonal_mapping)

      pair_idx += 1

  return mapping


def _get_pair_index(i: int, j: int, n: int) -> int:
  """Get the index in upper triangle storage for protein pair (i,j)."""
  if i >= j:
    msg = f"Invalid pair indices: i={i} must be < j={j}"
    raise ValueError(msg)
  return i * n - (i * (i + 1)) // 2 + j - i - 1


def batch_and_pad_proteins(
  proteins: Sequence[Protein],
  sequences_to_score: Sequence[jax.Array] | None = None,
  *,
  calculate_cross_diff: bool = False,
) -> tuple[ProteinEnsemble, jax.Array | None]:
  """Batch and pad a list of Protein objects to create a ProteinEnsemble.

  This function takes a list of Protein objects, determines the maximum length
  among them, and pads each protein's data to this length. The padded data is
  then combined into a single ProteinEnsemble object using JAX-idiomatic operations.

  Args:
    proteins (Sequence[Protein]): A list of Protein objects to batch and pad.
    sequences_to_score (Sequence[jax.Array] | None): Optional pre-tokenized sequences
      to batch and pad alongside the proteins.
    calculate_cross_diff (bool): Whether to calculate cross-protein mapping for
      position comparisons. Defaults to False to avoid expensive computation.

  Returns:
    tuple[ProteinEnsemble, jax.Array | None]: A tuple containing:
      - ProteinEnsemble: The batched and padded proteins with optional mapping
      - jax.Array | None: Batched sequences if sequences_to_score was provided

  Raises:
    ValueError: If the input protein list is empty.

  Example:
    >>> protein1 = Protein(coordinates=jnp.ones((10, 37, 3)), ...)
    >>> protein2 = Protein(coordinates=jnp.ones((15, 37, 3)), ...)
    >>> ensemble, _ = batch_and_pad_proteins([protein1, protein2])
    >>> ensemble.coordinates.shape
    (2, 15, 37, 3)

  """
  if not proteins:
    msg = "Cannot process an empty list of proteins."
    raise ValueError(msg)

  # Extract all attributes as lists for vectorized operations
  coords_list = [p.coordinates for p in proteins]
  aatype_list = [p.aatype for p in proteins]
  atom_mask_list = [p.atom_mask for p in proteins]
  residue_index_list = [p.residue_index for p in proteins]
  chain_index_list = [p.chain_index for p in proteins]
  one_hot_list = [p.one_hot_sequence for p in proteins]

  # Compute max length vectorized
  lengths = jnp.array([coords.shape[0] for coords in coords_list])
  max_len = int(lengths.max())

  # Pad all arrays using vmap for efficient vectorization
  def pad_array(arr: jax.Array, target_len: int, fill_value: float = 0) -> jax.Array:
    """Pad a single array to target length."""
    padding_needed = target_len - arr.shape[0]
    if arr.ndim == _1D_ARRAY:
      return jnp.pad(arr, (0, padding_needed), constant_values=fill_value)
    if arr.ndim == _2D_ARRAY:
      return jnp.pad(arr, ((0, padding_needed), (0, 0)), constant_values=fill_value)
    if arr.ndim == _3D_ARRAY:
      return jnp.pad(arr, ((0, padding_needed), (0, 0), (0, 0)), constant_values=fill_value)
    msg = f"Unsupported array dimensionality: {arr.ndim}"
    raise ValueError(msg)

  # Vectorized padding operations
  padded_coords = jnp.stack([pad_array(coords, max_len) for coords in coords_list])
  padded_aatype = jnp.stack([pad_array(aatype, max_len, -1) for aatype in aatype_list])
  padded_atom_mask = jnp.stack([pad_array(mask, max_len) for mask in atom_mask_list])
  padded_residue = jnp.stack([pad_array(res, max_len) for res in residue_index_list])
  padded_chain = jnp.stack([pad_array(chain, max_len) for chain in chain_index_list])
  padded_one_hot = jnp.stack([pad_array(oh, max_len) for oh in one_hot_list])

  # Generate cross-protein mapping if requested
  mapping = None
  if calculate_cross_diff:
    # Use milder gap penalties for similar sequences
    mapping = _generate_cross_protein_mapping(
      proteins,
      gap_open=-0.5,  # Much milder gap opening penalty
      gap_extend=-0.1,  # Mild gap extension penalty
      temp=0.5,  # Higher temperature for softer alignment
    )

  # Handle sequences if provided
  batched_sequences = None
  if sequences_to_score is not None:
    batched_sequences, _ = batch_and_pad_sequences(sequences_to_score)

  ensemble = ProteinEnsemble(
    coordinates=padded_coords,
    aatype=padded_aatype,
    atom_mask=padded_atom_mask,
    residue_index=padded_residue,
    chain_index=padded_chain,
    dihedrals=None,
    one_hot_sequence=padded_one_hot,
    mapping=mapping,
  )

  return ensemble, batched_sequences
