"""Utility functions for batching protein sequences for MPNN processing."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from biotite.sequence import ProteinSequence as BiotiteProteinSequence
from biotite.sequence import align

from prxteinmpnn.utils.aa_convert import protein_sequence_to_string, string_to_protein_sequence
from prxteinmpnn.utils.data_structures import Protein


def _pad_protein_to_length(protein: Protein, new_length: int, mapping: jax.Array) -> Protein:
  """Pad a Protein object to a new length based on an alignment mapping.

  Designed to be JAX-transformable (e.g., with vmap) when used with JAX arrays.
  """
  padded_coords = jnp.zeros((new_length, 37, 3), dtype=protein.coordinates.dtype)
  padded_aatype = jnp.full((new_length,), -1, dtype=protein.aatype.dtype)
  padded_one_hot_sequence = jnp.zeros((new_length, 21), dtype=jnp.float32)
  padded_atom_mask = jnp.zeros((new_length, 37), dtype=protein.atom_mask.dtype)
  padded_residue_index = jnp.zeros((new_length,), dtype=protein.residue_index.dtype)
  padded_chain_index = jnp.zeros((new_length,), dtype=protein.chain_index.dtype)

  valid_indices = mapping != -1
  original_indices = mapping[valid_indices]

  # Clip original_indices to prevent out-of-bounds access if mapping contains invalid values
  # for the current protein's actual length (should ideally not happen with correct alignment)
  # but adds robustness for JAX's static shape requirements.
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
  padded_one_hot_sequence = padded_one_hot_sequence.at[valid_indices].set(
    protein.one_hot_sequence[original_indices],
  )

  return Protein(
    coordinates=padded_coords,
    aatype=padded_aatype,
    atom_mask=padded_atom_mask,
    residue_index=padded_residue_index,
    chain_index=padded_chain_index,
    dihedrals=None,  # Dihedrals are dropped as they are non-trivial to align
    one_hot_sequence=padded_one_hot_sequence,
  )


def batch_and_pad_sequences(
  sequences: Sequence[str],
) -> tuple[jax.Array, jax.Array]:
  """Convert string sequences to a padded JAX array of MPNN indices."""
  if not sequences:
    msg = "Cannot process an empty list of sequences."
    raise ValueError(msg)

  tokenized_sequences = [string_to_protein_sequence(s) for s in sequences]
  max_len = max(seq.shape[0] for seq in tokenized_sequences)
  masks = [jnp.ones(seq.shape[0], dtype=jnp.bool_) for seq in tokenized_sequences]

  def _pad(array: jax.Array, pad_value: int = -1) -> jax.Array:
    """Pad a single JAX array to the max_len."""
    padding_needed = max_len - array.shape[0]
    return jnp.pad(array, (0, padding_needed), "constant", constant_values=pad_value)

  batched_tokens = jnp.stack([_pad(s) for s in tokenized_sequences])
  batched_masks = jnp.stack([_pad(m, pad_value=0) for m in masks])

  return batched_tokens, batched_masks


def _perform_star_alignment(
  unique_biotite_seqs: list[BiotiteProteinSequence],
) -> tuple[int, dict[tuple[int, int], int], int, list[align.Alignment | None]]:
  """Perform a star alignment on unique sequences and returns MSA metadata."""
  ref_unique_idx = int(np.argmax([len(s) for s in unique_biotite_seqs]))
  ref_biotite_seq = unique_biotite_seqs[ref_unique_idx]

  pairwise_alignments: list[align.Alignment | None] = [
    align.align_optimal(  # type: ignore[attr-access]
      ref_biotite_seq,
      s,
      align.SubstitutionMatrix.std_protein_matrix(),
      gap_penalty=(-10, -1),
      terminal_penalty=False,
      local=True,
    )[0]
    if i != ref_unique_idx
    else None
    for i, s in enumerate(unique_biotite_seqs)
  ]

  msa_cols = [(i, 0) for i in range(len(ref_biotite_seq))]

  for aln in pairwise_alignments:
    if aln is None:
      continue
    ref_trace = aln.trace[0]
    last_ref_idx = -1
    insertion_count = 0
    for ref_idx_in_aln in ref_trace:
      if ref_idx_in_aln != -1:
        last_ref_idx = ref_idx_in_aln
        insertion_count = 0
      else:
        insertion_count += 1
        new_col = (last_ref_idx, insertion_count)
        if new_col not in msa_cols:
          msa_cols.append(new_col)

  msa_cols.sort()
  msa_length = len(msa_cols)
  col_to_msa_pos = {col: i for i, col in enumerate(msa_cols)}

  return ref_unique_idx, col_to_msa_pos, msa_length, pairwise_alignments


def _compute_unique_mappings(
  unique_biotite_seqs: list[BiotiteProteinSequence],
  ref_unique_idx: int,
  col_to_msa_pos: dict[tuple[int, int], int],
  msa_length: int,
  pairwise_alignments: list[align.Alignment | None],
) -> dict[int, np.ndarray]:
  """Compute mapping arrays for each unique sequence to the MSA.

  Args:
      unique_biotite_seqs: List of unique BiotiteProteinSequence objects.
      ref_unique_idx: Index of the reference sequence in the unique sequences list.
      col_to_msa_pos: Dictionary mapping (ref_idx, insertion_count) to MSA positions.
      msa_length: Total length of the MSA.
      pairwise_alignments: List of pairwise alignments with the reference sequence.

  Returns:
      A dictionary mapping each unique sequence index to its MSA mapping array.

  """
  unique_mappings = {}
  for i, biotite_seq in enumerate(unique_biotite_seqs):
    mapping = np.full(msa_length, -1, dtype=int)
    if i == ref_unique_idx:
      # Reference sequence: direct mapping to MSA columns based on its own residues
      for ref_idx in range(len(biotite_seq)):
        msa_pos = col_to_msa_pos.get((ref_idx, 0))
        if msa_pos is not None:
          mapping[msa_pos] = ref_idx
    else:
      # Non-reference sequences: use pairwise alignment to map to MSA columns
      aln = pairwise_alignments[i]
      if aln is None:
        # This should ideally not happen for non-reference sequences
        continue

      ref_trace = aln.trace[:, 0]  # First column of the trace is for the reference
      tgt_trace = aln.trace[:, 1]
      current_ref_idx = -1  # Tracks the last actual residue index in the reference
      current_insertion_count = 0  # Tracks insertions relative to the current_ref_idx

      for aln_col_idx in range(len(ref_trace)):
        ref_idx_in_aln = ref_trace[aln_col_idx]
        tgt_idx_in_aln = tgt_trace[aln_col_idx]

        col_key = None
        if ref_idx_in_aln != -1:
          # This alignment column corresponds to a residue in the reference sequence
          current_ref_idx = ref_idx_in_aln
          current_insertion_count = 0
          col_key = (current_ref_idx, 0)
        else:
          # This alignment column is an insertion relative to the reference sequence
          # It's an insertion if ref_idx_in_aln is -1
          current_insertion_count += 1
          col_key = (current_ref_idx, current_insertion_count)

        msa_pos = col_to_msa_pos.get(col_key)

        if msa_pos is not None and tgt_idx_in_aln != -1:
          # If this MSA position exists and the target sequence has a residue here,
          # map the target residue's original index to this MSA position.
          mapping[msa_pos] = tgt_idx_in_aln
    unique_mappings[i] = mapping
  return unique_mappings


def _pad_single_token_seq(
  tokenized_seq: jax.Array,
  mapping: jnp.ndarray,
  msa_length: int,
) -> jax.Array:
  """Pad a single tokenized sequence using a given mapping."""
  padded_tokens = jnp.full((msa_length,), -1, dtype=jnp.int8)
  valid_indices = mapping != -1
  valid_indices_clipped = jnp.where(valid_indices & (mapping < tokenized_seq.shape[0]), True, False)  # noqa: FBT003
  original_indices = mapping[valid_indices_clipped]

  return padded_tokens.at[valid_indices_clipped].set(tokenized_seq[original_indices])


async def batch_and_pad_proteins(
  proteins: list[Protein],
  sequences_to_score: Sequence[str] | None = None,
) -> tuple[Protein, jax.Array | None]:
  """Consume a ProteinEnsemble stream, then pad and stack the proteins into a single batched Pytree.

  This function is responsible for aligning both the protein structures and any
  additional sequences provided. It performs a star alignment on the unique
  sequences present in the ensemble and `sequences_to_score`, then pads all
  proteins and sequences to match the resulting multiple sequence alignment.

  Args:
      ensemble: An async generator of Protein objects to batch.
      sequences_to_score: An optional list of protein sequences (strings) to
        align and batch alongside the protein structures.

  Returns:
      A tuple containing:
        - A single Protein object where each attribute is a JAX array with a
          leading batch dimension.
        - A list of source identifiers for each protein in the batch.
        - A JAX array of tokenized and padded sequences from `sequences_to_score`,
          or None if not provided.

  """
  protein_seq_strs = [protein_sequence_to_string(p.aatype) for p in proteins]
  all_seq_strs = list(protein_seq_strs)
  if sequences_to_score:
    all_seq_strs.extend(sequences_to_score)

  unique_sequences, inverse_indices = np.unique(all_seq_strs, return_inverse=True)
  unique_biotite_seqs = [BiotiteProteinSequence(s) for s in unique_sequences]

  ref_unique_idx, col_to_msa_pos, msa_length, pairwise_alignments = _perform_star_alignment(
    unique_biotite_seqs,
  )
  unique_mappings = _compute_unique_mappings(
    unique_biotite_seqs,
    ref_unique_idx,
    col_to_msa_pos,
    msa_length,
    pairwise_alignments,
  )

  padded_proteins = [
    _pad_protein_to_length(
      protein,
      msa_length,
      jnp.array(unique_mappings[inverse_indices[i]]),
    )
    for i, protein in enumerate(proteins)
  ]

  def stack_leaves(*leaves: jax.Array) -> jax.Array | None:
    if not leaves or leaves[0] is None:
      return None
    return jnp.stack(leaves)

  padded_batched_protein = jax.tree_util.tree_map(stack_leaves, *padded_proteins)

  aligned_sequences_tokens = None
  if sequences_to_score:
    num_proteins = len(proteins)
    tokenized_seqs_to_score = [string_to_protein_sequence(s) for s in sequences_to_score]

    padded_seqs = []
    for i, token_seq in enumerate(tokenized_seqs_to_score):
      # Get the mapping for this specific sequence
      mapping_idx = inverse_indices[num_proteins + i]
      mapping = jnp.array(unique_mappings[mapping_idx])

      # Pad the sequence according to the alignment
      padded_seq = _pad_single_token_seq(token_seq, mapping, msa_length)
      padded_seqs.append(padded_seq)

    if padded_seqs:
      aligned_sequences_tokens = jnp.stack(padded_seqs)
    else:
      aligned_sequences_tokens = jnp.empty((0, msa_length), dtype=jnp.int8)

  return padded_batched_protein, aligned_sequences_tokens
