"""Utilities for aligning protein structure ensembles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np
from biotite.sequence import ProteinSequence as BiotiteProteinSequence
from biotite.sequence import align

from prxteinmpnn.io.parsing import protein_sequence_to_string
from prxteinmpnn.utils.data_structures import Protein

if TYPE_CHECKING:
  from collections.abc import Sequence


def _pad_protein_to_length(protein: Protein, new_length: int, mapping: np.ndarray) -> Protein:
  """Pad a Protein object to a new length based on an alignment mapping.

  Args:
      protein: The original Protein object.
      new_length: The target length after padding.
      mapping: An array where mapping[i] is the original index for the new
        position i, or -1 for a gap.

  Returns:
      A new, padded Protein object.

  """
  padded_coords = jnp.zeros((new_length, 37, 3), dtype=protein.coordinates.dtype)
  padded_aatype = jnp.full((new_length,), -1, dtype=protein.aatype.dtype)
  padded_atom_mask = jnp.zeros((new_length, 37), dtype=protein.atom_mask.dtype)
  padded_residue_index = jnp.zeros((new_length,), dtype=protein.residue_index.dtype)
  padded_chain_index = jnp.zeros((new_length,), dtype=protein.chain_index.dtype)

  valid_indices = mapping != -1
  original_indices = mapping[valid_indices]

  padded_coords = padded_coords.at[valid_indices].set(protein.coordinates[original_indices])
  padded_aatype = padded_aatype.at[valid_indices].set(protein.aatype[original_indices])
  padded_atom_mask = padded_atom_mask.at[valid_indices].set(protein.atom_mask[original_indices])
  padded_residue_index = padded_residue_index.at[valid_indices].set(
    protein.residue_index[original_indices],
  )
  padded_chain_index = padded_chain_index.at[valid_indices].set(
    protein.chain_index[original_indices],
  )

  return Protein(
    coordinates=padded_coords,
    aatype=padded_aatype,
    atom_mask=padded_atom_mask,
    residue_index=padded_residue_index,
    chain_index=padded_chain_index,
    dihedrals=None,  # Dihedrals are dropped as they are non-trivial to align
  )


def align_ensemble(
  ensemble: Sequence[Protein],
  strategy: Literal["sequence", "structure"] = "sequence",
  reference_index: int = 0,
) -> list[Protein]:
  """Aligns an ensemble of protein structures.

  This function aligns all structures in an ensemble to a reference structure,
  producing a new list of Protein objects with padded arrays to ensure all
  structures have the same length corresponding to the alignment. This is
  achieved by performing a star-alignment using `biotite`.

  To optimize for cases with identical sequences (e.g., MD trajectories), this
  function first identifies unique sequences. Alignments are performed only on
  this unique subset, and the results are then mapped back to all proteins in
  the original ensemble.

  Args:
      ensemble: A list of Protein structures.
      strategy: The alignment strategy to use. Currently, only "sequence" is
          supported.
      reference_index: The index of the structure in the ensemble to use as
          the reference for alignment.

  Returns:
      A new list of aligned and padded Protein structures.

  Raises:
      NotImplementedError: If 'structure' alignment is requested.
      ValueError: If the ensemble is empty.

  """
  if not ensemble:
    msg = "Cannot align an empty ensemble."
    raise ValueError(msg)

  if strategy == "structure":
    msg = "Structural alignment is not yet implemented."
    raise NotImplementedError(msg)

  # Identify unique sequences to avoid redundant alignments
  sequences_str = [protein_sequence_to_string(p.aatype) for p in ensemble]
  unique_sequences, inverse_indices = np.unique(sequences_str, return_inverse=True)

  # If all sequences are identical, no alignment is needed.
  if len(unique_sequences) == 1:
    return list(ensemble)

  # Align the unique sequences
  unique_biotite_seqs = [BiotiteProteinSequence(s) for s in unique_sequences]
  ref_unique_idx = inverse_indices[reference_index]
  ref_biotite_seq = unique_biotite_seqs[ref_unique_idx]

  pairwise_alignments = [
    align.align_optimal(  # type: ignore[attr-access]
      ref_biotite_seq,
      s,
      align.SubstitutionMatrix.std_protein_matrix(),
      gap_penalty=(-10, -1),
      terminal_penalty=False,
    )[0]
    if i != ref_unique_idx
    else None
    for i, s in enumerate(unique_biotite_seqs)
  ]

  # Build the master alignment columns from all unique pairwise alignments
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

  # Compute the mapping from original to aligned indices for each unique sequence
  unique_mappings = {}
  for i, biotite_seq in enumerate(unique_biotite_seqs):
    mapping = np.full(msa_length, -1, dtype=int)
    if i == ref_unique_idx:
      for ref_idx in range(len(biotite_seq)):
        msa_pos = col_to_msa_pos.get((ref_idx, 0))
        if msa_pos is not None:
          mapping[msa_pos] = ref_idx
    else:
      aln = pairwise_alignments[i]
      ref_trace, tgt_trace = aln.trace[0], aln.trace[1]  # type: ignore[attr-access]
      last_ref_idx, insertion_count = -1, 0
      for _, (ref_idx, tgt_idx) in enumerate(zip(ref_trace, tgt_trace, strict=False)):
        if ref_idx != -1:
          last_ref_idx, insertion_count = ref_idx, 0
          col = (ref_idx, 0)
        else:
          insertion_count += 1
          col = (last_ref_idx, insertion_count)
        msa_pos = col_to_msa_pos.get(col)
        if msa_pos is not None and tgt_idx != -1:
          mapping[msa_pos] = tgt_idx
    unique_mappings[i] = mapping

  # Apply the pre-computed mappings to the full ensemble
  aligned_proteins: list[Protein | None] = [None] * len(ensemble)
  for i, protein in enumerate(ensemble):
    unique_idx = inverse_indices[i]
    mapping = unique_mappings[unique_idx]
    padded_protein = _pad_protein_to_length(protein, msa_length, mapping)
    aligned_proteins[i] = padded_protein

  return aligned_proteins  # type: ignore[return-value]
