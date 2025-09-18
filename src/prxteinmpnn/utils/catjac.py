"""Utilities for computing categorical Jacobians."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from prxteinmpnn.utils.align import align_sequences
from prxteinmpnn.utils.types import (
  CategoricalJacobian,
  InterproteinMapping,
  OneHotProteinSequence,
  ProteinSequence,
)

CombineCatJacPairFn = Callable[
  [
    CategoricalJacobian,  # Jacobian for protein A
    CategoricalJacobian,  # Jacobian for protein B (to be mapped)
    InterproteinMapping,  # The (L, 2) index map from B to A
    jax.Array,  # The scalar weight for the pair
  ],
  CategoricalJacobian,  # The combined Jacobian
]

# The public-facing function that operates on the whole batch
CombineCatJacFn = Callable[
  [
    jax.Array,  # Full batch of Jacobians, shape (N, ...)
    ProteinSequence,  # Full batch of sequences, shape (N, L)
    jax.Array | None,  # Per-protein weights, shape (N,)
  ],
  tuple[CategoricalJacobian, jax.Array],  # Final combined Jacobians, shape (N*(N-1)/2, ...)
]

# The factory that creates the public-facing function
CombineCatJacFnFactory = Callable[
  [
    CombineCatJacPairFn | Literal["add", "subtract"],
  ],
  CombineCatJacFn,
]


def _gather_mapped_jacobian(
  jacobian_to_map: CategoricalJacobian,
  mapping: jax.Array,
) -> CategoricalJacobian:
  """Remaps a Jacobian using an index map from an alignment."""
  k_indices = mapping[:, 1]
  valid_mask = k_indices != -1
  k_indices_clipped = jnp.maximum(0, k_indices)

  mapped_jac = jacobian_to_map[:, k_indices_clipped][:, :, :, k_indices_clipped]

  mask_i = valid_mask[:, None, None, None]
  mask_j = valid_mask[None, None, :, None]

  return mapped_jac * mask_i * mask_j


def _add_jacobians_mapped(
  jac1: CategoricalJacobian,
  jac2: CategoricalJacobian,
  mapping: jax.Array,
  weights: jax.Array,
) -> CategoricalJacobian:
  """Add jac1 to the mapped version of jac2."""
  mapped_jac2 = _gather_mapped_jacobian(jac2, mapping)
  return jac1 + mapped_jac2 * weights


def _subtract_jacobians_mapped(
  jac1: CategoricalJacobian,
  jac2: CategoricalJacobian,
  mapping: jax.Array,
  weights: jax.Array,
) -> CategoricalJacobian:
  """Subtracts the mapped version of jac2 from jac1."""
  mapped_jac2 = _gather_mapped_jacobian(jac2, mapping)
  return jac1 - mapped_jac2 * weights


_MIN_NUMBER_PROTEINS = 2


def make_combine_jac(
  combine_fn: Literal["add", "subtract"] | CombineCatJacPairFn,
  fn_kwargs: dict[str, Any] | None = None,
  batch_size: int = 1,
) -> CombineCatJacFn:
  """Create a function to combine Jacobians using a specified operation."""
  if combine_fn == "add":
    _combine_fn = _add_jacobians_mapped
  elif combine_fn == "subtract":
    _combine_fn = _subtract_jacobians_mapped
  else:
    _combine_fn = partial(combine_fn, **(fn_kwargs or {}))

  def combine_jacobians(
    jacobians: jax.Array,
    sequences: ProteinSequence | OneHotProteinSequence,
    weights: jax.Array | None = None,
  ) -> tuple[CategoricalJacobian, jax.Array]:
    """Compute cross-protein Jacobian combinations for all pairs."""
    if weights is None:
      weights = jnp.ones((jacobians.shape[0],), dtype=jacobians.dtype)
    n_proteins = jacobians.shape[0]
    if n_proteins < _MIN_NUMBER_PROTEINS:
      return jnp.empty((0, *jacobians.shape[1:]), dtype=jacobians.dtype), jnp.empty(
        (0, 2),
        dtype=jnp.int32,
      )

    i_indices, j_indices = jnp.triu_indices(n_proteins, k=1)

    jac_i = jacobians[i_indices]
    jac_j = jacobians[j_indices]
    pair_weights = weights[i_indices]

    all_mappings = align_sequences(sequences)

    xs = (jac_i, jac_j, all_mappings, pair_weights)

    return jax.lax.map(
      lambda x: _combine_fn(*x),
      xs,
      batch_size=batch_size,
    ), all_mappings

  return combine_jacobians


def combine_jacobians_h5_stream(
  h5_path: str | Path,
  combine_fn: CombineCatJacPairFn,
  fn_kwargs: dict,
  batch_size: int,
  weights: jax.Array,
) -> None:
  """Combine all pairs of Jacobians from an HDF5 file and save to the same file."""
  with h5py.File(str(h5_path), "a") as f:
    if "combined_catjac" in f:
      del f["combined_catjac"]
    if "mappings" in f:
      del f["mappings"]

    jacobians_dset = f["categorical_jacobians"]
    sequences_dset = f["one_hot_sequences"]

    n_samples = jacobians_dset.shape[0]  # pyright: ignore[reportAttributeAccessIssue]
    if n_samples != sequences_dset.shape[0] or n_samples != weights.shape[0]:  # pyright: ignore[reportAttributeAccessIssue]
      msg = "Jacobian, sequence, and weights arrays must have the same length."
      raise ValueError(msg)

    i_indices_all, j_indices_all = jnp.triu_indices(n_samples, k=1)
    total_pairs = len(i_indices_all)

    if total_pairs == 0:
      f.create_dataset("combined_catjac", shape=(0, *jacobians_dset.shape[1:]), dtype="float32")  # pyright: ignore[reportAttributeAccessIssue]
      f.create_dataset("mappings", shape=(0, sequences_dset.shape[1], 2), dtype="int32")  # pyright: ignore[reportAttributeAccessIssue]
      return

    _combine_fn_with_kwargs = partial(combine_fn, **(fn_kwargs or {}))

    def _process_pair(
      jac_i: CategoricalJacobian,
      jac_j: CategoricalJacobian,
      seq_i: OneHotProteinSequence,
      seq_j: OneHotProteinSequence,
      w_i: jax.Array,
    ) -> tuple[CategoricalJacobian, jax.Array]:
      mapping = align_sequences(jnp.stack([seq_i, seq_j]))[0]
      return _combine_fn_with_kwargs(jac_i, jac_j, mapping, w_i), mapping

    _vmapped_process_pairs = jax.jit(jax.vmap(_process_pair))

    out_shape = jacobians_dset.shape[1:]  # pyright: ignore[reportAttributeAccessIssue]
    n_residues = sequences_dset.shape[1]  # pyright: ignore[reportAttributeAccessIssue]

    chunk_size = min(batch_size, 100, total_pairs)
    dset = f.create_dataset(
      "combined_catjac",
      (total_pairs, *out_shape),
      dtype="float32",
      chunks=(chunk_size, *out_shape),
    )
    mapping_dset = f.create_dataset(
      "mappings",
      (total_pairs, n_residues, 2),
      dtype="int32",
      chunks=(chunk_size, n_residues, 2),
    )

    write_idx = 0
    # ** NEW STRATEGY: Iterate through each 'i' **
    for i in range(n_samples - 1):
      # For each i, the corresponding j's are always an increasing sequence
      j_indices_for_i = np.arange(i + 1, n_samples)
      if len(j_indices_for_i) == 0:
        continue

      # Load the data for 'i' once
      jac_i = jax.device_put(jacobians_dset[i])  # pyright: ignore[reportIndexIssue]
      seq_i = jax.device_put(sequences_dset[i])  # pyright: ignore[reportIndexIssue]
      w_i = jax.device_put(weights[i])

      # Process the corresponding 'j's in batches to manage memory
      for j_start in range(0, len(j_indices_for_i), batch_size):
        j_end = j_start + batch_size
        j_idx_batch = j_indices_for_i[j_start:j_end]

        # This read is now guaranteed to be sorted, fixing the h5py error
        jac_batch_j = jax.device_put(jacobians_dset[j_idx_batch])  # pyright: ignore[reportIndexIssue]
        seq_batch_j = jax.device_put(sequences_dset[j_idx_batch])  # pyright: ignore[reportIndexIssue]

        # Prepare batch for vmap by repeating the 'i' data
        current_batch_size = len(j_idx_batch)
        jac_batch_i = jnp.repeat(jac_i[None, ...], current_batch_size, axis=0)
        seq_batch_i = jnp.repeat(seq_i[None, ...], current_batch_size, axis=0)
        w_batch_i = jnp.repeat(w_i, current_batch_size, axis=0)

        # Compute combinations for the batch
        combined_block, mapping_block = _vmapped_process_pairs(
          jac_batch_i,
          jac_batch_j,
          seq_batch_i,
          seq_batch_j,
          w_batch_i,
        )

        # Write results to the correct slice in the output file
        write_end_idx = write_idx + current_batch_size
        dset[write_idx:write_end_idx] = np.array(combined_block)
        mapping_dset[write_idx:write_end_idx] = np.array(mapping_block)

        write_idx = write_end_idx
        f.flush()
