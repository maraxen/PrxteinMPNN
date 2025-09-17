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
  jax.Array,  # Final combined Jacobians, shape (N*(N-1)/2, ...)
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
  ) -> jax.Array:
    """Compute cross-protein Jacobian combinations for all pairs."""
    if weights is None:
      weights = jnp.ones((jacobians.shape[0],), dtype=jacobians.dtype)
    n_proteins = jacobians.shape[0]
    if n_proteins < _MIN_NUMBER_PROTEINS:
      return jnp.empty((0, *jacobians.shape[1:]), dtype=jacobians.dtype)

    i_indices, j_indices = jnp.triu_indices(n_proteins, k=1)

    jac_i = jacobians[i_indices]
    jac_j = jacobians[j_indices]
    pair_weights = weights[i_indices]

    # align_sequences expects a batch of sequences and returns mappings for triu pairs
    all_mappings = align_sequences(sequences)

    xs = (jac_i, jac_j, all_mappings, pair_weights)

    return jax.lax.map(
      lambda x: _combine_fn(*x),
      xs,
      batch_size=batch_size,
    )

  return combine_jacobians


def combine_jacobians_h5_stream(
  h5_path: str | Path,
  combine_fn: CombineCatJacPairFn,
  fn_kwargs: dict,
  batch_size: int,
  weights: jax.Array,
) -> None:
  """Combine all pairs of Jacobians from an HDF5 file and save to the same file.

  This function efficiently streams data from and to the HDF5 file without loading
  all data into memory at once. It processes data in batches and writes results
  incrementally.

  Args:
    h5_path (str | Path): Path to the input HDF5 file.
    combine_fn (CombineCatJacPairFn): Function to combine Jacobian pairs.
    fn_kwargs (dict): Additional keyword arguments for combine_fn.
    batch_size (int): Batch size for processing blocks of pairs.
    weights (jax.Array): Per-protein weights, shape (N,).

  Raises:
    ValueError: If input arrays have mismatched lengths.

  """
  with h5py.File(str(h5_path), "a") as f:
    if "combined_catjac" in f:
      del f["combined_catjac"]

    # Get dataset references without loading data
    jacobians_dset = f["categorical_jacobians"]
    sequences_dset = f["one_hot_sequences"]

    n_samples = jacobians_dset.shape[0]  # type: ignore[reportAttributeAccessIssue]
    if n_samples != sequences_dset.shape[0]:  # type: ignore[reportAttributeAccessIssue]
      msg = "Jacobian and sequence arrays must have the same length."
      raise ValueError(msg)
    if n_samples != weights.shape[0]:
      msg = "Weights array must match number of samples."
      raise ValueError(msg)

    if n_samples == 0:
      f.create_dataset("combined_catjac", shape=(0,), dtype="float32")
      return

    _combine_fn_with_kwargs = partial(combine_fn, **(fn_kwargs or {}))

    def _process_pair(
      jac_i: CategoricalJacobian,
      jac_j: CategoricalJacobian,
      seq_i: OneHotProteinSequence,
      seq_j: OneHotProteinSequence,
      w_i: jax.Array,
    ) -> CategoricalJacobian:
      mapping = align_sequences(jnp.stack([seq_i, seq_j]))[0]
      return _combine_fn_with_kwargs(jac_i, jac_j, mapping, w_i)

    _vmapped_j = jax.vmap(_process_pair, in_axes=(None, 0, None, 0, None))
    _vmapped_i_j = jax.jit(jax.vmap(_vmapped_j, in_axes=(0, None, 0, None, 0)))

    out_shape = jacobians_dset.shape[1:]  # type: ignore[reportAttributeAccessIssue]
    total_pairs = n_samples * n_samples

    # Create output dataset with chunking for efficient streaming
    chunk_size = min(total_pairs, 1000)
    dset = f.create_dataset(
      "combined_catjac",
      shape=(total_pairs, *out_shape),
      dtype="float32",
      chunks=(chunk_size, *out_shape),
      compression="gzip",
      compression_opts=1,
    )

    idx = 0
    for i in range(0, n_samples, batch_size):
      i_end = min(i + batch_size, n_samples)

      # Stream jacobians and sequences for batch i
      jac_batch_i = jax.device_put(jacobians_dset[i:i_end])  # type: ignore[reportIndexIssue]
      seq_batch_i = jax.device_put(sequences_dset[i:i_end])  # type: ignore[reportIndexIssue]
      w_batch_i = jax.device_put(weights[i:i_end])

      for j in range(0, n_samples, batch_size):
        j_end = min(j + batch_size, n_samples)

        if (i_end - i == 0) or (j_end - j == 0):
          continue

        # Stream jacobians and sequences for batch j
        jac_batch_j = jax.device_put(jacobians_dset[j:j_end])  # type: ignore[reportIndexIssue]
        seq_batch_j = jax.device_put(sequences_dset[j:j_end])  # type: ignore[reportIndexIssue]

        # Compute all combinations between the two batches
        combined_block = _vmapped_i_j(
          jac_batch_i,
          jac_batch_j,
          seq_batch_i,
          seq_batch_j,
          w_batch_i,
        )

        n_pairs = (i_end - i) * (j_end - j)
        combined_flat = combined_block.reshape((n_pairs, *out_shape))

        # Stream write to HDF5
        dset[idx : idx + n_pairs] = np.array(combined_flat)

        # Force flush to disk to free memory
        f.flush()

        idx += n_pairs
