"""Utilities for computing categorical Jacobians."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

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


def _combine_mapped_jacobians(
  jac1: CategoricalJacobian,
  jac2: CategoricalJacobian,
  mapping: jax.Array,
  weights: jax.Array | None = None,
  batch_size: int = 1,
) -> CategoricalJacobian:
  """Add jac1 to the mapped version of jac2, weighted by weights."""
  mapped_jac2 = _gather_mapped_jacobian(jac2, mapping)

  weights = (
    jnp.ones((2,), dtype=jac1.dtype) if weights is None else jnp.array(weights, dtype=jac1.dtype)
  )

  def combine_slice(jac1_slice: jax.Array) -> jax.Array:
    return (jac1_slice * weights[0]) + (mapped_jac2 * weights[1])

  combined = jax.lax.map(combine_slice, jac1, batch_size=batch_size)

  noise_levels = jac1.shape[0]
  return combined.reshape(noise_levels * noise_levels, *combined.shape[2:])


_MIN_NUMBER_PROTEINS = 2


def make_combine_jac(
  combine_fn: CombineCatJacPairFn | None = None,
  fn_kwargs: dict[str, Any] | None = None,
  batch_size: int = 1,
  combine_noise_batch_size: int = 1,
) -> CombineCatJacFn:
  """Create a function to combine Jacobians using a specified operation."""
  if combine_fn is None:
    _combine_fn = _combine_mapped_jacobians
  else:
    _combine_fn = partial(combine_fn, **(fn_kwargs or {}))

  _combine_fn = partial(_combine_fn, batch_size=combine_noise_batch_size)

  def combine_jacobians(
    jacobians: jax.Array,
    sequences: ProteinSequence | OneHotProteinSequence,
    weights: jax.Array | None = None,
  ) -> tuple[CategoricalJacobian, jax.Array]:
    """Compute cross-protein Jacobian combinations for all pairs.

    Args:
      jacobians: Shape (N, K, L, A, L, A) for N proteins.
      sequences: Shape (N, L) or (N, L, A) for N proteins.
      weights: The weights to apply. Can be one of:
        - None: Defaults to addition ([1.0, 1.0]) for all pairs.
        - Scalar: Applied to every protein (e.g., w -> [w, w] for each pair).
        - 1D Array (shape N,): Per-protein weights [w_0, w_1, ...].

    Returns:
      A tuple containing:
        - Combined Jacobians, shape (num_pairs, K*K, L, A, L, A).
        - Mappings, shape (num_pairs, L, 2).

    """
    n_proteins = jacobians.shape[0]
    if n_proteins < _MIN_NUMBER_PROTEINS:
      return jnp.empty((0, *jacobians.shape[1:])), jnp.empty((0, 2), dtype=jnp.int32)

    i_indices, j_indices = jnp.triu_indices(n_proteins, k=1)
    num_pairs = len(i_indices)

    if weights is None:
      combination_weights = jnp.ones((num_pairs, 2), dtype=jacobians.dtype)
    else:
      weights = jnp.asarray(weights, dtype=jacobians.dtype)
      if weights.ndim == 0:
        per_protein = jnp.full((n_proteins,), weights)
        combination_weights = jnp.stack(
          [per_protein[i_indices], per_protein[j_indices]],
          axis=-1,
        )
      elif weights.ndim == 1:
        if weights.shape[0] != n_proteins:
          msg = f"Invalid weights shape {weights.shape}, must be (N,) where N={n_proteins}."
          raise ValueError(msg)
        combination_weights = jnp.stack(
          [weights[i_indices], weights[j_indices]],
          axis=-1,
        )
      else:
        msg = f"Invalid weights shape {weights.shape}, must be scalar or (N,)."
        raise ValueError(msg)

    jac_i = jacobians[i_indices]
    jac_j = jacobians[j_indices]
    all_mappings = align_sequences(sequences)

    xs = (jac_i, jac_j, all_mappings, combination_weights)

    combined_jacs = jax.lax.map(
      lambda x: _combine_fn(*x),
      xs,
      batch_size=batch_size,
    )

    return combined_jacs, all_mappings

  return combine_jacobians


def combine_jacobians_h5_stream(
  h5_path: str | Path,
  combine_fn: CombineCatJacPairFn | None = None,
  fn_kwargs: dict | None = None,
  batch_size: int = 1,
  combine_noise_batch_size: int = 1,
  weights: jax.Array | None = None,
) -> None:
  """Combine all pairs of Jacobians from an HDF5 file and save to the same file."""
  with h5py.File(str(h5_path), "a") as f:
    if "combined_catjac" in f:
      del f["combined_catjac"]
    if "mappings" in f:
      del f["mappings"]

    jacobians_dset = cast("np.ndarray", f["categorical_jacobians"])
    sequences_dset = cast("np.ndarray", f["one_hot_sequences"])

    n_samples = jacobians_dset.shape[0]
    if n_samples != sequences_dset.shape[0]:
      msg = "Jacobian and sequence arrays must have the same length."
      raise ValueError(msg)

    i_indices_all, j_indices_all = jnp.triu_indices(n_samples, k=1)
    total_pairs = len(i_indices_all)

    if total_pairs == 0:
      f.create_dataset("combined_catjac", shape=(0, *jacobians_dset.shape[1:]), dtype="float32")
      f.create_dataset("mappings", shape=(0, sequences_dset.shape[1], 2), dtype="int32")
    _combine_fn = _combine_mapped_jacobians if combine_fn is None else combine_fn
    _combine_fn = partial(_combine_fn, **(fn_kwargs or {}))
    _combine_fn = partial(_combine_fn, batch_size=combine_noise_batch_size)

    @jax.jit
    @jax.vmap
    def _process_pair(
      jac_i: CategoricalJacobian,
      jac_j: CategoricalJacobian,
      seq_i: OneHotProteinSequence,
      seq_j: OneHotProteinSequence,
      pair_weights: jax.Array,
    ) -> tuple[CategoricalJacobian, jax.Array]:
      mapping = align_sequences(jnp.stack([seq_i, seq_j]))[0]
      return _combine_fn(jac_i, jac_j, mapping, pair_weights), mapping

    out_shape = (jacobians_dset.shape[1] ** 2, *jacobians_dset.shape[2:])  # pyright: ignore[reportAttributeAccessIssue]
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

    if weights is None:
      weights = jnp.ones(n_samples)

    for start_idx in range(0, total_pairs, batch_size):
      end_idx = min(start_idx + batch_size, total_pairs)
      if start_idx >= end_idx:
        continue

      i_idx_batch = i_indices_all[start_idx:end_idx]
      j_idx_batch = j_indices_all[start_idx:end_idx]

      unique_indices, inverse_indices = np.unique(
        np.concatenate([i_idx_batch, j_idx_batch]),
        return_inverse=True,
      )
      inv_i, inv_j = np.split(inverse_indices, 2)

      jac_data = jax.device_put(jacobians_dset[unique_indices])
      seq_data = jax.device_put(sequences_dset[unique_indices])
      w_data = jax.device_put(weights[unique_indices])

      jac_batch_i, jac_batch_j = jac_data[inv_i], jac_data[inv_j]
      seq_batch_i, seq_batch_j = seq_data[inv_i], seq_data[inv_j]

      w_batch_i, w_batch_j = w_data[inv_i], w_data[inv_j]
      combination_weights_batch = jnp.stack([w_batch_i, w_batch_j], axis=-1)

      combined_block, mapping_block = _process_pair(
        jac_batch_i,
        jac_batch_j,
        seq_batch_i,
        seq_batch_j,
        combination_weights_batch,
      )

      dset[start_idx:end_idx] = np.array(combined_block)
      mapping_dset[start_idx:end_idx] = np.array(mapping_block)
      f.flush()
