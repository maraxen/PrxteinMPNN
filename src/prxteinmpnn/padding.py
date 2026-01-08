"""Padding utilities for static-shape JAX compilation in PrxteinMPNN.

Provides bucketed shapes for protein length to avoid XLA recompilation
when processing proteins of different sizes.

Bucketing Strategy:
    - LENGTH_BUCKETS: [100, 200, 400, 800, 1200]

All input proteins are padded to bucket ceilings, with masks used to exclude
padded positions from model computations.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

# Protein length buckets (designed for typical protein sizes)
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 400, 800, 1200)


def get_length_bucket(n_residues: int) -> int:
  """Find the smallest bucket that can fit the protein length.

  Args:
      n_residues: Actual number of residues.

  Returns:
      The bucket size to use.

  Raises:
      ValueError: If length exceeds all buckets.

  """
  for bucket in LENGTH_BUCKETS:
    if n_residues <= bucket:
      return bucket
  raise ValueError(f"Protein length {n_residues} exceeds all buckets {LENGTH_BUCKETS}")


def pad_to_bucket(
  coordinates: Array,
  sequence: Array,
  mask: Array,
  target_length: int,
) -> tuple[Array, Array, Array]:
  """Pad protein data to the target bucket length.

  Args:
      coordinates: Backbone coordinates of shape (N, 4, 3) or (N, 3).
      sequence: Sequence indices or one-hot of shape (N,) or (N, 21).
      mask: Valid residue mask of shape (N,).
      target_length: Target padded length.

  Returns:
      Tuple of (padded_coordinates, padded_sequence, padded_mask).

  """
  real_n = coordinates.shape[0]
  n_pad = target_length - real_n

  # Pad coordinates
  if coordinates.ndim == 2:
    # Shape: (N, 3)
    padded_coords = jnp.pad(coordinates, ((0, n_pad), (0, 0)), constant_values=0.0)
  else:
    # Shape: (N, 4, 3) for backbone atoms
    padded_coords = jnp.pad(coordinates, ((0, n_pad), (0, 0), (0, 0)), constant_values=0.0)

  # Pad sequence
  if sequence.ndim == 1:
    # Shape: (N,) - indices
    padded_seq = jnp.pad(sequence, (0, n_pad), constant_values=0)
  else:
    # Shape: (N, 21) - one-hot
    padded_seq = jnp.pad(sequence, ((0, n_pad), (0, 0)), constant_values=0.0)

  # Pad mask (padded positions are False)
  padded_mask = jnp.pad(mask, (0, n_pad), constant_values=False)

  return padded_coords, padded_seq, padded_mask


def create_residue_mask(real_n: int, padded_n: int) -> Bool[Array, " padded_n"]:
  """Create a mask for valid residue positions.

  Args:
      real_n: Actual number of residues.
      padded_n: Padded number of residues.

  Returns:
      Boolean mask of shape (padded_n,), True for valid positions.

  """
  return jnp.arange(padded_n) < real_n


def masked_mean(values: Array, mask: Array) -> Float[Array, ""]:
  """Compute mean over valid (masked) elements.

  Args:
      values: Array of values.
      mask: Boolean mask (True for valid elements).

  Returns:
      Mean over valid elements.

  """
  return jnp.sum(values * mask) / jnp.sum(mask)
