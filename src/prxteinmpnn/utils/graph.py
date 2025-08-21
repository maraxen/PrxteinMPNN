"""Graph utilities for the PrxteinMPNN model.

prxteinmpnn.utils.graph
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

from prxteinmpnn.utils.types import (
  NeighborIndices,
  ResidueIndex,
)

NeighborOffsets = Int[Array, "num_residues num_neighbors"]


@jax.jit
def compute_neighbor_offsets(
  residue_indices: ResidueIndex,
  neighbor_indices: NeighborIndices,
) -> jax.Array:
  """Compute offsets between residues for neighbor indices.

  Args:
    residue_indices: Residue indices for each atom.
    neighbor_indices: Indices of neighboring atoms.

  Returns:
    A 2D array of offsets where each row corresponds to a residue and each column
    corresponds to a neighbor. The values represent the difference in residue indices.

  """
  offset = residue_indices[:, None] - residue_indices[None, :]
  return jnp.take_along_axis(offset, neighbor_indices, axis=1)
