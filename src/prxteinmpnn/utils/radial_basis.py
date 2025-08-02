"""Radial basis functions for distance encoding.

prxteinmpnn.utils.radial_basis
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .types import AtomIndexPair, BackboneCoordinates, NeighborIndices

AllAtomRBF = Float[Array, "L K R D"]
AtomPairRBF = Float[Array, "K R D"]

RADIAL_BASES = 16
RADIAL_BASE_MINIMUM, RADIAL_BASE_MAXIMUM = 2.0, 22.0
RBF_CENTERS = jnp.linspace(RADIAL_BASE_MINIMUM, RADIAL_BASE_MAXIMUM, RADIAL_BASES)
RBF_SIGMA = (RADIAL_BASE_MAXIMUM - RADIAL_BASE_MINIMUM) / RADIAL_BASES

BACKBONE_PAIRS = jnp.array(
  [
    [1, 1],
    [0, 0],
    [2, 2],
    [3, 3],
    [4, 4],
    [1, 0],
    [1, 2],
    [1, 3],
    [1, 4],
    [0, 2],
    [0, 3],
    [0, 4],
    [4, 2],
    [4, 3],
    [3, 2],
    [0, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [2, 0],
    [3, 0],
    [4, 0],
    [2, 4],
    [3, 4],
    [2, 3],
  ],
)

DISTANCE_EPSILON = 1e-6


@jax.jit
def compute_radial_basis(
  backbone_coordinates: BackboneCoordinates,
  neighbor_indices: NeighborIndices,
) -> AllAtomRBF:
  """Compute the radial basis functions for backbone coordinates."""

  def _rbf(pair: AtomIndexPair, neighbor_indices: NeighborIndices) -> AtomPairRBF:
    """Compute the radial basis function for a given pair of atoms."""
    atom1, atom2 = backbone_coordinates[:, pair[0], :], backbone_coordinates[:, pair[1], :]
    delta_coords = atom1[:, None, :] - atom2[None, :, :]
    distance_sq = jnp.sum(jnp.square(delta_coords), axis=-1)
    distance = jnp.sqrt(DISTANCE_EPSILON + distance_sq)
    neighbor_distances = jnp.take_along_axis(distance, neighbor_indices, axis=1)
    return jnp.exp(
      -(jnp.square((neighbor_distances[..., None] - RBF_CENTERS) / RBF_SIGMA)),
    )

  return (
    jax.vmap(lambda pair: _rbf(pair, neighbor_indices))(BACKBONE_PAIRS)
    .transpose((1, 2, 0, 3))
    .reshape(
      backbone_coordinates.shape[0],
      neighbor_indices.shape[1],
      -1,
    )
  )
