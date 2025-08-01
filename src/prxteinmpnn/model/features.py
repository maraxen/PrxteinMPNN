"""Feature extraction for protein structures in the PrxteinMPNN model."""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray

from prxteinmpnn.utils.coordinates import (
  apply_noise_to_coordinates,
  compute_backbone_coordinates,
  compute_backbone_distance,
)
from prxteinmpnn.utils.graph import NeighborOffsets, compute_neighbor_offsets
from prxteinmpnn.utils.normalize import layer_normalization
from prxteinmpnn.utils.radial_basis import compute_radial_basis
from prxteinmpnn.utils.types import (
  AtomChainIndex,
  AtomMask,
  AtomResidueIndex,
  EdgeFeatures,
  ModelParameters,
  NeighborIndices,
  StructureAtomicCoordinates,
)

EdgeChainNeighbors = Int[Array, "num_atoms num_neighbors"]
EncodedPositions = Int[Array, "num_atoms num_neighbors (2 * MAXIMUM_RELATIVE_FEATURES + 2)"]

MAXIMUM_RELATIVE_FEATURES = 32


@jax.jit
def get_edge_chains_neighbors(
  chain_indices: AtomChainIndex,
  neighbor_indices: NeighborIndices,
) -> EdgeChainNeighbors:
  """Compute edge chains for neighbors."""
  edge_chains = (chain_indices[:, None] == chain_indices[None, :]).astype(int)
  return jnp.take_along_axis(edge_chains, neighbor_indices, axis=1)


@jax.jit
def encode_positions(
  neighbor_offsets: NeighborOffsets,
  edge_chains_neighbors: EdgeChainNeighbors,
  model_parameters: ModelParameters,
) -> EncodedPositions:
  """Encode positions based on neighbor offsets and edge chains."""
  neighbor_offset_factor = jnp.clip(
    neighbor_offsets + MAXIMUM_RELATIVE_FEATURES,
    0,
    2 * MAXIMUM_RELATIVE_FEATURES,
  )
  edge_chain_factor = (1 - edge_chains_neighbors) * (2 * MAXIMUM_RELATIVE_FEATURES + 1)
  encoded_offset = neighbor_offset_factor * edge_chains_neighbors + edge_chain_factor
  encoded_offset_one_hot = jax.nn.one_hot(encoded_offset, 2 * MAXIMUM_RELATIVE_FEATURES + 2)
  pos_enc_params = model_parameters[
    "protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear"
  ]
  return jnp.dot(encoded_offset_one_hot, pos_enc_params["w"]) + pos_enc_params["b"]


@jax.jit
def embed_edges(
  edge_features: EncodedPositions,
  model_parameters: ModelParameters,
) -> EdgeFeatures:
  """Embed edge features using model parameters."""
  edge_emb_params = model_parameters["protein_mpnn/~/protein_features/~/edge_embedding"]
  return jnp.dot(edge_features, edge_emb_params["w"]) + edge_emb_params["b"]


@partial(jax.jit, static_argnames=("k_neighbors", "augment_eps"))
def extract_features(
  structure_coordinates: StructureAtomicCoordinates,
  mask: AtomMask,
  residue_indices: AtomResidueIndex,
  chain_indices: AtomChainIndex,
  mpnn_parameters: ModelParameters,
  prng_key: PRNGKeyArray,
  k_neighbors: int = 48,
  augment_eps: float = 0.0,
) -> tuple[EdgeFeatures, NeighborIndices]:
  """Extract features from protein structure coordinates.

  Args:
    structure_coordinates: Atomic coordinates of the protein structure.
    mask: Mask indicating valid atoms in the structure.
    residue_indices: Residue indices for each atom.
    chain_indices: Chain indices for each atom.
    mpnn_parameters: Model parameters for the feature extraction.
    prng_key: JAX random key for stochastic operations.
    k_neighbors: Maximum number of neighbors to consider for each atom.
    augment_eps: Standard deviation for Gaussian noise augmentation.

  Returns:
    edge_features: Edge features after concatenation and normalization.
    edge_indices: Indices of neighboring atoms.

  """
  noised_coordinates = apply_noise_to_coordinates(
    structure_coordinates,
    prng_key,
    augment_eps=augment_eps,
  )
  backbone_atom_coordinates = compute_backbone_coordinates(noised_coordinates)
  distances = compute_backbone_distance(backbone_atom_coordinates)
  distances_masked = jnp.array(
    jnp.where(
      (mask[:, None] * mask[None, :]).astype(bool),
      distances[..., 1],
      jnp.inf,
    ),
  )
  k = min(k_neighbors, structure_coordinates.shape[0])
  _, neighbor_indices = jax.lax.top_k(-distances_masked, k)
  rbf = compute_radial_basis(backbone_atom_coordinates, neighbor_indices)
  neighbor_offsets = compute_neighbor_offsets(residue_indices, neighbor_indices)
  edge_chains_neighbors = get_edge_chains_neighbors(
    chain_indices,
    neighbor_indices,
  )
  encoded_positions = encode_positions(
    neighbor_offsets,
    edge_chains_neighbors,
    mpnn_parameters,
  )
  edges = jnp.concatenate([encoded_positions, rbf], axis=-1)
  edge_features = embed_edges(edges, mpnn_parameters)
  norm_edge_params = mpnn_parameters["protein_mpnn/~/protein_features/~/norm_edges"]
  edge_features = layer_normalization(edge_features, norm_edge_params)
  return edge_features, neighbor_indices


@jax.jit
def project_features(mpnn_parameters: ModelParameters, edge_features: EdgeFeatures) -> EdgeFeatures:
  """Project edge features using model parameters."""
  w_e, b_e = mpnn_parameters["protein_mpnn/~/W_e"]["w"], mpnn_parameters["protein_mpnn/~/W_e"]["b"]
  return jnp.dot(edge_features, w_e) + b_e
