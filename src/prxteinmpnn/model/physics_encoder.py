"""PhysicsMPNN: ProteinMPNN with physics-based electrostatic node features.

This module provides a lightweight wrapper around the standard ProteinMPNN model
that incorporates physics-based electrostatic features as initial node representations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:
  from prxteinmpnn.model.encoder import Encoder
  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    EdgeFeatures,
    NeighborIndices,
    NodeFeatures,
  )


class PhysicsEncoder(eqx.Module):
  """Encoder with optional initial electrostatic node features.

  Wraps a standard Encoder to optionally accept precomputed physics-based features
  as initial node representations. If no initial features are provided, behaves
  identically to the standard encoder (initializing nodes to zeros).

  This allows using the same model weights with or without physics features.

  """

  base_encoder: Encoder
  use_initial_features: bool = eqx.field(static=True)

  def __init__(self, base_encoder: Encoder, *, use_initial_features: bool = False) -> None:
    """Initialize PhysicsEncoder.

    Args:
        base_encoder: Standard ProteinMPNN encoder to wrap
        use_initial_features: If True, accept initial_node_features parameter.
          If False, ignore initial features and use zeros (standard behavior).

    """
    self.base_encoder = base_encoder
    self.use_initial_features = use_initial_features

  def __call__(
    self,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    initial_node_features: NodeFeatures | None = None,
  ) -> tuple[NodeFeatures, EdgeFeatures]:
    """Forward pass with optional initial node features.

    Args:
        edge_features: Edge features from structure
        neighbor_indices: Neighbor connectivity
        mask: Valid residue mask
        initial_node_features: Optional precomputed physics features, shape (n_residues, 5).
          If provided and use_initial_features=True, these will be projected to the
          node feature dimension and used as initial node representations.

    Returns:
        Tuple of (node_features, edge_features) after encoding

    """
    n_residues = edge_features.shape[0]
    node_dim = self.base_encoder.node_feature_dim

    if self.use_initial_features and initial_node_features is not None:
      feat_dim = initial_node_features.shape[-1]
      if feat_dim < node_dim:
        repeats = node_dim // feat_dim
        remainder = node_dim % feat_dim
        repeated = jnp.tile(initial_node_features, (1, repeats))
        if remainder > 0:
          padding = initial_node_features[:, :remainder]
          node_features = jnp.concatenate([repeated, padding], axis=-1)
        else:
          node_features = repeated
      elif feat_dim > node_dim:
        node_features = initial_node_features[:, :node_dim]
      else:
        node_features = initial_node_features
    else:
      node_features = jnp.zeros((n_residues, node_dim))

    mask_2d = mask[:, None] * mask[None, :]
    mask_attend = jnp.take_along_axis(mask_2d, neighbor_indices, axis=1)

    for layer in self.base_encoder.layers:
      node_features, edge_features = layer(
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        mask_attend,
      )

    return node_features, edge_features


def create_physics_encoder(
  base_encoder: Encoder,
  *,
  use_initial_features: bool = True,
) -> PhysicsEncoder:
  """Create a PhysicsEncoder from an existing Encoder using model surgery.

  Args:
      base_encoder: Existing ProteinMPNN encoder
      use_initial_features: Whether to use physics features (default: True)

  Returns:
      PhysicsEncoder that wraps the base encoder

  Example:
      >>> # Load pretrained model
      >>> model = load_proteinmpnn_model("path/to/checkpoint.pkl")
      >>> # Create physics-enhanced encoder
      >>> physics_encoder = create_physics_encoder(
      ...     model.encoder,
      ...     use_initial_features=True
      ... )
      >>> # Replace encoder in model using equinox.tree_at
      >>> physics_model = eqx.tree_at(
      ...     lambda m: m.encoder,
      ...     model,
      ...     physics_encoder
      ... )

  """
  return PhysicsEncoder(base_encoder, use_initial_features=use_initial_features)
