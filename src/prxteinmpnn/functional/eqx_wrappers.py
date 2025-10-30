"""Functional wrappers for Equinox modules.

This module provides backward-compatible functional interfaces that internally
use Equinox modules. This allows us to maintain a single implementation while
supporting both functional and object-oriented APIs.

Example:
  >>> from prxteinmpnn.functional import get_functional_model, make_encoder_eqx
  >>> params = get_functional_model()
  >>> encoder_fn = make_encoder_eqx(params, num_encoder_layers=3)
  >>> nodes, edges = encoder_fn(edge_features, neighbor_indices, mask)

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

from prxteinmpnn import conversion

if TYPE_CHECKING:
  from collections.abc import Callable

  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    EdgeFeatures,
    ModelParameters,
    NeighborIndices,
    NodeFeatures,
  )


def make_encoder_eqx(
  model_parameters: ModelParameters,
  num_encoder_layers: int = 3,
  scale: float = 30.0,
  *,
  key: jax.Array | None = None,
) -> Callable[[EdgeFeatures, NeighborIndices, AlphaCarbonMask], tuple[NodeFeatures, EdgeFeatures]]:
  """Create encoder function using Equinox implementation.

  This is a drop-in replacement for the legacy `make_encoder()` that uses the
  Equinox implementation internally. It provides the same functional interface
  while using the unified Equinox codebase.

  Args:
    model_parameters: Model parameters PyTree.
    num_encoder_layers: Number of encoder layers (default: 3).
    scale: Scaling factor for message aggregation (default: 30.0).
    key: PRNG key for initialization (optional, will create one if not provided).

  Returns:
    A function that takes (edge_features, neighbor_indices, mask) and returns
    (node_features, edge_features).

  Example:
    >>> import jax
    >>> from prxteinmpnn.functional import get_functional_model
    >>> params = get_functional_model()
    >>> encoder_fn = make_encoder_eqx(params, num_encoder_layers=3)
    >>> nodes, edges = encoder_fn(edge_features, neighbor_indices, mask)

  """
  # Create Equinox encoder module
  if key is None:
    key = jax.random.PRNGKey(0)

  encoder = conversion.create_encoder(
    model_parameters,
    num_layers=num_encoder_layers,
    scale=scale,
    key=key,
  )

  # Return functional interface
  @jax.jit
  def encoder_fn(
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
  ) -> tuple[NodeFeatures, EdgeFeatures]:
    """Encode edge features into node and edge representations.

    Args:
      edge_features: Edge features (num_atoms, num_neighbors, edge_dim).
      neighbor_indices: Neighbor indices (num_atoms, num_neighbors).
      mask: Atom mask (num_atoms,).

    Returns:
      Tuple of (node_features, edge_features).

    """
    return encoder(edge_features, neighbor_indices, mask)

  return encoder_fn


def make_decoder_eqx(
  model_parameters: ModelParameters,
  num_decoder_layers: int = 3,
  scale: float = 30.0,
  *,
  key: jax.Array | None = None,
) -> Callable[[NodeFeatures, EdgeFeatures, AlphaCarbonMask], NodeFeatures]:
  """Create decoder function using Equinox implementation.

  This is a drop-in replacement for the legacy `make_decoder()` that uses the
  Equinox implementation internally. It provides the same functional interface
  while using the unified Equinox codebase.

  Args:
    model_parameters: Model parameters PyTree.
    num_decoder_layers: Number of decoder layers (default: 3).
    scale: Scaling factor for message aggregation (default: 30.0).
    key: PRNG key for initialization (optional, will create one if not provided).

  Returns:
    A function that takes (node_features, edge_features, mask) and returns
    updated node_features.

  Example:
    >>> import jax
    >>> from prxteinmpnn.functional import get_functional_model
    >>> params = get_functional_model()
    >>> decoder_fn = make_decoder_eqx(params, num_decoder_layers=3)
    >>> updated_nodes = decoder_fn(node_features, edge_features, mask)

  """
  # Create Equinox decoder module
  if key is None:
    key = jax.random.PRNGKey(0)

  decoder = conversion.create_decoder(
    model_parameters,
    num_layers=num_decoder_layers,
    scale=scale,
    key=key,
  )

  # Return functional interface
  @jax.jit
  def decoder_fn(
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    mask: AlphaCarbonMask,
  ) -> NodeFeatures:
    """Decode node and edge features.

    Args:
      node_features: Node features (num_atoms, node_dim).
      edge_features: Edge features (num_atoms, num_neighbors, edge_dim).
      mask: Atom mask (num_atoms,).

    Returns:
      Updated node features (num_atoms, node_dim).

    """
    return decoder(node_features, edge_features, mask)

  return decoder_fn


def make_model_eqx(
  model_parameters: ModelParameters,
  num_encoder_layers: int = 3,
  num_decoder_layers: int = 3,
  scale: float = 30.0,
  *,
  key: jax.Array | None = None,
) -> Callable[[EdgeFeatures, NeighborIndices, AlphaCarbonMask], jax.Array]:
  """Create full model function using Equinox implementation.

  This creates a complete ProteinMPNN model that goes from edge features to
  amino acid logits in a single function call.

  Args:
    model_parameters: Model parameters PyTree.
    num_encoder_layers: Number of encoder layers (default: 3).
    num_decoder_layers: Number of decoder layers (default: 3).
    scale: Scaling factor for message aggregation (default: 30.0).
    key: PRNG key for initialization (optional, will create one if not provided).

  Returns:
    A function that takes (edge_features, neighbor_indices, mask) and returns
    amino acid logits.

  Example:
    >>> import jax
    >>> from prxteinmpnn.functional import get_functional_model
    >>> params = get_functional_model()
    >>> model_fn = make_model_eqx(params)
    >>> logits = model_fn(edge_features, neighbor_indices, mask)

  """
  # Create Equinox full model
  if key is None:
    key = jax.random.PRNGKey(0)

  model = conversion.create_prxteinmpnn(
    model_parameters,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    scale=scale,
    key=key,
  )

  # Return functional interface
  @jax.jit
  def model_fn(
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
  ) -> jax.Array:
    """Run full ProteinMPNN model.

    Args:
      edge_features: Edge features (num_atoms, num_neighbors, edge_dim).
      neighbor_indices: Neighbor indices (num_atoms, num_neighbors).
      mask: Atom mask (num_atoms,).

    Returns:
      Amino acid logits (num_atoms, 21).

    """
    return model(edge_features, neighbor_indices, mask)

  return model_fn
