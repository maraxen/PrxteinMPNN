"""Main ProteinMPNN model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from prxteinmpnn.model.capabilities import (
  PRXTEIN_MPNN_CAPABILITIES,
  ModelCapabilities,
)
from prxteinmpnn.model.decoder import Decoder
from prxteinmpnn.model.encoder import (
  Encoder,
  PhysicsEncoder,
)
from prxteinmpnn.model.features import ProteinFeatures

from prxteinmpnn.types.bundles import InferenceBundle
from prxteinmpnn.types.configs import InferenceConfig

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import PRNGKeyArray

class PrxteinMPNN(eqx.Module):
  """The complete end-to-end ProteinMPNN model."""

  features: ProteinFeatures
  encoder: Encoder | PhysicsEncoder
  decoder: Decoder

  w_s_embed: eqx.nn.Embedding
  w_out: eqx.nn.Linear

  node_features_dim: int = eqx.field(static=True)
  edge_features_dim: int = eqx.field(static=True)
  num_decoder_layers: int = eqx.field(static=True)
  capabilities: ModelCapabilities = eqx.field(static=True, default=PRXTEIN_MPNN_CAPABILITIES)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    k_neighbors: int,
    num_positional_embeddings: int = 16,
    physics_feature_dim: int | None = None,
    num_amino_acids: int = 21,
    vocab_size: int = 21,
    dropout_rate: float = 0.1,
    *,
    key: PRNGKeyArray,
  ) -> None:
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features
    self.num_decoder_layers = num_decoder_layers

    keys = jax.random.split(key, 5)

    self.features = ProteinFeatures(
      node_features,
      edge_features,
      k_neighbors,
      num_positional_embeddings=num_positional_embeddings,
      key=keys[0],
    )
    self.encoder = (
      Encoder(
        node_features,
        edge_features,
        hidden_features,
        num_encoder_layers,
        dropout_rate=dropout_rate,
        key=keys[1],
      )
      if physics_feature_dim is None
      else PhysicsEncoder(
        node_features,
        edge_features,
        hidden_features,
        num_encoder_layers,
        dropout_rate,
        physics_feature_dim,
        key=keys[1],
      )
    )
    self.decoder = Decoder(
      node_features,
      edge_features,
      hidden_features,
      num_decoder_layers,
      dropout_rate=dropout_rate,
      key=keys[2],
    )
    self.w_s_embed = eqx.nn.Embedding(
      num_embeddings=vocab_size,
      embedding_size=node_features,
      key=keys[3],
    )
    self.w_out = eqx.nn.Linear(node_features, num_amino_acids, key=keys[4])

  def __call__(
    self,
    prng_key: PRNGKeyArray,
    bundle: InferenceBundle,
    config: InferenceConfig,
    **kwargs: Any,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Encoder-only entry point using structured bundles.
    
    Returns:
        3-tuple of (node_features, edge_features, edge_indices).
    """
    # 1. Featurize
    # Extract from bundle
    geo = bundle.geometry
    
    edge_features, edge_indices, node_features, _ = self.features(
      prng_key,
      geo.coords[0] if geo.coords.ndim == 4 else geo.coords,
      geo.mask[0] if geo.mask.ndim == 2 else geo.mask,
      geo.residue_index[0] if geo.residue_index.ndim == 2 else geo.residue_index,
      geo.chain_index[0] if geo.chain_index.ndim == 2 else geo.chain_index,
      bundle.backbone_noise,
      backbone_noise_mode=config.backbone_noise_mode,
      structure_mapping=geo.structure_mapping[0] if geo.structure_mapping is not None else None,
      initial_node_features=kwargs.get("initial_node_features"),
      rbf_features=kwargs.get("rbf_features"),
      neighbor_indices=kwargs.get("neighbor_indices"),
    )

    # 2. Encode
    node_features, edge_features = self.encoder(
      edge_features,
      edge_indices,
      geo.mask[0] if geo.mask.ndim == 2 else geo.mask,
      node_features,
      key=prng_key,
    )

    return node_features, edge_features, edge_indices
