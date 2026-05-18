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
  from prxteinmpnn.types.arrays import PRNGKeyArray
else:
  PRNGKeyArray = Any

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
    structure_coordinates: jax.Array | PRNGKeyArray,
    mask_or_bundle: jax.Array | InferenceBundle | None = None,
    residue_index: jax.Array | None = None,
    chain_index: jax.Array | None = None,
    *,
    decoding_approach: str | None = None,
    prng_key: PRNGKeyArray | None = None,
    ar_mask: jax.Array | None = None,
    one_hot_sequence: jax.Array | None = None,
    backbone_noise: jax.Array | float | None = None,
    bundle: InferenceBundle | None = None,
    config: InferenceConfig | None = None,
    **kwargs: Any,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single-state encode: features + encoder for one conformational state.

    This is the low-level building block called by make_encode_fn (inference/encode.py).
    It is NOT an inference entry point — use make_inference_plan, make_score_sequence,
    or make_sample_sequences for full inference paths.

    Args:
        structure_coordinates: (L, 4, 3) atom37 coordinates for one state.
        mask_or_bundle: (L,) residue mask.
        residue_index: (L,) residue indices.
        chain_index: (L,) chain indices.
        prng_key: PRNG key for encoder dropout.
        backbone_noise: scalar backbone noise level.

    Returns:
        (node_features, edge_features, neighbor_indices) for this state.
    """
    coords = structure_coordinates
    mask = mask_or_bundle
    if prng_key is None:
      import jax
      prng_key = jax.random.PRNGKey(0)
    noise = backbone_noise if backbone_noise is not None else 0.0
    sm = kwargs.get("structure_mapping", None)

    edge_features, neighbor_indices, node_features, _ = self.features(
      prng_key,
      coords,
      mask,
      residue_index,
      chain_index,
      noise,
      backbone_noise_mode=kwargs.get("backbone_noise_mode", "direct"),
      structure_mapping=sm,
    )
    node_features, edge_features = self.encoder(
      edge_features,
      neighbor_indices,
      mask,
      initial_node_features=node_features,
      key=prng_key,
    )
    return node_features, edge_features, neighbor_indices
