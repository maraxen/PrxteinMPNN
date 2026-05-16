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
  ) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Unified entry point supporting both old and new APIs.

    Old API: model(coords, mask, res_idx, chain_idx, decoding_approach="conditional", ...)
    New API: model(prng_key, bundle, config)

    Returns:
        Old API: (logits, logits) for conditional decoding
        New API: (node_features, edge_features, edge_indices) for encoder-only
    """
    from prxteinmpnn.inference.bundle_builder import build_inference_bundle
    from prxteinmpnn.inference.driver import decode
    from prxteinmpnn.inference.encode import make_encode_fn

    # Detect which API is being used
    if isinstance(mask_or_bundle, InferenceBundle) or bundle is not None:
      # New API: model(prng_key, bundle, config, ...)
      _prng_key = structure_coordinates
      _bundle = mask_or_bundle if isinstance(mask_or_bundle, InferenceBundle) else bundle
      _config = residue_index if isinstance(residue_index, InferenceConfig) else config

      geo = _bundle.geometry

      edge_features, edge_indices, node_features, _ = self.features(
        _prng_key,
        geo.coords[0],
        geo.mask[0],
        geo.residue_index[0],
        geo.chain_index[0],
        _bundle.backbone_noise,
        backbone_noise_mode=_config.backbone_noise_mode,
        structure_mapping=geo.structure_mapping[0] if geo.structure_mapping is not None else None,
        initial_node_features=kwargs.get("initial_node_features"),
        rbf_features=kwargs.get("rbf_features"),
        neighbor_indices=kwargs.get("neighbor_indices"),
      )

      node_features, edge_features = self.encoder(
        edge_features,
        edge_indices,
        geo.mask[0],
        node_features,
        key=_prng_key,
      )

      return node_features, edge_features, edge_indices

    else:
      # Old API: model(coords, mask, res_idx, chain_idx, decoding_approach="conditional", ...)
      if prng_key is None:
        raise ValueError("prng_key is required for old API")

      coords = structure_coordinates
      _mask = mask_or_bundle
      _residue_index = residue_index
      _chain_index = chain_index

      # Build bundle from geometry
      _bundle, _config, stage_set = build_inference_bundle(
        coords=coords,
        mask=_mask,
        residue_index=_residue_index,
        chain_index=_chain_index,
        sequence=one_hot_sequence if one_hot_sequence is not None else None,
        ar_mask=ar_mask,
        backbone_noise=backbone_noise if backbone_noise is not None else 0.0,
        mode="score_conditional" if decoding_approach == "conditional" else "score_unconditional",
        inference=True,
      )

      # Encode
      k_enc, k_dec = jax.random.split(prng_key)
      encode_fn = make_encode_fn(self, use_rolling_state=_config.use_rolling_state)
      enc = encode_fn(_bundle, k_enc, _config)

      # Decode
      logits = decode(self, k_dec, enc, _bundle.conditioning, _bundle.wave, _config, stage_set)

      # Return (logits, logits) for compatibility with test expectations
      return logits, logits
