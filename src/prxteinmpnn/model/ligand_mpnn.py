"""Ligand-conditioned MPNN (:class:`PrxteinLigandMPNN`) and ligand stack helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from prxteinmpnn.model.capabilities import (
  PRXTEIN_LIGAND_MPNN_CAPABILITIES,
  ModelCapabilities,
)
from prxteinmpnn.model.decoder import Decoder, DecoderLayer
from prxteinmpnn.model.dropout import Dropout
from prxteinmpnn.model.encoder import Encoder
from prxteinmpnn.model.ligand_features import ProteinFeaturesLigand
from prxteinmpnn.model.ligand_tiling import map_chunks_axis0, map_chunks_axis0_multi

if TYPE_CHECKING:
  from prxteinmpnn.types.arrays import PRNGKeyArray

class PrxteinLigandMPNN(eqx.Module):
  features: ProteinFeaturesLigand
  encoder: Encoder
  decoder: Decoder

  context_encoder: tuple[DecoderLayer, ...]
  y_context_encoder: tuple[DecoderLayer, ...]

  w_v: eqx.nn.Linear
  w_c: eqx.nn.Linear
  w_nodes_y: eqx.nn.Linear
  w_edges_y: eqx.nn.Linear
  v_c: eqx.nn.Linear
  v_c_norm: eqx.nn.LayerNorm

  w_s_embed: eqx.nn.Embedding
  w_out: eqx.nn.Linear
  dropout: Dropout

  node_features_dim: int = eqx.field(static=True)
  edge_features_dim: int = eqx.field(static=True)
  hidden_features_dim: int = eqx.field(static=True)
  num_decoder_layers: int = eqx.field(static=True)
  ligand_mpnn_use_side_chain_context: bool = eqx.field(static=True)
  capabilities: ModelCapabilities = eqx.field(static=True, default=PRXTEIN_LIGAND_MPNN_CAPABILITIES)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    k_neighbors: int,
    num_context_layers: int = 2,
    num_positional_embeddings: int = 16,
    num_amino_acids: int = 21,
    vocab_size: int = 21,
    dropout_rate: float = 0.1,
    ligand_l_chunk: int = 16,
    *,
    ligand_mpnn_use_side_chain_context: bool = False,
    key: PRNGKeyArray,
  ) -> None:
    keys = jax.random.split(key, 5)
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features
    self.hidden_features_dim = hidden_features
    self.num_decoder_layers = num_decoder_layers
    self.ligand_mpnn_use_side_chain_context = ligand_mpnn_use_side_chain_context

    self.features = ProteinFeaturesLigand(
      node_features=node_features,
      edge_features=edge_features,
      k_neighbors=k_neighbors,
      num_positional_embeddings=num_positional_embeddings,
      use_side_chains=ligand_mpnn_use_side_chain_context,
      ligand_l_chunk=ligand_l_chunk,
      key=keys[0],
    )

    self.encoder = Encoder(
      node_features=node_features,
      edge_features=edge_features,
      hidden_features=hidden_features,
      num_layers=num_encoder_layers,
      dropout_rate=dropout_rate,
      key=keys[1],
    )

    self.decoder = Decoder(
      node_features=node_features,
      edge_features=edge_features,
      hidden_features=hidden_features,
      num_layers=num_decoder_layers,
      dropout_rate=dropout_rate,
      key=keys[2],
    )

    context_keys = jax.random.split(keys[3], num_context_layers)
    y_context_keys = jax.random.split(keys[4], num_context_layers)

    # Extra keys for projections
    proj_keys = jax.random.split(jax.random.fold_in(key, 100), 7)

    self.context_encoder = tuple(
      DecoderLayer(
        node_features, node_features * 2, hidden_features, dropout_rate=dropout_rate, key=k,
      )
      for k in context_keys
    )
    # y_context_encoder takes num_in = hidden_dim
    self.y_context_encoder = tuple(
      DecoderLayer(node_features, node_features, hidden_features, dropout_rate=dropout_rate, key=k)
      for k in y_context_keys
    )

    self.w_v = eqx.nn.Linear(node_features, node_features, key=proj_keys[0])
    self.w_c = eqx.nn.Linear(node_features, node_features, key=proj_keys[1])
    self.w_nodes_y = eqx.nn.Linear(node_features, node_features, key=proj_keys[2])
    self.w_edges_y = eqx.nn.Linear(node_features, node_features, key=proj_keys[3])
    self.v_c = eqx.nn.Linear(node_features, node_features, key=proj_keys[4])
    self.v_c_norm = eqx.nn.LayerNorm(node_features)

    self.dropout = Dropout(dropout_rate)

    self.w_s_embed = eqx.nn.Embedding(vocab_size, node_features, key=proj_keys[5])
    self.w_out = eqx.nn.Linear(node_features, num_amino_acids, key=proj_keys[6])


  @classmethod
  def stage_schema(cls) -> dict[str, type | None]:
    from prxteinmpnn.types.stages import (
      ConditionalDecodeFn,
      UnconditionalDecodeFn,
      FeaturizeFn,
      LigandEncodeFn,
      LogitTransformFn,
      ARLogitTransformFn,
      EncoderStateFn,
    )
    return {
        "featurize": FeaturizeFn,
        "encode": LigandEncodeFn,
        "decode": ConditionalDecodeFn | UnconditionalDecodeFn,
        "logit_transform": LogitTransformFn,
        "ar_logit_transform": ARLogitTransformFn,
        "encoder_state_fn": EncoderStateFn | None,
    }

  def __call__(
    self,
    coords: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    *,
    prng_key: PRNGKeyArray | None = None,
    backbone_noise: float | jax.Array = 0.0,
    backbone_noise_mode: str = "direct",
    structure_mapping: jax.Array | None = None,
    y: jax.Array | None = None,
    y_t: jax.Array | None = None,
    y_m: jax.Array | None = None,
    inference: bool = True,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pure forward: features -> encode -> return encoded representation.

    Args:
        coords: (L, 4, 3) atom coordinates for one state.
        mask: (L,) residue mask.
        residue_index: (L,) residue indices.
        chain_index: (L,) chain indices.
        prng_key: PRNG key for encoding (optional, defaults to PRNGKey(0)).
        backbone_noise: scalar backbone noise level.
        backbone_noise_mode: accepted but not used by LigandMPNN features.
        structure_mapping: optional structure isolation mapping.
        y: (M, 4, 3) ligand coordinates.
        y_t: (M, 4) ligand atom type tokens.
        y_m: (M, 4) ligand atom mask.
        inference: unused, accepted for API uniformity.

    Returns:
        (node_features, edge_features, neighbor_indices) for this state.
    """
    if prng_key is None:
      prng_key = jax.random.PRNGKey(0)
    keys = jax.random.split(prng_key, 2)

    # Default ligand arrays to empty if not provided
    if y is None:
        y = jnp.zeros((0, 4, 3))
    if y_t is None:
        y_t = jnp.zeros((0, 4), dtype=jnp.int32)
    if y_m is None:
        y_m = jnp.zeros((0, 4))

    V, E, E_idx, Y_nodes, Y_edges, Y_m_out = self.features(
      _key=keys[0],
      structure_coordinates=coords,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      Y=y,
      Y_t=y_t,
      Y_m=y_m,
      backbone_noise=backbone_noise,
      structure_mapping=structure_mapping,
    )
    
    h_V = jnp.zeros((E.shape[0], self.node_features_dim))
    h_E = E

    mask_2d = mask[:, None] * mask[None, :]
    mask_attend = jnp.take_along_axis(mask_2d, E_idx.astype(jnp.int32), axis=1)

    for layer in self.encoder.layers:
      h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend, inference=True)

    h_V_C = jax.vmap(self.w_c)(h_V)
    h_E_context = jax.vmap(jax.vmap(self.w_v))(V)

    lig_chunk = self.features.ligand_l_chunk
    Y_m_edges = Y_m_out[..., None] * Y_m_out[..., None, :]

    if lig_chunk <= 0:
      Y_nodes = jax.vmap(jax.vmap(self.w_nodes_y))(Y_nodes)
      Y_edges = jax.vmap(jax.vmap(jax.vmap(self.w_edges_y)))(Y_edges)

      for i in range(len(self.context_encoder)):
        Y_nodes = jax.vmap(
          lambda node, edge, mask_l, mask_e: self.y_context_encoder[i](
            node, edge, mask_l, attention_mask=mask_e, inference=True,
          ),
        )(Y_nodes, Y_edges, Y_m_out, Y_m_edges)

        h_E_context_cat = jnp.concatenate([h_E_context, Y_nodes], axis=-1)
        h_V_C = self.context_encoder[i](
          h_V_C, h_E_context_cat, mask, attention_mask=Y_m_out, inference=True,
        )
    else:
      Y_nodes = map_chunks_axis0(
        Y_nodes,
        chunk_size=lig_chunk,
        fn=lambda s: jax.vmap(jax.vmap(self.w_nodes_y))(s),
      )
      Y_edges = map_chunks_axis0(
        Y_edges,
        chunk_size=lig_chunk,
        fn=lambda s: jax.vmap(jax.vmap(jax.vmap(self.w_edges_y)))(s),
      )

      for i in range(len(self.context_encoder)):
        y_layer = self.y_context_encoder[i]
        ctx_layer = self.context_encoder[i]

        def slab_fn(
          Yn: jax.Array,
          Ye: jax.Array,
          Ymm: jax.Array,
          Yme: jax.Array,
          hv: jax.Array,
          hec: jax.Array,
          msk: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
          Yn_out = jax.vmap(
            lambda node, edge, mask_l, mask_e: y_layer(
              node, edge, mask_l, attention_mask=mask_e, inference=True,
            ),
          )(Yn, Ye, Ymm, Yme)
          he_cat = jnp.concatenate([hec, Yn_out], axis=-1)
          hv_out = ctx_layer(hv, he_cat, msk, attention_mask=Ymm, inference=True)
          return Yn_out, hv_out

        Y_nodes, h_V_C = map_chunks_axis0_multi(
          slab_fn,
          lig_chunk,
          (Y_nodes, Y_edges, Y_m_out, Y_m_edges, h_V_C, h_E_context, mask),
        )

    h_V_C = jax.vmap(self.v_c)(h_V_C)
    h_V = h_V + jax.vmap(self.v_c_norm)(self.dropout(h_V_C, key=keys[1], inference=True))

    return h_V, h_E, E_idx
