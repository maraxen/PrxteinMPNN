"""Ligand-conditioned MPNN (:class:`PrxteinLigandMPNN`) and ligand stack helpers.

Extracted in roadmap Phase **5e** from :mod:`prxteinmpnn.model.mpnn`; that module re-exports
these symbols for backwards-compatible ``from prxteinmpnn.model.mpnn import …`` paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from prxteinmpnn.model._shared import (
  apply_multistate_to_all_logits,
  combine_logits_multistate_idx,
  create_group_index_table,
)
from prxteinmpnn.model.capabilities import (
  PRXTEIN_LIGAND_MPNN_CAPABILITIES,
  ModelCapabilities,
)
from prxteinmpnn.model.decoder import Decoder, DecoderLayer
from prxteinmpnn.model.encoder import (
  Encoder,
  pack_encoder_context,
)
from prxteinmpnn.model.ligand_features import ProteinFeaturesLigand
from prxteinmpnn.model.ligand_tiling import map_chunks_axis0, map_chunks_axis0_multi
from prxteinmpnn.model.mpnn_core import (
  autoregressive_decoding_context,
  edge_sequence_features_autoregressive,
)
from prxteinmpnn.model.multistate_stack import gather_flat_to_stack, scatter_stack_to_flat
from prxteinmpnn.payloads import LigandStack, MultistateStackPayload
from prxteinmpnn.registry import combine_strategy_to_index, multistate_mode_descriptor
from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes
from prxteinmpnn.utils.ste import straight_through_estimator

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    AutoRegressiveMask,
    ChainIndex,
    EdgeFeatures,
    Float,
    Int,
    Logits,
    NeighborIndices,
    NodeFeatures,
    OneHotProteinSequence,
    PRNGKeyArray,
    ResidueIndex,
    StructureAtomicCoordinates,
    TieGroupMap,
  )

DecodingApproach = Literal["unconditional", "conditional", "autoregressive"]

class PrxteinLigandMPNN(eqx.Module):
  """Ligand-aware ProteinMPNN model."""

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
  dropout: eqx.nn.Dropout

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
    ligand_mpnn_use_side_chain_context: bool = False,
    ligand_l_chunk: int = 16,
    *,
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

    self.dropout = eqx.nn.Dropout(dropout_rate)

    self.w_s_embed = eqx.nn.Embedding(vocab_size, node_features, key=proj_keys[5])
    self.w_out = eqx.nn.Linear(node_features, num_amino_acids, key=proj_keys[6])

  def __call__(
    self,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    Y: jnp.ndarray,
    Y_t: jnp.ndarray,
    Y_m: jnp.ndarray,
    decoding_approach: DecodingApproach = "conditional",
    *,
    prng_key: PRNGKeyArray | None = None,
    ar_mask: AutoRegressiveMask | None = None,
    one_hot_sequence: OneHotProteinSequence | None = None,
    temperature: float | None = None,
    bias: Logits | None = None,
    fixed_mask: jnp.ndarray | None = None,
    fixed_tokens: jnp.ndarray | None = None,
    backbone_noise: float = 0.0,
    inference: bool = True,
    xyz_37: jnp.ndarray | None = None,
    xyz_37_m: jnp.ndarray | None = None,
    chain_mask: jnp.ndarray | None = None,
    tie_group_map: jnp.ndarray | None = None,
    group_indices_table: jnp.ndarray | None = None,
    group_valid_table: jnp.ndarray | None = None,
    multi_state_strategy: Literal[
      "arithmetic_mean",
      "geometric_mean",
      "product",
    ] = "arithmetic_mean",
    structure_mapping: jnp.ndarray | None = None,
    multi_state_temperature: float = 1.0,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
    precomputed_Y_nodes: jnp.ndarray | None = None,
    precomputed_Y_edges: jnp.ndarray | None = None,
    precomputed_Y_m: jnp.ndarray | None = None,
    num_groups: int | None = None,
    multistate_mode: Literal["flat", "state_vmap", "state_vmap_exact"] = "flat",
    wave_group_ids: jnp.ndarray | None = None,
    wave_group_positions: jnp.ndarray | None = None,
    wave_group_valid: jnp.ndarray | None = None,
    wave_position_valid: jnp.ndarray | None = None,
    coords_stack: jnp.ndarray | None = None,
    mask_stack: jnp.ndarray | None = None,
    residue_index_stack: jnp.ndarray | None = None,
    chain_index_stack: jnp.ndarray | None = None,
    y_stack: jnp.ndarray | None = None,
    y_t_stack: jnp.ndarray | None = None,
    y_m_stack: jnp.ndarray | None = None,
    state_flat_rows: jnp.ndarray | None = None,
    n_flat: int | None = None,
    ar_mask_stack: jnp.ndarray | None = None,
    states_chunk_size: int | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Forward pass for LigandMPNN sequence scoring or sampling."""
    if prng_key is None:
      prng_key = jax.random.PRNGKey(0)

    ms_route = multistate_mode_descriptor(multistate_mode)

    if ms_route.uses_stacked_exact_model_call:
      if decoding_approach == "autoregressive":
        msg = (
          "PrxteinLigandMPNN.__call__ does not run state_vmap_exact autoregressive decoding; "
          "use sample_autoregressive_state_vmap_exact or prxteinmpnn.sampling.sample.make_sample_sequences."
        )
        raise ValueError(msg)
      if (
        precomputed_Y_nodes is not None
        or precomputed_Y_edges is not None
        or precomputed_Y_m is not None
      ):
        msg = (
          "precomputed_Y_nodes / precomputed_Y_edges / precomputed_Y_m are not supported with "
          "multistate_mode='state_vmap_exact' on __call__."
        )
        raise ValueError(msg)
      need_lm = {
        "coords_stack": coords_stack,
        "mask_stack": mask_stack,
        "residue_index_stack": residue_index_stack,
        "chain_index_stack": chain_index_stack,
        "y_stack": y_stack,
        "y_t_stack": y_t_stack,
        "y_m_stack": y_m_stack,
        "state_flat_rows": state_flat_rows,
      }
      missing_lm = [k for k, v in need_lm.items() if v is None]
      if missing_lm:
        msg = (
          "multistate_mode='state_vmap_exact' requires coords_stack, mask_stack, "
          "residue_index_stack, chain_index_stack, y_stack, y_t_stack, y_m_stack, "
          f"state_flat_rows, and n_flat; missing: {', '.join(missing_lm)}"
        )
        raise ValueError(msg)
      if n_flat is None:
        msg = "multistate_mode='state_vmap_exact' requires n_flat (flat logits length)."
        raise ValueError(msg)

      multi_state_strategy_idx_lm = jnp.array(
        combine_strategy_to_index(multi_state_strategy),
        dtype=jnp.int32,
      )
      ms_temp_lm = jnp.asarray(multi_state_temperature, dtype=jnp.float32)
      chunk_kw: dict[str, int] = {}
      if states_chunk_size is not None:
        chunk_kw["states_chunk_size"] = states_chunk_size

      if decoding_approach == "unconditional":
        logits_lm = self.score_unconditional_state_vmap_exact(
          prng_key,
          coords_stack,
          mask_stack,
          residue_index_stack,
          chain_index_stack,
          y_stack,
          y_t_stack,
          y_m_stack,
          state_flat_rows,
          n_flat,
          tie_group_map=tie_group_map,
          multi_state_strategy_idx=multi_state_strategy_idx_lm,
          multi_state_temperature=ms_temp_lm,
          state_weights=state_weights,
          state_mapping=state_mapping,
          **chunk_kw,
        )
        zseq_lm = jnp.zeros((n_flat, self.w_s_embed.num_embeddings), dtype=logits_lm.dtype)
        return zseq_lm, logits_lm

      if decoding_approach == "conditional":
        if one_hot_sequence is None:
          msg = (
            "decoding_approach='conditional' with multistate_mode='state_vmap_exact' requires "
            "one_hot_sequence as (n_flat, num_embeddings) one-hot or (n_flat,) aa indices."
          )
          raise ValueError(msg)
        s_dim_lm, p_dim_lm = mask_stack.shape[0], mask_stack.shape[1]
        arm_lm = (
          jnp.zeros((s_dim_lm, p_dim_lm, p_dim_lm), dtype=jnp.int32)
          if ar_mask_stack is None
          else ar_mask_stack
        )
        if one_hot_sequence.ndim == 1:
          oh_lm = jax.nn.one_hot(
            one_hot_sequence.astype(jnp.int32),
            self.w_s_embed.num_embeddings,
          )
        else:
          oh_lm = one_hot_sequence
        seq_stack_lm = gather_flat_to_stack(oh_lm, state_flat_rows)
        logits_lm = self.score_conditional_state_vmap_exact(
          prng_key,
          coords_stack,
          mask_stack,
          residue_index_stack,
          chain_index_stack,
          y_stack,
          y_t_stack,
          y_m_stack,
          seq_stack_lm,
          arm_lm,
          state_flat_rows,
          n_flat,
          tie_group_map=tie_group_map,
          multi_state_strategy_idx=multi_state_strategy_idx_lm,
          multi_state_temperature=ms_temp_lm,
          state_weights=state_weights,
          state_mapping=state_mapping,
          bias_flat=bias,
          inference=inference,
          **chunk_kw,
        )
        return oh_lm, logits_lm

      raise ValueError(f"Unsupported decoding_approach for PrxteinLigandMPNN stacked path: {decoding_approach!r}")

    del (
      coords_stack,
      mask_stack,
      residue_index_stack,
      chain_index_stack,
      y_stack,
      y_t_stack,
      y_m_stack,
      state_flat_rows,
      ar_mask_stack,
    )

    if not ms_route.allows_ligand_flat_encoder_path:
      msg = (
        f"{type(self).__name__}.__call__ only supports multistate_mode='flat' "
        f"(got {multistate_mode!r}); use multistate_mode='state_vmap_exact' with stacked inputs "
        f"for logits scoring, or sampling helpers for autoregressive decoding."
      )
      raise ValueError(msg)
    del wave_group_ids, wave_group_positions, wave_group_valid, wave_position_valid

    keys = jax.random.split(prng_key, 2)

    # 1. Feature Extraction
    # When precomputed ligand features are provided, skip the expensive ligand feature computation.
    # Protein features (V, E, E_idx) are always computed since they depend on the current sequence.
    if precomputed_Y_nodes is not None and precomputed_Y_edges is not None and precomputed_Y_m is not None:
      # Use cached ligand features; still compute protein features
      V, E, E_idx, _, _, _ = self.features(
        keys[0],
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        Y,
        Y_t,
        Y_m,
        backbone_noise,
        structure_mapping=structure_mapping,
        xyz_37=xyz_37,
        xyz_37_m=xyz_37_m,
        chain_mask=chain_mask,
      )
      Y_nodes = precomputed_Y_nodes
      Y_edges = precomputed_Y_edges
      Y_m = precomputed_Y_m
    else:
      # returns: V (protein nodes), E (protein/protein edges), E_idx,
      #          Y_nodes (ligand nodes), Y_edges (ligand/ligand edges), Y_m (mask)
      V, E, E_idx, Y_nodes, Y_edges, Y_m = self.features(
        keys[0],
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        Y,
        Y_t,
        Y_m,
        backbone_noise,
        structure_mapping=structure_mapping,
        xyz_37=xyz_37,
        xyz_37_m=xyz_37_m,
        chain_mask=chain_mask,
      )

    # 2. Base Model Encoder (Protein internal communication)
    h_V = jnp.zeros((E.shape[0], self.node_features_dim))
    h_E = E

    mask_2d = mask[:, None] * mask[None, :]
    mask_attend = jnp.take_along_axis(mask_2d, E_idx.astype(jnp.int32), axis=1)

    for layer in self.encoder.layers:
      h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend, inference=inference)

    # 3. Context Integration (Ligand-Protein communication)
    h_V_C = jax.vmap(self.w_c)(h_V)
    h_E_context = jax.vmap(jax.vmap(self.w_v))(V)

    # Initial projections + context integration for ligand (DecLayerJ stack + protein context DecoderLayers).
    lig_chunk = self.features.ligand_l_chunk

    # Precompute ligand edge masks
    Y_m_edges = Y_m[..., None] * Y_m[..., None, :]

    if lig_chunk <= 0:
      Y_nodes = jax.vmap(jax.vmap(self.w_nodes_y))(Y_nodes)
      Y_edges = jax.vmap(jax.vmap(jax.vmap(self.w_edges_y)))(Y_edges)

      for i in range(len(self.context_encoder)):
        # Ligand-Ligand communication (DecLayerJ in reference): vmap over residue axis L
        Y_nodes = jax.vmap(
          lambda node, edge, mask_l, mask_e: self.y_context_encoder[i](
            node, edge, mask_l, attention_mask=mask_e, inference=inference,
          ),
        )(Y_nodes, Y_edges, Y_m, Y_m_edges)

        h_E_context_cat = jnp.concatenate([h_E_context, Y_nodes], axis=-1)
        h_V_C = self.context_encoder[i](
          h_V_C, h_E_context_cat, mask, attention_mask=Y_m, inference=inference,
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
              node, edge, mask_l, attention_mask=mask_e, inference=inference,
            ),
          )(Yn, Ye, Ymm, Yme)
          he_cat = jnp.concatenate([hec, Yn_out], axis=-1)
          hv_out = ctx_layer(hv, he_cat, msk, attention_mask=Ymm, inference=inference)
          return Yn_out, hv_out

        Y_nodes, h_V_C = map_chunks_axis0_multi(
          slab_fn,
          lig_chunk,
          (Y_nodes, Y_edges, Y_m, Y_m_edges, h_V_C, h_E_context, mask),
        )

    # Final context combination
    h_V_C = jax.vmap(self.v_c)(h_V_C)
    h_V = h_V + jax.vmap(self.v_c_norm)(self.dropout(h_V_C, key=keys[1], inference=inference))

    # 4. Decoding (Sequence prediction)
    if decoding_approach == "unconditional":
      return self._call_unconditional(
        h_V,
        h_E,
        E_idx,
        mask,
        None,
        None,
        None,
        None,
        None,
        tie_group_map,
        jnp.asarray(combine_strategy_to_index(multi_state_strategy), dtype=jnp.int32),
        multi_state_temperature,
        None,
        state_weights,
        state_mapping,
        None,
        None,
        group_indices_table,
        group_valid_table,
      )

    if decoding_approach == "conditional":
      if one_hot_sequence is None:
        raise ValueError("one_hot_sequence MUST be provided for conditional decoding approach")

      # Scoring/Feedback
      node_decoded = self.decoder.call_conditional(
        h_V,
        h_E,
        E_idx,
        mask,
        ar_mask,
        one_hot_sequence,
        self.w_s_embed.weight,
        inference=inference,
      )

      all_logits = jax.vmap(self.w_out)(node_decoded)

      if bias is not None:
        all_logits = all_logits + bias

      if tie_group_map is not None:
        strategy_idx = jnp.asarray(
          combine_strategy_to_index(multi_state_strategy),
          dtype=jnp.int32,
        )
        all_logits = apply_multistate_to_all_logits(
          all_logits,
          tie_group_map,
          strategy_idx,
          multi_state_temperature,
          state_weights,
          state_mapping,
        )

      return one_hot_sequence, all_logits

    if decoding_approach == "autoregressive":
      if temperature is None:
        temperature = 1.0
      if bias is None:
        bias = jnp.zeros((mask.shape[0], 21))
      if ar_mask is None:
        # Standard decoding order (sum of ar_mask rows defines order)
        ar_mask = jnp.zeros((mask.shape[0], mask.shape[0]))

      return self._run_autoregressive_scan(
        keys[0],
        h_V,
        h_E,
        E_idx,
        mask,
        ar_mask,
        temperature,
        bias,
        tie_group_map=tie_group_map,
        multi_state_strategy_idx=jnp.asarray(
          combine_strategy_to_index(multi_state_strategy),
          dtype=jnp.int32,
        ),
        multi_state_temperature=multi_state_temperature,
        state_weights=state_weights,
        state_mapping=state_mapping,
        fixed_mask=fixed_mask,
        fixed_tokens=fixed_tokens,
        group_indices_table=group_indices_table,
        group_valid_table=group_valid_table,
        num_groups=num_groups,
        inference=inference,
      )

    return None, None

  def _call_unconditional(
    self,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    _ar_mask: AutoRegressiveMask,
    _one_hot_sequence: OneHotProteinSequence,
    _prng_key: PRNGKeyArray,
    _temperature: Float,
    _bias: Logits,
    _tie_group_map: TieGroupMap | None,
    _multi_state_strategy_idx: Int,
    _multi_state_temperature: Float,
    _initial_node_features: NodeFeatures | None,
    _state_weights: jnp.ndarray | None,
    _state_mapping: jnp.ndarray | None,
    _fixed_mask: jnp.ndarray | None,
    _fixed_tokens: jnp.ndarray | None,
    group_indices_table: jnp.ndarray | None,
    group_valid_table: jnp.ndarray | None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the unconditional (scoring) path for LigandMPNN.

    The ligand context is already baked into the encoded node/edge features.
    This method runs the decoder without sequence conditioning and projects
    through w_out, mirroring PrxteinMPNN._call_unconditional.

    Args:
      node_features: Node features from encoding (with ligand context).
      edge_features: Edge features from encoding (with ligand context).
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask.
      _ar_mask: Unused, required for signature uniformity.
      _one_hot_sequence: Unused, required for signature uniformity.
      _prng_key: Unused, required for signature uniformity.
      _temperature: Unused, required for signature uniformity.
      _bias: Unused, required for signature uniformity.
      _tie_group_map: Unused, required for signature uniformity.
      _multi_state_strategy_idx: Unused, required for signature uniformity.
      _multi_state_temperature: Unused, required for signature uniformity.
      _initial_node_features: Unused.
      _state_weights: Unused.
      _state_mapping: Unused.
      _fixed_mask: Unused.
      _fixed_tokens: Unused.
      group_indices_table: Unused.
      group_valid_table: Unused.

    Returns:
      Tuple of (dummy sequence, logits).
    """
    decoded_node_features = self.decoder(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      key=_prng_key,
    )

    logits = jax.vmap(self.w_out)(decoded_node_features)

    # Multi-state logit combining — mirrors PrxteinMPNN._call_unconditional.
    # LigandMPNN's conditional path already uses this static call (line 1744).
    if _tie_group_map is not None:
      logits = apply_multistate_to_all_logits(
        logits,
        _tie_group_map,
        _multi_state_strategy_idx,
        _multi_state_temperature,
        _state_weights,
        _state_mapping,
      )

    # Return dummy sequence to match PyTree shape
    dummy_seq = jnp.zeros(
      (logits.shape[0], self.w_s_embed.num_embeddings),
      dtype=logits.dtype,
    )
    return dummy_seq, logits

  def _run_autoregressive_scan(
    self,
    prng_key: PRNGKeyArray,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    autoregressive_mask: AutoRegressiveMask,
    temperature: float,
    bias: Logits,
    tie_group_map: TieGroupMap | None = None,
    multi_state_strategy_idx: Int = 0,
    multi_state_temperature: Float = 1.0,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
    fixed_mask: jnp.ndarray | None = None,
    fixed_tokens: jnp.ndarray | None = None,
    group_indices_table: jnp.ndarray | None = None,
    group_valid_table: jnp.ndarray | None = None,
    num_groups: int | None = None,
    inference: bool = True,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Autoregressive scan for LigandMPNN with optional tied-position decoding."""
    num_residues = node_features.shape[0]
    fixed_mask_array = (
      jnp.zeros((num_residues,), dtype=jnp.bool_)
      if fixed_mask is None
      else fixed_mask.astype(jnp.bool_)
    )
    fixed_tokens_array = (
      jnp.zeros((num_residues,), dtype=jnp.int32)
      if fixed_tokens is None
      else fixed_tokens.astype(jnp.int32)
    )

    logit_accum_dtype = self.w_out.weight.dtype
    lig_oh_fixed = jax.nn.one_hot(
      fixed_tokens_array,
      self.w_s_embed.num_embeddings,
      dtype=logit_accum_dtype,
    )
    lig_seq_zero = jnp.zeros((num_residues, self.w_s_embed.num_embeddings), dtype=logit_accum_dtype)
    lig_initial_sequence_from_fixed = jnp.where(fixed_mask_array[:, None], lig_oh_fixed, lig_seq_zero)
    lig_emb_w = self.w_s_embed.weight.astype(logit_accum_dtype)
    lig_initial_s_embed_from_fixed = lig_initial_sequence_from_fixed @ lig_emb_w

    # Precompute masks and order
    attention_mask = jnp.take_along_axis(
      autoregressive_mask, neighbor_indices.astype(jnp.int32), axis=1,
    )
    mask_1d = mask[:, None]
    mask_bw = mask_1d * attention_mask
    mask_fw = mask_1d * (1 - attention_mask)
    decoding_order = jnp.argsort(jnp.sum(autoregressive_mask, axis=1))

    # Encoder context (pre-weighted by mask_fw)
    encoder_context = pack_encoder_context(
      node_features,
      edge_features,
      neighbor_indices,
      mask_fw,
    )

    def _decode_position(
      position_all_layers_h: jax.Array,
      s_embed: jax.Array,
      position: Int,
    ) -> tuple[jax.Array, jax.Array]:
      """Decode one position and return updated hidden state + logits."""
      edge_sequence_features = edge_sequence_features_autoregressive(
        s_embed,
        edge_features,
        neighbor_indices,
        position,
      )

      for layer_idx, layer in enumerate(self.decoder.layers):
        h_in_pos = position_all_layers_h[layer_idx, position]
        decoding_context = autoregressive_decoding_context(
          position_all_layers_h[layer_idx],
          edge_sequence_features,
          neighbor_indices[position],
          encoder_context[position],
          mask_bw[position],
        )

        h_out_pos = layer(
          h_in_pos[None],
          decoding_context[None],
          mask=mask[position],
          key=None,
          inference=inference,
        )
        position_all_layers_h = position_all_layers_h.at[layer_idx + 1, position].set(
          jnp.squeeze(h_out_pos),
        )

      final_h_pos = position_all_layers_h[-1, position]
      logits_pos = self.w_out(final_h_pos)
      return position_all_layers_h, logits_pos

    def _sample_and_broadcast_to_group(
      avg_logits: jax.Array,
      group_mask: jax.Array,
      key: jax.Array,
      all_logits: jax.Array,
      s_embed: jax.Array,
      sequence: jax.Array,
      fixed_mask_local: jax.Array,
      fixed_tokens_local: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      """Sample one token from group-combined logits and broadcast to all group positions."""
      if state_weights is not None and state_mapping is not None:
        w = state_weights[state_mapping]
        group_bias = jnp.sum(
          jnp.where(group_mask[:, None], bias * w[:, None], 0.0),
          axis=0,
          keepdims=True,
        ) / jnp.sum(jnp.where(group_mask, w, 0.0))
      else:
        group_count = jnp.maximum(jnp.sum(group_mask.astype(jnp.float32)), 1.0)
        group_bias = jnp.sum(
          jnp.where(group_mask[:, None], bias, 0.0),
          axis=0,
          keepdims=True,
        ) / group_count

      logits_with_bias = avg_logits + group_bias
      group_fixed_mask = group_mask & fixed_mask_local
      has_fixed_token = jnp.any(group_fixed_mask)

      def _sample_group(_: None) -> jax.Array:
        sampled_logits = (logits_with_bias / temperature) + jax.random.gumbel(
          key,
          logits_with_bias.shape,
          dtype=logits_with_bias.dtype,
        )
        one_hot_sample = straight_through_estimator(sampled_logits[..., :20])
        return jnp.concatenate([one_hot_sample, jnp.zeros_like(one_hot_sample[..., :1])], axis=-1)

      def _fixed_group(_: None) -> jax.Array:
        fixed_token = jnp.max(jnp.where(group_fixed_mask, fixed_tokens_local, -1))
        return jax.nn.one_hot(
          fixed_token,
          self.w_s_embed.num_embeddings,
          dtype=logits_with_bias.dtype,
        )[None, :]

      one_hot_seq = jax.lax.cond(has_fixed_token, _fixed_group, _sample_group, operand=None)
      s_embed_new = one_hot_seq @ self.w_s_embed.weight
      all_logits = jnp.where(group_mask[:, None], jnp.squeeze(avg_logits), all_logits)
      s_embed = jnp.where(group_mask[:, None], jnp.squeeze(s_embed_new), s_embed)
      sequence = jnp.where(group_mask[:, None], jnp.squeeze(one_hot_seq), sequence)
      return all_logits, s_embed, sequence

    def _initial_carry() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
      initial_all_layers_h = jnp.zeros(
        (len(self.decoder.layers) + 1, num_residues, self.node_features_dim),
      )
      initial_all_layers_h = initial_all_layers_h.at[0].set(node_features)
      zeros_logits = jnp.zeros((num_residues, 21), dtype=logit_accum_dtype)
      return (
        initial_all_layers_h,
        lig_initial_s_embed_from_fixed,
        zeros_logits,
        lig_initial_sequence_from_fixed,
      )

    if tie_group_map is None:
      def autoregressive_step(carry, scan_inputs):
        all_layers_h, s_embed, all_logits, sequence = carry
        position, key = scan_inputs
        all_layers_h, logits_pos = _decode_position(all_layers_h, s_embed, position)
        all_logits = all_logits.at[position].set(logits_pos)
        logits_with_bias = logits_pos + bias[position]

        def _sample_position(_: None) -> jax.Array:
          sampled_logits = (logits_with_bias / temperature) + jax.random.gumbel(
            key,
            logits_with_bias.shape,
            dtype=logits_with_bias.dtype,
          )
          one_hot_sample = straight_through_estimator(sampled_logits[:20])
          return jnp.concatenate(
            [one_hot_sample, jnp.zeros(1, dtype=one_hot_sample.dtype)],
            axis=-1,
          )

        def _fixed_position(_: None) -> jax.Array:
          return jax.nn.one_hot(
            fixed_tokens_array[position],
            self.w_s_embed.num_embeddings,
            dtype=logits_with_bias.dtype,
          )

        one_hot_seq_pos = jax.lax.cond(
          fixed_mask_array[position],
          _fixed_position,
          _sample_position,
          operand=None,
        )
        s_embed_pos = one_hot_seq_pos @ self.w_s_embed.weight
        s_embed = s_embed.at[position].set(s_embed_pos)
        sequence = sequence.at[position].set(one_hot_seq_pos)
        return (all_layers_h, s_embed, all_logits, sequence), None

      final_carry, _ = jax.lax.scan(
        autoregressive_step,
        _initial_carry(),
        (decoding_order, jax.random.split(prng_key, num_residues)),
      )
      return final_carry[3], final_carry[2]

    # Use pre-computed group tables if provided, else compute them
    if group_indices_table is None or group_valid_table is None:
      # Compute max group size from tie_group_map
      unique_groups, counts = jnp.unique(tie_group_map[tie_group_map >= 0], return_counts=True)
      max_group_size = int(counts.max()) if len(counts) > 0 else 1
      group_indices_table, group_valid_table = create_group_index_table(
        tie_group_map,
        max_group_size,
      )
    else:
      # Infer max_group_size from the shape of the pre-computed tables
      max_group_size = group_indices_table.shape[1]

    groups_in_order = tie_group_map[decoding_order]
    position_indices = jnp.arange(num_residues)
    is_before_mask = position_indices[:, None] > position_indices[None, :]
    group_matches = groups_in_order[:, None] == groups_in_order[None, :]
    appeared_before = jnp.any(group_matches & is_before_mask, axis=1)
    is_first_occurrence = ~appeared_before
    compress_size = num_groups if num_groups is not None else num_residues
    group_decoding_order = jnp.compress(
      is_first_occurrence,
      groups_in_order,
      size=compress_size,
      fill_value=-1,
    )

    def group_autoregressive_step(carry, scan_inputs):
      all_layers_h, s_embed, all_logits, sequence = carry
      group_id, key = scan_inputs

      def _skip_group(_: None) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
        return (all_layers_h, s_embed, all_logits, sequence), None

      def _decode_group(_: None) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
        group_indices = group_indices_table[group_id]
        valid_mask = group_valid_table[group_id]
        computed_logits = jnp.zeros((num_residues, 21), dtype=all_logits.dtype)

        def process_one_member(i: Int, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
          position_all_layers_h, position_logits = state
          idx = group_indices[i]
          is_valid = valid_mask[i]

          def _process(_: None) -> tuple[jax.Array, jax.Array]:
            updated_h, logits_pos = _decode_position(position_all_layers_h, s_embed, idx)
            return updated_h, position_logits.at[idx].set(logits_pos)

          return jax.lax.cond(
            is_valid,
            _process,
            lambda _: (position_all_layers_h, position_logits),
            operand=None,
          )

        all_layers_h_updated, computed_logits = jax.lax.fori_loop(
          0,
          max_group_size,
          process_one_member,
          (all_layers_h, computed_logits),
        )
        group_mask = tie_group_map == group_id
        combined_logits = combine_logits_multistate_idx(
          computed_logits,
          group_mask,
          multi_state_strategy_idx,
          multi_state_temperature,
          state_weights,
          state_mapping,
        )
        all_logits_updated, s_embed_updated, sequence_updated = _sample_and_broadcast_to_group(
          combined_logits,
          group_mask,
          key,
          all_logits,
          s_embed,
          sequence,
          fixed_mask_array,
          fixed_tokens_array,
        )
        return (all_layers_h_updated, s_embed_updated, all_logits_updated, sequence_updated), None

      return jax.lax.cond(group_id < 0, _skip_group, _decode_group, operand=None)

    n_groups = group_decoding_order.shape[0]
    final_carry, _ = jax.lax.scan(
      group_autoregressive_step,
      _initial_carry(),
      (group_decoding_order, jax.random.split(prng_key, n_groups)),
      unroll=1,
    )
    return final_carry[3], final_carry[2]

  def _ligand_encode_stack_one(
    self,
    coords: jax.Array,
    ma: jax.Array,
    ri: jax.Array,
    ci: jax.Array,
    yy: jax.Array,
    yt: jax.Array,
    ym: jax.Array,
    hk: PRNGKeyArray,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Encode one stacked-structure row with ``structure_mapping`` zeros (LigandMPNN ``state_vmap_exact`` stacks)."""
    return ligand_encode_stack_row(self, coords, ma, ri, ci, yy, yt, ym, hk)

  def score_unconditional_state_vmap_exact(
    self,
    prng_key: PRNGKeyArray,
    coords_stack: jax.Array,
    mask_stack: jax.Array,
    residue_index_stack: jax.Array,
    chain_index_stack: jax.Array,
    y_stack: jax.Array,
    y_t_stack: jax.Array,
    y_m_stack: jax.Array,
    state_flat_rows: jax.Array,
    n_flat: int,
    *,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    multi_state_temperature: Float | float,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    _dropout_inference: bool = True,
    states_chunk_size: int | None = None,
  ) -> Logits:
    """LigandMPNN unconditional logits: per-row encode + decoder, scatter+fuse (``state_vmap_exact``)."""
    del _dropout_inference
    k_enc, k_feat = jax.random.split(prng_key)
    s_tot = int(coords_stack.shape[0])
    scs = int(states_chunk_size) if states_chunk_size is not None else 0
    log_dim = int(self.w_out.out_features)
    log_dtype = self.w_out.weight.dtype

    def _one_shot() -> jax.Array:
      def enc_one(coords, ma, ri, ci, yy, yt, ym):
        return ligand_encode_stack_row(self, coords, ma, ri, ci, yy, yt, ym, k_feat)

      node_b, edge_b, nei_b = jax.vmap(enc_one)(
        coords_stack,
        mask_stack,
        residue_index_stack,
        chain_index_stack,
        y_stack,
        y_t_stack,
        y_m_stack,
      )

      def decode_one(nb, eb, nei, mk):
        # Match :meth:`PrxteinLigandMPNN._call_unconditional`: ``key=None`` disables decoder dropout
        # ( unconditional ``__call__`` passes ``None`` for the decoder PRNG slot ).
        return self.decoder(nb, eb, nei, mk, key=None)

      decoded = jax.vmap(decode_one)(node_b, edge_b, nei_b, mask_stack)
      logits_s = jax.vmap(jax.vmap(self.w_out))(decoded)
      return scatter_stack_to_flat(logits_s, state_flat_rows, n_flat)

    if scs <= 0 or scs >= s_tot:
      logits_flat = _one_shot()
    else:
      logits_flat = jnp.zeros((n_flat, log_dim), dtype=log_dtype)
      for s0 in range(0, s_tot, scs):
        c_coords, c_mask, c_ri, c_ci, c_y, c_yt, c_ym, c_rows = _ligand_slice_pad_state_batch(
          s0,
          scs,
          s_tot,
          coords_stack,
          mask_stack,
          residue_index_stack,
          chain_index_stack,
          y_stack,
          y_t_stack,
          y_m_stack,
          state_flat_rows,
        )
        logits_flat = logits_flat + ligand_score_unconditional_state_vmap_one_chunk(
          self,
          k_enc,
          k_feat,
          c_coords,
          c_mask,
          c_ri,
          c_ci,
          c_y,
          c_yt,
          c_ym,
          c_rows,
          n_flat,
        )

    if tie_group_map is not None:
      logits_flat = apply_multistate_to_all_logits(
        logits_flat,
        tie_group_map,
        multi_state_strategy_idx,
        jnp.asarray(multi_state_temperature, jnp.float32),
        state_weights,
        state_mapping,
      )
    return logits_flat

  def score_unconditional_state_vmap_exact_from_payload(
    self,
    prng_key: PRNGKeyArray,
    stack: MultistateStackPayload,
    ligand: LigandStack,
    *,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    multi_state_temperature: Float | float,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    _dropout_inference: bool = True,
    states_chunk_size: int | None = None,
  ) -> Logits:
    """Same as :meth:`score_unconditional_state_vmap_exact` with ``stack`` + :class:`~prxteinmpnn.payloads.LigandStack`."""
    return self.score_unconditional_state_vmap_exact(
      prng_key,
      stack.coords_stack,
      stack.mask_stack,
      stack.residue_index_stack,
      stack.chain_index_stack,
      ligand.y_stack,
      ligand.y_t_stack,
      ligand.y_m_stack,
      stack.state_flat_rows,
      stack.n_flat,
      tie_group_map=tie_group_map,
      multi_state_strategy_idx=multi_state_strategy_idx,
      multi_state_temperature=multi_state_temperature,
      state_weights=state_weights,
      state_mapping=state_mapping,
      _dropout_inference=_dropout_inference,
      states_chunk_size=states_chunk_size,
    )

  def score_conditional_state_vmap_exact(
    self,
    prng_key: PRNGKeyArray,
    coords_stack: jax.Array,
    mask_stack: jax.Array,
    residue_index_stack: jax.Array,
    chain_index_stack: jax.Array,
    y_stack: jax.Array,
    y_t_stack: jax.Array,
    y_m_stack: jax.Array,
    seq_oh_stack: jax.Array,
    ar_mask_stack: jax.Array,
    state_flat_rows: jax.Array,
    n_flat: int,
    *,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    multi_state_temperature: Float | float,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    bias_flat: jax.Array | None = None,
    inference: bool = True,
    states_chunk_size: int | None = None,
  ) -> Logits:
    """LigandMPNN stacked conditional logits; optional ``bias_flat`` added before fuse."""
    k_enc, k_feat = jax.random.split(prng_key)
    s_tot = int(coords_stack.shape[0])
    scs = int(states_chunk_size) if states_chunk_size is not None else 0
    log_dim = int(self.w_out.out_features)
    log_dtype = self.w_out.weight.dtype

    def _one_shot() -> jax.Array:
      def enc_row(coords, ma, ri, ci, yy, yt, ym):
        return ligand_encode_stack_row(self, coords, ma, ri, ci, yy, yt, ym, k_feat)

      node_b, edge_b, nei_b = jax.vmap(enc_row)(
        coords_stack,
        mask_stack,
        residue_index_stack,
        chain_index_stack,
        y_stack,
        y_t_stack,
        y_m_stack,
      )

      def dec_one(nb, eb, nei, mk, arm, oh):
        return self.decoder.call_conditional(
          nb,
          eb,
          nei,
          mk,
          arm,
          oh,
          self.w_s_embed.weight,
          inference=inference,
          key=k_enc,
        )

      decoded = jax.vmap(dec_one)(
        node_b,
        edge_b,
        nei_b,
        mask_stack,
        ar_mask_stack,
        seq_oh_stack,
      )

      logits_s = jax.vmap(jax.vmap(self.w_out))(decoded)
      if bias_flat is not None:
        logits_s = logits_s + gather_flat_to_stack(bias_flat, state_flat_rows)

      return scatter_stack_to_flat(logits_s, state_flat_rows, n_flat)

    if scs <= 0 or scs >= s_tot:
      logits_flat = _one_shot()
    else:
      logits_flat = jnp.zeros((n_flat, log_dim), dtype=log_dtype)
      for s0 in range(0, s_tot, scs):
        c_coords, c_mask, c_ri, c_ci, c_y, c_yt, c_ym, c_rows = _ligand_slice_pad_state_batch(
          s0,
          scs,
          s_tot,
          coords_stack,
          mask_stack,
          residue_index_stack,
          chain_index_stack,
          y_stack,
          y_t_stack,
          y_m_stack,
          state_flat_rows,
        )
        c_oh, c_arm = _ligand_slice_pad_cond_batch(s0, scs, s_tot, seq_oh_stack, ar_mask_stack)
        logits_flat = logits_flat + ligand_score_conditional_state_vmap_one_chunk(
          self,
          k_enc,
          k_feat,
          c_coords,
          c_mask,
          c_ri,
          c_ci,
          c_y,
          c_yt,
          c_ym,
          c_oh,
          c_arm,
          c_rows,
          n_flat,
          bias_flat,
          inference,
        )

    if tie_group_map is not None:
      logits_flat = apply_multistate_to_all_logits(
        logits_flat,
        tie_group_map,
        multi_state_strategy_idx,
        jnp.asarray(multi_state_temperature, jnp.float32),
        state_weights,
        state_mapping,
      )
    return logits_flat

  def score_conditional_state_vmap_exact_from_payload(
    self,
    prng_key: PRNGKeyArray,
    stack: MultistateStackPayload,
    ligand: LigandStack,
    seq_oh_stack: jax.Array,
    ar_mask_stack: jax.Array,
    *,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    multi_state_temperature: Float | float,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    bias_flat: jax.Array | None = None,
    inference: bool = True,
    states_chunk_size: int | None = None,
  ) -> Logits:
    """Same as :meth:`score_conditional_state_vmap_exact` with ``stack`` + :class:`~prxteinmpnn.payloads.LigandStack`."""
    return self.score_conditional_state_vmap_exact(
      prng_key,
      stack.coords_stack,
      stack.mask_stack,
      stack.residue_index_stack,
      stack.chain_index_stack,
      ligand.y_stack,
      ligand.y_t_stack,
      ligand.y_m_stack,
      seq_oh_stack,
      ar_mask_stack,
      stack.state_flat_rows,
      stack.n_flat,
      tie_group_map=tie_group_map,
      multi_state_strategy_idx=multi_state_strategy_idx,
      multi_state_temperature=multi_state_temperature,
      state_weights=state_weights,
      state_mapping=state_mapping,
      bias_flat=bias_flat,
      inference=inference,
      states_chunk_size=states_chunk_size,
    )

  def sample_autoregressive_state_vmap_exact(
    self,
    prng_key: PRNGKeyArray,
    coords_stack: jax.Array,
    mask_stack: jax.Array,
    residue_index_stack: jax.Array,
    chain_index_stack: jax.Array,
    autoregressive_mask_stack: jax.Array,
    tie_group_map_stack: jax.Array | None,
    bias_stack: jax.Array,
    temperature: Float | float,
    multi_state_strategy_idx: Int,
    multi_state_temperature: Float,
    state_weights: jnp.ndarray | None,
    fixed_mask_stack: jax.Array,
    fixed_tokens_stack: jax.Array,
    y_stack: jax.Array,
    y_t_stack: jax.Array,
    y_m_stack: jax.Array,
    wave_group_ids_local: jax.Array,
    wave_group_positions_local: jax.Array,
    wave_group_valid_local: jax.Array,
    wave_position_valid_local: jax.Array,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Stacked-graph wave sampler for LigandMPNN (``state_vmap_exact``)."""
    del tie_group_map_stack
    pk_cur, fk = jax.random.split(prng_key)
    temp = jnp.asarray(temperature, dtype=jnp.float32)
    feat_keys = jax.random.split(fk, coords_stack.shape[0])

    def lig_encode_one(coords, ma, ri, ci, yy, yt, ym, hk):
      return ligand_encode_stack_row(self, coords, ma, ri, ci, yy, yt, ym, hk)

    node_b, edge_b, nei_b = jax.vmap(lig_encode_one)(
      coords_stack,
      mask_stack,
      residue_index_stack,
      chain_index_stack,
      y_stack,
      y_t_stack,
      y_m_stack,
      feat_keys,
    )

    attn_b = jax.vmap(
      lambda am, nei_i: jnp.take_along_axis(am, nei_i.astype(jnp.int32), axis=1),
    )(autoregressive_mask_stack, nei_b)
    mask_bw = mask_stack[:, :, jnp.newaxis] * attn_b
    mask_fw = mask_stack[:, :, jnp.newaxis] * (jnp.float32(1.0) - jnp.asarray(attn_b, jnp.float32))
    encoder_stack = jax.vmap(pack_encoder_context)(node_b, edge_b, nei_b, mask_fw)

    log_dtype = self.w_out.weight.dtype
    num_dec = len(self.decoder.layers) + 1
    s_dim, p_pad = int(coords_stack.shape[0]), int(coords_stack.shape[1])

    ah = jnp.zeros((s_dim, num_dec, p_pad, self.node_features_dim), dtype=log_dtype)
    ah = ah.at[:, 0, :, :].set(node_b.astype(log_dtype))

    fixed_b = fixed_mask_stack.astype(jnp.bool_)
    fixed_toks = fixed_tokens_stack.astype(jnp.int32)
    oh_fix = jax.nn.one_hot(fixed_toks, self.w_s_embed.num_embeddings, dtype=log_dtype)
    seq_z = jnp.zeros((s_dim, p_pad, self.w_s_embed.num_embeddings), dtype=log_dtype)
    seq_carry = jnp.where(fixed_b[..., jnp.newaxis], oh_fix, seq_z)
    emb_w = self.w_s_embed.weight.astype(log_dtype)
    se = seq_carry @ emb_w

    logits_acc = jnp.zeros((s_dim, p_pad, self.w_out.out_features), dtype=log_dtype)

    nw_i = jnp.int32(wave_group_valid_local.shape[0])
    nslot_i = jnp.int32(wave_group_valid_local.shape[1])
    max_gs_tr = jnp.int32(wave_position_valid_local.shape[-1])
    max_gs = int(wave_position_valid_local.shape[-1])
    strat_idx = jnp.asarray(multi_state_strategy_idx, dtype=jnp.int32)
    ms_temp = jnp.asarray(multi_state_temperature, dtype=jnp.float32)
    if state_weights is None:
      sw_use = jnp.ones((s_dim,), dtype=jnp.float32) / jnp.float32(max(s_dim, 1))
    else:
      sw_use = jnp.asarray(state_weights, dtype=jnp.float32)
    row_state_map = jnp.arange(max_gs, dtype=jnp.int32)

    def decode_site(
      ah_s: jax.Array,
      se_s: jax.Array,
      lid_s: jax.Array,
      _ignore_key: jax.Array,
      *,
      eb: jax.Array,
      nei: jax.Array,
      mw_atom: jax.Array,
      ecx: jax.Array,
      bw: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
      del _ignore_key
      ecx_row = jax.lax.dynamic_index_in_dim(ecx, lid_s, axis=0).squeeze(0)
      nei_row = jax.lax.dynamic_index_in_dim(nei, lid_s, axis=0).squeeze(0)
      mask_pos = jax.lax.dynamic_index_in_dim(mw_atom, lid_s, axis=0).squeeze(0)
      bw_row = jax.lax.dynamic_index_in_dim(bw, lid_s, axis=0).squeeze(0)
      edge_slice = jax.lax.dynamic_index_in_dim(eb, lid_s, axis=0).squeeze(0)
      seq_nf = concatenate_neighbor_nodes(se_s, edge_slice, nei_row)
      ah_cur = ah_s
      for lyr_i, layer in enumerate(self.decoder.layers):
        h_in = jax.lax.dynamic_index_in_dim(ah_cur[lyr_i], lid_s, axis=0).squeeze(0)
        dctx = concatenate_neighbor_nodes(ah_cur[lyr_i], seq_nf, nei_row)
        dcomb = bw_row[..., None] * dctx + ecx_row
        h_out = layer(
          jnp.expand_dims(h_in, 0),
          jnp.expand_dims(dcomb, 0),
          mask=mask_pos,
          key=None,
          inference=True,
        )
        ah_cur = ah_cur.at[lyr_i + 1, lid_s].set(jnp.squeeze(h_out, axis=0))
      lg = self.w_out(jax.lax.dynamic_index_in_dim(ah_cur[-1], lid_s, axis=0).squeeze(0))
      return ah_cur, lg

    def outer_wave(w_ix: jax.Array, cw: tuple) -> tuple:
      wi_here = jnp.int32(w_ix)
      pk_w, ah_w, se_w, seq_w, log_w = cw

      def inner_slot(mi_xx: jax.Array, cms: tuple) -> tuple:
        mi_here = jnp.int32(mi_xx)
        pk_mi, ah_mi, se_mi, seq_mi, logits_mi = cms

        lane_wgv = jax.lax.dynamic_index_in_dim(wave_group_valid_local, wi_here, axis=0).squeeze(0)
        wgv_here = jax.lax.dynamic_index_in_dim(lane_wgv, mi_here, axis=0).squeeze()
        lane_gid = jax.lax.dynamic_index_in_dim(wave_group_ids_local, wi_here, axis=0).squeeze(0)
        gid_here = jax.lax.dynamic_index_in_dim(lane_gid, mi_here, axis=0).squeeze()
        wl_plane = jax.lax.dynamic_index_in_dim(wave_group_positions_local, wi_here, axis=0).squeeze(0)
        wl_lane = jax.lax.dynamic_index_in_dim(wl_plane, mi_here, axis=0).squeeze(0)
        pv_plane = jax.lax.dynamic_index_in_dim(wave_position_valid_local, wi_here, axis=0).squeeze(0)
        pv_lane = jax.lax.dynamic_index_in_dim(pv_plane, mi_here, axis=0).squeeze(0)
        active_slot = jnp.logical_and(wgv_here.astype(jnp.bool_), gid_here >= jnp.int32(0))
        has_member = jnp.any(pv_lane.astype(jnp.bool_))
        first_g = jnp.argmax(pv_lane.astype(jnp.int32))
        lid_pick = jax.lax.dynamic_index_in_dim(wl_lane, first_g).astype(jnp.int32)
        cand_lid = jnp.reshape(lid_pick, ())
        lid_here = jax.lax.select(
          jnp.logical_and(active_slot.squeeze(), has_member.squeeze()),
          cand_lid,
          jnp.int32(0),
        )

        def skip_slot(__: None) -> tuple:
          del __
          return pk_mi, ah_mi, se_mi, seq_mi, logits_mi

        def do_slot(__: None) -> tuple:
          del __
          logits_rows_init = jnp.zeros((max_gs, self.w_out.out_features), dtype=log_dtype)
          cmask_init = jnp.zeros((max_gs,), dtype=jnp.bool_)

          def body_g(g: jax.Array, carry_g: tuple) -> tuple:
            pk_g, ah_g, se_g, lrows_g, cg_g = carry_g
            g_idx = jnp.int32(g)
            pv_g = jax.lax.dynamic_index_in_dim(pv_lane, g_idx, axis=0).squeeze(axis=0)
            si = g_idx
            mask_here = jax.lax.dynamic_index_in_dim(
              jax.lax.dynamic_index_in_dim(mask_stack.astype(jnp.float32), si, axis=0).squeeze(axis=0),
              lid_here,
              axis=0,
            ).squeeze()
            go_decode = jnp.logical_and(
              jnp.logical_and(pv_g.astype(jnp.bool_), active_slot.squeeze()),
              mask_here > jnp.float32(0.0),
            )

            def do_dec(___: None) -> tuple:
              del ___
              pk_dec, dk_dummy = jax.random.split(pk_g)
              ah_si = jax.lax.dynamic_index_in_dim(ah_g, si, axis=0).squeeze(axis=0)
              se_si = jax.lax.dynamic_index_in_dim(se_g, si, axis=0).squeeze(axis=0)
              eb_si = jax.lax.dynamic_index_in_dim(edge_b, si, axis=0).squeeze(axis=0)
              nei_si = jax.lax.dynamic_index_in_dim(nei_b, si, axis=0).squeeze(axis=0)
              mw_si = jax.lax.dynamic_index_in_dim(mask_stack.astype(jnp.float32), si, axis=0).squeeze(axis=0)
              ecx_si = jax.lax.dynamic_index_in_dim(encoder_stack, si, axis=0).squeeze(axis=0)
              bw_si = jax.lax.dynamic_index_in_dim(mask_bw, si, axis=0).squeeze(axis=0)
              ah_new, logits_row = decode_site(
                ah_si,
                se_si,
                lid_here,
                dk_dummy,
                eb=eb_si,
                nei=nei_si,
                mw_atom=mw_si,
                ecx=ecx_si,
                bw=bw_si,
              )
              ah_upd = ah_g.at[si].set(ah_new)
              cg_upd = cg_g.at[g_idx].set(True)
              lr_upd = lrows_g.at[g_idx].set(logits_row.astype(log_dtype))
              return pk_dec, ah_upd, se_g, lr_upd, cg_upd

            return jax.lax.cond(go_decode.squeeze(), do_dec, lambda _: (pk_g, ah_g, se_g, lrows_g, cg_g), None)

          pk_in, ah_in = pk_mi, ah_mi
          pk_step, ah_step, _, lrows_fin, cmask_fin = jax.lax.fori_loop(
            jnp.int32(0),
            max_gs_tr,
            body_g,
            (pk_in, ah_in, se_mi, logits_rows_init, cmask_init),
          )
          any_cmem = jnp.any(cmask_fin)

          def noop(__: None) -> tuple:
            del __
            return pk_step, ah_step, se_mi, seq_mi, logits_mi

          def contrib(__: None) -> tuple:
            del __
            combined = combine_logits_multistate_idx(
              lrows_fin,
              cmask_fin.astype(jnp.bool_),
              strat_idx,
              ms_temp,
              sw_use,
              row_state_map,
            )
            comb_vec = jnp.squeeze(combined, axis=0).astype(log_dtype)

            def gb_body(g_i: jax.Array, acc_gb: tuple) -> tuple:
              num_acc, den_acc = acc_gb
              gi = jnp.int32(g_i)
              take = jax.lax.dynamic_index_in_dim(cmask_fin.astype(jnp.bool_), gi).squeeze(axis=0)
              bw_row_here = jax.lax.dynamic_index_in_dim(
                jax.lax.dynamic_index_in_dim(bias_stack, gi, axis=0).squeeze(axis=0),
                lid_here,
                axis=0,
              ).squeeze()
              ww = jax.lax.dynamic_index_in_dim(sw_use, gi).squeeze(axis=0)
              inc_num = jax.lax.select(take, bw_row_here * ww, jnp.zeros_like(bw_row_here))
              inc_den = jax.lax.select(take, ww, jnp.float32(0.0))
              return (num_acc + inc_num.astype(log_dtype), den_acc + inc_den)

            num_b_v, den_b_v = jax.lax.fori_loop(
              jnp.int32(0),
              max_gs_tr,
              gb_body,
              (jnp.zeros((self.w_out.out_features,), dtype=log_dtype), jnp.float32(0.0)),
            )
            group_bias_flat = num_b_v / jnp.maximum(den_b_v, jnp.float32(1e-8))
            logits_wb = comb_vec + group_bias_flat

            def orb_fix(g_fix: jax.Array, hv_fix: jax.Array) -> jax.Array:
              gx = jnp.int32(g_fix)
              take_f = jax.lax.dynamic_index_in_dim(cmask_fin.astype(jnp.bool_), gx).squeeze(axis=0)
              fx_slot = jax.lax.dynamic_index_in_dim(
                jax.lax.dynamic_index_in_dim(fixed_b.astype(jnp.bool_), gx, axis=0).squeeze(axis=0),
                lid_here,
                axis=0,
              ).squeeze(axis=0)
              return hv_fix | jnp.logical_and(take_f, fx_slot)

            has_any_fixed = jax.lax.fori_loop(jnp.int32(0), max_gs_tr, orb_fix, jnp.bool_(False))

            pk_samp, dk_s = jax.random.split(pk_step)

            def samp_vec(*_unused: object) -> jax.Array:
              del _unused
              lo_row = jnp.expand_dims(logits_wb, axis=0)
              samp_logits = lo_row / temp + jax.random.gumbel(dk_s, shape=lo_row.shape, dtype=log_dtype)
              one_hot_sample = straight_through_estimator(samp_logits[..., :20])
              pad_b = jnp.zeros_like(one_hot_sample[..., :1])
              oh = jnp.concatenate([one_hot_sample, pad_b], axis=-1)
              return oh[0, :]

            def fixed_vec(*_unused2: object) -> jax.Array:
              del _unused2

              def fv_body(g_ff: jax.Array, best_tok: jax.Array) -> jax.Array:
                gx_ff = jnp.int32(g_ff)
                cand = jax.lax.dynamic_index_in_dim(cmask_fin.astype(jnp.bool_), gx_ff).squeeze(axis=0)
                tok_g = jax.lax.dynamic_index_in_dim(
                  jax.lax.dynamic_index_in_dim(fixed_toks.astype(jnp.int32), gx_ff, axis=0).squeeze(axis=0),
                  lid_here,
                  axis=0,
                ).squeeze(axis=0)
                best_i = jax.lax.select(cand, jnp.maximum(best_tok, tok_g), best_tok)
                return best_i.astype(jnp.int32)

              fixed_idx = jax.lax.fori_loop(jnp.int32(0), max_gs_tr, fv_body, jnp.int32(-1))
              return jax.nn.one_hot(fixed_idx, self.w_s_embed.num_embeddings, dtype=log_dtype)

            oh_broadcast = jax.lax.cond(has_any_fixed, fixed_vec, samp_vec)
            pk_post = jax.lax.select(has_any_fixed, pk_step, pk_samp)

            emb_new_vec = jnp.matmul(
              oh_broadcast.astype(log_dtype),
              emb_w.astype(log_dtype),
            )

            def upd_si(si_ix: jax.Array, bags: tuple) -> tuple:
              seq_lp, logits_lp, emb_lp = bags
              si_j = jnp.int32(si_ix)
              in_rng = jnp.logical_and(si_j >= jnp.int32(0), si_j < max_gs_tr)
              grp_mem = jax.lax.select(
                in_rng.squeeze(),
                jax.lax.dynamic_index_in_dim(cmask_fin.astype(jnp.bool_), si_j, axis=0).squeeze(axis=0),
                jnp.bool_(False),
              )

              row_seq = seq_lp[si_j]
              new_seq_row = jax.lax.select(grp_mem, row_seq.at[lid_here].set(oh_broadcast.astype(row_seq.dtype)), row_seq)
              seq_lp2 = seq_lp.at[si_j].set(new_seq_row)

              row_logits = logits_lp[si_j]
              li_next = jax.lax.select(grp_mem, row_logits.at[lid_here].set(comb_vec), row_logits)
              logits_lp2 = logits_lp.at[si_j].set(li_next)

              row_emb = emb_lp[si_j]
              em_next = jax.lax.select(grp_mem, row_emb.at[lid_here].set(emb_new_vec.astype(row_emb.dtype)), row_emb)
              emb_lp2 = emb_lp.at[si_j].set(em_next)
              return seq_lp2, logits_lp2, emb_lp2

            seq_nf, logits_nf, se_nf = jax.lax.fori_loop(
              jnp.int32(0),
              jnp.int32(s_dim),
              upd_si,
              (seq_mi, logits_mi, se_mi),
            )
            return pk_post, ah_step, se_nf, seq_nf, logits_nf

          return jax.lax.cond(any_cmem, contrib, noop, None)

        active_decode = jnp.logical_and(active_slot.squeeze(), has_member)
        return jax.lax.cond(active_decode, do_slot, skip_slot, None)

      return jax.lax.fori_loop(jnp.int32(0), nslot_i, inner_slot, (pk_w, ah_w, se_w, seq_w, log_w))

    _pk_out, ah_out, se_out, seq_out, log_out = jax.lax.fori_loop(
      jnp.int32(0),
      nw_i,
      outer_wave,
      (pk_cur, ah, se, seq_carry, logits_acc),
    )
    del _pk_out
    return seq_out.astype(log_dtype), log_out.astype(log_dtype)

  def sample_autoregressive_state_vmap_exact_from_payload(
    self,
    prng_key: PRNGKeyArray,
    stack: MultistateStackPayload,
    ligand: LigandStack,
    autoregressive_mask_stack: jax.Array,
    bias_stack: jax.Array,
    temperature: Float | float,
    multi_state_strategy_idx: Int,
    multi_state_temperature: Float,
    state_weights: jnp.ndarray | None,
    wave_group_ids_local: jax.Array,
    wave_group_positions_local: jax.Array,
    wave_group_valid_local: jax.Array,
    wave_position_valid_local: jax.Array,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Same as :meth:`sample_autoregressive_state_vmap_exact` with ``stack`` + :class:`~prxteinmpnn.payloads.LigandStack`."""
    return self.sample_autoregressive_state_vmap_exact(
      prng_key,
      stack.coords_stack,
      stack.mask_stack,
      stack.residue_index_stack,
      stack.chain_index_stack,
      autoregressive_mask_stack,
      stack.tie_group_map_stack,
      bias_stack,
      temperature,
      multi_state_strategy_idx,
      multi_state_temperature,
      state_weights,
      stack.fixed_mask_stack,
      stack.fixed_tokens_stack,
      ligand.y_stack,
      ligand.y_t_stack,
      ligand.y_m_stack,
      wave_group_ids_local,
      wave_group_positions_local,
      wave_group_valid_local,
      wave_position_valid_local,
    )


def ligand_encode_stack_row(
  model: PrxteinLigandMPNN,
  coords: jax.Array,
  ma: jax.Array,
  ri: jax.Array,
  ci: jax.Array,
  yy: jax.Array,
  yt: jax.Array,
  ym: jax.Array,
  hk: PRNGKeyArray,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Module-level LigandMPNN stacked-row encoder (callable from ``jax.vmap`` closures; JAX-tracing-safe)."""
  fe_k, dn_k = jax.random.split(hk)
  zeros_map = jnp.zeros((coords.shape[0],), dtype=jnp.int32)
  V, E, E_idx, Y_nodes, Y_edges, Y_m = model.features(
    fe_k,
    coords,
    ma,
    ri,
    ci,
    yy,
    yt,
    ym,
    jnp.asarray(0.0, jnp.float32),
    structure_mapping=zeros_map,
    xyz_37=None,
    xyz_37_m=None,
    chain_mask=None,
  )
  h_V = jnp.zeros((E.shape[0], model.node_features_dim))
  h_E = E
  mask_2d = ma[:, None] * ma[None, :]
  mask_attend = jnp.take_along_axis(mask_2d, E_idx.astype(jnp.int32), axis=1)
  for enc in model.encoder.layers:
    h_V, h_E = enc(h_V, h_E, E_idx, ma, mask_attend, inference=True)

  h_V_C = jax.vmap(model.w_c)(h_V)
  h_E_context = jax.vmap(jax.vmap(model.w_v))(V)
  lc = model.features.ligand_l_chunk
  Y_m_edges = Y_m[..., None] * Y_m[..., None, :]

  if lc <= 0:
    Y_proj = jax.vmap(jax.vmap(model.w_nodes_y))(Y_nodes)
    Y_edges_p = jax.vmap(jax.vmap(jax.vmap(model.w_edges_y)))(Y_edges)

    for ix in range(len(model.context_encoder)):
      y_enc = model.y_context_encoder[ix]
      ctx_enc = model.context_encoder[ix]

      def y_layer(nd, eg, gm, ge):
        return y_enc(nd, eg, gm, attention_mask=ge, inference=True)

      Y_proj = jax.vmap(y_layer)(Y_proj, Y_edges_p, Y_m, Y_m_edges)
      h_E_cat = jnp.concatenate([h_E_context, Y_proj], axis=-1)
      h_V_C = ctx_enc(h_V_C, h_E_cat, ma, attention_mask=Y_m, inference=True)
  else:
    Y_proj = map_chunks_axis0(
      Y_nodes,
      chunk_size=lc,
      fn=lambda s: jax.vmap(jax.vmap(model.w_nodes_y))(s),
    )
    Y_edges_p = map_chunks_axis0(
      Y_edges,
      chunk_size=lc,
      fn=lambda s: jax.vmap(jax.vmap(jax.vmap(model.w_edges_y)))(s),
    )

    for ix in range(len(model.context_encoder)):
      y_layer = model.y_context_encoder[ix]
      ctx_layer = model.context_encoder[ix]

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

      Y_proj, h_V_C = map_chunks_axis0_multi(
        slab_fn,
        lc,
        (Y_proj, Y_edges_p, Y_m, Y_m_edges, h_V_C, h_E_context, ma),
      )

  h_V_C = jax.vmap(model.v_c)(h_V_C)
  h_fin = h_V + jax.vmap(model.v_c_norm)(
    model.dropout(h_V_C, key=dn_k, inference=True),
  )
  return h_fin, h_E, E_idx.astype(jnp.int32)


def _ligand_slice_pad_state_batch(
  s0: int,
  states_chunk_size: int,
  s_tot: int,
  coords_stack: jax.Array,
  mask_stack: jax.Array,
  residue_index_stack: jax.Array,
  chain_index_stack: jax.Array,
  y_stack: jax.Array,
  y_t_stack: jax.Array,
  y_m_stack: jax.Array,
  state_flat_rows: jax.Array,
) -> tuple[jax.Array, ...]:
  """Slice ``[s0, s0+cs)`` along state axis and pad to length ``states_chunk_size``.

  Padded rows: ``mask=0``, ``state_flat_rows=-1`` (scatter skips). Other tensors tile the
  last real row so ligand feature shapes stay valid under ``ligand_l_chunk`` tiling.
  """
  cs = int(states_chunk_size)
  s1 = min(s0 + cs, s_tot)
  n_real = s1 - s0

  def pad_first_axis(a: jax.Array) -> jax.Array:
    slab = a[s0:s1]
    if n_real >= cs:
      return slab
    pad_n = cs - n_real
    last = slab[-1:]
    tail = jnp.tile(last, (pad_n,) + (1,) * (slab.ndim - 1))
    return jnp.concatenate([slab, tail], axis=0)

  rows = state_flat_rows[s0:s1]
  if n_real < cs:
    rows = jnp.concatenate(
      [rows, jnp.full((cs - n_real, rows.shape[1]), -1, dtype=rows.dtype)],
      axis=0,
    )
  m = mask_stack[s0:s1]
  if n_real < cs:
    m = jnp.concatenate([m, jnp.zeros((cs - n_real, m.shape[1]), dtype=m.dtype)], axis=0)

  return (
    pad_first_axis(coords_stack),
    m,
    pad_first_axis(residue_index_stack),
    pad_first_axis(chain_index_stack),
    pad_first_axis(y_stack),
    pad_first_axis(y_t_stack),
    pad_first_axis(y_m_stack),
    rows,
  )


def _ligand_slice_pad_cond_batch(
  s0: int,
  states_chunk_size: int,
  s_tot: int,
  seq_oh_stack: jax.Array,
  ar_mask_stack: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  cs = int(states_chunk_size)
  s1 = min(s0 + cs, s_tot)
  n_real = s1 - s0

  def pad_first_axis(a: jax.Array) -> jax.Array:
    slab = a[s0:s1]
    if n_real >= cs:
      return slab
    pad_n = cs - n_real
    last = slab[-1:]
    tail = jnp.tile(last, (pad_n,) + (1,) * (slab.ndim - 1))
    return jnp.concatenate([slab, tail], axis=0)

  arm = ar_mask_stack[s0:s1]
  if n_real < cs:
    arm = jnp.concatenate(
      [arm, jnp.zeros((cs - n_real,) + arm.shape[1:], dtype=arm.dtype)],
      axis=0,
    )
  return pad_first_axis(seq_oh_stack), arm


def ligand_score_unconditional_state_vmap_one_chunk(
  model: PrxteinLigandMPNN,
  k_enc: jax.Array,
  k_feat: jax.Array,
  coords_stack: jax.Array,
  mask_stack: jax.Array,
  residue_index_stack: jax.Array,
  chain_index_stack: jax.Array,
  y_stack: jax.Array,
  y_t_stack: jax.Array,
  y_m_stack: jax.Array,
  state_flat_rows: jax.Array,
  n_flat: int,
) -> jax.Array:
  """Single fixed-size state batch (padded to leading dim of ``coords_stack``); scatter only, no fuse."""

  def enc_one(coords, ma, ri, ci, yy, yt, ym):
    return ligand_encode_stack_row(model, coords, ma, ri, ci, yy, yt, ym, k_feat)

  node_b, edge_b, nei_b = jax.vmap(enc_one)(
    coords_stack,
    mask_stack,
    residue_index_stack,
    chain_index_stack,
    y_stack,
    y_t_stack,
    y_m_stack,
  )

  def decode_one(nb, eb, nei, mk):
    return model.decoder(nb, eb, nei, mk, key=None)

  decoded = jax.vmap(decode_one)(node_b, edge_b, nei_b, mask_stack)
  logits_s = jax.vmap(jax.vmap(model.w_out))(decoded)
  return scatter_stack_to_flat(logits_s, state_flat_rows, n_flat)


def ligand_score_conditional_state_vmap_one_chunk(
  model: PrxteinLigandMPNN,
  k_enc: jax.Array,
  k_feat: jax.Array,
  coords_stack: jax.Array,
  mask_stack: jax.Array,
  residue_index_stack: jax.Array,
  chain_index_stack: jax.Array,
  y_stack: jax.Array,
  y_t_stack: jax.Array,
  y_m_stack: jax.Array,
  seq_oh_stack: jax.Array,
  ar_mask_stack: jax.Array,
  state_flat_rows: jax.Array,
  n_flat: int,
  bias_flat: jax.Array | None,
  inference: bool,
) -> jax.Array:
  """Conditional path for one fixed-size state batch; scatter only, no fuse."""

  def enc_row(coords, ma, ri, ci, yy, yt, ym):
    return ligand_encode_stack_row(model, coords, ma, ri, ci, yy, yt, ym, k_feat)

  node_b, edge_b, nei_b = jax.vmap(enc_row)(
    coords_stack,
    mask_stack,
    residue_index_stack,
    chain_index_stack,
    y_stack,
    y_t_stack,
    y_m_stack,
  )

  def dec_one(nb, eb, nei, mk, arm, oh):
    return model.decoder.call_conditional(
      nb,
      eb,
      nei,
      mk,
      arm,
      oh,
      model.w_s_embed.weight,
      inference=inference,
      key=k_enc,
    )

  decoded = jax.vmap(dec_one)(
    node_b,
    edge_b,
    nei_b,
    mask_stack,
    ar_mask_stack,
    seq_oh_stack,
  )

  logits_s = jax.vmap(jax.vmap(model.w_out))(decoded)
  if bias_flat is not None:
    logits_s = logits_s + gather_flat_to_stack(bias_flat, state_flat_rows)
  return scatter_stack_to_flat(logits_s, state_flat_rows, n_flat)
