"""Ligand-conditioned MPNN (:class:`PrxteinLigandMPNN`) and ligand stack helpers.

Extracted in roadmap Phase **5e** from :mod:`prxteinmpnn.model.mpnn`; that module re-exports
these symbols for backwards-compatible ``from prxteinmpnn.model.mpnn import …`` paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

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
from prxteinmpnn.model.multistate_stack import gather_flat_to_stack
from prxteinmpnn.payloads import LigandStack, MultistateStackPayload
from prxteinmpnn.registry import combine_strategy_to_index, multistate_mode_descriptor
from prxteinmpnn.utils.ste import straight_through_estimator

if TYPE_CHECKING:
  from prxteinmpnn.model_inputs import LogitTransformFn
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
    _group_indices_table: jnp.ndarray | None,
    _group_valid_table: jnp.ndarray | None,
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
    *,
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
    *,
    inference: bool = True,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Encode one stacked-structure row with ``structure_mapping`` zeros (LigandMPNN ``state_vmap_exact`` stacks)."""
    return ligand_encode_stack_row(self, coords, ma, ri, ci, yy, yt, ym, hk, inference=inference)

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
    inference: bool = True,
    states_chunk_size: int | None = None,
  ) -> Logits:
    """LigandMPNN unconditional logits: per-row encode + decoder, scatter+fuse (``state_vmap_exact``)."""
    from prxteinmpnn.model.mpnn_scoring_state_vmap_exact_ligand import (  # noqa: PLC0415
      run_score_unconditional_state_vmap_exact_ligand,
    )

    return run_score_unconditional_state_vmap_exact_ligand(
      self,
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
      multi_state_strategy_idx=multi_state_strategy_idx,
      multi_state_temperature=multi_state_temperature,
      state_weights=state_weights,
      state_mapping=state_mapping,
      inference=inference,
      states_chunk_size=states_chunk_size,
    )

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
    inference: bool = True,
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
      inference=inference,
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
    logit_transform_fn: LogitTransformFn | None = None,
  ) -> Logits:
    """LigandMPNN stacked conditional logits; optional ``bias_flat`` added before fuse."""
    from prxteinmpnn.model.mpnn_scoring_state_vmap_exact_ligand import (  # noqa: PLC0415
      run_score_conditional_state_vmap_exact_ligand,
    )

    return run_score_conditional_state_vmap_exact_ligand(
      self,
      prng_key,
      coords_stack,
      mask_stack,
      residue_index_stack,
      chain_index_stack,
      y_stack,
      y_t_stack,
      y_m_stack,
      seq_oh_stack,
      ar_mask_stack,
      state_flat_rows,
      n_flat,
      tie_group_map=tie_group_map,
      multi_state_strategy_idx=multi_state_strategy_idx,
      multi_state_temperature=multi_state_temperature,
      state_weights=state_weights,
      state_mapping=state_mapping,
      bias_flat=bias_flat,
      inference=inference,
      states_chunk_size=states_chunk_size,
      logit_transform_fn=logit_transform_fn,
    )

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
    logit_transform_fn: LogitTransformFn | None = None,
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
      logit_transform_fn=logit_transform_fn,
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
    from prxteinmpnn.model.mpnn_autoregressive_state_vmap_exact_ligand import (  # noqa: PLC0415
      run_sample_autoregressive_state_vmap_exact_ligand,
    )

    return run_sample_autoregressive_state_vmap_exact_ligand(
      self,
      prng_key,
      coords_stack,
      mask_stack,
      residue_index_stack,
      chain_index_stack,
      autoregressive_mask_stack,
      tie_group_map_stack,
      bias_stack,
      temperature,
      multi_state_strategy_idx,
      multi_state_temperature,
      state_weights,
      fixed_mask_stack,
      fixed_tokens_stack,
      y_stack,
      y_t_stack,
      y_m_stack,
      wave_group_ids_local,
      wave_group_positions_local,
      wave_group_valid_local,
      wave_position_valid_local,
    )

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
  *,
  inference: bool = True,
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
    h_V, h_E = enc(h_V, h_E, E_idx, ma, mask_attend, inference=inference)

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
        return y_enc(nd, eg, gm, attention_mask=ge, inference=inference)

      Y_proj = jax.vmap(y_layer)(Y_proj, Y_edges_p, Y_m, Y_m_edges)
      h_E_cat = jnp.concatenate([h_E_context, Y_proj], axis=-1)
      h_V_C = ctx_enc(h_V_C, h_E_cat, ma, attention_mask=Y_m, inference=inference)
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
            node, edge, mask_l, attention_mask=mask_e, inference=inference,
          ),
        )(Yn, Ye, Ymm, Yme)
        he_cat = jnp.concatenate([hec, Yn_out], axis=-1)
        hv_out = ctx_layer(hv, he_cat, msk, attention_mask=Ymm, inference=inference)
        return Yn_out, hv_out

      Y_proj, h_V_C = map_chunks_axis0_multi(
        slab_fn,
        lc,
        (Y_proj, Y_edges_p, Y_m, Y_m_edges, h_V_C, h_E_context, ma),
      )

  h_V_C = jax.vmap(model.v_c)(h_V_C)
  h_fin = h_V + jax.vmap(model.v_c_norm)(
    model.dropout(h_V_C, key=dn_k, inference=inference),
  )
  return h_fin, h_E, E_idx.astype(jnp.int32)
