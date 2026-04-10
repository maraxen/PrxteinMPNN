"""Main ProteinMPNN model implementation.

This module contains the top-level PrxteinMPNN model that combines all components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp

from prxteinmpnn.model.decoder import DecoderLayer, Decoder
from prxteinmpnn.model.encoder import EncoderLayer, Encoder, PhysicsEncoder
from prxteinmpnn.model.features import ProteinFeatures
from prxteinmpnn.model.ligand_features import ProteinFeaturesLigand
from prxteinmpnn.model.multi_state_sampling import (
  arithmetic_mean_logits,
  geometric_mean_logits,
  product_of_probabilities_logits,
)
from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes
from prxteinmpnn.utils.ste import straight_through_estimator

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    AutoRegressiveMask,
    BackboneNoise,
    ChainIndex,
    DecodingOrder,
    EdgeFeatures,
    Float,
    GroupMask,
    Int,
    LinkMask,
    Logits,
    NeighborIndices,
    NodeEdgeFeatures,
    NodeFeatures,
    OneHotProteinSequence,
    PRNGKeyArray,
    ResidueIndex,
    StructureAtomicCoordinates,
    TieGroupMap,
  )

DecodingApproach = Literal["unconditional", "conditional", "autoregressive"]


class PrxteinMPNN(eqx.Module):
  """The complete end-to-end ProteinMPNN model."""

  features: ProteinFeatures
  encoder: Encoder | PhysicsEncoder
  decoder: Decoder

  w_s_embed: eqx.nn.Embedding  # For sequence

  w_out: eqx.nn.Linear

  node_features_dim: int = eqx.field(static=True)
  edge_features_dim: int = eqx.field(static=True)
  num_decoder_layers: int = eqx.field(static=True)

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
    vocab_size: int = 21,  # for w_s
    dropout_rate: float = 0.1,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the complete model.

    Args:
      node_features: Dimension of node features (e.g., 128).
      edge_features: Dimension of edge features (e.g., 128).
      hidden_features: Dimension of hidden layer in encoder/decoder.
      num_encoder_layers: Number of encoder layers.
      num_decoder_layers: Number of decoder layers.
      k_neighbors: Number of nearest neighbors for graph construction.
      physics_feature_dim: Dimension of physical features (if any).
      num_amino_acids: Number of amino acid types (default: 21).
      vocab_size: Size of sequence vocabulary (default: 21).
      dropout_rate: Dropout rate (default: 0.1).
      key: PRNG key for initialization.

    Returns:
      None

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> model = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=key)

    """
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features
    self.num_decoder_layers = num_decoder_layers

    keys = jax.random.split(key, 5)  # 1 for features, 4 for main model

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
    _initial_node_features: NodeFeatures | None = None,
    _state_weights: jnp.ndarray | None = None,
    _state_mapping: jnp.ndarray | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the unconditional (scoring) path."""
    decoded_node_features = self.decoder(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      key=_prng_key,
    )

    logits = jax.vmap(self.w_out)(decoded_node_features)

    if _tie_group_map is not None:
      logits = self._apply_multistate_to_all_logits(
        logits,
        _tie_group_map,
        _multi_state_strategy_idx,
        _multi_state_temperature,
        _state_weights,
        _state_mapping,
      )

    dummy_seq = jnp.zeros(
      (logits.shape[0], self.w_s_embed.num_embeddings),
      dtype=logits.dtype,
    )
    return dummy_seq, logits

  def _call_conditional(
    self,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    ar_mask: AutoRegressiveMask,
    one_hot_sequence: OneHotProteinSequence,
    prng_key: PRNGKeyArray,
    _temperature: Float,
    _bias: Logits,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    multi_state_temperature: Float,
    _initial_node_features: NodeFeatures | None = None,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the conditional (scoring) path."""
    decoded_node_features = self.decoder.call_conditional(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      one_hot_sequence,
      self.w_s_embed.weight,
      key=prng_key,
    )
    logits = jax.vmap(self.w_out)(decoded_node_features)

    if tie_group_map is not None:
      logits = self._apply_multistate_to_all_logits(
        logits,
        tie_group_map,
        multi_state_strategy_idx,
        multi_state_temperature,
        state_weights,
        state_mapping,
      )

    return one_hot_sequence.astype(logits.dtype), logits

  def _call_autoregressive(
    self,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    ar_mask: AutoRegressiveMask,
    _one_hot_sequence: OneHotProteinSequence,
    prng_key: PRNGKeyArray,
    temperature: Float,
    bias: Logits,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    multi_state_temperature: Float = 1.0,
    _initial_node_features: NodeFeatures | None = None,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the autoregressive (sampling) path."""
    seq, logits = self._run_autoregressive_scan(
      prng_key,
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      temperature,
      bias,
      tie_group_map,
      multi_state_strategy_idx,
      multi_state_temperature,
      state_weights,
      state_mapping,
    )
    return seq, logits

  @staticmethod
  def _combine_logits_multistate(
    logits: Logits,
    group_mask: GroupMask,
    strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
    temperature: float = 1.0,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
  ) -> Logits:
    """Combine logits across tied positions using different multi-state strategies."""
    if strategy == "arithmetic_mean":
      return arithmetic_mean_logits(logits, group_mask, state_weights, state_mapping)
    if strategy == "geometric_mean":
      return geometric_mean_logits(logits, group_mask, temperature, state_weights, state_mapping)
    if strategy == "product":
      return product_of_probabilities_logits(logits, group_mask, state_weights, state_mapping)
    msg = f"Unknown multi-state strategy: {strategy}"
    raise ValueError(msg)

  @staticmethod
  def _apply_multistate_to_all_logits(
    logits: Logits,
    tie_group_map: TieGroupMap,
    strategy_idx: Int,
    temperature: float = 1.0,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
  ) -> Logits:
    """Apply multi-state combination strategies across ALL groups in parallel."""
    num_total = tie_group_map.shape[0]
    
    def apply_arithmetic(l: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
      if state_weights is not None and state_mapping is not None:
        w = state_weights[state_mapping]
        log_w = jnp.log(jnp.where(w > 0, w, 1e-9))
        weighted_l = l + log_w
        
        max_per_group = jax.ops.segment_max(weighted_l, g, num_segments=num_total)
        l_shifted = weighted_l - max_per_group[g]
        exp_l = jnp.exp(l_shifted)
        sum_exp = jax.ops.segment_sum(exp_l, g, num_segments=num_total)
        sum_w = jax.ops.segment_sum(w, g, num_segments=num_total)
        log_avg = jnp.log(sum_exp / jnp.where(sum_w > 0, sum_w, 1.0))
        return (log_avg + max_per_group)[g]
      
      max_per_group = jax.ops.segment_max(l, g, num_segments=num_total)
      l_shifted = l - max_per_group[g]
      exp_l = jnp.exp(l_shifted)
      sum_exp = jax.ops.segment_sum(exp_l, g, num_segments=num_total)
      count = jax.ops.segment_sum(jnp.ones_like(g, dtype=jnp.float32), g, num_segments=num_total)
      log_avg = jnp.log(sum_exp / jnp.where(count > 0, count, 1.0))
      return (log_avg + max_per_group)[g]

    def apply_geometric(l: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
      if state_weights is not None and state_mapping is not None:
        w = state_weights[state_mapping]
        sum_wl = jax.ops.segment_sum(l * w, g, num_segments=num_total)
        sum_w = jax.ops.segment_sum(w, g, num_segments=num_total)
        return (sum_wl / (jnp.where(sum_w > 0, sum_w, 1.0) * temperature))[g]
        
      sum_l = jax.ops.segment_sum(l, g, num_segments=num_total)
      count = jax.ops.segment_sum(jnp.ones_like(g, dtype=jnp.float32), g, num_segments=num_total)
      return (sum_l / (jnp.where(count > 0, count, 1.0) * temperature))[g]

    def apply_product(l: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
      if state_weights is not None and state_mapping is not None:
        w = state_weights[state_mapping]
        return jax.ops.segment_sum(l * w, g, num_segments=num_total)[g]
      return jax.ops.segment_sum(l, g, num_segments=num_total)[g]

    def switch_strategy(l, g, idx):
      return jax.lax.switch(
        idx,
        [
          lambda x: apply_arithmetic(x[0], x[1]),
          lambda x: apply_geometric(x[0], x[1]),
          lambda x: apply_product(x[0], x[1]),
        ],
        (l, g)
      )

    return jax.vmap(switch_strategy, in_axes=(1, None, None), out_axes=1)(
      logits, tie_group_map, strategy_idx
    )

  def _combine_logits_multistate_idx(
    self,
    logits: Logits,
    group_mask: GroupMask,
    strategy_idx: Int,
    temperature: float = 1.0,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
  ) -> Logits:
    """Combine logits using strategy index (JAX-traceable version)."""
    def arithmetic_mean_fn(_: tuple) -> jnp.ndarray:
      return arithmetic_mean_logits(logits, group_mask, state_weights, state_mapping)

    def geometric_mean_fn(_: tuple) -> jnp.ndarray:
      return geometric_mean_logits(logits, group_mask, temperature, state_weights, state_mapping)

    def product_fn(_: tuple) -> jnp.ndarray:
      return product_of_probabilities_logits(logits, group_mask, state_weights, state_mapping)

    branches = [arithmetic_mean_fn, geometric_mean_fn, product_fn]
    return jax.lax.switch(strategy_idx, branches, ())

  def _process_group_positions(
    self,
    group_mask: GroupMask,
    all_layers_h: NodeFeatures,
    s_embed: NodeFeatures,
    encoder_context: NodeEdgeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    mask_bw: LinkMask,
  ) -> tuple[NodeFeatures, Logits]:
    """Process all positions in a group through decoder and collect logits."""
    num_residues = all_layers_h.shape[1]
    computed_logits = jnp.zeros((num_residues, 21))

    def process_one_position(idx: Int, state: tuple) -> tuple:
      position_all_layers_h, position_logits = state
      is_in_group = group_mask[idx]

      encoder_context_pos = encoder_context[idx]
      neighbor_indices_pos = neighbor_indices[idx]
      mask_pos = mask[idx]
      mask_bw_pos = mask_bw[idx]

      edge_sequence_features = concatenate_neighbor_nodes(
        s_embed,
        edge_features[idx],
        neighbor_indices_pos,
      )

      for layer_idx, layer in enumerate(self.decoder.layers):
        h_in_pos = position_all_layers_h[layer_idx, idx]

        decoder_context_pos = concatenate_neighbor_nodes(
          position_all_layers_h[layer_idx],
          edge_sequence_features,
          neighbor_indices_pos,
        )

        decoding_context = mask_bw_pos[..., None] * decoder_context_pos + encoder_context_pos

        h_in_expanded = jnp.expand_dims(h_in_pos, axis=0)
        decoding_context_expanded = jnp.expand_dims(decoding_context, axis=0)

        h_out_pos = layer(
          h_in_expanded,
          decoding_context_expanded,
          mask=mask_pos,
          key=None,
        )

        position_all_layers_h = position_all_layers_h.at[layer_idx + 1, idx].set(
          jnp.squeeze(h_out_pos),
        )

      final_h_pos = position_all_layers_h[-1, idx]
      logits_pos = self.w_out(final_h_pos)

      position_logits = jnp.where(
        is_in_group,
        position_logits.at[idx].set(logits_pos),
        position_logits,
      )

      return position_all_layers_h, position_logits

    return jax.lax.fori_loop(
      0,
      num_residues,
      process_one_position,
      (all_layers_h, computed_logits),
    )

  def _run_tied_position_scan(
    self,
    prng_key: PRNGKeyArray,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    encoder_context: NodeEdgeFeatures,
    mask_bw: LinkMask,
    temperature: Float,
    bias: Logits,
    tie_group_map: TieGroupMap,
    decoding_order: DecodingOrder,
    multi_state_strategy_idx: Int = 0,
    multi_state_temperature: Float = 1.0,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run group-based autoregressive scan with logit combining."""
    num_residues = node_features.shape[0]
    if tie_group_map is None:
      tie_group_map = jnp.arange(num_residues)
    groups_in_order = tie_group_map[decoding_order]
    position_indices = jnp.arange(num_residues)
    is_before_mask = position_indices[:, None] > position_indices[None, :]
    group_matches = groups_in_order[:, None] == groups_in_order[None, :]
    appeared_before = jnp.any(group_matches & is_before_mask, axis=1)
    is_first_occurrence = ~appeared_before
    group_decoding_order = jnp.compress(
      is_first_occurrence,
      groups_in_order,
      size=num_residues,
      fill_value=-1,
    )

    def group_autoregressive_step(
      carry: tuple[NodeFeatures, NodeFeatures, Logits, OneHotProteinSequence],
      scan_inputs: tuple[Int, PRNGKeyArray],
    ) -> tuple[
      tuple[NodeFeatures, NodeFeatures, Logits, OneHotProteinSequence],
      None,
    ]:
      all_layers_h, s_embed, all_logits, sequence = carry
      group_id, key = scan_inputs

      group_mask = tie_group_map == group_id

      all_layers_h, computed_logits = self._process_group_positions(
        group_mask,
        all_layers_h,
        s_embed,
        encoder_context,
        edge_features,
        neighbor_indices,
        mask,
        mask_bw,
      )

      combined_logits = self._combine_logits_multistate_idx(
        computed_logits,
        group_mask,
        multi_state_strategy_idx,
        multi_state_temperature,
        state_weights,
        state_mapping,
      )
      all_logits, s_embed, sequence = self._sample_and_broadcast_to_group(
        combined_logits,
        group_mask,
        bias,
        temperature,
        key,
        all_logits,
        s_embed,
        sequence,
        state_weights,
        state_mapping,
      )

      return (all_layers_h, s_embed, all_logits, sequence), None

    initial_all_layers_h = jnp.zeros(
      (self.num_decoder_layers + 1, num_residues, self.node_features_dim),
    )
    initial_all_layers_h = initial_all_layers_h.at[0].set(node_features)

    initial_s_embed = jnp.zeros_like(node_features)
    initial_all_logits = jnp.zeros((num_residues, self.w_out.out_features))
    initial_sequence = jnp.zeros((num_residues, self.w_s_embed.num_embeddings))

    initial_carry = (
      initial_all_layers_h,
      initial_s_embed,
      initial_all_logits,
      initial_sequence,
    )

    actual_num_groups = group_decoding_order.shape[0]
    scan_inputs = (group_decoding_order, jax.random.split(prng_key, actual_num_groups))

    final_carry, _ = jax.lax.scan(
      group_autoregressive_step,
      initial_carry,
      scan_inputs,
    )

    return final_carry[3], final_carry[2]

  def _sample_and_broadcast_to_group(
    self,
    avg_logits: Logits,
    group_mask: GroupMask,
    bias: Logits,
    temperature: Float,
    key: PRNGKeyArray,
    all_logits: Logits,
    s_embed: NodeFeatures,
    sequence: OneHotProteinSequence,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
  ) -> tuple[Logits, NodeFeatures, OneHotProteinSequence]:
    """Sample once and broadcast token to all positions in a group."""
    if state_weights is not None and state_mapping is not None:
      w = state_weights[state_mapping]
      group_bias = jnp.sum(
        jnp.where(group_mask[:, None], bias * w[:, None], 0.0),
        axis=0,
        keepdims=True,
      ) / jnp.sum(jnp.where(group_mask, w, 0.0))
    else:
      group_bias = jnp.sum(
        jnp.where(group_mask[:, None], bias, 0.0),
        axis=0,
        keepdims=True,
      ) / jnp.sum(group_mask)
      
    logits_with_bias = avg_logits + group_bias

    sampled_logits = (logits_with_bias / temperature) + jax.random.gumbel(
      key,
      logits_with_bias.shape,
      dtype=logits_with_bias.dtype,
    )
    sampled_logits_no_pad = sampled_logits[..., :20]
    one_hot_sample = straight_through_estimator(sampled_logits_no_pad)
    padding = jnp.zeros_like(one_hot_sample[..., :1])
    one_hot_seq = jnp.concatenate([one_hot_sample, padding], axis=-1)

    s_embed_new = one_hot_seq @ self.w_s_embed.weight
    all_logits = jnp.where(group_mask[:, None], jnp.squeeze(avg_logits), all_logits)
    s_embed = jnp.where(group_mask[:, None], jnp.squeeze(s_embed_new), s_embed)
    sequence = jnp.where(group_mask[:, None], jnp.squeeze(one_hot_seq), sequence)

    return all_logits, s_embed, sequence

  def _run_autoregressive_scan(  # noqa: PLR0915
    self,
    prng_key: PRNGKeyArray,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    autoregressive_mask: AutoRegressiveMask,
    temperature: Float,
    bias: Logits,
    tie_group_map: TieGroupMap | None = None,
    multi_state_strategy_idx: Int = 0,
    multi_state_temperature: Float = 1.0,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run JAX scan loop for autoregressive sampling."""
    num_residues = node_features.shape[0]

    attention_mask = jnp.take_along_axis(
      autoregressive_mask,
      neighbor_indices,
      axis=1,
    )
    mask_1d = mask[:, None]
    mask_bw = mask_1d * attention_mask
    mask_fw = mask_1d * (1 - attention_mask)
    decoding_order = jnp.argsort(jnp.sum(autoregressive_mask, axis=1))
    encoder_edge_neighbors = concatenate_neighbor_nodes(
      jnp.zeros_like(node_features),
      edge_features,
      neighbor_indices,
    )
    encoder_context = concatenate_neighbor_nodes(
      node_features,
      encoder_edge_neighbors,
      neighbor_indices,
    )
    encoder_context = encoder_context * mask_fw[..., None]

    def autoregressive_step(
      carry: tuple[NodeFeatures, NodeFeatures, Logits, OneHotProteinSequence],
      scan_inputs: tuple[Int, PRNGKeyArray],
    ) -> tuple[
      tuple[NodeFeatures, NodeFeatures, Logits, OneHotProteinSequence],
      None,
    ]:
      all_layers_h, s_embed, all_logits, sequence = carry
      position, key = scan_inputs

      encoder_context_pos = encoder_context[position]
      neighbor_indices_pos = neighbor_indices[position]
      mask_pos = mask[position]
      mask_bw_pos = mask_bw[position]

      edge_sequence_features = concatenate_neighbor_nodes(
        s_embed,
        edge_features[position],
        neighbor_indices_pos,
      )

      layer_keys = jax.random.split(key, len(self.decoder.layers))

      for layer_idx, layer in enumerate(self.decoder.layers):
        h_in_pos = all_layers_h[layer_idx, position]
        decoder_context_pos = concatenate_neighbor_nodes(
          all_layers_h[layer_idx],
          edge_sequence_features,
          neighbor_indices_pos,
        )
        decoding_context = mask_bw_pos[..., None] * decoder_context_pos + encoder_context_pos
        h_in_expanded = jnp.expand_dims(h_in_pos, axis=0)
        decoding_context_expanded = jnp.expand_dims(decoding_context, axis=0)
        h_out_pos = layer(
          h_in_expanded,
          decoding_context_expanded,
          mask=mask_pos,
          key=layer_keys[layer_idx],
        )
        all_layers_h = (
          cast("jax.Array", all_layers_h).at[layer_idx + 1, position].set(jnp.squeeze(h_out_pos))
        )

      final_h_pos = all_layers_h[-1, position]
      logits_pos_vec = self.w_out(final_h_pos)
      logits_pos = jnp.expand_dims(logits_pos_vec, axis=0)

      next_all_logits = cast("jax.Array", all_logits).at[position, :].set(jnp.squeeze(logits_pos))

      bias_pos = jax.lax.dynamic_slice(
        bias,
        (position, 0),
        (1, bias.shape[-1]),
      )
      logits_with_bias = logits_pos + bias_pos

      sampled_logits = (logits_with_bias / temperature) + jax.random.gumbel(
        key,
        logits_with_bias.shape,
        dtype=logits_with_bias.dtype,
      )
      sampled_logits_no_pad = sampled_logits[..., :20]
      one_hot_sample = straight_through_estimator(sampled_logits_no_pad)
      padding = jnp.zeros_like(one_hot_sample[..., :1])
      one_hot_seq_pos = jnp.concatenate([one_hot_sample, padding], axis=-1)
      s_embed_pos = one_hot_seq_pos @ self.w_s_embed.weight
      next_s_embed = cast("jax.Array", s_embed).at[position, :].set(jnp.squeeze(s_embed_pos))
      next_sequence = cast("jax.Array", sequence).at[position, :].set(jnp.squeeze(one_hot_seq_pos))

      return (
        all_layers_h,
        next_s_embed,
        next_all_logits,
        next_sequence,
      ), None

    if tie_group_map is None:
      initial_all_layers_h = jnp.zeros(
        (self.num_decoder_layers + 1, num_residues, self.node_features_dim),
      )
      initial_all_layers_h = initial_all_layers_h.at[0].set(node_features)
      initial_s_embed = jnp.zeros_like(node_features)
      initial_all_logits = jnp.zeros((num_residues, self.w_out.out_features))
      initial_sequence = jnp.zeros((num_residues, self.w_s_embed.num_embeddings))
      initial_carry = (
        initial_all_layers_h,
        initial_s_embed,
        initial_all_logits,
        initial_sequence,
      )
      scan_inputs = (decoding_order, jax.random.split(prng_key, num_residues))
      final_carry, _ = jax.lax.scan(
        autoregressive_step,
        initial_carry,
        scan_inputs,
      )
      final_sequence = final_carry[3]
      final_all_logits = final_carry[2]
      return final_sequence, final_all_logits

    return self._run_tied_position_scan(
      prng_key,
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      encoder_context,
      cast("jax.Array", mask_bw),
      temperature,
      bias,
      tie_group_map,
      decoding_order,
      multi_state_strategy_idx,
      multi_state_temperature,
      state_weights,
      state_mapping,
    )

  def __call__(
    self,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    decoding_approach: DecodingApproach,
    *,
    prng_key: PRNGKeyArray | None = None,
    ar_mask: AutoRegressiveMask | None = None,
    one_hot_sequence: OneHotProteinSequence | None = None,
    temperature: Float | None = None,
    bias: Logits | None = None,
    backbone_noise: BackboneNoise | None = None,
    tie_group_map: jnp.ndarray | None = None,
    multi_state_strategy: Literal[
      "arithmetic_mean",
      "geometric_mean",
      "product",
    ] = "arithmetic_mean",
    structure_mapping: jnp.ndarray | None = None,
    initial_node_features: jnp.ndarray | None = None,
    rbf_features: jnp.ndarray | None = None,
    neighbor_indices: jnp.ndarray | None = None,
    membrane_per_residue_labels: jnp.ndarray | None = None,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
    inference: bool = True,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Forward pass for the complete model."""
    if prng_key is None:
      prng_key = jax.random.PRNGKey(0)

    if membrane_per_residue_labels is not None:
      initial_node_features = jax.nn.one_hot(membrane_per_residue_labels, 3)

    prng_key, feat_key = jax.random.split(prng_key)

    if backbone_noise is None:
      backbone_noise = jnp.array(0.0, dtype=jnp.float32)

    edge_features, new_neighbor_indices, node_features, _ = self.features(
      feat_key,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      backbone_noise,
      structure_mapping=structure_mapping,
      initial_node_features=initial_node_features,
      rbf_features=rbf_features,
      neighbor_indices=neighbor_indices,
    )
    neighbor_indices = cast("jax.Array", new_neighbor_indices)

    node_features, edge_features = self.encoder(
      edge_features,
      neighbor_indices,
      mask,
      initial_node_features=node_features,
      inference=inference,
      key=prng_key,
    )

    branch_indices = {
      "unconditional": 0,
      "conditional": 1,
      "autoregressive": 2,
    }
    branch_index = branch_indices[decoding_approach]

    if ar_mask is None:
      ar_mask = jnp.zeros((mask.shape[0], mask.shape[0]), dtype=jnp.int32)

    if one_hot_sequence is None:
      one_hot_sequence = jnp.zeros(
        (mask.shape[0], self.w_s_embed.num_embeddings),
      )

    if temperature is None:
      temperature = jnp.array(1.0)

    if bias is None:
      bias = jnp.zeros((mask.shape[0], 21), dtype=jnp.float32)

    strategy_map = {"arithmetic_mean": 0, "geometric_mean": 1, "product": 2}
    multi_state_strategy_idx = jnp.array(
      strategy_map[multi_state_strategy],
      dtype=jnp.int32,
    )

    branches = [
      self._call_unconditional,
      self._call_conditional,
      self._call_autoregressive,
    ]

    operands = (
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      one_hot_sequence,
      prng_key,
      temperature,
      bias,
      tie_group_map,
      multi_state_strategy_idx,
      temperature,
      initial_node_features,
      state_weights,
      state_mapping,
    )
    return jax.lax.switch(branch_index, branches, *operands)


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

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    k_neighbors: int,
    num_context_layers: int = 1,
    num_positional_embeddings: int = 16,
    num_amino_acids: int = 21,
    vocab_size: int = 21,
    dropout_rate: float = 0.1,
    *,
    key: PRNGKeyArray,
  ) -> None:
    keys = jax.random.split(key, 5)
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features
    self.hidden_features_dim = hidden_features
    self.num_decoder_layers = num_decoder_layers

    self.features = ProteinFeaturesLigand(
      node_features=node_features,
      edge_features=edge_features,
      k_neighbors=k_neighbors,
      num_positional_embeddings=num_positional_embeddings,
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
    
    proj_keys = jax.random.split(jax.random.fold_in(key, 100), 7)

    self.context_encoder = tuple(
      DecoderLayer(node_features, node_features * 2, hidden_features, dropout_rate=dropout_rate, key=k)
      for k in context_keys
    )
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

  def _combine_logits_multistate_idx(
    self,
    logits: Logits,
    group_mask: GroupMask,
    strategy_idx: Int,
    temperature: float = 1.0,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
  ) -> Logits:
    """Combine logits using strategy index."""
    def arithmetic_mean_fn(_: tuple) -> jnp.ndarray:
      return arithmetic_mean_logits(logits, group_mask, state_weights, state_mapping)

    def geometric_mean_fn(_: tuple) -> jnp.ndarray:
      return geometric_mean_logits(logits, group_mask, temperature, state_weights, state_mapping)

    def product_fn(_: tuple) -> jnp.ndarray:
      return product_of_probabilities_logits(logits, group_mask, state_weights, state_mapping)

    branches = [arithmetic_mean_fn, geometric_mean_fn, product_fn]
    return jax.lax.switch(strategy_idx, branches, ())

  def _process_group_positions(
    self,
    group_mask: GroupMask,
    all_layers_h: NodeFeatures,
    s_embed: NodeFeatures,
    encoder_context: NodeEdgeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    mask_bw: LinkMask,
    inference: bool = True,
  ) -> tuple[NodeFeatures, Logits]:
    """Process positions in a group through LigandMPNN decoder."""
    num_residues = all_layers_h.shape[1]
    computed_logits = jnp.zeros((num_residues, 21))

    def process_one_position(idx: Int, state: tuple) -> tuple:
      position_all_layers_h, position_logits = state
      is_in_group = group_mask[idx]

      edge_sequence_features = concatenate_neighbor_nodes(
        s_embed,
        edge_features[idx],
        neighbor_indices[idx],
      )

      for layer_idx, layer in enumerate(self.decoder.layers):
        h_in_pos = position_all_layers_h[layer_idx, idx]
        decoder_context_pos = concatenate_neighbor_nodes(
          position_all_layers_h[layer_idx],
          edge_sequence_features,
          neighbor_indices[idx],
        )
        decoding_context = mask_bw[idx][..., None] * decoder_context_pos + encoder_context[idx]
        h_out_pos = layer(
          h_in_pos[None],
          decoding_context[None],
          mask=mask[idx],
          key=None,
          inference=inference,
        )
        position_all_layers_h = position_all_layers_h.at[layer_idx + 1, idx].set(
          jnp.squeeze(h_out_pos),
        )

      final_h_pos = position_all_layers_h[-1, idx]
      logits_pos = self.w_out(final_h_pos)
      position_logits = jnp.where(
        is_in_group,
        position_logits.at[idx].set(logits_pos),
        position_logits,
      )
      return position_all_layers_h, position_logits

    return jax.lax.fori_loop(
      0,
      num_residues,
      process_one_position,
      (all_layers_h, computed_logits),
    )

  def _sample_and_broadcast_to_group(
    self,
    avg_logits: Logits,
    group_mask: GroupMask,
    bias: Logits,
    temperature: Float,
    key: PRNGKeyArray,
    all_logits: Logits,
    s_embed: NodeFeatures,
    sequence: OneHotProteinSequence,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
  ) -> tuple[Logits, NodeFeatures, OneHotProteinSequence]:
    """Sample once and broadcast token to all positions in a group."""
    if state_weights is not None and state_mapping is not None:
      w = state_weights[state_mapping]
      group_bias = jnp.sum(
        jnp.where(group_mask[:, None], bias * w[:, None], 0.0),
        axis=0,
        keepdims=True,
      ) / jnp.sum(jnp.where(group_mask, w, 0.0))
    else:
      group_bias = jnp.sum(
        jnp.where(group_mask[:, None], bias, 0.0),
        axis=0,
        keepdims=True,
      ) / jnp.sum(group_mask)
      
    logits_with_bias = avg_logits + group_bias
    sampled_logits = (logits_with_bias / temperature) + jax.random.gumbel(key, logits_with_bias.shape)
    one_hot_sample = straight_through_estimator(sampled_logits[..., :20])
    padding = jnp.zeros_like(one_hot_sample[..., :1])
    one_hot_seq = jnp.concatenate([one_hot_sample, padding], axis=-1)

    s_embed_new = one_hot_seq @ self.w_s_embed.weight
    all_logits = jnp.where(group_mask[:, None], jnp.squeeze(avg_logits), all_logits)
    s_embed = jnp.where(group_mask[:, None], jnp.squeeze(s_embed_new), s_embed)
    sequence = jnp.where(group_mask[:, None], jnp.squeeze(one_hot_seq), sequence)
    return all_logits, s_embed, sequence

  def _run_tied_position_scan(
    self,
    prng_key: PRNGKeyArray,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    encoder_context: NodeEdgeFeatures,
    mask_bw: LinkMask,
    temperature: Float,
    bias: Logits,
    tie_group_map: TieGroupMap,
    decoding_order: DecodingOrder,
    multi_state_strategy_idx: Int = 0,
    multi_state_temperature: Float = 1.0,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
    inference: bool = True,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run group-based autoregressive scan for LigandMPNN."""
    num_residues = node_features.shape[0]
    groups_in_order = tie_group_map[decoding_order]
    position_indices = jnp.arange(num_residues)
    is_before_mask = position_indices[:, None] > position_indices[None, :]
    group_matches = groups_in_order[:, None] == groups_in_order[None, :]
    appeared_before = jnp.any(group_matches & is_before_mask, axis=1)
    is_first_occurrence = ~appeared_before
    group_decoding_order = jnp.compress(
      is_first_occurrence,
      groups_in_order,
      size=num_residues,
      fill_value=-1,
    )

    def group_autoregressive_step(carry, scan_inputs):
      all_layers_h, s_embed, all_logits, sequence = carry
      group_id, key = scan_inputs
      group_mask = tie_group_map == group_id

      all_layers_h, computed_logits = self._process_group_positions(
        group_mask, all_layers_h, s_embed, encoder_context, edge_features, neighbor_indices, mask, mask_bw, inference=inference
      )

      combined_logits = self._combine_logits_multistate_idx(
        computed_logits, group_mask, multi_state_strategy_idx, multi_state_temperature, state_weights, state_mapping
      )
      all_logits, s_embed, sequence = self._sample_and_broadcast_to_group(
        combined_logits, group_mask, bias, temperature, key, all_logits, s_embed, sequence, state_weights, state_mapping
      )
      return (all_layers_h, s_embed, all_logits, sequence), None

    initial_all_layers_h = jnp.zeros((len(self.decoder.layers) + 1, num_residues, self.node_features_dim))
    initial_all_layers_h = initial_all_layers_h.at[0].set(node_features)
    initial_carry = (
      initial_all_layers_h,
      jnp.zeros_like(node_features),
      jnp.zeros((num_residues, 21)),
      jnp.zeros((num_residues, 21))
    )
    keys = jax.random.split(prng_key, num_residues)
    final_carry, _ = jax.lax.scan(group_autoregressive_step, initial_carry, (group_decoding_order, keys))
    return final_carry[3], final_carry[2]

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
    backbone_noise: float = 0.0,
    tie_group_map: jnp.ndarray | None = None,
    multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
    inference: bool = True,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Forward pass for LigandMPNN sequence scoring or sampling."""
    if prng_key is None:
      prng_key = jax.random.PRNGKey(0)
    
    keys = jax.random.split(prng_key, 2)
    
    V, E, E_idx, Y_nodes, Y_edges, Y_m = self.features(
      keys[0], structure_coordinates, mask, residue_index, chain_index,
      Y, Y_t, Y_m, backbone_noise
    )
    
    h_V = jnp.zeros((E.shape[0], self.node_features_dim))
    h_E = E
    mask_2d = mask[:, None] * mask[None, :]
    mask_attend = jnp.take_along_axis(mask_2d, E_idx.astype(jnp.int32), axis=1)
    
    for layer in self.encoder.layers:
      h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend, inference=inference)
      
    h_V_C = jax.vmap(self.w_c)(h_V)
    h_E_context = jax.vmap(self.w_v)(V)
    Y_nodes = jax.vmap(jax.vmap(self.w_nodes_y))(Y_nodes)
    Y_edges = jax.vmap(jax.vmap(jax.vmap(self.w_edges_y)))(Y_edges)
    Y_m_edges = Y_m[..., None] * Y_m[..., None, :]
    
    for i in range(len(self.context_encoder)):
      Y_nodes = jax.vmap(lambda node, edge, mask_l, mask_e: 
                         self.y_context_encoder[i](node, edge, mask_l, attention_mask=mask_e, inference=inference)
                        )(Y_nodes, Y_edges, Y_m, Y_m_edges)
      h_E_context_cat = jnp.concatenate([h_E_context, Y_nodes], axis=-1)
      h_V_C = self.context_encoder[i](h_V_C, h_E_context_cat, mask, attention_mask=Y_m, inference=inference)
      
    h_V_C = jax.vmap(self.v_c)(h_V_C)
    h_V = h_V + jax.vmap(self.v_c_norm)(self.dropout(h_V_C, key=keys[1], inference=inference))
    
    if decoding_approach == "conditional":
      if one_hot_sequence is None:
        raise ValueError("one_hot_sequence MUST be provided for conditional decoding approach")
      node_decoded = self.decoder.call_conditional(
        h_V, h_E, E_idx, mask, ar_mask, one_hot_sequence, self.w_s_embed.weight, inference=inference
      )
      all_logits = jax.vmap(self.w_out)(node_decoded)
      if bias is not None:
        all_logits = all_logits + bias
      return one_hot_sequence, all_logits
      
    if decoding_approach == "autoregressive":
      if temperature is None:
        temperature = 1.0
      if bias is None:
        bias = jnp.zeros((mask.shape[0], 21))
      if ar_mask is None:
        ar_mask = jnp.zeros((mask.shape[0], mask.shape[0]))

      strategy_map = {"arithmetic_mean": 0, "geometric_mean": 1, "product": 2}
      strategy_idx = jnp.array(strategy_map[multi_state_strategy], dtype=jnp.int32)

      if tie_group_map is not None:
        # Precompute masks and order for tied scan
        attention_mask = jnp.take_along_axis(ar_mask, E_idx.astype(jnp.int32), axis=1)
        mask_bw = mask[:, None] * attention_mask
        mask_fw = mask[:, None] * (1 - attention_mask)
        decoding_order = jnp.argsort(jnp.sum(ar_mask, axis=1))
        
        encoder_edge_neighbors = concatenate_neighbor_nodes(jnp.zeros_like(h_V), h_E, E_idx)
        encoder_context = concatenate_neighbor_nodes(h_V, encoder_edge_neighbors, E_idx)
        encoder_context = encoder_context * mask_fw[..., None]

        return self._run_tied_position_scan(
          keys[0], h_V, h_E, E_idx, mask, encoder_context, mask_bw, temperature, bias, 
          tie_group_map, decoding_order, strategy_idx, temperature, state_weights, state_mapping, inference=inference
        )

      return self._run_autoregressive_scan(
        keys[0], h_V, h_E, E_idx, mask, ar_mask, temperature, bias, inference=inference
      )

    return None, None
