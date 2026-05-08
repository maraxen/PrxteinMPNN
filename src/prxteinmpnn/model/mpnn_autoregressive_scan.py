"""Autoregressive tied-group and position scan loops for :class:`~prxteinmpnn.model.mpnn.PrxteinMPNN`.

Extracted from :mod:`prxteinmpnn.model.mpnn` (Phase 5e-cont) to shrink ``mpnn.py`` without
changing JIT boundaries or public method names on the model class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from prxteinmpnn.model._shared import create_group_index_table
from prxteinmpnn.model.encoder import pack_encoder_context
from prxteinmpnn.model.mpnn_core import (
  autoregressive_decoding_context,
  edge_sequence_features_autoregressive,
)
from prxteinmpnn.padding import LENGTH_BUCKETS
from prxteinmpnn.utils.ste import straight_through_estimator

if TYPE_CHECKING:
  from prxteinmpnn.model.mpnn import PrxteinMPNN
  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    AutoRegressiveMask,
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
    TieGroupMap,
  )


def run_tied_position_scan(
  model: PrxteinMPNN,
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
  state_weights: jnp.ndarray | None = None,
  state_mapping: jnp.ndarray | None = None,
  fixed_mask: jnp.ndarray | None = None,
  fixed_tokens: jnp.ndarray | None = None,
  group_indices_table: jnp.ndarray | None = None,
  group_valid_table: jnp.ndarray | None = None,
  n_canonical: int | None = None,
) -> tuple[OneHotProteinSequence, Logits]:
  """Run group-based autoregressive scan with logit combining."""
  num_residues = node_features.shape[0]
  if tie_group_map is None:
    tie_group_map = jnp.arange(num_residues)

  if group_indices_table is None or group_valid_table is None:
    msg = "group_indices_table and group_valid_table must be provided for tied decoding."
    raise ValueError(msg)

  tied_fixed_mask = (
    jnp.zeros((num_residues,), dtype=jnp.bool_)
    if fixed_mask is None
    else fixed_mask.astype(jnp.bool_)
  )
  tied_fixed_tokens = (
    jnp.zeros((num_residues,), dtype=jnp.int32)
    if fixed_tokens is None
    else fixed_tokens.astype(jnp.int32)
  )
  logit_accum_dtype = model.w_out.weight.dtype
  oh_tied = jax.nn.one_hot(
    tied_fixed_tokens,
    model.w_s_embed.num_embeddings,
    dtype=logit_accum_dtype,
  )
  seq_zero_tied = jnp.zeros((num_residues, model.w_s_embed.num_embeddings), dtype=logit_accum_dtype)
  initial_sequence_from_fixed = jnp.where(tied_fixed_mask[:, None], oh_tied, seq_zero_tied)
  emb_w_tied = model.w_s_embed.weight.astype(logit_accum_dtype)
  initial_s_embed_from_fixed = initial_sequence_from_fixed @ emb_w_tied

  groups_in_order = tie_group_map[decoding_order]
  position_indices = jnp.arange(num_residues)
  is_before_mask = position_indices[:, None] > position_indices[None, :]
  group_matches = groups_in_order[:, None] == groups_in_order[None, :]
  appeared_before = jnp.any(group_matches & is_before_mask, axis=1)
  is_first_occurrence = ~appeared_before
  compress_size = n_canonical if n_canonical is not None else num_residues
  group_decoding_order = jnp.compress(
    is_first_occurrence,
    groups_in_order,
    size=compress_size,
    fill_value=-1,
  )

  def group_autoregressive_step(
    carry: tuple[NodeFeatures, NodeFeatures, Logits, OneHotProteinSequence],
    scan_inputs: tuple[Int, PRNGKeyArray],
  ) -> tuple[
    tuple[NodeFeatures, NodeFeatures, Logits, OneHotProteinSequence],
    None,
  ]:
    """Process one group at a time with logit averaging."""
    all_layers_h, s_embed, all_logits, sequence = carry
    group_id, key = scan_inputs

    def _skip_group(_: None) -> tuple:
      return (all_layers_h, s_embed, all_logits, sequence), None

    def _decode_group(_: None) -> tuple:
      group_indices = group_indices_table[group_id]
      valid_mask = group_valid_table[group_id]

      all_layers_h_updated, computed_logits = model._process_group_positions(  # noqa: SLF001
        group_indices,
        valid_mask,
        all_layers_h,
        s_embed,
        encoder_context,
        edge_features,
        neighbor_indices,
        mask,
        mask_bw,
      )

      group_mask = tie_group_map == group_id
      combined_logits = model._combine_logits_multistate_idx(  # noqa: SLF001
        computed_logits,
        group_mask,
        multi_state_strategy_idx,
        jnp.float32(1.0),
        state_weights,
        state_mapping,
      )
      all_logits_updated, s_embed_updated, sequence_updated = sample_and_broadcast_to_group(
        model,
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
        fixed_mask,
        fixed_tokens,
      )
      return (
        all_layers_h_updated,
        s_embed_updated,
        all_logits_updated,
        sequence_updated,
      ), None

    return jax.lax.cond(group_id < 0, _skip_group, _decode_group, operand=None)

  initial_all_layers_h = jnp.zeros(
    (model.num_decoder_layers + 1, num_residues, model.node_features_dim),
  )
  initial_all_layers_h = initial_all_layers_h.at[0].set(node_features)

  initial_s_embed = initial_s_embed_from_fixed
  initial_all_logits = jnp.zeros((num_residues, model.w_out.out_features), dtype=logit_accum_dtype)
  initial_sequence = initial_sequence_from_fixed

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
    unroll=1,
  )

  return final_carry[3], final_carry[2]


def sample_and_broadcast_to_group(
  model: PrxteinMPNN,
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
  fixed_mask: jnp.ndarray | None = None,
  fixed_tokens: jnp.ndarray | None = None,
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

  fixed_mask_array = (
    jnp.zeros_like(group_mask, dtype=jnp.bool_)
    if fixed_mask is None
    else fixed_mask.astype(jnp.bool_)
  )
  fixed_tokens_array = (
    jnp.zeros_like(group_mask, dtype=jnp.int32)
    if fixed_tokens is None
    else fixed_tokens.astype(jnp.int32)
  )
  group_fixed_mask = group_mask & fixed_mask_array
  has_fixed_token = jnp.any(group_fixed_mask)

  def _sample_group(_: None) -> jnp.ndarray:
    sampled_logits = (logits_with_bias / temperature) + jax.random.gumbel(
      key,
      logits_with_bias.shape,
      dtype=logits_with_bias.dtype,
    )
    sampled_logits_no_pad = sampled_logits[..., :20]
    one_hot_sample = straight_through_estimator(sampled_logits_no_pad)
    padding = jnp.zeros_like(one_hot_sample[..., :1])
    return jnp.concatenate([one_hot_sample, padding], axis=-1)

  def _fixed_group(_: None) -> jnp.ndarray:
    fixed_token = jnp.max(jnp.where(group_fixed_mask, fixed_tokens_array, -1))
    return jax.nn.one_hot(
      fixed_token,
      model.w_s_embed.num_embeddings,
      dtype=logits_with_bias.dtype,
    )[None, :]

  one_hot_seq = jax.lax.cond(has_fixed_token, _fixed_group, _sample_group, operand=None)

  s_embed_new = one_hot_seq @ model.w_s_embed.weight
  all_logits = jnp.where(group_mask[:, None], jnp.squeeze(avg_logits), all_logits)
  s_embed = jnp.where(group_mask[:, None], jnp.squeeze(s_embed_new), s_embed)
  sequence = jnp.where(group_mask[:, None], jnp.squeeze(one_hot_seq), sequence)

  return all_logits, s_embed, sequence


def run_autoregressive_scan(  # noqa: PLR0915
  model: PrxteinMPNN,
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
  state_weights: jnp.ndarray | None = None,
  state_mapping: jnp.ndarray | None = None,
  fixed_mask: jnp.ndarray | None = None,
  fixed_tokens: jnp.ndarray | None = None,
  group_indices_table: jnp.ndarray | None = None,
  group_valid_table: jnp.ndarray | None = None,
  num_groups: int | None = None,
) -> tuple[OneHotProteinSequence, Logits]:
  """Run JAX scan loop for autoregressive sampling with optional tied positions."""
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

  logit_accum_dtype = model.w_out.weight.dtype
  oh_fixed = jax.nn.one_hot(
    fixed_tokens_array,
    model.w_s_embed.num_embeddings,
    dtype=logit_accum_dtype,
  )
  seq_zero = jnp.zeros((num_residues, model.w_s_embed.num_embeddings), dtype=logit_accum_dtype)
  initial_sequence_from_fixed = jnp.where(fixed_mask_array[:, None], oh_fixed, seq_zero)
  emb_w = model.w_s_embed.weight.astype(logit_accum_dtype)
  initial_s_embed_from_fixed = initial_sequence_from_fixed @ emb_w

  attention_mask = jnp.take_along_axis(
    autoregressive_mask,
    neighbor_indices,
    axis=1,
  )
  mask_1d = mask[:, None]
  mask_bw = mask_1d * attention_mask
  mask_fw = mask_1d * (1 - attention_mask)
  decoding_order = jnp.argsort(jnp.sum(autoregressive_mask, axis=1))
  encoder_context = pack_encoder_context(
    node_features,
    edge_features,
    neighbor_indices,
    mask_fw,
  )

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

    edge_sequence_features = edge_sequence_features_autoregressive(
      s_embed,
      edge_features,
      neighbor_indices,
      position,
    )

    layer_keys = jax.random.split(key, len(model.decoder.layers))

    for layer_idx, layer in enumerate(model.decoder.layers):
      h_in_pos = all_layers_h[layer_idx, position]

      decoding_context = autoregressive_decoding_context(
        all_layers_h[layer_idx],
        edge_sequence_features,
        neighbor_indices_pos,
        encoder_context_pos,
        mask_bw_pos,
      )

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
    logits_pos_vec = model.w_out(final_h_pos)
    logits_pos = jnp.expand_dims(logits_pos_vec, axis=0)

    next_all_logits = cast("jax.Array", all_logits).at[position, :].set(jnp.squeeze(logits_pos))

    bias_pos = jax.lax.dynamic_slice(
      bias,
      (position, 0),
      (1, bias.shape[-1]),
    )
    logits_with_bias = logits_pos + bias_pos

    def _sample_position(_: None) -> jax.Array:
      sampled_logits = (logits_with_bias / temperature) + jax.random.gumbel(
        key,
        logits_with_bias.shape,
        dtype=logits_with_bias.dtype,
      )
      sampled_logits_no_pad = sampled_logits[..., :20]
      one_hot_sample = straight_through_estimator(sampled_logits_no_pad)
      padding = jnp.zeros_like(one_hot_sample[..., :1])
      return jnp.concatenate([one_hot_sample, padding], axis=-1)

    def _fixed_position(_: None) -> jax.Array:
      return jax.nn.one_hot(
        fixed_tokens_array[position],
        model.w_s_embed.num_embeddings,
        dtype=logits_with_bias.dtype,
      )[None, :]

    one_hot_seq_pos = jax.lax.cond(
      fixed_mask_array[position],
      _fixed_position,
      _sample_position,
      operand=None,
    )

    s_embed_pos = one_hot_seq_pos @ model.w_s_embed.weight

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
      (model.num_decoder_layers + 1, num_residues, model.node_features_dim),
    )
    initial_all_layers_h = initial_all_layers_h.at[0].set(node_features)

    initial_s_embed = initial_s_embed_from_fixed
    initial_all_logits = jnp.zeros(
      (num_residues, model.w_out.out_features), dtype=logit_accum_dtype,
    )
    initial_sequence = initial_sequence_from_fixed

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
      unroll=1,
    )
    final_sequence = final_carry[3]
    final_all_logits = final_carry[2]

    return final_sequence, final_all_logits

  if tie_group_map is not None and (group_indices_table is None or group_valid_table is None):
    max_bucket_size = max(LENGTH_BUCKETS)
    group_indices_table, group_valid_table = create_group_index_table(
      tie_group_map,
      max_bucket_size,
    )

  return run_tied_position_scan(
    model,
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
    state_weights,
    state_mapping,
    fixed_mask,
    fixed_tokens,
    group_indices_table,
    group_valid_table,
    n_canonical=num_groups,
  )
