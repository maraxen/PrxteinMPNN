"""Main ProteinMPNN model implementation.

This module contains the top-level PrxteinMPNN model that combines all components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from prxteinmpnn.model.decoder import Decoder
from prxteinmpnn.model.encoder import Encoder, PhysicsEncoder
from prxteinmpnn.model.features import ProteinFeatures
from prxteinmpnn.model.multi_state_sampling import (
  max_min_over_group_logits,
  min_over_group_logits,
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
    physics_feature_dim: int | None = None,
    num_amino_acids: int = 21,
    vocab_size: int = 21,  # for w_s
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
      key=keys[0],
    )
    self.encoder = (
      Encoder(
        node_features,
        edge_features,
        hidden_features,
        num_encoder_layers,
        key=keys[1],
      )
      if physics_feature_dim is None
      else PhysicsEncoder(
        node_features,
        edge_features,
        hidden_features,
        num_encoder_layers,
        physics_feature_dim,
        key=keys[1],
      )
    )
    self.decoder = Decoder(
      node_features,
      edge_features,
      hidden_features,
      num_decoder_layers,
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
    _tie_group_map: jnp.ndarray | None,
    _multi_state_strategy_idx: Int,
    _multi_state_alpha: float,
    _initial_node_features: NodeFeatures | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the unconditional (scoring) path.

    Args:
      node_features: Node features from encoding.
      edge_features: Edge features from encoding.
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask.
      _ar_mask: Unused, required for jax.lax.switch signature.
      _one_hot_sequence: Unused, required for jax.lax.switch signature.
      prng_key: Unused, required for jax.lax.switch signature.
      _temperature: Unused, required for jax.lax.switch signature.
      _bias: Unused, required for jax.lax.switch signature.
      _tie_group_map: Unused, required for jax.lax.switch signature.
      _multi_state_strategy_idx: Unused, required for jax.lax.switch signature.
      _multi_state_alpha: Unused, required for jax.lax.switch signature.

    Returns:
      Tuple of (dummy sequence, logits).

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> model = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=key)
      >>> edge_feats = jnp.ones((10, 30, 128))
      >>> neighbor_idx = jnp.arange(300).reshape(10, 30)
      >>> mask = jnp.ones((10,))
      >>> seq, logits = model._call_unconditional(edge_feats, neighbor_idx, mask)

    """
    decoded_node_features = self.decoder(
      node_features,
      edge_features,
      neighbor_indices,  # Pass neighbor indices for correct context
      mask,
    )

    logits = jax.vmap(self.w_out)(decoded_node_features)

    # Return dummy sequence to match PyTree shape
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
    temperature: Float,
    bias: Logits,
    tie_group_map: jnp.ndarray | None,
    multi_state_strategy_idx: Int,
    multi_state_alpha: float,
    initial_node_features: NodeFeatures | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the conditional (scoring) path.

    Args:
      node_features: Node features from encoding.
      edge_features: Edge features from encoding.
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask.
      _ar_mask: Autoregressive mask for conditional decoding.
      one_hot_sequence: One-hot encoded protein sequence.
      prng_key: Unused, required for jax.lax.switch signature.
      _temperature: Unused, required for jax.lax.switch signature.
      _bias: Unused, required for jax.lax.switch signature.
      _tie_group_map: Unused, required for jax.lax.switch signature.
      _multi_state_strategy_idx: Unused, required for jax.lax.switch signature.
      _multi_state_alpha: Unused, required for jax.lax.switch signature.

    Returns:
      Tuple of (input sequence, logits).

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> model = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=key)
      >>> edge_feats = jnp.ones((10, 30, 128))
      >>> neighbor_idx = jnp.arange(300).reshape(10, 30)
      >>> mask = jnp.ones((10,))
      >>> ar_mask = jnp.ones((10, 10))
      >>> seq = jax.nn.one_hot(jnp.arange(10), 21)
      >>> out_seq, logits = model._call_conditional(
      ...     edge_feats, neighbor_idx, mask, ar_mask, seq
      ... )

    """
    decoded_node_features = self.decoder.call_conditional(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      one_hot_sequence,
      self.w_s_embed.weight,
    )
    logits = jax.vmap(self.w_out)(decoded_node_features)

    # Return input sequence to match PyTree shape
    return one_hot_sequence, logits

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
    tie_group_map: jnp.ndarray | None,
    multi_state_strategy_idx: Int,
    multi_state_alpha: float = 0.5,
    _initial_node_features: NodeFeatures | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the autoregressive (sampling) path.

    Args:
      node_features: Node features from encoding.
      edge_features: Edge features from encoding.
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask.
      ar_mask: Autoregressive mask for sampling.
      _one_hot_sequence: Unused, required for jax.lax.switch signature.
      prng_key: PRNG key for sampling.
      temperature: Temperature for Gumbel-max sampling.
      bias: Bias to add to logits before sampling (N, 21).
      tie_group_map: Optional (N,) array mapping each position to a group ID.
          When provided, positions in the same group sample identical amino acids.
      multi_state_strategy_idx: Integer index for strategy (0=mean, 1=min, 2=product, 3=max_min).
      multi_state_alpha: Weight for min component when strategy="max_min".

    Returns:
      Tuple of (sampled sequence, logits).

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> model = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=key)
      >>> edge_feats = jnp.ones((10, 30, 128))
      >>> neighbor_idx = jnp.arange(300).reshape(10, 30)
      >>> mask = jnp.ones((10,))
      >>> ar_mask = jnp.ones((10, 10))
      >>> temp = jnp.array(1.0)
      >>> bias = jnp.zeros((10, 21))
      >>> seq, logits = model._call_autoregressive(
      ...     edge_feats, neighbor_idx, mask, ar_mask, None, key, temp, bias, None
      ... )

    """
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
      multi_state_alpha,
    )
    return seq, logits

  @staticmethod
  def _average_logits_over_group(
    logits: Logits,
    group_mask: jnp.ndarray,
  ) -> jnp.ndarray:
    """Average logits across positions in a tie group using log-sum-exp.

    This implements numerically stable logit averaging for tied positions.
    Given logits of shape (N, 21) and a boolean mask indicating which
    positions belong to the current group, returns averaged logits of shape (1, 21).

    Args:
      logits: Logits array of shape (N, 21).
      group_mask: Boolean mask of shape (N,) indicating group membership.

    Returns:
      Averaged logits of shape (1, 21).

    Raises:
      None

    Example:
      >>> logits = jnp.array([[0.1, 0.9], [0.3, 0.7]])
      >>> group_mask = jnp.array([True, True])
      >>> avg_logits = PrxteinMPNN._average_logits_over_group(logits, group_mask)

    """
    max_logits = jnp.max(
      logits,
      where=group_mask[:, None],
      initial=-1e9,
      axis=0,
      keepdims=True,
    )  # (1, 21)

    shifted_logits = logits - max_logits  # (N, 21)
    exp_logits = jnp.exp(shifted_logits)  # (N, 21)

    masked_exp_logits = jnp.where(group_mask[:, None], exp_logits, 0.0)  # (N, 21)
    sum_exp_logits = jnp.sum(masked_exp_logits, axis=0, keepdims=True)  # (1, 21)

    num_in_group = jnp.sum(group_mask)
    avg_exp_logits = sum_exp_logits / num_in_group  # (1, 21)
    return jnp.log(avg_exp_logits) + max_logits  # (1, 21)

  @staticmethod
  def _combine_logits_multistate(
    logits: Logits,
    group_mask: jnp.ndarray,
    strategy: Literal["mean", "min", "product", "max_min"] = "mean",
    alpha: float = 0.5,
  ) -> jnp.ndarray:
    """Combine logits across tied positions using different multi-state strategies.

    Args:
      logits: Logits array of shape (N, 21).
      group_mask: Boolean mask of shape (N,) indicating group membership.
      strategy: Strategy for combining logits:
        - "mean": Average logits (consensus prediction, default)
        - "min": Minimum logits (worst-case robust design)
        - "product": Sum of logits (multiply probabilities)
        - "max_min": Weighted combination of min and mean (alpha controls weight)
      alpha: Weight for min component when strategy="max_min" (0=pure mean, 1=pure min).

    Returns:
      Combined logits of shape (1, 21).

    Example:
      >>> logits = jnp.array([[10.0, -5.0], [8.0, -3.0]])
      >>> group_mask = jnp.array([True, True])
      >>> # Average strategy (compromise)
      >>> avg = PrxteinMPNN._combine_logits_multistate(logits, group_mask, "mean")
      >>> # Min strategy (robust to worst case)
      >>> robust = PrxteinMPNN._combine_logits_multistate(logits, group_mask, "min")

    """
    if strategy == "mean":
      return PrxteinMPNN._average_logits_over_group(logits, group_mask)
    if strategy == "min":
      return min_over_group_logits(logits, group_mask)
    if strategy == "product":
      return product_of_probabilities_logits(logits, group_mask)
    if strategy == "max_min":
      return max_min_over_group_logits(logits, group_mask, alpha)
    msg = f"Unknown multi-state strategy: {strategy}"
    raise ValueError(msg)

  @staticmethod
  def _combine_logits_multistate_idx(
    logits: Logits,
    group_mask: jnp.ndarray,
    strategy_idx: Int,
    alpha: float = 0.5,
  ) -> jnp.ndarray:
    """Combine logits using strategy index (JAX-traceable version).

    This is a JAX-traceable wrapper around _combine_logits_multistate that
    accepts an integer strategy index instead of a string. Used internally
    when the function needs to be JIT-compiled.

    Args:
      logits: Logits array of shape (N, 21).
      group_mask: Boolean mask of shape (N,) indicating group membership.
      strategy_idx: Integer strategy index (0=mean, 1=min, 2=product, 3=max_min).
      alpha: Weight for min component when strategy_idx=3 (0=pure mean, 1=pure min).

    Returns:
      Combined logits of shape (1, 21).

    """

    def mean_fn(_: tuple) -> jnp.ndarray:
      return PrxteinMPNN._average_logits_over_group(logits, group_mask)

    def min_fn(_: tuple) -> jnp.ndarray:
      return min_over_group_logits(logits, group_mask)

    def product_fn(_: tuple) -> jnp.ndarray:
      return product_of_probabilities_logits(logits, group_mask)

    def max_min_fn(_: tuple) -> jnp.ndarray:
      return max_min_over_group_logits(logits, group_mask, alpha)

    branches = [mean_fn, min_fn, product_fn, max_min_fn]
    return jax.lax.switch(strategy_idx, branches, ())

  def _process_group_positions(
    self,
    group_mask: jnp.ndarray,
    all_layers_h: NodeFeatures,
    s_embed: NodeFeatures,
    encoder_context: jnp.ndarray,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    mask_bw: jnp.ndarray,
  ) -> tuple[NodeFeatures, jnp.ndarray]:
    """Process all positions in a group through decoder and collect logits.

    Args:
      group_mask: Boolean mask (N,) for positions in current group.
      all_layers_h: Hidden states (num_layers+1, N, C).
      s_embed: Sequence embeddings (N, C).
      encoder_context: Precomputed encoder context (N, K, features).
      edge_features: Edge features (N, K, C).
      neighbor_indices: Neighbor indices (N, K).
      mask: Alpha carbon mask (N,).
      mask_bw: Backward mask (N, K).

    Returns:
      Tuple of (updated all_layers_h, computed logits (N, 21)).

    """
    num_residues = all_layers_h.shape[1]
    computed_logits = jnp.zeros((num_residues, 21))

    def process_one_position(idx: Int, state: tuple) -> tuple:
      """Process one position through decoder layers."""
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

        h_out_pos = layer(h_in_expanded, decoding_context_expanded, mask=mask_pos)

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
    encoder_context: jnp.ndarray,
    mask_bw: jnp.ndarray,
    temperature: Float,
    bias: Logits,
    tie_group_map: jnp.ndarray,
    decoding_order: jnp.ndarray,
    multi_state_strategy_idx: Int = 0,
    multi_state_alpha: float = 0.5,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run group-based autoregressive scan with logit combining.

    Args:
      prng_key: PRNG key.
      node_features: Node features (N, C).
      edge_features: Edge features (N, K, C).
      neighbor_indices: Neighbor indices (N, K).
      mask: Alpha carbon mask (N,).
      encoder_context: Precomputed encoder context (N, K, features).
      mask_bw: Backward mask (N, K).
      temperature: Sampling temperature.
      bias: Logits array (N, 21).
      tie_group_map: Group mapping (N,).
      decoding_order: Position decoding order (N,).
      multi_state_strategy_idx: Integer strategy index (0=mean, 1=min, 2=product, 3=max_min).
      multi_state_alpha: Weight for min component when strategy_idx=3.

    Returns:
      Tuple of (final sequence, final logits).

    """
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
      """Process one group at a time with logit averaging."""
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
        multi_state_alpha,
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
    avg_logits: jnp.ndarray,
    group_mask: jnp.ndarray,
    bias: Logits,
    temperature: Float,
    key: PRNGKeyArray,
    all_logits: Logits,
    s_embed: NodeFeatures,
    sequence: OneHotProteinSequence,
  ) -> tuple[Logits, NodeFeatures, OneHotProteinSequence]:
    """Sample once and broadcast token to all positions in a group.

    Args:
      avg_logits: Averaged logits (1, 21).
      group_mask: Boolean mask (N,) for group positions.
      bias: Bias array (N, 21).
      temperature: Sampling temperature.
      key: PRNG key.
      all_logits: Current logits array (N, 21).
      s_embed: Current sequence embeddings (N, C).
      sequence: Current sequence (N, 21).

    Returns:
      Tuple of (updated all_logits, updated s_embed, updated sequence).

    """
    group_bias = jnp.sum(
      jnp.where(group_mask[:, None], bias, 0.0),
      axis=0,
      keepdims=True,
    ) / jnp.sum(group_mask)
    logits_with_bias = avg_logits + group_bias

    sampled_logits = (logits_with_bias / temperature) + jax.random.gumbel(
      key,
      logits_with_bias.shape,
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

  def _run_autoregressive_scan(
    self,
    prng_key: PRNGKeyArray,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    autoregressive_mask: AutoRegressiveMask,
    temperature: Float,
    bias: Logits,
    tie_group_map: jnp.ndarray | None = None,
    multi_state_strategy_idx: Int = 0,
    multi_state_alpha: float = 0.5,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run JAX scan loop for autoregressive sampling with optional tied positions.

    When tie_group_map is provided, the scan iterates over groups instead of
    individual positions. For each group:
    1. Decoder processes all positions in the group
    2. Logits are computed for all group members
    3. Logits are averaged across the group (log-sum-exp)
    4. A single token is sampled from the averaged logits
    5. The token is broadcast to all positions in the group

    Args:
      prng_key: PRNG key for sampling.
      node_features: Node features from encoder.
      edge_features: Edge features from encoder.
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask.
      autoregressive_mask: Mask defining decoding order.
      temperature: Temperature for Gumbel-max sampling.
      bias: Bias to add to logits before sampling (N, 21).
      tie_group_map: Optional (N,) array mapping each position to a group ID.
          When provided, positions in the same group are sampled together
          using combined logits.
      multi_state_strategy_idx: Integer strategy index (0=mean, 1=min, 2=product, 3=max_min).
      multi_state_alpha: Weight for min component when strategy_idx=3.

    Returns:
      Tuple of (sampled sequence, final logits).

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> model = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=key)
      >>> node_feats = jnp.ones((10, 128))
      >>> edge_feats = jnp.ones((10, 30, 128))
      >>> neighbor_idx = jnp.arange(300).reshape(10, 30)
      >>> mask = jnp.ones((10,))
      >>> ar_mask = jnp.ones((10, 10))
      >>> temp = jnp.array(1.0)
      >>> bias = jnp.zeros((10, 21))
      >>> seq, logits = model._run_autoregressive_scan(
      ...     key, node_feats, edge_feats, neighbor_idx, mask, ar_mask, temp, bias
      ... )

    """
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
    )  # [e_ij, 0_j]
    encoder_context = concatenate_neighbor_nodes(
      node_features,
      encoder_edge_neighbors,
      neighbor_indices,
    )  # [[e_ij, 0_j], h_j] = [e_ij, 0_j, h_j]
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

      encoder_context_pos = encoder_context[position]  # (K, 384)
      neighbor_indices_pos = neighbor_indices[position]  # (K,)
      mask_pos = mask[position]  # scalar
      mask_bw_pos = mask_bw[position]  # (K,)

      edge_sequence_features = concatenate_neighbor_nodes(
        s_embed,
        edge_features[position],
        neighbor_indices_pos,
      )  # (K, 256)

      for layer_idx, layer in enumerate(self.decoder.layers):
        # Get node features for this layer at current position
        h_in_pos = all_layers_h[layer_idx, position]  # [C]

        # Compute decoder context for this position
        decoder_context_pos = concatenate_neighbor_nodes(
          all_layers_h[layer_idx],
          edge_sequence_features,
          neighbor_indices_pos,
        )  # (K, 384)

        # Combine with encoder context using backward mask
        decoding_context = (
          mask_bw_pos[..., None] * decoder_context_pos + encoder_context_pos
        )  # (K, 384)

        # Expand dims for layer forward pass
        h_in_expanded = jnp.expand_dims(h_in_pos, axis=0)  # [1, C]
        decoding_context_expanded = jnp.expand_dims(decoding_context, axis=0)  # [1, K, 384]

        # Call DecoderLayer
        h_out_pos = layer(
          h_in_expanded,
          decoding_context_expanded,
          mask=mask_pos,
        )  # [1, C]

        # Update the state for next layer
        all_layers_h = all_layers_h.at[layer_idx + 1, position].set(jnp.squeeze(h_out_pos))

      # Sampling Step
      # Get final layer output for this position
      final_h_pos = all_layers_h[-1, position]  # [C]
      logits_pos_vec = self.w_out(final_h_pos)  # [21]
      logits_pos = jnp.expand_dims(logits_pos_vec, axis=0)  # [1, 21]

      next_all_logits = all_logits.at[position, :].set(jnp.squeeze(logits_pos))

      # Apply bias before sampling
      bias_pos = jax.lax.dynamic_slice(
        bias,
        (position, 0),
        (1, bias.shape[-1]),
      )
      logits_with_bias = logits_pos + bias_pos

      # Gumbel-max trick
      sampled_logits = (logits_with_bias / temperature) + jax.random.gumbel(
        key,
        logits_with_bias.shape,
      )
      sampled_logits_no_pad = sampled_logits[..., :20]  # Exclude padding

      one_hot_sample = straight_through_estimator(sampled_logits_no_pad)
      padding = jnp.zeros_like(one_hot_sample[..., :1])

      one_hot_seq_pos = jnp.concatenate([one_hot_sample, padding], axis=-1)

      s_embed_pos = one_hot_seq_pos @ self.w_s_embed.weight  # [1, C]

      next_s_embed = s_embed.at[position, :].set(jnp.squeeze(s_embed_pos))
      next_sequence = sequence.at[position, :].set(jnp.squeeze(one_hot_seq_pos))

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
      mask_bw,
      temperature,
      bias,
      tie_group_map,
      decoding_order,
      multi_state_strategy_idx,
      multi_state_alpha,
    )

  def __call__(  # noqa: PLR0913
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
    multi_state_strategy: Literal["mean", "min", "product", "max_min"] = "mean",
    multi_state_alpha: float = 0.5,
    structure_mapping: jnp.ndarray | None = None,
    initial_node_features: jnp.ndarray | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Forward pass for the complete model.

    Dispatches to one of three modes:
    1. "unconditional": Scores all positions in parallel.
    2. "conditional": Scores a given sequence.
    3. "autoregressive": Samples a new sequence.

    Args:
      structure_coordinates: Raw atomic coordinates of protein structure.
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices for each position.
      chain_index: Chain indices for each position.
      decoding_approach: One of "unconditional", "conditional", or "autoregressive".
      prng_key: PRNG key for random operations (optional).
      ar_mask: Autoregressive mask for decoding order (optional).
      one_hot_sequence: One-hot encoded sequence for conditional mode (optional).
      temperature: Temperature for autoregressive sampling (optional).
      bias: Optional bias to add to logits before sampling (N, 21) (optional).
      backbone_noise: Noise level for backbone coordinates (optional).
      tie_group_map: Optional (N,) array mapping each position to a group ID.
          When provided, positions in the same group sample identical amino acids
          using logit combining. Only used in "autoregressive" mode (optional).
      multi_state_strategy: Strategy for combining logits across tied positions.
          Options: "mean" (default, average), "min" (worst-case robust),
          "product" (multiply probabilities), "max_min" (weighted combination).
          Only used in "autoregressive" mode with tied positions (optional).
      multi_state_alpha: Weight for min component when multi_state_strategy="max_min".
          Range [0, 1] where 0=pure mean, 1=pure min (optional).
      structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                        When provided (multi-state mode), prevents cross-structure
                        neighbors to avoid information leakage between conformational states.
      initial_node_features: Optional (n_residues, feature_dim) physics features
                to use as initial node representations. If provided and the encoder
                is a PhysicsEncoder with use_initial_features=True, these will be
                used instead of zeros. Typically contains electrostatic features
                with shape (n_residues, 5).
      use_electrostatics: Whether to use electrostatic features in the physics encoder.
      _use_vdw: Whether to use van der Waals features in the physics encoder (not implemented).


    Returns:
      A tuple of (OneHotProteinSequence, Logits).
        - For "unconditional", the sequence is a zero-tensor.
        - For "conditional", the sequence is the input sequence.
        - For "autoregressive", the sequence is the newly sampled one.

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> model = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=key)
      >>> coords = jnp.ones((10, 4, 3))
      >>> mask = jnp.ones((10,))
      >>> residue_idx = jnp.arange(10)
      >>> chain_idx = jnp.zeros((10,), dtype=jnp.int32)
      >>> seq, logits = model(
      ...     coords, mask, residue_idx, chain_idx, "unconditional", prng_key=key
      ... )

    """
    if prng_key is None:
      prng_key = jax.random.PRNGKey(0)

    prng_key, feat_key = jax.random.split(prng_key)

    if backbone_noise is None:
      backbone_noise = jnp.array(0.0, dtype=jnp.float32)

    edge_features, neighbor_indices, node_features, _ = self.features(
      feat_key,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      backbone_noise,
      structure_mapping=structure_mapping,
      initial_node_features=initial_node_features,
    )

    node_features, edge_features = self.encoder(
      edge_features,
      neighbor_indices,
      mask,
      node_features,
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

    strategy_map = {"mean": 0, "min": 1, "product": 2, "max_min": 3}
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
        multi_state_alpha,
        initial_node_features,
    )
    return jax.lax.switch(branch_index, branches, *operands)
