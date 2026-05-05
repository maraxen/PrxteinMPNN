# TODO: Explore internal state-batching (K, N, ...) instead of super-sequence concatenation to optimize attention complexity.
"""Main ProteinMPNN model implementation.

This module contains the top-level PrxteinMPNN model that combines all components.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp

from prxteinmpnn.model.decoder import Decoder, DecoderLayer
from prxteinmpnn.model.encoder import Encoder, PhysicsEncoder
from prxteinmpnn.model.features import ProteinFeatures
from prxteinmpnn.model.ligand_features import ProteinFeaturesLigand
from prxteinmpnn.model.multi_state_sampling import (
  arithmetic_mean_logits,
  geometric_mean_logits,
  product_of_probabilities_logits,
)
from prxteinmpnn.padding import LENGTH_BUCKETS
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

def _create_group_index_table(
  tie_group_map: jnp.ndarray,
  max_group_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Create a table of indices belonging to each group.

  Args:
    tie_group_map: (N,) array of group IDs.
    max_group_size: Static maximum number of residues per group.

  Returns:
    group_indices: (N, max_group_size) table of member indices.
    valid_mask: (N, max_group_size) boolean mask for valid indices.
  """
  num_residues = tie_group_map.shape[0]
  # mask_matrix[g, i] = i if tie_group_map[i] == g else -1
  mask_matrix = jnp.where(
    tie_group_map[None, :] == jnp.arange(num_residues)[:, None],
    jnp.arange(num_residues)[None, :],
    -1,
  )

  def sort_row(row):
    is_valid = row >= 0
    return jnp.sort(jnp.where(is_valid, row, num_residues + 1))

  sorted_indices = jax.vmap(sort_row)(mask_matrix)
  group_indices = sorted_indices[:, :max_group_size]
  valid_mask = group_indices < num_residues

  return group_indices, valid_mask


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
    _initial_node_features: NodeFeatures | None,
    _state_weights: jnp.ndarray | None,
    _state_mapping: jnp.ndarray | None,
    _fixed_mask: jnp.ndarray | None,
    _fixed_tokens: jnp.ndarray | None,
    group_indices_table: jnp.ndarray | None,
    group_valid_table: jnp.ndarray | None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the unconditional (scoring) path.

    Args:
      node_features: Node features from encoding.
      edge_features: Edge features from encoding.
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask.
      _ar_mask: Unused, required for jax.lax.switch signature.
      _one_hot_sequence: Unused, required for jax.lax.switch signature.
      _prng_key: Unused, required for jax.lax.switch signature.
      _temperature: Unused, required for jax.lax.switch signature.
      _bias: Unused, required for jax.lax.switch signature.
      _tie_group_map: Unused, required for jax.lax.switch signature.
      _multi_state_strategy_idx: Unused, required for jax.lax.switch signature.
      _multi_state_temperature: Unused, required for jax.lax.switch signature.
      _initial_node_features: Unused.
      _state_weights: Weights for each structural state.
      _state_mapping: Mapping of each residue to its state index.

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
      key=_prng_key,
    )

    logits = jax.vmap(self.w_out)(decoded_node_features)

    # Multi-state logit combining for unconditional mode (consensus likelihood)
    if _tie_group_map is not None:
      logits = self._apply_multistate_to_all_logits(
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
    _initial_node_features: NodeFeatures | None,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    _fixed_mask: jnp.ndarray | None,
    _fixed_tokens: jnp.ndarray | None,
    group_indices_table: jnp.ndarray | None,
    group_valid_table: jnp.ndarray | None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the conditional (scoring) path.

    Args:
      node_features: Node features from encoding.
      edge_features: Edge features from encoding.
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask.
      ar_mask: Autoregressive mask for conditional decoding.
      one_hot_sequence: One-hot encoded protein sequence.
      prng_key: PRNG Key.
      _temperature: Unused, required for jax.lax.switch signature.
      _bias: Unused, required for jax.lax.switch signature.
      tie_group_map: Group mapping for multi-state scoring (consensus).
      multi_state_strategy_idx: Strategy index for combining logits.
      multi_state_temperature: Temperature for geometric_mean strategy.
      _initial_node_features: Unused.
      state_weights: Weights for each structural state.
      state_mapping: Mapping of each residue to its state index.

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
      key=prng_key,
    )
    logits = jax.vmap(self.w_out)(decoded_node_features)

    # Multi-state logit combining for consensus likelihood
    if tie_group_map is not None:
      logits = self._apply_multistate_to_all_logits(
        logits,
        tie_group_map,
        multi_state_strategy_idx,
        multi_state_temperature,
        state_weights,
        state_mapping,
      )

    # Return input sequence to match PyTree shape
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
    multi_state_temperature: Float,
    _initial_node_features: NodeFeatures | None,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    fixed_mask: jnp.ndarray | None,
    fixed_tokens: jnp.ndarray | None,
    group_indices_table: jnp.ndarray | None,
    group_valid_table: jnp.ndarray | None,
    *,
    num_groups: int | None = None,
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
      multi_state_strategy_idx: Integer index for strategy
          (0=arithmetic_mean, 1=geometric_mean, 2=product).
      multi_state_temperature: Temperature for geometric_mean strategy.
      _initial_node_features: Unused.
      state_weights: Weights for each structural state.
      state_mapping: Mapping of each residue to its state index.
      group_indices_table: Pre-calculated table of group indices.
      group_valid_table: Pre-calculated boolean valid mask for group indices.

    Returns:
      Tuple of (sampled sequence, logits).

    Raises:
      None

    Warning:
      This path is NOT differentiable. Sequence sampling uses Gumbel-max noise +
      straight-through estimator (STE), creating a hard gradient barrier. Do not
      wrap calls with jax.grad / jax.value_and_grad / eqx.filter_grad — gradients
      will be zero or incorrect. For differentiable sequence optimization, use
      decoding_approach="conditional" (teacher-forced parallel decode).

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
      multi_state_temperature,
      state_weights,
      state_mapping,
      fixed_mask,
      fixed_tokens,
      group_indices_table,
      group_valid_table,
      num_groups=num_groups,
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
    """Combine logits across tied positions using different multi-state strategies.

    Args:
      logits: Logits array of shape (N, 21).
      group_mask: Boolean mask of shape (N,) indicating group membership.
      strategy: Strategy for combining logits:
        - "arithmetic_mean": Average logits using log-sum-exp (consensus prediction, default)
        - "geometric_mean": Geometric mean of probabilities with temperature scaling
        - "product": Sum of logits (multiply probabilities)
      temperature: Temperature for geometric_mean strategy.
      state_weights: Weights for each structural state.
      state_mapping: Mapping of each residue to its state index.

    Returns:
      Combined logits of shape (1, 21).

    Example:
      >>> logits = jnp.array([[10.0, -5.0], [8.0, -3.0]])
      >>> group_mask = jnp.array([True, True])
      >>> # Arithmetic mean strategy (compromise)
      >>> avg = PrxteinMPNN._combine_logits_multistate(logits, group_mask, "arithmetic_mean")
      >>> # Product strategy (multiply probabilities)
      >>> prod = PrxteinMPNN._combine_logits_multistate(logits, group_mask, "product")

    """
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
    """Apply multi-state combination strategies across ALL groups in parallel.

    Args:
      logits: Logits array of shape (N, 21).
      tie_group_map: Group mapping of shape (N,).
      strategy_idx: Integer strategy index.
      temperature: Temperature for geometric_mean strategy.
      state_weights: Weights for each structural state.
      state_mapping: Mapping of each residue to its state index.

    Returns:
      Combined logits of shape (N, 21). Each position in a group will contain
      the SAME combined/consensus logit.
    """
    num_total = tie_group_map.shape[0]

    def apply_arithmetic(l: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
      # Segment-wise LogSumExp
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
        avg_l = sum_wl / (jnp.where(sum_w > 0, sum_w, 1.0) * temperature)
        return avg_l[g]

      sum_l = jax.ops.segment_sum(l, g, num_segments=num_total)
      count = jax.ops.segment_sum(jnp.ones_like(g, dtype=jnp.float32), g, num_segments=num_total)
      avg_l = sum_l / (jnp.where(count > 0, count, 1.0) * temperature)
      return avg_l[g]

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
        (l, g),
      )

    # vmap over the 21 classes
    return jax.vmap(switch_strategy, in_axes=(1, None, None), out_axes=1)(
      logits,
      tie_group_map,
      strategy_idx,
    )

  @staticmethod
  def _combine_logits_multistate_idx(
    logits: Logits,
    group_mask: GroupMask,
    strategy_idx: Int,
    temperature: float = 1.0,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
  ) -> Logits:
    """Combine logits using strategy index (JAX-traceable version).

    This is a JAX-traceable wrapper around _combine_logits_multistate that
    accepts an integer strategy index instead of a string. Used internally
    when the function needs to be JIT-compiled.

    Args:
      logits: Logits array of shape (N, 21).
      group_mask: Boolean mask of shape (N,) indicating group membership.
      strategy_idx: Integer strategy index (0=arithmetic_mean, 1=geometric_mean, 2=product).
      temperature: Temperature for geometric_mean strategy.
      state_weights: Weights for each structural state.
      state_mapping: Mapping of each residue to its state index.

    Returns:
      Combined logits of shape (1, 21).

    """

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
    group_indices: jnp.ndarray,
    valid_mask: jnp.ndarray,
    all_layers_h: NodeFeatures,
    s_embed: NodeFeatures,
    encoder_context: NodeEdgeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    mask_bw: LinkMask,
  ) -> tuple[NodeFeatures, Logits]:
    """Process only positions belonging to the current group through decoder.

    Args:
      group_indices: (S,) array of indices in the current tie group.
      valid_mask: (S,) boolean mask indicating valid indices.
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
    max_group_size = group_indices.shape[0]
    computed_logits = jnp.zeros((num_residues, 21), dtype=self.w_out.weight.dtype)

    def _decode_position(
      position_all_layers_h: jax.Array,
      idx: Int,
    ) -> tuple[jax.Array, jax.Array]:
      """Decode one position and return updated hidden state + logits."""
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
          key=None,  # Dropout will use inference mode set on model
        )

        position_all_layers_h = position_all_layers_h.at[layer_idx + 1, idx].set(
          jnp.squeeze(h_out_pos),
        )

      final_h_pos = position_all_layers_h[-1, idx]
      logits_pos = self.w_out(final_h_pos)
      return position_all_layers_h, logits_pos

    def process_one_member(i: Int, state: tuple) -> tuple:
      """Process i-th member of the group if valid."""
      position_all_layers_h, position_logits = state
      idx = group_indices[i]
      is_valid = valid_mask[i]

      def _process(_: None) -> tuple:
        updated_h, logits_pos = _decode_position(position_all_layers_h, idx)
        return updated_h, position_logits.at[idx].set(logits_pos)

      return jax.lax.cond(
        is_valid,
        _process,
        lambda _: (position_all_layers_h, position_logits),
        operand=None,
      )

    return jax.lax.fori_loop(
      0,
      max_group_size,
      process_one_member,
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
    fixed_mask: jnp.ndarray | None = None,
    fixed_tokens: jnp.ndarray | None = None,
    group_indices_table: jnp.ndarray | None = None,
    group_valid_table: jnp.ndarray | None = None,
    n_canonical: int | None = None,
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
      multi_state_strategy_idx: Integer strategy index
          (0=arithmetic_mean, 1=geometric_mean, 2=product).
      multi_state_temperature: Temperature for geometric_mean strategy.
      state_weights: Weights for each structural state.
      state_mapping: Mapping of each residue to its state index.
      max_group_size: Static maximum number of residues per group.
      n_canonical: Static number of unique canonical groups. Must be provided
          for multistate designs to avoid iterating over padded positions.

    Returns:
      Tuple of (final sequence, final logits).

    """
    num_residues = node_features.shape[0]
    if tie_group_map is None:
      tie_group_map = jnp.arange(num_residues)

    # If tables aren't provided (e.g. from standard ProteinMPNN path), compute them.
    # Note: This might still trigger tracing issues if not careful,
    # but for design grid we ensure they are passed.
    if group_indices_table is None or group_valid_table is None:
        # Fallback to O(N) internal computation if tables not passed
        # We need a static max_group_size here.
        # For simplicity and to avoid tracing issues in general use,
        # we can just use a large enough static size if not in design grid.
        # But here we assume they ARE passed in the performance critical path.
        msg = "group_indices_table and group_valid_table must be provided for tied decoding."
        raise ValueError(msg)

    max_group_size = group_indices_table.shape[1]

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
    logit_accum_dtype = self.w_out.weight.dtype
    oh_tied = jax.nn.one_hot(
      tied_fixed_tokens,
      self.w_s_embed.num_embeddings,
      dtype=logit_accum_dtype,
    )
    seq_zero_tied = jnp.zeros((num_residues, self.w_s_embed.num_embeddings), dtype=logit_accum_dtype)
    initial_sequence_from_fixed = jnp.where(tied_fixed_mask[:, None], oh_tied, seq_zero_tied)
    emb_w_tied = self.w_s_embed.weight.astype(logit_accum_dtype)
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

        all_layers_h_updated, computed_logits = self._process_group_positions(
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
        combined_logits = self._combine_logits_multistate_idx(
          computed_logits,
          group_mask,
          multi_state_strategy_idx,
          multi_state_temperature,
          state_weights,
          state_mapping,
        )
        all_logits_updated, s_embed_updated, sequence_updated = self._sample_and_broadcast_to_group(
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
      (self.num_decoder_layers + 1, num_residues, self.node_features_dim),
    )
    initial_all_layers_h = initial_all_layers_h.at[0].set(node_features)

    initial_s_embed = initial_s_embed_from_fixed
    initial_all_logits = jnp.zeros((num_residues, self.w_out.out_features), dtype=logit_accum_dtype)
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
    fixed_mask: jnp.ndarray | None = None,
    fixed_tokens: jnp.ndarray | None = None,
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
      state_weights: Weights for each structural state.
      state_mapping: Mapping of each residue to its state index.

    Returns:
      Tuple of (updated all_logits, updated s_embed, updated sequence).

    """
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
        self.w_s_embed.num_embeddings,
        dtype=logits_with_bias.dtype,
      )[None, :]

    one_hot_seq = jax.lax.cond(has_fixed_token, _fixed_group, _sample_group, operand=None)

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
    fixed_mask: jnp.ndarray | None = None,
    fixed_tokens: jnp.ndarray | None = None,
    group_indices_table: jnp.ndarray | None = None,
    group_valid_table: jnp.ndarray | None = None,
    num_groups: int | None = None,
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
      multi_state_strategy_idx: Integer strategy index
          (0=arithmetic_mean, 1=geometric_mean, 2=product).
      multi_state_temperature: Temperature for geometric_mean strategy.
      state_weights: Weights for each structural state.
      state_mapping: Mapping of each residue to its state index.

    Returns:
      Tuple of (sampled sequence, final logits).

    Raises:
      None

    Note:
      Non-differentiable: Gumbel sampling + STE blocks gradient flow. See
      _call_autoregressive for details.

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
    oh_fixed = jax.nn.one_hot(
      fixed_tokens_array,
      self.w_s_embed.num_embeddings,
      dtype=logit_accum_dtype,
    )
    seq_zero = jnp.zeros((num_residues, self.w_s_embed.num_embeddings), dtype=logit_accum_dtype)
    initial_sequence_from_fixed = jnp.where(fixed_mask_array[:, None], oh_fixed, seq_zero)
    emb_w = self.w_s_embed.weight.astype(logit_accum_dtype)
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
        # Get node features for this layer at current position
        h_in_pos = all_layers_h[layer_idx, position]

        # Compute decoder context for this position
        decoder_context_pos = concatenate_neighbor_nodes(
          all_layers_h[layer_idx],
          edge_sequence_features,
          neighbor_indices_pos,
        )

        # Combine with encoder context using backward mask
        decoding_context = mask_bw_pos[..., None] * decoder_context_pos + encoder_context_pos

        # Expand dims for layer forward pass
        h_in_expanded = jnp.expand_dims(h_in_pos, axis=0)
        decoding_context_expanded = jnp.expand_dims(decoding_context, axis=0)

        # Call DecoderLayer
        h_out_pos = layer(
          h_in_expanded,
          decoding_context_expanded,
          mask=mask_pos,
          key=layer_keys[layer_idx],
        )

        # Update the state for next layer
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

      def _sample_position(_: None) -> jax.Array:
        sampled_logits = (logits_with_bias / temperature) + jax.random.gumbel(
          key,
          logits_with_bias.shape,
          dtype=logits_with_bias.dtype,
        )
        sampled_logits_no_pad = sampled_logits[..., :20]  # Exclude padding
        one_hot_sample = straight_through_estimator(sampled_logits_no_pad)
        padding = jnp.zeros_like(one_hot_sample[..., :1])
        return jnp.concatenate([one_hot_sample, padding], axis=-1)

      def _fixed_position(_: None) -> jax.Array:
        return jax.nn.one_hot(
          fixed_tokens_array[position],
          self.w_s_embed.num_embeddings,
          dtype=logits_with_bias.dtype,
        )[None, :]

      one_hot_seq_pos = jax.lax.cond(
        fixed_mask_array[position],
        _fixed_position,
        _sample_position,
        operand=None,
      )

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

      initial_s_embed = initial_s_embed_from_fixed
      initial_all_logits = jnp.zeros((num_residues, self.w_out.out_features), dtype=logit_accum_dtype)
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
      # all_logits is returned as final_carry[2]; caller reads it on the host and
      # writes to disk in one bulk call after the scan returns. No io_callback needed.
      final_sequence = final_carry[3]
      final_all_logits = final_carry[2]

      return final_sequence, final_all_logits

    if tie_group_map is not None and (group_indices_table is None or group_valid_table is None):
        # We need a static max_group_size for the table shape.
        # Use LENGTH_BUCKETS ceiling to avoid recompilation across different protein lengths.
        max_bucket_size = max(LENGTH_BUCKETS)
        group_indices_table, group_valid_table = _create_group_index_table(
            tie_group_map, max_bucket_size
        )

    # all_logits is returned as final_carry[2]; caller reads it on the host and
    # writes to disk in one bulk call after the scan returns. No io_callback needed.
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
      fixed_mask,
      fixed_tokens,
      group_indices_table,
      group_valid_table,
      n_canonical=num_groups,
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
    fixed_mask: jnp.ndarray | None = None,
    fixed_tokens: jnp.ndarray | None = None,
    backbone_noise: BackboneNoise | None = None,
    tie_group_map: jnp.ndarray | None = None,
    group_indices_table: jnp.ndarray | None = None,
    group_valid_table: jnp.ndarray | None = None,
    multi_state_strategy: Literal[
      "arithmetic_mean",
      "geometric_mean",
      "product",
    ] = "arithmetic_mean",
    multi_state_temperature: Float = 1.0,
    multistate_mode: Literal["flat", "state_vmap", "state_vmap_exact"] = "flat",
    structure_mapping: jnp.ndarray | None = None,
    initial_node_features: jnp.ndarray | None = None,
    rbf_features: jnp.ndarray | None = None,
    neighbor_indices: jnp.ndarray | None = None,
    precomputed_node_features: jnp.ndarray | None = None,
    precomputed_edge_features: jnp.ndarray | None = None,
    precomputed_neighbor_indices: jnp.ndarray | None = None,
    membrane_per_residue_labels: jnp.ndarray | None = None,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
    num_groups: int | None = None,
    wave_group_ids: jnp.ndarray | None = None,
    wave_group_positions: jnp.ndarray | None = None,
    wave_group_valid: jnp.ndarray | None = None,
    wave_position_valid: jnp.ndarray | None = None,
    inference: bool = True,
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
          Options: "arithmetic_mean" (default, log-sum-exp average),
          "geometric_mean" (geometric mean with temperature scaling),
          "product" (multiply probabilities).
          Only used in "autoregressive" mode with tied positions (optional).
      structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                        When provided (multi-state mode), prevents cross-structure
                        neighbors to avoid information leakage between conformational states.
      initial_node_features: Optional (n_residues, feature_dim) physics features
                to use as initial node representations. If provided and the encoder
                is a PhysicsEncoder with use_initial_features=True, these will be
                used instead of zeros. Typically contains electrostatic features
                with shape (n_residues, 5).
      state_weights: Weights for each structural state.
      state_mapping: Mapping of each residue to its state index.

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

    # Automatically handle membrane labels if provided
    if membrane_per_residue_labels is not None:
      # Assume 3 classes for standard membrane MPNN (0: unknown/rest, 1: interface, 2: buried)
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

    ar_fn = (
      partial(self._call_autoregressive, num_groups=num_groups)
      if num_groups is not None
      else self._call_autoregressive
    )
    branches = [
      self._call_unconditional,
      self._call_conditional,
      ar_fn,
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
      multi_state_temperature,
      initial_node_features,
      state_weights,
      state_mapping,
      fixed_mask,
      fixed_tokens,
      group_indices_table,
      group_valid_table,
    )
    return jax.lax.switch(branch_index, branches, *operands)

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
    wave_group_ids_local: jax.Array,
    wave_group_positions_local: jax.Array,
    wave_group_valid_local: jax.Array,
    wave_position_valid_local: jax.Array,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Stacked-graph wave-parallel sampler for ProteinMPNN (``state_vmap_exact``).

    Outputs one-hot sequences and logits with shape ``(n_states, n_pad, ·)``.
    """
    del tie_group_map_stack
    pk_cur, fk = jax.random.split(prng_key)
    temp = jnp.asarray(temperature, dtype=jnp.float32)
    feat_keys = jax.random.split(fk, coords_stack.shape[0])

    def encode_one(coords, ma, ri, ci, k):
      ef, nei, nf, _ = self.features(
        k,
        coords,
        ma,
        ri,
        ci,
        jnp.asarray(0.0, jnp.float32),
        structure_mapping=None,
        initial_node_features=None,
        rbf_features=None,
        neighbor_indices=None,
      )
      nf2, ef2 = self.encoder(ef, nei, ma, initial_node_features=nf, inference=True, key=k)
      return nf2, ef2, nei.astype(jnp.int32)

    node_b, edge_b, nei_b = jax.vmap(encode_one)(
      coords_stack,
      mask_stack,
      residue_index_stack,
      chain_index_stack,
      feat_keys,
    )

    attn_b = jax.vmap(
      lambda am, nei_i: jnp.take_along_axis(am, nei_i.astype(jnp.int32), axis=1),
    )(autoregressive_mask_stack, nei_b)
    mask_bw = mask_stack[:, :, jnp.newaxis] * attn_b
    mask_fw = mask_stack[:, :, jnp.newaxis] * (jnp.float32(1.0) - jnp.asarray(attn_b, jnp.float32))
    enc_neighbors = jax.vmap(concatenate_neighbor_nodes)(
      jnp.zeros_like(node_b),
      edge_b,
      nei_b,
    )
    encoder_stack = jax.vmap(concatenate_neighbor_nodes)(node_b, enc_neighbors, nei_b)
    encoder_stack *= mask_fw[..., jnp.newaxis]

    log_dtype = self.w_out.weight.dtype
    num_dec = len(self.decoder.layers) + 1
    d_hid = node_b.shape[-1]
    s_dim, p_pad = int(coords_stack.shape[0]), int(coords_stack.shape[1])

    ah = jnp.zeros((s_dim, num_dec, p_pad, d_hid), dtype=log_dtype)
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
      dk: jax.Array,
      *,
      eb: jax.Array,
      nei: jax.Array,
      mw_atom: jax.Array,
      ecx: jax.Array,
      bw: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
      ecx_row = jax.lax.dynamic_index_in_dim(ecx, lid_s, axis=0).squeeze(0)
      nei_row = jax.lax.dynamic_index_in_dim(nei, lid_s, axis=0).squeeze(0)
      mask_pos = jax.lax.dynamic_index_in_dim(mw_atom, lid_s, axis=0).squeeze(0)
      bw_row = jax.lax.dynamic_index_in_dim(bw, lid_s, axis=0).squeeze(0)
      edge_slice = jax.lax.dynamic_index_in_dim(eb, lid_s, axis=0).squeeze(0)
      seq_nf = concatenate_neighbor_nodes(se_s, edge_slice, nei_row)
      sk_l = jax.random.split(dk, len(self.decoder.layers))
      ah_cur = ah_s
      for lyr_i, layer in enumerate(self.decoder.layers):
        h_in = jax.lax.dynamic_index_in_dim(ah_cur[lyr_i], lid_s, axis=0).squeeze(0)
        dctx = concatenate_neighbor_nodes(ah_cur[lyr_i], seq_nf, nei_row)
        dcomb = bw_row[..., None] * dctx + ecx_row
        h_out = layer(
          jnp.expand_dims(h_in, 0),
          jnp.expand_dims(dcomb, 0),
          mask=mask_pos,
          key=sk_l[lyr_i],
          inference=False,
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
              pk_dec, dk = jax.random.split(pk_g)
              ah_si = jax.lax.dynamic_index_in_dim(ah_g, si, axis=0).squeeze(axis=0)
              se_si = jax.lax.dynamic_index_in_dim(se_g, si, axis=0).squeeze(axis=0)
              eb_si = jax.lax.dynamic_index_in_dim(edge_b, si, axis=0).squeeze(axis=0)
              nei_si = jax.lax.dynamic_index_in_dim(nei_b, si, axis=0).squeeze(axis=0)
              mw_si = jax.lax.dynamic_index_in_dim(mask_stack.astype(jnp.float32), si, axis=0).squeeze(axis=0)
              ecx_si = jax.lax.dynamic_index_in_dim(encoder_stack, si, axis=0).squeeze(axis=0)
              bw_si = jax.lax.dynamic_index_in_dim(mask_bw, si, axis=0).squeeze(axis=0)
              ah_new, logits_row = decode_site(
                ah_si, se_si, lid_here, dk, eb=eb_si, nei=nei_si, mw_atom=mw_si, ecx=ecx_si, bw=bw_si,
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
            combined = PrxteinMPNN._combine_logits_multistate_idx(
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
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Forward pass for LigandMPNN sequence scoring or sampling."""
    if prng_key is None:
      prng_key = jax.random.PRNGKey(0)

    if multistate_mode != "flat":
      msg = (
        f"{type(self).__name__}.__call__ only supports multistate_mode='flat' "
        f"(got {multistate_mode!r}); use stacked multi-state sampling helpers otherwise."
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

    # Initial projections for ligand features
    Y_nodes = jax.vmap(jax.vmap(self.w_nodes_y))(Y_nodes)
    Y_edges = jax.vmap(jax.vmap(jax.vmap(self.w_edges_y)))(Y_edges)

    # Precompute ligand edge masks
    Y_m_edges = Y_m[..., None] * Y_m[..., None, :]

    # Iterate through context integration layers
    for i in range(len(self.context_encoder)):
      # Ligand-Ligand communication (DecLayerJ in reference)
      # We vmap over the protein residue dimension (L)
      Y_nodes = jax.vmap(
        lambda node, edge, mask_l, mask_e: self.y_context_encoder[i](
          node, edge, mask_l, attention_mask=mask_e, inference=inference,
        ),
      )(Y_nodes, Y_edges, Y_m, Y_m_edges)

      # Protein-Ligand communication
      # h_E_context_cat combines protein-v-projections and projected ligand nodes
      h_E_context_cat = jnp.concatenate([h_E_context, Y_nodes], axis=-1)
      h_V_C = self.context_encoder[i](
        h_V_C, h_E_context_cat, mask, attention_mask=Y_m, inference=inference,
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
        jnp.asarray({"arithmetic_mean": 0, "geometric_mean": 1, "product": 2}[multi_state_strategy], dtype=jnp.int32),
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
        strategy_map = {"arithmetic_mean": 0, "geometric_mean": 1, "product": 2}
        strategy_idx = jnp.asarray(strategy_map[multi_state_strategy], dtype=jnp.int32)
        all_logits = PrxteinMPNN._apply_multistate_to_all_logits(
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
          {"arithmetic_mean": 0, "geometric_mean": 1, "product": 2}[multi_state_strategy],
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
      logits = PrxteinMPNN._apply_multistate_to_all_logits(
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
    # [E_ij, 0_j, h_j]
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

    def _decode_position(
      position_all_layers_h: jax.Array,
      s_embed: jax.Array,
      position: Int,
    ) -> tuple[jax.Array, jax.Array]:
      """Decode one position and return updated hidden state + logits."""
      edge_sequence_features = concatenate_neighbor_nodes(
        s_embed,
        edge_features[position],
        neighbor_indices[position],
      )

      for layer_idx, layer in enumerate(self.decoder.layers):
        h_in_pos = position_all_layers_h[layer_idx, position]
        decoder_context_pos = concatenate_neighbor_nodes(
          position_all_layers_h[layer_idx],
          edge_sequence_features,
          neighbor_indices[position],
        )
        decoding_context = mask_bw[position][..., None] * decoder_context_pos + encoder_context[position]

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
      group_indices_table, group_valid_table = _create_group_index_table(
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
        combined_logits = PrxteinMPNN._combine_logits_multistate_idx(
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
      fe_k, dn_k = jax.random.split(hk)
      zeros_map = jnp.zeros((coords.shape[0],), dtype=jnp.int32)
      V, E, E_idx, Y_nodes, Y_edges, Y_m = self.features(
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
      h_V = jnp.zeros((E.shape[0], self.node_features_dim))
      h_E = E
      mask_2d = ma[:, None] * ma[None, :]
      mask_attend = jnp.take_along_axis(mask_2d, E_idx.astype(jnp.int32), axis=1)
      for enc in self.encoder.layers:
        h_V, h_E = enc(h_V, h_E, E_idx, ma, mask_attend, inference=True)

      h_V_C = jax.vmap(self.w_c)(h_V)
      h_E_context = jax.vmap(jax.vmap(self.w_v))(V)
      Y_proj = jax.vmap(jax.vmap(self.w_nodes_y))(Y_nodes)
      Y_edges_p = jax.vmap(jax.vmap(jax.vmap(self.w_edges_y)))(Y_edges)
      Y_m_edges = Y_m[..., None] * Y_m[..., None, :]

      for ix in range(len(self.context_encoder)):
        y_enc = self.y_context_encoder[ix]
        ctx_enc = self.context_encoder[ix]

        def y_layer(nd, eg, gm, ge):
          return y_enc(nd, eg, gm, attention_mask=ge, inference=True)

        Y_proj = jax.vmap(y_layer)(Y_proj, Y_edges_p, Y_m, Y_m_edges)
        h_E_cat = jnp.concatenate([h_E_context, Y_proj], axis=-1)
        h_V_C = ctx_enc(h_V_C, h_E_cat, ma, attention_mask=Y_m, inference=True)

      h_V_C = jax.vmap(self.v_c)(h_V_C)
      h_fin = h_V + jax.vmap(self.v_c_norm)(
        self.dropout(h_V_C, key=dn_k, inference=True),
      )
      return h_fin, h_E, E_idx.astype(jnp.int32)

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
    enc_neighbors = jax.vmap(concatenate_neighbor_nodes)(
      jnp.zeros_like(node_b),
      edge_b,
      nei_b,
    )
    encoder_stack = jax.vmap(concatenate_neighbor_nodes)(node_b, enc_neighbors, nei_b)
    encoder_stack *= mask_fw[..., jnp.newaxis]

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
            combined = PrxteinMPNN._combine_logits_multistate_idx(
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
