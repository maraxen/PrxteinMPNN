"""Main ProteinMPNN model implementation.

This module contains the top-level PrxteinMPNN model that combines all components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from prxteinmpnn.model.decoder import Decoder
from prxteinmpnn.model.encoder import Encoder
from prxteinmpnn.model.features import ProteinFeatures
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

# Define decoding approach type
DecodingApproach = Literal["unconditional", "conditional", "autoregressive"]


class PrxteinMPNN(eqx.Module):
  """The complete end-to-end ProteinMPNN model."""

  features: ProteinFeatures
  encoder: Encoder
  decoder: Decoder

  # Feature embedding layers
  w_s_embed: eqx.nn.Embedding  # For sequence

  # Final projection
  w_out: eqx.nn.Linear

  # Store dimensions as static metadata
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
    self.encoder = Encoder(
      node_features,
      edge_features,
      hidden_features,
      num_encoder_layers,
      key=keys[1],
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
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    _ar_mask: AutoRegressiveMask,
    _one_hot_sequence: OneHotProteinSequence,
    _prng_key: PRNGKeyArray,
    _temperature: Float,
    _bias: Logits,
    _tie_group_map: jnp.ndarray | None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the unconditional (scoring) path.

    Args:
      edge_features: Edge features from feature extraction.
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask.
      _ar_mask: Unused, required for jax.lax.switch signature.
      _one_hot_sequence: Unused, required for jax.lax.switch signature.
      _prng_key: Unused, required for jax.lax.switch signature.
      _temperature: Unused, required for jax.lax.switch signature.
      _bias: Unused, required for jax.lax.switch signature.
      _tie_group_map: Unused, required for jax.lax.switch signature.

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
    node_features, processed_edge_features = self.encoder(
      edge_features,
      neighbor_indices,
      mask,
    )

    decoded_node_features = self.decoder(
      node_features,
      processed_edge_features,
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
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    ar_mask: AutoRegressiveMask,
    one_hot_sequence: OneHotProteinSequence,
    _prng_key: PRNGKeyArray,
    _temperature: Float,
    _bias: Logits,
    _tie_group_map: jnp.ndarray | None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the conditional (scoring) path.

    Args:
      edge_features: Edge features from feature extraction.
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask.
      ar_mask: Autoregressive mask for conditional decoding.
      one_hot_sequence: One-hot encoded protein sequence.
      _prng_key: Unused, required for jax.lax.switch signature.
      _temperature: Unused, required for jax.lax.switch signature.
      _bias: Unused, required for jax.lax.switch signature.
      _tie_group_map: Unused, required for jax.lax.switch signature.

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
    node_features, processed_edge_features = self.encoder(
      edge_features,
      neighbor_indices,
      mask,
    )
    decoded_node_features = self.decoder.call_conditional(
      node_features,
      processed_edge_features,
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
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    ar_mask: AutoRegressiveMask,
    _one_hot_sequence: OneHotProteinSequence,
    prng_key: PRNGKeyArray,
    temperature: Float,
    bias: Logits,
    tie_group_map: jnp.ndarray | None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the autoregressive (sampling) path.

    Args:
      edge_features: Edge features from feature extraction.
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask.
      ar_mask: Autoregressive mask for sampling.
      _one_hot_sequence: Unused, required for jax.lax.switch signature.
      prng_key: PRNG key for sampling.
      temperature: Temperature for Gumbel-max sampling.
      bias: Bias to add to logits before sampling (N, 21).
      tie_group_map: Optional (N,) array mapping each position to a group ID.
          When provided, positions in the same group sample identical amino acids.

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
    node_features, processed_edge_features = self.encoder(
      edge_features,
      neighbor_indices,
      mask,
    )

    seq, logits = self._run_autoregressive_scan(
      prng_key,
      node_features,
      processed_edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      temperature,
      bias,
      tie_group_map,
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
    # Find max logits within the group for numerical stability
    max_logits = jnp.max(
      logits,
      where=group_mask[:, None],
      initial=-1e9,
      axis=0,
      keepdims=True,
    )  # (1, 21)

    # Shift logits and compute exp
    shifted_logits = logits - max_logits  # (N, 21)
    exp_logits = jnp.exp(shifted_logits)  # (N, 21)

    # Mask out non-group positions and sum
    masked_exp_logits = jnp.where(group_mask[:, None], exp_logits, 0.0)  # (N, 21)
    sum_exp_logits = jnp.sum(masked_exp_logits, axis=0, keepdims=True)  # (1, 21)

    # Compute average and convert back to log space
    num_in_group = jnp.sum(group_mask)
    avg_exp_logits = sum_exp_logits / num_in_group  # (1, 21)
    return jnp.log(avg_exp_logits) + max_logits  # (1, 21)

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
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run group-based autoregressive scan with logit averaging.

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

    Returns:
      Tuple of (final sequence, final logits).

    """
    num_residues = node_features.shape[0]

    # Find group decoding order based on first appearance in decoding_order
    # Get the group ID at each position in decoding_order
    groups_in_order = tie_group_map[decoding_order]

    # Find unique groups preserving order of first appearance using cumulative comparison
    # For each position, check if this group appeared earlier in the sequence
    # Create a lower triangular mask to check all previous positions
    position_indices = jnp.arange(num_residues)
    # Create mask: mask[i,j] = True if i > j (i.e., j is before i)
    is_before_mask = position_indices[:, None] > position_indices[None, :]
    # Check if current group (at each position) matches any previous group
    # groups_in_order[:, None] has shape (N, 1), groups_in_order[None, :] has shape (1, N)
    group_matches = groups_in_order[:, None] == groups_in_order[None, :]
    # For each position i, check if its group appears in any position j < i
    appeared_before = jnp.any(group_matches & is_before_mask, axis=1)
    # Mark first occurrences (positions where group hasn't appeared before)
    is_first_occurrence = ~appeared_before

    # Extract the group IDs at their first occurrences
    # Use compress with static size to avoid boolean indexing issues
    # Maximum possible size is num_residues (all positions in different groups)
    group_decoding_order = jnp.compress(
      is_first_occurrence,
      groups_in_order,
      size=num_residues,
      fill_value=-1,
    )
    # Note: group_decoding_order now contains the group IDs followed by -1 padding
    # The scan will iterate over this, and steps with group_id=-1 will be no-ops

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

      # Process positions and collect logits
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

      # Average logits and sample
      avg_logits = self._average_logits_over_group(computed_logits, group_mask)
      all_logits, s_embed, sequence = self._sample_and_broadcast_to_group(
        avg_logits,
        group_mask,
        bias,
        temperature,
        key,
        all_logits,
        s_embed,
        sequence,
      )

      return (all_layers_h, s_embed, all_logits, sequence), None

    # Initialize scan
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

    # Use the length of group_decoding_order to determine how many groups we have
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
    # Average bias across group
    group_bias = jnp.sum(
      jnp.where(group_mask[:, None], bias, 0.0),
      axis=0,
      keepdims=True,
    ) / jnp.sum(group_mask)
    logits_with_bias = avg_logits + group_bias

    # Sample once for the entire group
    sampled_logits = (logits_with_bias / temperature) + jax.random.gumbel(
      key,
      logits_with_bias.shape,
    )
    sampled_logits_no_pad = sampled_logits[..., :20]
    one_hot_sample = straight_through_estimator(sampled_logits_no_pad)
    padding = jnp.zeros_like(one_hot_sample[..., :1])
    one_hot_seq = jnp.concatenate([one_hot_sample, padding], axis=-1)

    # Broadcast to all group positions
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
          using averaged logits.

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

    # Precompute encoder context: [e_ij, 0_j, h_j]
    # This matches the unconditional decoder structure
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

      # Direct indexing at current position
      encoder_context_pos = encoder_context[position]  # (K, 384)
      neighbor_indices_pos = neighbor_indices[position]  # (K,)
      mask_pos = mask[position]  # scalar
      mask_bw_pos = mask_bw[position]  # (K,)

      # Compute edge sequence features for this position
      edge_sequence_features = concatenate_neighbor_nodes(
        s_embed,
        edge_features[position],
        neighbor_indices_pos,
      )  # (K, 256)

      # Decoder Layer Loop
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

    # Branch based on whether we have tied positions
    if tie_group_map is None:
      # Standard position-by-position sampling
      # Initialize Scan
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

    # Group-based sampling with logit averaging
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
          using logit averaging. Only used in "autoregressive" mode (optional).

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
    # 1. Prepare keys and noise
    if prng_key is None:
      prng_key = jax.random.PRNGKey(0)  # Use a default key if none provided

    prng_key, feat_key = jax.random.split(prng_key)

    if backbone_noise is None:
      backbone_noise = jnp.array(0.0, dtype=jnp.float32)

    # 2. Run Feature Extraction
    edge_features, neighbor_indices, _ = self.features(
      feat_key,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      backbone_noise,
    )

    # 3. Prepare inputs for jax.lax.switch
    branch_indices = {
      "unconditional": 0,
      "conditional": 1,
      "autoregressive": 2,
    }
    branch_index = branch_indices[decoding_approach]

    # All branches must accept the same (super-set) of arguments.
    # We fill in defaults for modes that don't use them.
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

    # Note: tie_group_map is only used in autoregressive mode, but all branches
    # must accept the same arguments for jax.lax.switch compatibility

    # 4. Define the branches for jax.lax.switch
    branches = [
      self._call_unconditional,
      self._call_conditional,
      self._call_autoregressive,
    ]

    # 5. Collect all operands
    operands = (
      edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      one_hot_sequence,
      prng_key,
      temperature,
      bias,
      tie_group_map,
    )

    # 6. Run the switch
    return jax.lax.switch(branch_index, branches, *operands)
