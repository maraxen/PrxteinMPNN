"""Equinox-based neural network modules for PrxteinMPNN.

This module contains Equinox implementations of the core neural network
components used in ProteinMPNN, enabling a more modular and composable
architecture.

prxteinmpnn.eqx
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

# Import necessary utilities from your project
from prxteinmpnn.model.ste import straight_through_estimator
from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes
from prxteinmpnn.utils.coordinates import (
  apply_noise_to_coordinates,
  compute_backbone_coordinates,
  compute_backbone_distance,
)
from prxteinmpnn.utils.graph import compute_neighbor_offsets
from prxteinmpnn.utils.radial_basis import compute_radial_basis
from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
  AutoRegressiveMask,
  BackboneNoise,
  ChainIndex,
  EdgeFeatures,
  Logits,
  NeighborIndices,
  NodeFeatures,
  OneHotProteinSequence,
  ResidueIndex,
  StructureAtomicCoordinates,
)

# Import necessary utilities from your project

# A more specific type for PRNG keys
PRNGKeyArray = jax.Array

# Layer normalization with a standard epsilon
LayerNorm = eqx.nn.LayerNorm
_gelu = partial(jax.nn.gelu, approximate=False)
STANDARD_EPSILON = 1e-5

# Define decoding approach type
DecodingApproach = Literal["unconditional", "conditional", "autoregressive"]


# --- Feature Extraction Constants ---
MAXIMUM_RELATIVE_FEATURES = 32
POS_EMBED_DIM = 16  # Output dim for positional encoding (w_pos)
top_k = jax.jit(jax.lax.top_k, static_argnames=("k",))


class ProteinFeatures(eqx.Module):
  """Extracts and projects features from raw protein coordinates.

  This module encapsulates the logic from `features.py`, including
  k-NN, RBF, positional encodings, and edge projections.
  """

  w_pos: eqx.nn.Linear
  w_e: eqx.nn.Linear
  norm_edges: LayerNorm
  w_e_proj: eqx.nn.Linear
  k_neighbors: int = eqx.field(static=True)
  rbf_dim: int = eqx.field(static=True)
  pos_embed_dim: int = eqx.field(static=True)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    k_neighbors: int,
    *,
    key: PRNGKeyArray,
  ):
    """Initialize feature extraction layers."""
    keys = jax.random.split(key, 3)

    self.k_neighbors = k_neighbors
    self.rbf_dim = 16
    self.pos_embed_dim = POS_EMBED_DIM  # Always use constant for output dim

    pos_one_hot_dim = 2 * MAXIMUM_RELATIVE_FEATURES + 2  # 66
    edge_embed_in_dim = 416  # Match original model's edge embedding input size

    self.w_pos = eqx.nn.Linear(pos_one_hot_dim, POS_EMBED_DIM, key=keys[0])
    self.w_e = eqx.nn.Linear(edge_embed_in_dim, edge_features, use_bias=False, key=keys[1])
    self.norm_edges = LayerNorm(edge_features)
    self.w_e_proj = eqx.nn.Linear(edge_features, edge_features, key=keys[2])

  def __call__(
    self,
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    backbone_noise: BackboneNoise | None,
  ) -> tuple[EdgeFeatures, NeighborIndices, PRNGKeyArray]:
    """Implement the logic from `extract_features` and `project_features`."""
    # --- `extract_features` logic ---
    if backbone_noise is None:
      backbone_noise = jnp.array(0.0, dtype=jnp.float32)

    noised_coordinates, prng_key = apply_noise_to_coordinates(
      prng_key,
      structure_coordinates,
      backbone_noise=backbone_noise,
    )
    backbone_atom_coordinates = compute_backbone_coordinates(noised_coordinates)
    distances = compute_backbone_distance(backbone_atom_coordinates)

    distances_masked = jnp.array(
      jnp.where(
        (mask[:, None] * mask[None, :]).astype(bool),
        distances,
        jnp.inf,
      ),
    )

    k = min(self.k_neighbors, structure_coordinates.shape[0])
    _, neighbor_indices = top_k(-distances_masked, k)
    neighbor_indices = jnp.array(neighbor_indices, dtype=jnp.int32)

    rbf = compute_radial_basis(backbone_atom_coordinates, neighbor_indices)
    neighbor_offsets = compute_neighbor_offsets(residue_index, neighbor_indices)

    # --- `get_edge_chains_neighbors` logic ---
    edge_chains = (chain_index[:, None] == chain_index[None, :]).astype(int)
    edge_chains_neighbors = jnp.take_along_axis(
      edge_chains,
      neighbor_indices,
      axis=1,
    )

    # --- `encode_positions` logic ---
    neighbor_offset_factor = jnp.clip(
      neighbor_offsets + MAXIMUM_RELATIVE_FEATURES,
      0,
      2 * MAXIMUM_RELATIVE_FEATURES,
    )
    edge_chain_factor = (1 - edge_chains_neighbors) * (2 * MAXIMUM_RELATIVE_FEATURES + 1)
    encoded_offset = neighbor_offset_factor * edge_chains_neighbors + edge_chain_factor
    encoded_offset_one_hot = jax.nn.one_hot(
      encoded_offset,
      2 * MAXIMUM_RELATIVE_FEATURES + 2,
    )

    # vmap over (N, K)
    encoded_positions = jax.vmap(jax.vmap(self.w_pos))(encoded_offset_one_hot)

    # --- `embed_edges` logic ---
    edges = jnp.concatenate([encoded_positions, rbf], axis=-1)
    edge_features = jax.vmap(jax.vmap(self.w_e))(edges)
    edge_features = jax.vmap(jax.vmap(self.norm_edges))(edge_features)

    # --- `project_features` logic ---
    edge_features = jax.vmap(jax.vmap(self.w_e_proj))(edge_features)

    return edge_features, neighbor_indices, prng_key


class EncoderLayer(eqx.Module):
  """A single encoder layer for the ProteinMPNN model."""

  edge_message_mlp: eqx.nn.MLP
  norm1: LayerNorm
  dense: eqx.nn.MLP  # Use MLP with two layers (dense_W_in and dense_W_out)
  norm2: LayerNorm
  edge_update_mlp: eqx.nn.MLP
  norm3: LayerNorm

  # Store dimensions as static metadata
  node_features_dim: int = eqx.field(static=True)
  edge_features_dim: int = eqx.field(static=True)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    *,
    key: PRNGKeyArray,
  ):
    """Initialize the encoder layer."""
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features

    keys = jax.random.split(key, 4)
    # Input: [h_i (128), e_ij (128)] = 256 (original model)
    self.edge_message_mlp = eqx.nn.MLP(
      in_size=384,
      out_size=128,
      width_size=128,
      depth=2,
      activation=_gelu,
      key=keys[0],
    )
    self.norm1 = LayerNorm(node_features)
    # Dense layer is an MLP: node_features -> hidden_features -> node_features
    self.dense = eqx.nn.MLP(
      in_size=node_features,
      out_size=node_features,
      width_size=hidden_features,
      depth=1,
      activation=_gelu,
      key=keys[1],
    )
    self.norm2 = LayerNorm(node_features)
    # Edge update MLP: 384 -> 128 -> 128 -> 128 (width=edge_features, not hidden_features)
    self.edge_update_mlp = eqx.nn.MLP(
      in_size=node_features * 2 + edge_features,
      out_size=edge_features,
      width_size=edge_features,  # 128, matches functional W11/W12/W13
      depth=2,
      activation=_gelu,
      key=keys[2],
    )
    self.norm3 = LayerNorm(edge_features)

  def _get_mlp_input(
    self,
    h: NodeFeatures,
    e: EdgeFeatures,
    neighbor_indices: NeighborIndices,
  ) -> Array:
    """Return the input tensor [h_i, e_ij, h_j] for edge_message_mlp."""
    # First concatenate neighbor node features with edge features: [e_ij, h_j]
    e_with_neighbors = concatenate_neighbor_nodes(h, e, neighbor_indices)
    # Then expand central node features and concatenate: [h_i, e_ij, h_j]
    node_expanded = jnp.tile(jnp.expand_dims(h, -2), [1, e_with_neighbors.shape[-2], 1])
    return jnp.concatenate([node_expanded, e_with_neighbors], -1)

  def __call__(
    self,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    scale: float = 30.0,
  ) -> tuple[NodeFeatures, EdgeFeatures]:
    """Forward pass for the encoder layer."""
    # 1. Update nodes
    mlp_input = self._get_mlp_input(
      node_features,
      edge_features,
      neighbor_indices,
    )
    # Apply MLP to each (atom, neighbor) pair: vmap over atoms, then over neighbors
    message = jax.vmap(jax.vmap(self.edge_message_mlp))(mlp_input)

    aggregated_message = jnp.sum(message, -2) / scale
    node_features = node_features + aggregated_message
    node_features = jax.vmap(self.norm1)(node_features)
    # Dense layer: node_features -> hidden -> node_features (residual connection)
    node_features = node_features + jax.vmap(self.dense)(node_features)
    node_features = jax.vmap(self.norm2)(node_features)
    node_features = mask[:, None] * node_features

    # 2. Update edges
    # Gather neighbor node features: [e_ij, h_j]
    edge_features_cat = concatenate_neighbor_nodes(node_features, edge_features, neighbor_indices)
    # Expand central node features and concatenate: [h_i, e_ij, h_j]
    node_features_expand = jnp.tile(
      jnp.expand_dims(node_features, -2),
      [1, edge_features_cat.shape[-2], 1],
    )
    mlp_input_edge_update = jnp.concatenate([node_features_expand, edge_features_cat], -1)
    # Apply MLP to each (atom, neighbor) pair: vmap over atoms, then over neighbors
    edge_message = jax.vmap(jax.vmap(self.edge_update_mlp))(mlp_input_edge_update)
    edge_features = edge_features + edge_message
    # vmap over (N, K)
    edge_features = jax.vmap(jax.vmap(self.norm3))(edge_features)

    return node_features, edge_features


class DecoderLayer(eqx.Module):
  """A single decoder layer for the ProteinMPNN model."""

  message_mlp: eqx.nn.MLP
  norm1: LayerNorm
  dense: eqx.nn.MLP  # Use eqx.nn.MLP directly
  norm2: LayerNorm

  def __init__(
    self,
    node_features: int,
    edge_context_features: int,  # This will be 384
    hidden_features: int,
    *,
    key: PRNGKeyArray,
  ):
    """Initialize the decoder layer."""
    keys = jax.random.split(key, 2)

    # Input dim is [h_i (128), e_context (384)] = 512
    mlp_input_dim = node_features + edge_context_features

    # Message MLP: 512 -> 128 -> 128 -> 128 (width=node_features, not hidden_features)
    self.message_mlp = eqx.nn.MLP(
      in_size=mlp_input_dim,
      out_size=node_features,
      width_size=node_features,  # 128, matches functional W1/W2/W3
      depth=2,
      activation=_gelu,
      key=keys[0],
    )
    self.norm1 = LayerNorm(node_features)
    # Use eqx.nn.MLP for the dense layer
    self.dense = eqx.nn.MLP(
      in_size=node_features,
      out_size=node_features,
      width_size=hidden_features,
      depth=1,
      activation=_gelu,
      key=keys[1],
    )
    self.norm2 = LayerNorm(node_features)

  def __call__(
    self,
    node_features: NodeFeatures,
    layer_edge_features: EdgeFeatures,  # This is the (N, K, 384) context
    mask: AlphaCarbonMask,
    scale: float = 30.0,
    attention_mask: Array | None = None,  # Optional attention mask for conditional decoding
  ) -> NodeFeatures:
    """Forward pass for the decoder layer.

    Works for both N-batch (N, C) and single-node (1, C) inputs.
    """
    # Tile central node features [h_i (N, 1, C)]
    node_features_expand = jnp.tile(
      jnp.expand_dims(node_features, -2),
      [1, layer_edge_features.shape[1], 1],
    )

    # Concat with context [h_i (N, K, C), e_context (N, K, 384)]
    mlp_input = jnp.concatenate([node_features_expand, layer_edge_features], -1)

    # Apply MLP to each (atom, neighbor) pair: vmap over atoms, then over neighbors
    message = jax.vmap(jax.vmap(self.message_mlp))(mlp_input)

    # Apply attention mask if provided (for conditional decoding)
    if attention_mask is not None:
      message = jnp.expand_dims(attention_mask, -1) * message

    # Aggregate messages
    aggregated_message = jnp.sum(message, -2) / scale
    node_features = node_features + aggregated_message

    # vmap over N
    node_features_norm1 = jax.vmap(self.norm1)(node_features)
    dense_output = jax.vmap(self.dense)(node_features_norm1)  # This works
    node_features = node_features_norm1 + dense_output
    node_features_norm2 = jax.vmap(self.norm2)(node_features)

    # Handle both batched (N,) mask and scalar mask
    if jnp.ndim(mask) == 0:
      return mask * node_features_norm2
    return mask[:, None] * node_features_norm2


class Encoder(eqx.Module):
  """The complete encoder module for ProteinMPNN."""

  layers: tuple[EncoderLayer, ...]
  node_feature_dim: int = eqx.field(static=True)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    num_layers: int = 3,
    *,
    key: PRNGKeyArray,
  ):
    """Initialize the encoder."""
    self.node_feature_dim = node_features
    keys = jax.random.split(key, num_layers)
    self.layers = tuple(
      EncoderLayer(node_features, edge_features, hidden_features, key=k) for k in keys
    )

  def __call__(
    self,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
  ) -> tuple[NodeFeatures, EdgeFeatures]:
    """Forward pass for the encoder."""
    node_features = jnp.zeros(
      (edge_features.shape[0], self.node_feature_dim),
    )

    # This is the functional state-passing loop
    for layer in self.layers:
      node_features, edge_features = layer(
        node_features,
        edge_features,
        neighbor_indices,
        mask,
      )
    return node_features, edge_features


class Decoder(eqx.Module):
  """The complete decoder module for ProteinMPNN."""

  layers: tuple[DecoderLayer, ...]
  node_features_dim: int = eqx.field(static=True)
  edge_features_dim: int = eqx.field(static=True)

  def __init__(
    self,
    node_features: int,
    edge_features: int,  # This is the raw edge_features dim (128)
    hidden_features: int,
    num_layers: int = 3,
    *,
    key: PRNGKeyArray,
  ):
    """Initialize the decoder."""
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features

    keys = jax.random.split(key, num_layers)

    # The context dim is 384 ([h_i/s_i, e_ij, h_j/s_j])
    edge_context_features = 384

    self.layers = tuple(
      DecoderLayer(node_features, edge_context_features, hidden_features, key=k) for k in keys
    )

  def __call__(
    self,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,  # Raw 128-dim edges
    mask: AlphaCarbonMask,
  ) -> NodeFeatures:
    """Forward pass for UNCONDITIONAL decoding."""
    # Prepare 384-dim context tensor *once*
    nodes_expanded = jnp.tile(
      jnp.expand_dims(node_features, -2),
      [1, edge_features.shape[1], 1],
    )
    zeros_expanded = jnp.tile(
      jnp.expand_dims(jnp.zeros_like(node_features), -2),
      [1, edge_features.shape[1], 1],
    )
    layer_edge_features = jnp.concatenate(
      [nodes_expanded, zeros_expanded, edge_features],
      -1,
    )

    loop_node_features = node_features
    for layer in self.layers:
      loop_node_features = layer(
        loop_node_features,
        layer_edge_features,
        mask,
      )
    return loop_node_features

  def call_conditional(
    self,
    node_features: NodeFeatures,  # h_i from encoder
    edge_features: EdgeFeatures,  # e_ij from encoder
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    ar_mask: AutoRegressiveMask,
    one_hot_sequence: OneHotProteinSequence,
    w_s_weight: Array,  # Sequence embedding weight
  ) -> NodeFeatures:
    """Forward pass for CONDITIONAL decoding (scoring)."""
    # 1. Embed the sequence
    embedded_sequence = one_hot_sequence @ w_s_weight  # s_i    # 2. Initialize context features
    # Following functional implementation (decoder.py lines 127-141)

    # First: [0, e_ij, h_j] -> (N, K, 256)
    temp_node_edge = concatenate_neighbor_nodes(
      jnp.zeros_like(node_features),
      edge_features,
      neighbor_indices,
    )
    # Second: [h_i, [0, e_ij, h_j]] -> (N, K, 384)
    node_edge_features = concatenate_neighbor_nodes(
      node_features,
      temp_node_edge,
      neighbor_indices,
    )

    # [e_ij, s_j] -> (N, K, 256)
    # Note: concatenate_neighbor_nodes returns [edge_features, neighbor_features]
    sequence_edge_features = concatenate_neighbor_nodes(
      embedded_sequence,
      edge_features,
      neighbor_indices,
    )

    # 3. Prepare masks
    attention_mask = jnp.take_along_axis(ar_mask, neighbor_indices, axis=1)
    mask_bw = mask[:, None] * attention_mask
    mask_fw = mask[:, None] * (1 - attention_mask)
    masked_node_edge_features = mask_fw[..., None] * node_edge_features

    # 4. Run the decoder loop
    # Following functional implementation (decoder.py lines 480-497)
    loop_node_features = node_features
    for layer in self.layers:
      # Construct the decoder context for this layer
      # concatenate_neighbor_nodes: [edge_features, neighbor_features]
      # Gathers h_j from loop_node_features, concatenates with sequence_edge_features
      # Result: [[e_ij, s_j], h_j] -> (N, K, 384)
      current_features = concatenate_neighbor_nodes(
        loop_node_features,  # (N, 128) -> gather neighbors -> (N, K, 128) = h_j
        sequence_edge_features,  # (N, K, 256) = [e_ij, s_j]
        neighbor_indices,
      )  # Result: (N, K, 384) = [e_ij, s_j, h_j]

      layer_edge_features = (mask_bw[..., None] * current_features) + masked_node_edge_features

      # Run the layer with attention masking for conditional decoding
      loop_node_features = layer(
        loop_node_features,
        layer_edge_features,
        mask,
        attention_mask=attention_mask,  # Pass attention mask for conditional decoding
      )

    return loop_node_features


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
  ):
    """Initialize the complete model."""
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
      edge_features,  # Pass raw edge dim (128)
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
    **kwargs,  # To accept unused args
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Runs the unconditional (scoring) path."""
    node_features, processed_edge_features = self.encoder(
      edge_features,
      neighbor_indices,
      mask,
    )
    decoded_node_features = self.decoder(
      node_features,
      processed_edge_features,
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
    **kwargs,  # To accept unused args
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Runs the conditional (scoring) path."""
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
    prng_key: PRNGKeyArray,
    temperature: Float,
    **kwargs,  # To accept unused args
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Runs the autoregressive (sampling) path."""
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
    )
    return seq, logits

  def _run_autoregressive_scan(
    self,
    prng_key: PRNGKeyArray,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    autoregressive_mask: AutoRegressiveMask,
    temperature: Float,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Internal JAX scan loop for autoregressive sampling."""
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

    # Precompute encoder context
    encoder_edge_neighbors = concatenate_neighbor_nodes(
      jnp.zeros_like(node_features),
      edge_features,
      neighbor_indices,
    )
    encoder_context = jnp.concatenate(
      [
        jnp.tile(
          jnp.expand_dims(node_features, -2),
          [1, edge_features.shape[1], 1],
        ),
        encoder_edge_neighbors,
      ],
      -1,
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

      # Direct indexing at current position - following functional implementation
      encoder_context_pos = encoder_context[position]  # (K, 384)
      neighbor_indices_pos = neighbor_indices[position]  # (K,)
      mask_pos = mask[position]  # scalar
      mask_bw_pos = mask_bw[position]  # (K,)

      # Compute edge sequence features for this position (used in decoder layers)
      # s_embed is (N, C), so concatenate_neighbor_nodes gathers neighbors correctly
      edge_sequence_features = concatenate_neighbor_nodes(
        s_embed,
        edge_features[position],
        neighbor_indices_pos,
      )  # (K, 256)

      # --- Decoder Layer Loop ---
      # Run decoder layers (use Python loop since we only have 3 layers)
      # Following functional implementation pattern from decoder.py lines 357-387
      for layer_idx, layer in enumerate(self.decoder.layers):
        # Get node features for this layer at current position
        h_in_pos = all_layers_h[layer_idx, position]  # [C]

        # Compute decoder context for this position
        # h_in_pos is a single vector, but concatenate_neighbor_nodes expects to gather
        # from a full (N, ...) tensor. So we need to use the full all_layers_h tensor.
        decoder_context_pos = concatenate_neighbor_nodes(
          all_layers_h[layer_idx],  # Use full (N, C) tensor for gathering
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

      # --- Sampling Step ---
      # Get final layer output for this position
      final_h_pos = all_layers_h[-1, position]  # [C]
      logits_pos_vec = self.w_out(final_h_pos)  # [21]
      logits_pos = jnp.expand_dims(logits_pos_vec, axis=0)  # [1, 21]

      next_all_logits = all_logits.at[position, :].set(jnp.squeeze(logits_pos))

      # Gumbel-max trick
      sampled_logits = (logits_pos / temperature) + jax.random.gumbel(
        key,
        logits_pos.shape,
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

    # --- Initialize Scan ---
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

  def __call__(
    self,
    # Raw structure inputs
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    decoding_approach: DecodingApproach,
    *,  # Make subsequent args keyword-only
    prng_key: PRNGKeyArray | None = None,
    ar_mask: AutoRegressiveMask | None = None,
    one_hot_sequence: OneHotProteinSequence | None = None,
    temperature: Float | None = None,
    backbone_noise: BackboneNoise | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Forward pass for the complete model.

    Dispatches to one of three modes:
    1. "unconditional": Scores all positions in parallel.
    2. "conditional": Scores a given sequence.
    3. "autoregressive": Samples a new sequence.

    Returns:
        A tuple of (OneHotProteinSequence, Logits).
        - For "unconditional", the sequence is a zero-tensor.
        - For "conditional", the sequence is the input sequence.
        - For "autoregressive", the sequence is the newly sampled one.

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

    # `switch` requires an integer index.
    branch_indices = {
      "unconditional": 0,
      "conditional": 1,
      "autoregressive": 2,
    }
    branch_index = branch_indices[decoding_approach]

    # All branches must accept the same (super-set) of arguments.
    # We fill in defaults for modes that don't use them.

    if ar_mask is None:
      # Dummy mask
      ar_mask = jnp.zeros((mask.shape[0], mask.shape[0]), dtype=jnp.int32)

    if one_hot_sequence is None:
      # Dummy sequence
      one_hot_sequence = jnp.zeros(
        (mask.shape[0], self.w_s_embed.num_embeddings),
      )

    if temperature is None:
      temperature = jnp.array(1.0)

    # 4. Define the branches for jax.lax.switch
    branches = [
      self._call_unconditional,
      self._call_conditional,
      self._call_autoregressive,
    ]

    # 5. Collect all operands
    # `jax.lax.switch` requires operands to be passed as a tuple
    operands = (
      edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      one_hot_sequence,
      prng_key,  # Pass the *updated* prng_key
      temperature,
    )

    # 6. Run the switch
    # This will select one of the 3 functions and call it with `operands`.
    return jax.lax.switch(branch_index, branches, *operands)
