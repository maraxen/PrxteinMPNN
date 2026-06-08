"""JIT-pure neighbor-gather helpers shared across protein / ligand MPNN paths (Phase **5c**)."""

from __future__ import annotations

import jax

from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes


def edge_sequence_features_autoregressive(
  s_embed: jax.Array,
  edge_features: jax.Array,
  neighbor_indices: jax.Array,
  position: jax.Array,
) -> jax.Array:
  """Neighbor-packed sequence/edge features for one autoregressive position."""
  return concatenate_neighbor_nodes(
    s_embed,
    edge_features[position],
    neighbor_indices[position],
  )


def autoregressive_decoding_context(
  layer_hidden_all_positions: jax.Array,
  edge_sequence_features: jax.Array,
  neighbor_indices_position: jax.Array,
  encoder_context_position: jax.Array,
  mask_bw_position: jax.Array,
) -> jax.Array:
  """Blend decoder neighbor context with encoder context using the backward AR mask."""
  decoder_context = concatenate_neighbor_nodes(
    layer_hidden_all_positions,
    edge_sequence_features,
    neighbor_indices_position,
  )
  return mask_bw_position[..., None] * decoder_context + encoder_context_position
