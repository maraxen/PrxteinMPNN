"""Legacy functional API for PrxteinMPNN.

This module contains the original functional implementation of ProteinMPNN,
preserved during the migration to Equinox. All functions and logic are
maintained as-is to serve as a baseline for equivalence testing.

New Equinox-based functional wrappers are available for improved performance
and unified codebase.

prxteinmpnn.functional
"""

from .decoder import (
  decode_message,
  decoder_normalize,
  decoder_parameter_pytree,
  embed_sequence,
  initialize_conditional_decoder,
  make_decode_layer,
  make_decoder,
  setup_decoder,
)
from .dense import dense_layer
from .encoder import (
  encode,
  encoder_normalize,
  encoder_parameter_pytree,
  initialize_node_features,
  make_encode_layer,
  make_encoder,
  setup_encoder,
)
from .eqx_wrappers import make_decoder_eqx, make_encoder_eqx, make_model_eqx
from .features import (
  embed_edges,
  encode_positions,
  extract_features,
  get_edge_chains_neighbors,
  project_features,
)
from .model import get_functional_model
from .normalize import layer_normalization, normalize
from .projection import final_projection

__all__ = [
  "decode_message",
  "decoder_normalize",
  # Decoder
  "decoder_parameter_pytree",
  # Dense layer
  "dense_layer",
  "embed_edges",
  "embed_sequence",
  "encode",
  "encode_positions",
  "encoder_normalize",
  # Encoder
  "encoder_parameter_pytree",
  # Features
  "extract_features",
  # Projection
  "final_projection",
  "get_edge_chains_neighbors",
  # Model loading
  "get_functional_model",
  "initialize_conditional_decoder",
  "initialize_node_features",
  # Normalization
  "layer_normalization",
  "make_decode_layer",
  "make_decoder",
  # Equinox-based wrappers
  "make_decoder_eqx",
  "make_encode_layer",
  "make_encoder",
  "make_encoder_eqx",
  "make_model_eqx",
  "normalize",
  "project_features",
  "setup_decoder",
  "setup_encoder",
]
