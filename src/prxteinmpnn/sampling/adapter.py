"""Adapter functions for working with both PyTree and Equinox model architectures.

This module provides utility functions that detect the model type and route to the
appropriate implementation, enabling gradual migration from functional PyTree models
to Equinox PrxteinMPNN models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from collections.abc import Callable

  from prxteinmpnn.model.decoder import DecodingApproach
  from prxteinmpnn.model.masked_attention import MaskedAttentionType
  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    EdgeFeatures,
    Model,
    ModelParameters,
    NeighborIndices,
    NodeFeatures,
  )


def is_equinox_model(model: Model) -> bool:
  """Check if the model is a PrxteinMPNN Equinox instance.

  Args:
      model: Either a PyTree (dict) or PrxteinMPNN instance.

  Returns:
      True if model is a PrxteinMPNN instance, False if PyTree.

  Example:
      >>> from prxteinmpnn.io.weights import load_model
      >>> model = load_model()
      >>> is_equinox_model(model)
      True

  """
  # Import here to avoid circular dependency
  from prxteinmpnn.eqx_new import PrxteinMPNN  # noqa: PLC0415

  return isinstance(model, PrxteinMPNN)


def get_encoder_fn(
  model: Model,
  *,
  attention_mask_type: MaskedAttentionType | None = None,
  num_encoder_layers: int = 3,
  scale: float = 30.0,
) -> Callable[..., tuple[NodeFeatures, EdgeFeatures]]:
  """Get an encoder function that works with either model architecture.

  Args:
      model: Either a PyTree (ModelParameters) or PrxteinMPNN instance.
      attention_mask_type: Type of attention masking to use.
      num_encoder_layers: Number of encoder layers.
      scale: Scaling factor for edge features.

  Returns:
      A function that runs the encoder and returns (node_features, edge_features).

  Example:
      >>> encoder_fn = get_encoder_fn(model)
      >>> node_feats, edge_feats = encoder_fn(edge_features, neighbor_indices, mask)

  """
  if is_equinox_model(model):
    # New Equinox architecture - model is already PrxteinMPNN
    model_eqx = model  # type: ignore[assignment]

    def equinox_encoder(
      edge_features: EdgeFeatures,
      neighbor_indices: NeighborIndices,
      mask: AlphaCarbonMask,
    ) -> tuple[NodeFeatures, EdgeFeatures]:
      """Wrap Equinox encoder for consistent interface."""
      return model_eqx.encoder(edge_features, neighbor_indices, mask)

    return equinox_encoder

  # Legacy functional architecture
  from prxteinmpnn.model.encoder import make_encoder  # noqa: PLC0415

  model_params: ModelParameters = model  # type: ignore[assignment]
  return make_encoder(
    model_parameters=model_params,
    attention_mask_type=attention_mask_type,
    num_encoder_layers=num_encoder_layers,
    scale=scale,
  )


def get_decoder_fn(
  model: Model,
  *,
  attention_mask_type: MaskedAttentionType | None = None,
  decoding_approach: DecodingApproach = "conditional",
  num_decoder_layers: int = 3,
) -> Callable[..., Any]:
  """Get a decoder function that works with either model architecture.

  Args:
      model: Either a PyTree (ModelParameters) or PrxteinMPNN instance.
      attention_mask_type: Type of attention masking to use.
      decoding_approach: Either "conditional" or "autoregressive".
      num_decoder_layers: Number of decoder layers.

  Returns:
      A function that runs the decoder.

  Example:
      >>> decoder_fn = get_decoder_fn(model, decoding_approach="conditional")
      >>> output = decoder_fn(...)

  """
  if is_equinox_model(model):
    # New Equinox architecture - model is already PrxteinMPNN
    model_eqx = model  # type: ignore[assignment]

    if decoding_approach == "conditional":
      return model_eqx.decoder
    if decoding_approach == "autoregressive":
      # For autoregressive, we'd need to adapt the decoder
      # For now, use the conditional decoder as it's the most common
      return model_eqx.decoder
    msg = f"Unknown decoding approach: {decoding_approach}"
    raise ValueError(msg)

  # Legacy functional architecture
  from prxteinmpnn.model.decoder import make_decoder  # noqa: PLC0415

  model_params: ModelParameters = model  # type: ignore[assignment]
  return make_decoder(
    model_parameters=model_params,
    attention_mask_type=attention_mask_type,
    decoding_approach=decoding_approach,
    num_decoder_layers=num_decoder_layers,
  )


def get_model_parameters(model: Model) -> ModelParameters:
  """Extract ModelParameters from either architecture.

  Args:
      model: Either a PyTree (ModelParameters) or PrxteinMPNN instance.

  Returns:
      The model parameters as a PyTree.

  Raises:
      NotImplementedError: If model is PrxteinMPNN (not yet implemented).

  Example:
      >>> params = get_model_parameters(model)

  """
  if is_equinox_model(model):
    # For Equinox models, we could potentially extract parameters
    # but this is not typically needed in the new architecture
    msg = "Extracting parameters from Equinox model not yet implemented"
    raise NotImplementedError(msg)

  # Legacy functional architecture
  model_params: ModelParameters = model  # type: ignore[assignment]
  return model_params
