"""Adapter functions for working with both PyTree and Equinox model architectures.

This module provides utility functions that detect the model type and route to the
appropriate implementation, enabling gradual migration from functional PyTree models
to Equinox PrxteinMPNN models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from collections.abc import Callable

  from prxteinmpnn.types import DecodingApproach, MaskedAttentionType
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
  from prxteinmpnn.model import PrxteinMPNN  # noqa: PLC0415

  return isinstance(model, PrxteinMPNN)


def get_encoder_fn(
  model: Model,
  *,
  _attention_mask_type: MaskedAttentionType | None = None,
  _num_encoder_layers: int = 3,
  _scale: float = 30.0,
) -> Callable[..., tuple[NodeFeatures, EdgeFeatures]]:
  """Get an encoder function from PrxteinMPNN Equinox model.

  Args:
      model: PrxteinMPNN Equinox instance.
      _attention_mask_type: Deprecated, ignored (kept for compatibility).
      _num_encoder_layers: Deprecated, ignored (kept for compatibility).
      _scale: Deprecated, ignored (kept for compatibility).

  Returns:
      A function that runs the encoder and returns (node_features, edge_features).

  Raises:
      TypeError: If model is not a PrxteinMPNN instance.

  Example:
      >>> encoder_fn = get_encoder_fn(model)
      >>> node_feats, edge_feats = encoder_fn(edge_features, neighbor_indices, mask)

  """
  if not is_equinox_model(model):
    msg = "Only Equinox PrxteinMPNN models are supported. Legacy PyTree models have been removed."
    raise TypeError(msg)

  model_eqx = model  # type: ignore[assignment]

  def equinox_encoder(
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
  ) -> tuple[NodeFeatures, EdgeFeatures]:
    """Wrap Equinox encoder for consistent interface."""
    return model_eqx.encoder(edge_features, neighbor_indices, mask)

  return equinox_encoder


def get_decoder_fn(
  model: Model,
  *,
  _attention_mask_type: MaskedAttentionType | None = None,
  decoding_approach: DecodingApproach = "conditional",
  _num_decoder_layers: int = 3,
) -> Callable[..., Any]:
  """Get a decoder function from PrxteinMPNN Equinox model.

  Args:
      model: PrxteinMPNN Equinox instance.
      _attention_mask_type: Deprecated, ignored (kept for compatibility).
      decoding_approach: Either "conditional" or "unconditional" (autoregressive not supported yet).
      _num_decoder_layers: Deprecated, ignored (kept for compatibility).

  Returns:
      A function that runs the decoder.

  Raises:
      TypeError: If model is not a PrxteinMPNN instance.
      ValueError: If decoding_approach is not supported.

  Example:
      >>> decoder_fn = get_decoder_fn(model, decoding_approach="conditional")
      >>> output = decoder_fn(...)

  """
  if not is_equinox_model(model):
    msg = "Only Equinox PrxteinMPNN models are supported. Legacy PyTree models have been removed."
    raise TypeError(msg)

  model_eqx = model  # type: ignore[assignment]

  if decoding_approach == "conditional":
    return model_eqx.decoder.call_conditional
  if decoding_approach == "unconditional":
    return model_eqx.decoder
  msg = f"Decoding approach '{decoding_approach}' not supported by adapter"
  raise ValueError(msg)


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
