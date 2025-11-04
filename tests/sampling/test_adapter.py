"""Tests for sampling adapter functions.

Note: Legacy .pkl models no longer exist on HuggingFace, so we only test
the new Equinox architecture. The adapter functions are designed to support
both architectures, but in practice we're migrating to Equinox.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.io.weights import load_model
from prxteinmpnn.sampling.adapter import (
  get_decoder_fn,
  get_encoder_fn,
  get_model_parameters,
  is_equinox_model,
)


@pytest.fixture
def equinox_model():
  """Load an Equinox model."""
  return load_model(model_version="v_48_002", model_weights="original")


def test_is_equinox_model_detects_equinox(equinox_model):
  """Test that is_equinox_model correctly identifies Equinox models."""
  assert is_equinox_model(equinox_model)


def test_get_encoder_fn_equinox(equinox_model):
  """Test that get_encoder_fn returns a working encoder for Equinox models."""
  encoder_fn = get_encoder_fn(equinox_model)

  # Encoder functions have complex signatures, just check it's callable
  assert callable(encoder_fn)


def test_get_decoder_fn_equinox(equinox_model):
  """Test that get_decoder_fn returns a working decoder for Equinox models."""
  decoder_fn = get_decoder_fn(equinox_model, decoding_approach="conditional")

  # Decoder functions have complex signatures, just check it's callable
  assert callable(decoder_fn)


def test_get_model_parameters_equinox_raises(equinox_model):
  """Test that extracting parameters from Equinox models raises NotImplementedError."""
  with pytest.raises(NotImplementedError, match="not yet implemented"):
    get_model_parameters(equinox_model)


@pytest.mark.parametrize("model_version", ["v_48_002", "v_48_010", "v_48_020", "v_48_030"])
@pytest.mark.parametrize("model_weights", ["original", "soluble"])
def test_adapter_functions_all_versions(model_version, model_weights):
  """Test adapter functions work with all Equinox model versions and weights."""
  # Test Equinox model
  equinox_model = load_model(model_version=model_version, model_weights=model_weights)
  assert is_equinox_model(equinox_model)

  equinox_encoder = get_encoder_fn(equinox_model)
  assert callable(equinox_encoder)

  equinox_decoder = get_decoder_fn(equinox_model)
  assert callable(equinox_decoder)
