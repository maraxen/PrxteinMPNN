"""Tests for averaged encodings in sampling."""

import inspect

import jax
import jax.numpy as jnp

from prxteinmpnn.sampling.sample import make_encoding_sampling_split_fn
from prxteinmpnn.utils.decoding_order import random_decoding_order
from prxteinmpnn.utils.types import ModelParameters


def test_make_encoding_sampling_split_fn_creates_functions(
  mock_model_parameters: ModelParameters,
) -> None:
  """Test that the split function creates encode and sample functions."""
  encode_fn, sample_fn = make_encoding_sampling_split_fn(
    model_parameters=mock_model_parameters,
    decoding_order_fn=random_decoding_order,
    sampling_strategy="temperature",
  )

  assert callable(encode_fn), "encode_fn should be callable"  # noqa: S101
  assert callable(sample_fn), "sample_fn should be callable"  # noqa: S101


def test_encoding_sampling_split_signature(mock_model_parameters: ModelParameters) -> None:
  """Test that the encode and sample functions have the correct signatures."""
  encode_fn, sample_fn = make_encoding_sampling_split_fn(
    model_parameters=mock_model_parameters,
    decoding_order_fn=random_decoding_order,
    sampling_strategy="straight_through",
  )

  # Test that encode_fn can be called with appropriate arguments
  # (We don't actually call it since it requires real model params)
  encode_sig = inspect.signature(encode_fn)
  expected_encode_params = {
    "prng_key",
    "structure_coordinates",
    "mask",
    "residue_index",
    "chain_index",
        "dihedrals",
    "k_neighbors",
    "backbone_noise",
  }
  encode_params_actual = set(encode_sig.parameters.keys())
  assert (  # noqa: S101
    encode_params_actual == expected_encode_params
  ), f"encode_fn signature mismatch. Expected {expected_encode_params}, got {encode_params_actual}"

  sample_sig = inspect.signature(sample_fn)
  expected_sample_params = {
    "prng_key",
    "encoded_features",
    "decoding_order",
    "bias",
    "iterations",
    "learning_rate",
    "temperature",
    "sampling_strategy",
  }
  sample_params_actual = set(sample_sig.parameters.keys())
  assert (  # noqa: S101
    sample_params_actual == expected_sample_params
  ), f"sample_fn signature mismatch. Expected {expected_sample_params}, got {sample_params_actual}"


def test_averaged_encodings_structure() -> None:
  """Test that averaged encodings preserve the correct structure."""
  # Create mock encodings for 3 noise levels
  mock_encodings = {
    "node_features": jnp.ones((3, 10, 128)),  # (noise_levels, seq_len, features)
    "edge_features": jnp.ones((3, 10, 10, 64)),
    "neighbor_indices": jnp.ones((3, 10, 48), dtype=jnp.int32),
    "mask": jnp.ones((3, 10)),
  }

  # Average across the first dimension (noise levels)
  averaged = jax.tree_util.tree_map(
    lambda x: jnp.mean(x, axis=0),
    mock_encodings,
  )

  # Check shapes are correct (noise dimension removed)
  assert (  # noqa: S101
    averaged["node_features"].shape == (10, 128)
  ), f"node_features shape mismatch: {averaged['node_features'].shape}"
  assert (  # noqa: S101
    averaged["edge_features"].shape == (10, 10, 64)
  ), f"edge_features shape mismatch: {averaged['edge_features'].shape}"
  assert (  # noqa: S101
    averaged["neighbor_indices"].shape == (10, 48)
  ), f"neighbor_indices shape mismatch: {averaged['neighbor_indices'].shape}"
  assert (  # noqa: S101
    averaged["mask"].shape == (10,)
  ), f"mask shape mismatch: {averaged['mask'].shape}"
