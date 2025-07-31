"""Tests for sampling step implementations."""

import jax.numpy as jnp
import pytest
from prxteinmpnn.model.decoder import RunConditionalDecoderFn
from prxteinmpnn.sampling.sampling import (
  preload_sampling_step_decoder,
  sample_straight_through_estimator_step,
  sample_temperature_step,
)
from prxteinmpnn.utils.data_structures import SamplingEnum


def test_preload_sampling_step_decoder():
  """Test the preloading of sampling step decoders.

  Raises:
      NotImplementedError: For unimplemented sampling strategies.
      ValueError: For unknown sampling strategies.
  """
  # Mock a decoder function
  mock_decoder: RunConditionalDecoderFn = lambda *args: args[0]

  # Test Temperature sampling
  preloaded_fn_temp = preload_sampling_step_decoder(mock_decoder, SamplingEnum.TEMPERATURE)
  sampling_step_fn_temp = preloaded_fn_temp(None, None, None, None, 1.0)
  assert sampling_step_fn_temp.func is sample_temperature_step

  # Test Straight-Through sampling
  preloaded_fn_ste = preload_sampling_step_decoder(
    mock_decoder, SamplingEnum.STRAIGHT_THROUGH
  )
  sampling_step_fn_ste = preloaded_fn_ste(None, None, None, None, 0.01) # Example learning rate
  assert sampling_step_fn_ste.func is sample_straight_through_estimator_step

  # Test unimplemented strategies
  with pytest.raises(NotImplementedError):
    preload_sampling_step_decoder(mock_decoder, SamplingEnum.BEAM_SEARCH)

  with pytest.raises(NotImplementedError):
    preload_sampling_step_decoder(mock_decoder, SamplingEnum.GREEDY)

  # Test invalid strategy
  with pytest.raises(ValueError):
    preload_sampling_step_decoder(mock_decoder, "invalid_strategy") # type: ignore[call-arg]