"""Tests for sampling step implementations."""

import pytest
from prxteinmpnn.model.decoding_signatures import RunAutoregressiveDecoderFn
from prxteinmpnn.sampling.sampling_step import (
    preload_sampling_step_decoder,
    temperature_sample,
)
from prxteinmpnn.sampling.config import SamplingConfig


def test_preload_sampling_step_decoder():
    """Test the preloading of sampling step decoders.

    Tests that the preload_sampling_step_decoder correctly configures temperature
    sampling and raises appropriate errors for unsupported strategies.

    Raises:
        NotImplementedError: For unimplemented sampling strategies.
    """
    # Mock functions
    mock_decoder: RunAutoregressiveDecoderFn = lambda *args: args[0]  # type: ignore
    mock_model_pass = lambda *args, **kwargs: (None, None, None, None, None, None)  # type: ignore

    # Test Temperature sampling configuration
    temp_config = SamplingConfig(
        sampling_strategy="temperature",
        temperature=1.0
    )
    
    preloaded_fn = preload_sampling_step_decoder(
        decoder=mock_decoder,
        sample_model_pass_fn=mock_model_pass,
        sampling_config=temp_config
    )
    
    # Verify the preloaded function is using temperature_sample
    assert preloaded_fn.func is temperature_sample
    assert preloaded_fn.keywords["temperature"] == 1.0
    assert preloaded_fn.keywords["decoder"] is mock_decoder

    # Test unsupported strategy raises NotImplementedError
    unsupported_config = SamplingConfig(
        sampling_strategy="beam_search",
        temperature=1.0
    )
    
    with pytest.raises(NotImplementedError, match="Unsupported sampling strategy"):
        preload_sampling_step_decoder(
            decoder=mock_decoder,
            sample_model_pass_fn=mock_model_pass,
            sampling_config=unsupported_config
        )