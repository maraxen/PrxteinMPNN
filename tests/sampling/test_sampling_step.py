import jax.numpy as jnp
from prxteinmpnn.sampling.sampling_step import group_sampling_step

def test_group_sampling_step_logic():
    # Mock model: returns fixed logits
    def mock_model(S, mask):
        # logits shape (4, 2)
        logits = jnp.array([
            [0.1, 0.9],  # 0
            [0.2, 0.8],  # 1
            [0.3, 0.7],  # 2
            [0.4, 0.6],  # 3
        ])
        return logits
    key = jnp.array([0, 1], dtype=jnp.uint32)
    S = jnp.array([0, 0, 0, 0])
    tie_group_map = jnp.array([0, 1, 0, 2])
    mask = jnp.ones((4, 4), dtype=bool)
    temperature = 1.0
    carry = (key, S, mock_model, tie_group_map, mask, temperature)
    # Test group 0: indices 0 and 2
    new_carry, _ = group_sampling_step(carry, 0)
    _, S_new, *_ = new_carry
    # Both S_new[0] and S_new[2] should be the same token
    assert S_new[0] == S_new[2]
    # S_new[1] and S_new[3] should be unchanged
    assert S_new[1] == 0
    assert S_new[3] == 0
"""Tests for sampling step implementations."""

import pytest
from prxteinmpnn.model.decoding_signatures import RunAutoregressiveDecoderFn
from prxteinmpnn.sampling.sampling_step import (
    preload_sampling_step_decoder,
    temperature_sample,
)


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
    preloaded_fn = preload_sampling_step_decoder(
        decoder=mock_decoder,
        sample_model_pass_fn=mock_model_pass,
        sampling_strategy="temperature",
        temperature=1.0,
    )
    
    # Verify the preloaded function is using temperature_sample
    assert preloaded_fn.func is temperature_sample
    assert preloaded_fn.keywords["temperature"] == 1.0
    assert preloaded_fn.keywords["decoder"] is mock_decoder

    # Test unsupported strategy raises NotImplementedError    
    with pytest.raises(NotImplementedError, match="Unsupported sampling strategy"):
        preload_sampling_step_decoder(
            decoder=mock_decoder,
            sample_model_pass_fn=mock_model_pass,
            sampling_strategy="beam_search",  # type: ignore[arg-type]
            temperature=1.0,
        )