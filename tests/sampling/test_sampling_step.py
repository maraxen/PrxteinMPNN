import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.sampling.sampling_step import (
    group_sampling_step,
    preload_sampling_step_decoder,
)
from unittest.mock import Mock


def test_group_sampling_step():
    key = jax.random.PRNGKey(0)
    S = jnp.zeros(10, dtype=jnp.int32)
    model = Mock()
    model.return_value = jax.nn.one_hot(jnp.arange(10), 21)
    tie_group_map = jnp.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    ar_mask = jnp.ones((10, 10))
    temperature = 1.0
    carry = (key, S, model, tie_group_map, ar_mask, temperature)
    group_id_to_decode = 1

    new_carry, _ = group_sampling_step(carry, group_id_to_decode)
    new_key, S_new, _, _, _, _ = new_carry

    assert not jnp.array_equal(key, new_key)
    assert S_new.shape == (10,)
    assert not jnp.array_equal(S, S_new)


def test_preload_sampling_step_decoder():
    decoder = Mock()
    sample_model_pass_fn = Mock()
    temperature = 1.0

    sampling_step_fn = preload_sampling_step_decoder(
        decoder, sample_model_pass_fn, "temperature", temperature
    )
    assert callable(sampling_step_fn)

    with pytest.raises(NotImplementedError):
        preload_sampling_step_decoder(
            decoder, sample_model_pass_fn, "unsupported_strategy"
        )
