"""COMP-535: Tests for InferencePlan.encode() and .decode() methods."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import equinox as eqx
import jax.numpy as jnp
import pytest

from prxteinmpnn.host.plan import InferencePlan, InferenceComponents, make_inference_plan
from prxteinmpnn.inference.sample_autoregressive import SampleResult


class DummyModel(eqx.Module):
    """Minimal equinox module for make_inference_plan testing."""


class DummySpec:
    use_rolling_state = False
    multi_state_strategy = "arithmetic_mean"
    multi_state_temperature = 1.0
    state_weights = None
    sampling_strategy = "temperature"
    temperature = [1.0]


_DUMMY_SAMPLE_RESULT = SampleResult(
    sequence=jnp.zeros((5,), dtype=jnp.int32),
    logits=jnp.zeros((5, 21), dtype=jnp.float32),
)


def _make_plan_with_mocks():
    """Helper: InferencePlan with MagicMock encode_fn, driver, stage_set."""
    encode_fn = MagicMock(name="encode_fn")
    driver = MagicMock(name="driver", return_value=_DUMMY_SAMPLE_RESULT)
    stage_set = MagicMock(name="stage_set")
    components = InferenceComponents(encode_fn=encode_fn, driver=driver, stage_set=stage_set)
    model = MagicMock(name="model")
    return InferencePlan(model=model, components=components), encode_fn, driver, stage_set, model


def test_inference_plan_has_encode_method():
    assert hasattr(InferencePlan, "encode")
    assert callable(InferencePlan.encode)


def test_inference_plan_has_decode_method():
    assert hasattr(InferencePlan, "decode")
    assert callable(InferencePlan.decode)


def test_encode_method_calls_encode_fn():
    plan, encode_fn, _, _, _ = _make_plan_with_mocks()
    bundle = MagicMock(name="bundle")
    key = MagicMock(name="key")
    config = MagicMock(name="config")

    result = plan.encode(bundle, key, config)

    encode_fn.assert_called_once_with(bundle, key, config)
    assert result is encode_fn.return_value


def test_decode_method_calls_driver():
    plan, _, driver, stage_set, model = _make_plan_with_mocks()
    enc = MagicMock(name="enc")
    bundle = MagicMock(name="bundle")
    key = MagicMock(name="key")
    config = MagicMock(name="config")

    result = plan.decode(enc, bundle, key, config)

    driver.assert_called_once_with(
        model, key, enc, bundle.conditioning, bundle.wave, config, stage_set
    )
    assert result is driver.return_value


def test_sample_delegates_to_encode_decode():
    plan, _, _, _, _ = _make_plan_with_mocks()
    bundle = MagicMock(name="bundle")
    key = MagicMock(name="key")
    config = MagicMock(name="config")

    with patch.object(plan, "encode", wraps=plan.encode) as mock_encode, \
         patch.object(plan, "decode", wraps=plan.decode) as mock_decode:
        plan.sample(bundle, key, config)

    mock_encode.assert_called_once_with(bundle, key, config)
    mock_decode.assert_called_once()
    # Verify decode receives the output of encode
    # The first argument to decode should be the result from encode_fn (mocked in _make_plan_with_mocks)
    decode_call_args = mock_decode.call_args.args
    assert len(decode_call_args) >= 1
    # decode's first arg is the encoder output (which comes from plan.components.encode_fn)
    assert decode_call_args[0] is plan.components.encode_fn.return_value


def test_score_delegates_to_encode_decode():
    plan, _, _, _, _ = _make_plan_with_mocks()
    bundle = MagicMock(name="bundle")
    key = MagicMock(name="key")
    config = MagicMock(name="config")

    with patch.object(plan, "encode", wraps=plan.encode) as mock_encode, \
         patch.object(plan, "decode", wraps=plan.decode) as mock_decode:
        plan.score(bundle, key, config)

    mock_encode.assert_called_once_with(bundle, key, config)
    mock_decode.assert_called_once()
    # Verify decode receives the output of encode
    # The first argument to decode should be the result from encode_fn (mocked in _make_plan_with_mocks)
    decode_call_args = mock_decode.call_args.args
    assert len(decode_call_args) >= 1
    # decode's first arg is the encoder output (which comes from plan.components.encode_fn)
    assert decode_call_args[0] is plan.components.encode_fn.return_value


def test_encode_passthrough_identity():
    """encode() returns exactly what encode_fn returns (identity pass-through)."""
    from prxteinmpnn.types.encodings import EncoderOutput

    enc_instance = EncoderOutput(
        node_features=MagicMock(),
        edge_features=MagicMock(),
        neighbor_indices=MagicMock(),
        mask=None,
    )
    plan, encode_fn, _, _, _ = _make_plan_with_mocks()
    encode_fn.return_value = enc_instance

    bundle = MagicMock()
    key = MagicMock()
    config = MagicMock()

    result = plan.encode(bundle, key, config)
    assert result is enc_instance


def test_make_inference_plan_has_encode_decode():
    plan = make_inference_plan(DummyModel(), DummySpec())
    assert hasattr(plan, "encode") and callable(plan.encode)
    assert hasattr(plan, "decode") and callable(plan.decode)


def test_encode_once_decode_many_invariant():
    """A single encode() result can be passed to decode() twice — encode_fn called once."""
    plan, encode_fn, driver, _, _ = _make_plan_with_mocks()
    bundle = MagicMock(name="bundle")
    key = MagicMock(name="key")
    key1 = MagicMock(name="key1")
    key2 = MagicMock(name="key2")
    config = MagicMock(name="config")

    enc = plan.encode(bundle, key, config)
    plan.decode(enc, bundle, key1, config)
    plan.decode(enc, bundle, key2, config)

    assert encode_fn.call_count == 1
    assert driver.call_count == 2
    assert driver.call_args_list[0].args[2] is enc
    assert driver.call_args_list[1].args[2] is enc
