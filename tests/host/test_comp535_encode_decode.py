"""COMP-535: Tests for InferencePlan.encode() and .decode() methods."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import equinox as eqx
import jax.numpy as jnp
import pytest

from aminx.host.plan import InferencePlan, InferenceComponents, make_inference_plan
from aminx.inference.sample_autoregressive import SampleResult


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
    """Helper: InferencePlan with MagicMock encode_fn, stage_set, decode_fn."""
    encode_fn = MagicMock(name="encode_fn")
    stage_set = MagicMock(name="stage_set")
    # decode_fn should return real SampleResult to pass through jnp.argmax check
    decode_fn = MagicMock(name="decode_fn", return_value=_DUMMY_SAMPLE_RESULT)
    components = InferenceComponents(encode_fn=encode_fn, stage_set=stage_set)
    model = MagicMock(name="model")
    return InferencePlan(model=model, components=components, decode_fn=decode_fn), encode_fn, stage_set, model


def test_inference_plan_has_encode_method():
    assert hasattr(InferencePlan, "encode")
    assert callable(InferencePlan.encode)


def test_inference_plan_has_decode_method():
    assert hasattr(InferencePlan, "decode")
    assert callable(InferencePlan.decode)


def test_encode_method_calls_encode_fn():
    plan, encode_fn, _, _ = _make_plan_with_mocks()
    bundle = MagicMock(name="bundle")
    key = MagicMock(name="key")
    config = MagicMock(name="config")

    result = plan.encode(bundle, key, config)

    encode_fn.assert_called_once_with(bundle, key, config)
    assert result is encode_fn.return_value


def test_decode_method_calls_driver():
    plan, _, stage_set, model = _make_plan_with_mocks()
    enc = MagicMock(name="enc")
    bundle = MagicMock(name="bundle")
    key = MagicMock(name="key")
    config = MagicMock(name="config")

    result = plan.decode(enc, bundle, key, config)

    # decode_fn now called with (key, enc, bundle, config, stage_set)
    plan.decode_fn.assert_called_once_with(
        key, enc, bundle, config, stage_set
    )
    # Result should be the SampleResult returned by decode_fn
    assert isinstance(result, SampleResult)


def test_sample_delegates_to_encode_decode():
    plan, encode_fn, _, _ = _make_plan_with_mocks()
    bundle = MagicMock(name="bundle")
    key = MagicMock(name="key")
    config = MagicMock(name="config")

    plan.sample(bundle, key, config)

    # Verify encode_fn was called once with correct args
    encode_fn.assert_called_once_with(bundle, key, config)
    # Verify decode_fn was called once with the result from encode_fn as second arg
    plan.decode_fn.assert_called_once()
    decode_call_args = plan.decode_fn.call_args.args
    assert len(decode_call_args) >= 2
    # Second arg to decode_fn should be the encoder output
    assert decode_call_args[1] is encode_fn.return_value


def test_score_delegates_to_encode_decode():
    plan, encode_fn, _, _ = _make_plan_with_mocks()
    bundle = MagicMock(name="bundle")
    key = MagicMock(name="key")
    config = MagicMock(name="config")

    plan.score(bundle, key, config)

    # Verify encode_fn was called once with correct args
    encode_fn.assert_called_once_with(bundle, key, config)
    # Verify decode_fn was called once with the result from encode_fn as second arg
    plan.decode_fn.assert_called_once()
    decode_call_args = plan.decode_fn.call_args.args
    assert len(decode_call_args) >= 2
    # Second arg to decode_fn should be the encoder output
    assert decode_call_args[1] is encode_fn.return_value


def test_encode_passthrough_equality():
    """encode() returns a value equal to what encode_fn returns (logical pass-through)."""
    from aminx.types.encodings import EncoderOutput

    enc_instance = EncoderOutput(
        node_features=MagicMock(),
        edge_features=MagicMock(),
        neighbor_indices=MagicMock(),
        mask=None,
    )
    plan, encode_fn, _, _ = _make_plan_with_mocks()
    encode_fn.return_value = enc_instance

    bundle = MagicMock()
    key = MagicMock()
    config = MagicMock()

    result = plan.encode(bundle, key, config)
    # Use equality (==) instead of identity (is) because filter_jit may restructure the result
    assert result == enc_instance


def test_make_inference_plan_has_encode_decode():
    plan = make_inference_plan(DummyModel(), DummySpec())
    assert hasattr(plan, "encode") and callable(plan.encode)
    assert hasattr(plan, "decode") and callable(plan.decode)


def test_encode_once_decode_many_invariant():
    """A single encode() result can be passed to decode() twice — encode_fn called once."""
    plan, encode_fn, _, _ = _make_plan_with_mocks()
    bundle = MagicMock(name="bundle")
    key = MagicMock(name="key")
    key1 = MagicMock(name="key1")
    key2 = MagicMock(name="key2")
    config = MagicMock(name="config")

    enc = plan.encode(bundle, key, config)
    plan.decode(enc, bundle, key1, config)
    plan.decode(enc, bundle, key2, config)

    assert encode_fn.call_count == 1
    # decode_fn now called twice (instead of driver)
    assert plan.decode_fn.call_count == 2
    assert plan.decode_fn.call_args_list[0].args[1] is enc
    assert plan.decode_fn.call_args_list[1].args[1] is enc
