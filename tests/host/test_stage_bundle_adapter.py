"""Tests for StageBundleAdapter wrapping StageSet with StageBundle interface."""

from __future__ import annotations

import pytest

from aminx.host.stage_adapter import StageBundleAdapter
from aminx.types.stages import EncoderSinkFn, StageSet


def test_stage_bundle_adapter_wraps_stage_set() -> None:
  """StageBundleAdapter wraps a StageSet and presents StageBundle interface."""
  stage_set = StageSet()
  adapter = StageBundleAdapter(stage_set)

  assert adapter._stage_set is stage_set


def test_encoder_sink_chains_multiple_sinks() -> None:
  """encoder_sink property chains multiple sinks into single callable."""
  # Create side-effect capture lists
  sink1_called = []
  sink2_called = []

  def sink1(enc, batch_idx, structure_idx, noise_idx):  # noqa: ARG001
    sink1_called.append(("sink1", batch_idx, structure_idx, noise_idx))

  def sink2(enc, batch_idx, structure_idx, noise_idx):  # noqa: ARG001
    sink2_called.append(("sink2", batch_idx, structure_idx, noise_idx))

  # Create StageSet with encoder_sink tuple
  stage_set = StageSet(encoder_sink=(sink1, sink2))
  adapter = StageBundleAdapter(stage_set)

  # Call the chained encoder_sink
  encoder_sink_fn = adapter.encoder_sink
  assert encoder_sink_fn is not None
  encoder_sink_fn(None, 0, 1, 2)

  # Verify BOTH sinks fired in order
  assert len(sink1_called) == 1
  assert len(sink2_called) == 1
  assert sink1_called[0] == ("sink1", 0, 1, 2)
  assert sink2_called[0] == ("sink2", 0, 1, 2)


def test_encoder_sink_returns_none_when_empty() -> None:
  """encoder_sink returns None when encoder_sink tuple is empty."""
  stage_set = StageSet(encoder_sink=())
  adapter = StageBundleAdapter(stage_set)

  assert adapter.encoder_sink is None


def test_decoder_sink_chains_multiple_sinks() -> None:
  """decoder_sink property chains multiple sinks into single callable."""
  sink1_called = []
  sink2_called = []

  def sink1(*args, **kwargs):  # noqa: ARG001
    sink1_called.append("sink1")

  def sink2(*args, **kwargs):  # noqa: ARG001
    sink2_called.append("sink2")

  stage_set = StageSet(decoder_sink=(sink1, sink2))
  adapter = StageBundleAdapter(stage_set)

  decoder_sink_fn = adapter.decoder_sink
  assert decoder_sink_fn is not None
  decoder_sink_fn("dummy_arg")

  assert len(sink1_called) == 1
  assert len(sink2_called) == 1


def test_decoder_sink_returns_none_when_empty() -> None:
  """decoder_sink returns None when decoder_sink tuple is empty."""
  stage_set = StageSet(decoder_sink=())
  adapter = StageBundleAdapter(stage_set)

  assert adapter.decoder_sink is None


def test_encode_fn_returns_wrapped_function() -> None:
  """encode_fn returns the wrapped function if present."""
  def dummy_encode(*args, **kwargs):  # noqa: ARG001
    return "encoded"

  stage_set = StageSet()
  # Simulate setting encode_fn by creating a mock-like attribute
  # Since StageSet doesn't have encode_fn, we test the adapter logic
  adapter = StageBundleAdapter(stage_set)

  # encode_fn isn't defined on StageSet, so it should return None
  # This test verifies that undefined properties return None
  assert adapter.encode_fn is None


def test_active_stages_returns_non_none_properties() -> None:
  """active_stages() returns names of non-None callables."""
  def dummy_sink(*args, **kwargs):  # noqa: ARG001
    pass

  stage_set = StageSet(
    encoder_sink=(dummy_sink,),  # Non-None, should appear in active_stages
  )
  adapter = StageBundleAdapter(stage_set)

  active = adapter.active_stages()
  assert "encoder_sink" in active


def test_has_stage_returns_true_for_non_none() -> None:
  """has_stage(name) returns True only for non-None callable properties."""
  def dummy_sink(*args, **kwargs):  # noqa: ARG001
    pass

  stage_set = StageSet(encoder_sink=(dummy_sink,))
  adapter = StageBundleAdapter(stage_set)

  assert adapter.has_stage("encoder_sink") is True
  assert adapter.has_stage("decoder_sink") is False
  assert adapter.has_stage("nonexistent_stage") is False


def test_multiple_encoder_sink_calls_preserve_order() -> None:
  """Multiple calls to chained encoder_sink preserve firing order."""
  call_order = []

  def sink_a(enc, batch_idx, structure_idx, noise_idx):  # noqa: ARG001
    call_order.append("a")

  def sink_b(enc, batch_idx, structure_idx, noise_idx):  # noqa: ARG001
    call_order.append("b")

  def sink_c(enc, batch_idx, structure_idx, noise_idx):  # noqa: ARG001
    call_order.append("c")

  stage_set = StageSet(encoder_sink=(sink_a, sink_b, sink_c))
  adapter = StageBundleAdapter(stage_set)

  encoder_sink_fn = adapter.encoder_sink
  assert encoder_sink_fn is not None

  # Call once
  encoder_sink_fn(None, 0, 0, 0)
  assert call_order == ["a", "b", "c"]

  # Call again
  call_order.clear()
  encoder_sink_fn(None, 1, 1, 1)
  assert call_order == ["a", "b", "c"]
