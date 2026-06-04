"""Default :class:`~aminx.protocols.DesignSink` builders and host sessions (Phase 5).

Registers ``noop`` and ``streaming_tensor_staging`` under
:data:`~aminx.registry.OUTPUT_SINKS`. Import this module (side-effect) before
using ``OUTPUT_SINKS.get`` in host code — :mod:`aminx.run.sampling` does so.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.experimental
import numpy as np

from aminx.registry import OUTPUT_SINKS
from aminx.types.protocols import DesignSink

if TYPE_CHECKING:
  from aminx.types.bundles import EncoderOutput


class NoopDesignSink:
  """Default host sink: ignore tensor payloads (tests may replace registry entries)."""

  def on_sampling_sequences_logits(
    self,
    batch_idx: object,
    batch_count: object,
    chunk_start: object,
    chunk_count: object,
    sequences_host: object,
    logits_host: object,
  ) -> None:
    del batch_idx, batch_count, chunk_start, chunk_count, sequences_host, logits_host

  def on_scoring_scores_logits(
    self,
    batch_idx: object,
    batch_count: object,
    scores_host: object,
    logits_host: object,
  ) -> None:
    del batch_idx, batch_count, scores_host, logits_host


class StreamingTensorStagingSink:
  """Stages sampling ``io_callback`` tensor payloads for drain after :func:`sink_barrier`."""

  __slots__ = ("_pending",)

  def __init__(self) -> None:
    self._pending: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

  def on_sampling_sequences_logits(
    self,
    batch_idx: object,
    batch_count: object,
    chunk_start: object,
    chunk_count: object,
    sequences_host: object,
    logits_host: object,
  ) -> None:
    del batch_count
    key = (
      int(np.asarray(batch_idx)),
      int(np.asarray(chunk_start)),
      int(np.asarray(chunk_count)),
    )
    self._pending[key] = (np.asarray(sequences_host), np.asarray(logits_host))

  def on_scoring_scores_logits(
    self,
    batch_idx: object,
    batch_count: object,
    scores_host: object,
    logits_host: object,
  ) -> None:
    del batch_idx, batch_count, scores_host, logits_host

  def take_sequences_logits(
    self,
    batch_idx: int,
    chunk_start: int,
    chunk_count: int,
  ) -> tuple[np.ndarray, np.ndarray]:
    key = (batch_idx, chunk_start, chunk_count)
    try:
      return self._pending.pop(key)
    except KeyError as e:
      pending = list(self._pending.keys())
      msg = f"Streaming tensor sink missing entry for {key=}; pending={pending}"
      raise RuntimeError(msg) from e


_streaming_tensor_sink_ctx: ContextVar[StreamingTensorStagingSink | None] = ContextVar(
  "_streaming_tensor_sink_ctx",
  default=None,
)

_scoring_tensor_sink_ctx: ContextVar[DesignSink | None] = ContextVar(
  "_scoring_tensor_sink_ctx",
  default=None,
)


@contextmanager
def streaming_tensor_sink_session() -> Iterator[StreamingTensorStagingSink]:
  """Activate a fresh staging sink for one streaming sampling session."""
  factory = OUTPUT_SINKS.get("streaming_tensor_staging")
  sink_any = factory()
  if not isinstance(sink_any, StreamingTensorStagingSink):
    msg = "streaming_tensor_staging factory must return StreamingTensorStagingSink"
    raise TypeError(msg)
  token: Token[StreamingTensorStagingSink | None] = _streaming_tensor_sink_ctx.set(sink_any)
  try:
    yield sink_any
  finally:
    _streaming_tensor_sink_ctx.reset(token)


def take_staging_sequences_logits(
  batch_idx: int,
  chunk_start: int,
  chunk_count: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Drain tensor ``io_callback`` payloads for this streaming batch/chunk key."""
  sink = _streaming_tensor_sink_ctx.get()
  if sink is None:
    msg = "Streaming tensor sink is not active."
    raise RuntimeError(msg)
  return sink.take_sequences_logits(batch_idx, chunk_start, chunk_count)


def active_sampling_staging_sink() -> StreamingTensorStagingSink | None:
  """Return the active staging sink, if any (for advanced callers/tests)."""
  return _streaming_tensor_sink_ctx.get()


def active_scoring_sink() -> DesignSink | None:
  """Return the active scoring sink, if any."""
  return _scoring_tensor_sink_ctx.get()


@contextmanager
def scoring_tensor_sink_session(sink: DesignSink) -> Iterator[DesignSink]:
  """Optional host session for scoring tensor staging (streaming HDF5 follow-ups)."""
  token: Token[DesignSink | None] = _scoring_tensor_sink_ctx.set(sink)
  try:
    yield sink
  finally:
    _scoring_tensor_sink_ctx.reset(token)


class EncoderIntermediateStagingSink:
  """Stages encoder intermediate io_callback payloads.

  Keyed by (batch_idx, structure_idx, noise_idx).
  """

  __slots__ = ("_pending",)

  def __init__(self) -> None:
    self._pending: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

  def on_encoder_intermediate(
    self,
    batch_idx: object,
    structure_idx: object,
    noise_idx: object,
    node_features: object,
    edge_features: object,
  ) -> None:
    key = (
      int(np.asarray(batch_idx)),
      int(np.asarray(structure_idx)),
      int(np.asarray(noise_idx)),
    )
    self._pending[key] = (np.asarray(node_features), np.asarray(edge_features))

  def take_intermediates(
    self,
    batch_idx: int,
    structure_idx: int,
    noise_idx: int,
  ) -> tuple[np.ndarray, np.ndarray]:
    key = (batch_idx, structure_idx, noise_idx)
    try:
      return self._pending.pop(key)
    except KeyError as e:
      pending = list(self._pending.keys())
      msg = f"Encoder intermediate sink missing entry for {key=}; pending={pending}"
      raise RuntimeError(msg) from e


_encoder_intermediate_sink_ctx: ContextVar[EncoderIntermediateStagingSink | None] = ContextVar(
  "_encoder_intermediate_sink_ctx",
  default=None,
)


@contextmanager
def encoder_sink_session() -> Iterator[EncoderIntermediateStagingSink]:
  """Activate a fresh staging sink for encoder intermediate captures."""
  sink = EncoderIntermediateStagingSink()
  token: Token[EncoderIntermediateStagingSink | None] = _encoder_intermediate_sink_ctx.set(sink)
  try:
    yield sink
  finally:
    _encoder_intermediate_sink_ctx.reset(token)


def take_encoder_intermediates(
  batch_idx: int,
  structure_idx: int,
  noise_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Drain encoder intermediate payload for this (batch, structure, noise) key."""
  sink = _encoder_intermediate_sink_ctx.get()
  if sink is None:
    msg = "Encoder intermediate sink is not active."
    raise RuntimeError(msg)
  return sink.take_intermediates(batch_idx, structure_idx, noise_idx)


def active_encoder_staging_sink() -> EncoderIntermediateStagingSink | None:
  """Return the active encoder intermediate sink, if any."""
  return _encoder_intermediate_sink_ctx.get()


def _dispatch_encoder_intermediate_io(
  batch_idx: object,
  structure_idx: object,
  noise_idx: object,
  node_features: object,
  edge_features: object,
) -> None:
  """io_callback target: routes encoder intermediate to active staging sink."""
  sink = _encoder_intermediate_sink_ctx.get()
  if sink is None:
    return
  sink.on_encoder_intermediate(batch_idx, structure_idx, noise_idx, node_features, edge_features)


class IoCallbackEncoderSink(eqx.Module):
  """Fires jax.experimental.io_callback per noise-level encoding into the active encoder sink."""

  def __call__(
    self,
    enc: EncoderOutput,
    batch_idx: object,
    structure_idx: object,
    noise_idx: object,
  ) -> None:
    jax.experimental.io_callback(
      _dispatch_encoder_intermediate_io,
      None,
      batch_idx,
      structure_idx,
      noise_idx,
      enc.node_features,
      enc.edge_features,
      ordered=False,
    )


def _register_default_output_sinks() -> None:
  OUTPUT_SINKS.register("noop")(lambda: NoopDesignSink())
  OUTPUT_SINKS.register("streaming_tensor_staging")(lambda: StreamingTensorStagingSink())


_register_default_output_sinks()
