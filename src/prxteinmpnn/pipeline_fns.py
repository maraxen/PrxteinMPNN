"""PipelineFns: host-only frozen dataclass for pipeline hook UIDs.

PipelineFns stores UID strings (not callables) for LogitTransformFn,
EncoderPreFn, and EncoderPostFn. UIDs are safe to pass as static_argnames
to JIT — the callable is resolved at dispatch time via pipeline_registry.

Usage:
    fns = PipelineFns.default()                          # arithmetic mean
    fns = PipelineFns.from_callables(logit_transform=fn) # custom transform
    logit_fn = fns.resolve_logit_transform()             # get callable
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from prxteinmpnn.pipeline_registry import (
  DEFAULT_LOGIT_TRANSFORM_UID,
  register_ar_logit_transform_fn,
  register_encoder_post_fn,
  register_encoder_pre_fn,
  register_encoder_state_fn,
  register_logit_transform_fn,
  resolve_hook,
)

if TYPE_CHECKING:
  from prxteinmpnn.model_inputs import ARLogitTransformFn, LogitTransformFn
  from prxteinmpnn.protocols import EncoderPostFn, EncoderPreFn, EncoderStateFn


@dataclasses.dataclass(frozen=True)
class PipelineFns:
  """Host-only container for pipeline hook UIDs.

  All three hook types are optional except logit_transform (defaults to
  arithmetic mean). Encoder hooks are None by default — set them when
  custom encoder pre/post processing is needed (e.g. cosine similarity
  across states for multistate design).
  """

  logit_transform_uid: str
  encoder_pre_process_uid: str | None = None
  encoder_post_process_uid: str | None = None
  encoder_state_fn_uid: str | None = None
  ar_logit_transform_uid: str | None = None

  @classmethod
  def default(cls) -> PipelineFns:
    """Default PipelineFns: arithmetic mean logit transform, no encoder hooks."""
    return cls(logit_transform_uid=DEFAULT_LOGIT_TRANSFORM_UID)

  @classmethod
  def from_callables(
    cls,
    *,
    logit_transform: Any | None = None,
    encoder_pre_process: Any | None = None,
    encoder_post_process: Any | None = None,
    encoder_state_fn: "EncoderStateFn | None" = None,
    ar_logit_transform: "ARLogitTransformFn | None" = None,
  ) -> PipelineFns:
    """Register callables and return a PipelineFns with their UIDs.

    Re-registering the same callable is idempotent.
    """
    lt_uid = (
      register_logit_transform_fn(logit_transform)
      if logit_transform is not None
      else DEFAULT_LOGIT_TRANSFORM_UID
    )
    pre_uid = (
      register_encoder_pre_fn(encoder_pre_process)
      if encoder_pre_process is not None
      else None
    )
    post_uid = (
      register_encoder_post_fn(encoder_post_process)
      if encoder_post_process is not None
      else None
    )
    esf_uid = (
      register_encoder_state_fn(encoder_state_fn)
      if encoder_state_fn is not None
      else None
    )
    ar_uid = (
      register_ar_logit_transform_fn(ar_logit_transform)
      if ar_logit_transform is not None
      else None
    )
    return cls(
      logit_transform_uid=lt_uid,
      encoder_pre_process_uid=pre_uid,
      encoder_post_process_uid=post_uid,
      encoder_state_fn_uid=esf_uid,
      ar_logit_transform_uid=ar_uid,
    )

  def resolve_logit_transform(self) -> LogitTransformFn:
    """Return the registered LogitTransformFn callable."""
    return resolve_hook(self.logit_transform_uid)  # type: ignore[return-value]

  def resolve_encoder_pre_process(self) -> EncoderPreFn | None:
    """Return the registered EncoderPreFn callable, or None if not set."""
    if self.encoder_pre_process_uid is None:
      return None
    return resolve_hook(self.encoder_pre_process_uid)  # type: ignore[return-value]

  def resolve_encoder_post_process(self) -> EncoderPostFn | None:
    """Return the registered EncoderPostFn callable, or None if not set."""
    if self.encoder_post_process_uid is None:
      return None
    return resolve_hook(self.encoder_post_process_uid)  # type: ignore[return-value]

  def resolve_encoder_state_fn(self) -> "EncoderStateFn | None":
    """Return the registered EncoderStateFn callable, or None if not set."""
    if self.encoder_state_fn_uid is None:
      return None
    return resolve_hook(self.encoder_state_fn_uid)  # type: ignore[return-value]

  def resolve_ar_logit_transform(self) -> "ARLogitTransformFn | None":
    """Return the registered ARLogitTransformFn callable, or None if not set."""
    if self.ar_logit_transform_uid is None:
      return None
    return resolve_hook(self.ar_logit_transform_uid)  # type: ignore[return-value]


@dataclasses.dataclass(frozen=True)
class TrainingFns:
  """Host-only container for training-specific hook UIDs.

  Separate from PipelineFns. Currently empty — extend when training
  hook wiring is needed (loss fns, etc.).
  """

  pass


__all__ = ["PipelineFns", "TrainingFns"]
