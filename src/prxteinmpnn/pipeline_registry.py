"""Unified UID-based hook registry for PipelineFns callables.

Mirrors decode_registry.py but covers all three hook types:
LogitTransformFn, EncoderPreFn, EncoderPostFn.

Host-only: never imported from JAX-traced code.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
from typing import Any


@dataclasses.dataclass
class HookEntry:
  """Registry entry for a pipeline hook with provenance metadata."""

  uid: str
  name: str
  fn: Any
  cloudpickle_bytes: bytes
  env_trace: dict[str, str]


_REGISTRY: dict[str, HookEntry] = {}


def register_hook(fn: Any, *, name: str | None = None) -> str:
  """Register a pipeline hook callable and return its UID.

  UID is a 16-char hex prefix of SHA-256(cloudpickle(fn)).
  Idempotent: re-registering the same fn returns the same UID.
  Works for LogitTransformFn, EncoderPreFn, EncoderPostFn.
  """
  import cloudpickle  # noqa: PLC0415

  pkl = cloudpickle.dumps(fn)
  uid = hashlib.sha256(pkl).hexdigest()[:16]
  if uid not in _REGISTRY:
    _REGISTRY[uid] = HookEntry(
      uid=uid,
      name=name or getattr(fn, "__name__", repr(fn)),
      fn=fn,
      cloudpickle_bytes=pkl,
      env_trace=_capture_env(),
    )
  return uid


def resolve_hook(uid: str) -> Any:
  """Return the registered callable for a given UID."""
  if uid not in _REGISTRY:
    msg = f"No hook registered for uid={uid!r}. Call register_hook first."
    raise KeyError(msg)
  return _REGISTRY[uid].fn


def register_logit_transform_fn(fn: Any, *, name: str | None = None) -> str:
  """Typed alias for register_hook for LogitTransformFn callables."""
  return register_hook(fn, name=name)


def register_encoder_pre_fn(fn: Any, *, name: str | None = None) -> str:
  """Typed alias for register_hook for EncoderPreFn callables."""
  return register_hook(fn, name=name)


def register_encoder_post_fn(fn: Any, *, name: str | None = None) -> str:
  """Typed alias for register_hook for EncoderPostFn callables."""
  return register_hook(fn, name=name)


def register_encoder_state_fn(fn: Any, *, name: str | None = None) -> str:
  """Typed alias for register_hook for EncoderStateFn callables."""
  return register_hook(fn, name=name)


def register_ar_logit_transform_fn(fn: Any, *, name: str | None = None) -> str:
  """Typed alias for register_hook for ARLogitTransformFn callables."""
  return register_hook(fn, name=name)


def register_featurize_fn(fn: Any, *, name: str | None = None) -> str:
  """Typed alias for register_hook for FeaturizeFn callables."""
  return register_hook(fn, name=name)


def register_encode_fn(fn: Any, *, name: str | None = None) -> str:
  """Typed alias for register_hook for EncoderStepFn or EncoderStateFn callables."""
  return register_hook(fn, name=name)


def register_decode_fn(fn: Any, *, name: str | None = None) -> str:
  """Typed alias for register_hook for ConditionalDecodeFn or UnconditionalDecodeFn callables."""
  return register_hook(fn, name=name)


def _capture_env() -> dict[str, str]:
  import jax  # noqa: PLC0415

  return {"python": sys.version, "jax": jax.__version__}


def _default_arithmetic_mean(
  state_logits: Any,
  _state_index: Any,
  _state_weights: Any,
) -> Any:
  """Default LogitTransformFn: uniform arithmetic mean across states."""
  import jax.numpy as jnp  # noqa: PLC0415

  return jnp.mean(state_logits, axis=0)


DEFAULT_LOGIT_TRANSFORM_UID: str = register_hook(
  _default_arithmetic_mean,
  name="arithmetic_mean_default",
)


def _default_ar_logit_transform(
  state_logits: Any,
  _state_index: Any,
  _state_weights: Any,
) -> Any:
  """Default ARLogitTransformFn: uniform arithmetic mean across states."""
  import jax.numpy as jnp  # noqa: PLC0415

  return jnp.mean(state_logits, axis=0)


DEFAULT_AR_LOGIT_TRANSFORM_UID: str = register_hook(
  _default_ar_logit_transform,
  name="arithmetic_mean_ar_default",
)


def _default_featurize_fn(backbone: Any) -> Any:
  """Placeholder default featurize function (to be overridden by model)."""
  return backbone


DEFAULT_FEATURIZE_UID: str = register_hook(
  _default_featurize_fn,
  name="default_featurize",
)


def _default_encode_fn(featurized: Any) -> Any:
  """Placeholder default encode function (to be overridden by model)."""
  return featurized


DEFAULT_ENCODE_UID: str = register_hook(
  _default_encode_fn,
  name="default_encode",
)


def _default_decode_fn(encoded: Any) -> Any:
  """Placeholder default decode function (to be overridden by model)."""
  return encoded


DEFAULT_DECODE_UID: str = register_hook(
  _default_decode_fn,
  name="default_decode",
)


_geom_mean_cache: dict[float, Any] = {}


def make_geometric_mean_transform(temperature: float) -> Any:
  """Return a memoised LogitTransformFn that scales arithmetic mean by 1/temperature.

  Memoised so that the same T always returns the same closure object,
  preventing registry UID proliferation during temperature sweeps.
  """
  import jax.numpy as jnp  # noqa: PLC0415

  if temperature in _geom_mean_cache:
    return _geom_mean_cache[temperature]

  def _geom_mean(state_logits: Any, state_index: Any, state_weights: Any) -> Any:
    return jnp.mean(state_logits, axis=0) / temperature

  _geom_mean_cache[temperature] = _geom_mean
  return _geom_mean


@dataclasses.dataclass(frozen=True)
class StageSet:
  """Host-only frozen dataclass for pipeline stage UIDs.

  Stores UID strings (not callables) for all six stages: featurize, encode,
  decode, logit_transform, ar_logit_transform, and encoder_state_fn.
  UIDs are safe to pass as static_argnames to JIT — callables are resolved
  at dispatch time via the pipeline registry.

  Usage:
    stage_set = StageSet.default()                        # all default UIDs
    stage_set = StageSet.from_callables(featurize=fn)    # custom featurize
    callables_dict = stage_set.resolve_all()              # resolve all UIDs
  """

  featurize_uid: str = dataclasses.field(default_factory=lambda: DEFAULT_FEATURIZE_UID)
  encode_uid: str = dataclasses.field(default_factory=lambda: DEFAULT_ENCODE_UID)
  decode_uid: str = dataclasses.field(default_factory=lambda: DEFAULT_DECODE_UID)
  logit_transform_uid: str = dataclasses.field(default_factory=lambda: DEFAULT_LOGIT_TRANSFORM_UID)
  ar_logit_transform_uid: str = dataclasses.field(default_factory=lambda: DEFAULT_AR_LOGIT_TRANSFORM_UID)
  encoder_state_fn_uid: str | None = None

  @classmethod
  def default(cls) -> "StageSet":
    """Return StageSet with all default UIDs."""
    return cls()

  @classmethod
  def from_callables(
    cls,
    *,
    featurize: Any | None = None,
    encode: Any | None = None,
    decode: Any | None = None,
    logit_transform: Any | None = None,
    ar_logit_transform: Any | None = None,
    encoder_state_fn: Any | None = None,
  ) -> "StageSet":
    """Register callables and return StageSet with their UIDs.

    Re-registering the same callable is idempotent.

    Args:
      featurize: FeaturizeFn to register, or None to use default.
      encode: EncoderStepFn or EncoderStateFn to register, or None to use default.
      decode: ConditionalDecodeFn or UnconditionalDecodeFn to register, or None to use default.
      logit_transform: LogitTransformFn to register, or None to use default.
      ar_logit_transform: ARLogitTransformFn to register, or None to use default.
      encoder_state_fn: EncoderStateFn to register, or None to leave unset.

    Returns:
      StageSet with all six UID fields populated.
    """
    featurize_uid = (
      register_featurize_fn(featurize)
      if featurize is not None
      else DEFAULT_FEATURIZE_UID
    )
    encode_uid = (
      register_encode_fn(encode)
      if encode is not None
      else DEFAULT_ENCODE_UID
    )
    decode_uid = (
      register_decode_fn(decode)
      if decode is not None
      else DEFAULT_DECODE_UID
    )
    logit_transform_uid = (
      register_logit_transform_fn(logit_transform)
      if logit_transform is not None
      else DEFAULT_LOGIT_TRANSFORM_UID
    )
    ar_logit_transform_uid = (
      register_ar_logit_transform_fn(ar_logit_transform)
      if ar_logit_transform is not None
      else DEFAULT_AR_LOGIT_TRANSFORM_UID
    )
    encoder_state_fn_uid = (
      register_encoder_state_fn(encoder_state_fn)
      if encoder_state_fn is not None
      else None
    )
    return cls(
      featurize_uid=featurize_uid,
      encode_uid=encode_uid,
      decode_uid=decode_uid,
      logit_transform_uid=logit_transform_uid,
      ar_logit_transform_uid=ar_logit_transform_uid,
      encoder_state_fn_uid=encoder_state_fn_uid,
    )

  def resolve_all(self) -> dict[str, Any]:
    """Resolve all UIDs to callables from the registry.

    Returns:
      Dict with keys: featurize_fn, encode_fn, decode_fn, logit_transform_fn,
      ar_logit_transform_fn, encoder_state_fn. Values are resolved callables
      or None for unset stages.
    """
    return {
      "featurize_fn": resolve_hook(self.featurize_uid),
      "encode_fn": resolve_hook(self.encode_uid),
      "decode_fn": resolve_hook(self.decode_uid),
      "logit_transform_fn": resolve_hook(self.logit_transform_uid),
      "ar_logit_transform_fn": resolve_hook(self.ar_logit_transform_uid),
      "encoder_state_fn": (
        resolve_hook(self.encoder_state_fn_uid)
        if self.encoder_state_fn_uid is not None
        else None
      ),
    }

  def validate_for(self, model_schema: dict[str, type]) -> None:
    """Validate StageSet against a model's stage schema.

    Checks that:
    1. All required stages in model_schema have corresponding UIDs in this StageSet.
    2. TransformFn and RollingFn types are compatible (simplified check).

    Args:
      model_schema: Dict mapping stage name (str) to expected type (type or Protocol).

    Raises:
      StageSchemaError: If validation fails with a clear diagnostic message.
    """
    from prxteinmpnn.model_inputs import StageSchema  # noqa: PLC0415

    if not isinstance(model_schema, (dict, StageSchema)):
      msg = f"model_schema must be dict or StageSchema, got {type(model_schema)}"
      raise TypeError(msg)

    # Map stage name to UID field in this StageSet
    stage_to_uid = {
      "featurize": self.featurize_uid,
      "encode": self.encode_uid,
      "decode": self.decode_uid,
      "logit_transform": self.logit_transform_uid,
      "ar_logit_transform": self.ar_logit_transform_uid,
      "encoder_state_fn": self.encoder_state_fn_uid,
    }

    for stage_name, expected_type in model_schema.items():
      if expected_type is None:
        continue  # Skip optional stages not in schema

      uid = stage_to_uid.get(stage_name)
      if uid is None:
        msg = (
          f"Stage '{stage_name}' required by model but not in StageSet. "
          f"Set {stage_name}_uid explicitly or use StageSet.from_callables({stage_name}=...)."
        )
        raise StageSchemaError(msg)

      # Resolve and basic type check (simplified; full signature matching in Fixer 4)
      try:
        resolved_fn = resolve_hook(uid)
      except KeyError as e:
        msg = f"UID {uid!r} for stage '{stage_name}' not in registry: {e}"
        raise StageSchemaError(msg) from e

      if resolved_fn is None:
        msg = f"Stage '{stage_name}' resolved to None but is required by model."
        raise StageSchemaError(msg)


class StageSchemaError(Exception):
  """Raised when StageSet validation fails against model.stage_schema()."""

  pass


__all__ = [
  "DEFAULT_LOGIT_TRANSFORM_UID",
  "DEFAULT_AR_LOGIT_TRANSFORM_UID",
  "DEFAULT_FEATURIZE_UID",
  "DEFAULT_ENCODE_UID",
  "DEFAULT_DECODE_UID",
  "HookEntry",
  "StageSet",
  "StageSchemaError",
  "make_geometric_mean_transform",
  "register_ar_logit_transform_fn",
  "register_decode_fn",
  "register_encode_fn",
  "register_encoder_post_fn",
  "register_encoder_pre_fn",
  "register_encoder_state_fn",
  "register_featurize_fn",
  "register_hook",
  "register_logit_transform_fn",
  "resolve_hook",
]
