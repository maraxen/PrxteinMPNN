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


__all__ = [
  "DEFAULT_LOGIT_TRANSFORM_UID",
  "HookEntry",
  "make_geometric_mean_transform",
  "register_ar_logit_transform_fn",
  "register_encoder_post_fn",
  "register_encoder_pre_fn",
  "register_encoder_state_fn",
  "register_hook",
  "register_logit_transform_fn",
  "resolve_hook",
]
