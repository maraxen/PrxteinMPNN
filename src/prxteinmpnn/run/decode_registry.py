"""Decode function registry for tracking LogitTransformFn provenance.

Host-only: never imported from JAX-traced code. Functions are resolved here
and passed as static_argnames to JIT at dispatch time.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from prxteinmpnn.types.stages import LogitTransformFn


@dataclasses.dataclass
class DecodeFnEntry:
  """Registry entry for a LogitTransformFn with provenance metadata."""

  uid: str
  name: str
  fn: Any
  cloudpickle_bytes: bytes
  env_trace: dict[str, str]


_REGISTRY: dict[str, DecodeFnEntry] = {}


def register_decode_fn(fn: Any, name: str | None = None) -> str:
  """Register a LogitTransformFn and return its UID.

  UID is a 16-char hex prefix of SHA-256(cloudpickle(fn)).
  Idempotent: re-registering the same fn returns the same UID.
  """
  import cloudpickle  # noqa: PLC0415

  pkl = cloudpickle.dumps(fn)
  uid = hashlib.sha256(pkl).hexdigest()[:16]
  if uid not in _REGISTRY:
    _REGISTRY[uid] = DecodeFnEntry(
      uid=uid,
      name=name or getattr(fn, "__name__", repr(fn)),
      fn=fn,
      cloudpickle_bytes=pkl,
      env_trace=_capture_env(),
    )
  return uid


def resolve_decode_fn(uid: str) -> Any:
  """Return the registered LogitTransformFn for a given UID."""
  if uid not in _REGISTRY:
    msg = f"No decode fn registered for uid={uid!r}. Call register_decode_fn first."
    raise KeyError(msg)
  return _REGISTRY[uid].fn


def _capture_env() -> dict[str, str]:
  import jax  # noqa: PLC0415

  return {
    "python": sys.version,
    "jax": jax.__version__,
  }


__all__ = ["DecodeFnEntry", "register_decode_fn", "resolve_decode_fn"]
