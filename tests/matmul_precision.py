"""Scoped JAX matmul precision for the parity suites (debt #2428).

``jax.config.update`` sets the precision globally and leaks into later tests in the
same xdist worker. ``highest_matmul_precision`` scopes it to one test call instead, and
``matmul_precision_guard`` is the autouse check that fails a test which leaks it anyway.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from typing import Any

import jax

HIGHEST = "highest"


def highest_matmul_precision[F: Callable[..., Any]](fn: F) -> F:
  """Run the decorated test body inside ``jax.default_matmul_precision('highest')``."""

  @functools.wraps(fn)
  def wrapper(*args: Any, **kwargs: Any) -> Any:
    with jax.default_matmul_precision(HIGHEST):
      return fn(*args, **kwargs)

  return wrapper  # type: ignore[return-value]


def matmul_precision_guard() -> Iterator[None]:
  """Autouse body: assert the global precision is unchanged after each test."""
  before = jax.config.jax_default_matmul_precision
  yield
  after = jax.config.jax_default_matmul_precision
  if after != before:
    msg = (
      f"test leaked jax_default_matmul_precision: {before!r} -> {after!r}; "
      "use @highest_matmul_precision or a scoped context manager"
    )
    raise AssertionError(msg)
