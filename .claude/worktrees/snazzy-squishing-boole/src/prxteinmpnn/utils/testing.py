"""Testing helpers (tolerance helpers).

Vendored from jaxbeans ``utils/testing.py`` (small surface; see roadmap §3.6).
"""

from __future__ import annotations

import jax.numpy as jnp


def get_tolerances(dtype: jnp.dtype, multiplier: float = 100.0) -> tuple[float, float]:
  """Return ``(atol, rtol)`` from machine epsilon times ``multiplier``."""
  info = jnp.finfo(dtype)
  eps = float(info.eps)
  scale = eps * multiplier
  return scale, scale
