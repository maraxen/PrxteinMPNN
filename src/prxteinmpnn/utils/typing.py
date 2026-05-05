"""Runtime shape/dtype verification toggle.

Vendored pattern from jaxbeans ``utils/typing.py``. Uses env
``PRXTEINMPNN_VERIFY=1`` (not ``JAXBEANS_VERIFY``).
"""

from __future__ import annotations

import os
from collections.abc import Callable

import beartype
import jaxtyping

_ENABLE_VERIFICATION = os.environ.get("PRXTEINMPNN_VERIFY", "0") == "1"


def verify_types(f: Callable) -> Callable:
  """``jaxtyping.jaxtyped`` + beartype when ``PRXTEINMPNN_VERIFY=1``."""
  if _ENABLE_VERIFICATION:
    return jaxtyping.jaxtyped(f, typechecker=beartype.beartype)
  return f


jaxtyped = verify_types
