"""Central registries and frozen dispatch tables for prxteinmpnn.

``MULTISTATE_MODES`` is intentionally empty in the foundation slice: Phase 4
follow-ups will ``register`` callables (or use :func:`register_multistate_mode`)
so scoring/sampling can dispatch through a single registry surface instead of
ad-hoc ``if multistate_mode == ...`` ladders.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")

# --- Multistate mode strings (runtime API; keep in sync with type Literals) ---

MULTISTATE_MODE_FLAT: str = "flat"
MULTISTATE_MODE_STATE_VMAP: str = "state_vmap"
MULTISTATE_MODE_STATE_VMAP_EXACT: str = "state_vmap_exact"

FROZEN_MULTISTATE_MODES: frozenset[str] = frozenset(
  {
    MULTISTATE_MODE_FLAT,
    MULTISTATE_MODE_STATE_VMAP,
    MULTISTATE_MODE_STATE_VMAP_EXACT,
  },
)


def assert_known_multistate_mode(mode: str) -> str:
  """Return ``mode`` if it is a supported multistate mode string, else raise."""
  if mode in FROZEN_MULTISTATE_MODES:
    return mode
  known = ", ".join(sorted(FROZEN_MULTISTATE_MODES))
  msg = f"Unknown multistate_mode {mode!r}; expected one of: {known}"
  raise ValueError(msg)


_COMBINE_INDEX: OrderedDict[str, int] = OrderedDict(
  [("arithmetic_mean", 0), ("geometric_mean", 1), ("product", 2)],
)


def combine_strategy_to_index(name: str) -> int:
  return _COMBINE_INDEX[name]


class Registry(Generic[T]):
  def __init__(self, name: str) -> None:
    self.name = name
    self._items: dict[str, T] = {}

  def register(self, key: str) -> Callable[[T], T]:
    def deco(item: T) -> T:
      self._items[key] = item
      return item

    return deco

  def get(self, key: str) -> T:
    return self._items[key]

  def keys(self) -> list[str]:
    return list(self._items)


MULTISTATE_MODES: Registry[object] = Registry[object]("multistate_modes")


def register_multistate_mode(key: str, handler: object) -> None:
  """Register a multistate mode handler into :data:`MULTISTATE_MODES` (Phase 4+)."""
  MULTISTATE_MODES.register(key)(handler)
