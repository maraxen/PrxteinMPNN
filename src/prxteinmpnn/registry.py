"""Central registries and frozen dispatch tables for prxteinmpnn."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")

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
