"""Central registries and frozen dispatch tables for prxteinmpnn.

``SAMPLERS`` holds sampler **factories** (``model, …`` →
:class:`~prxteinmpnn.protocols.SamplerFn`). ``OUTPUT_SINKS`` holds **builders**
(``()`` → :class:`~prxteinmpnn.protocols.DesignSink`) for host tensor staging.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

from prxteinmpnn.types.protocols import DesignSink, SamplerFn

T = TypeVar("T")

SamplerFactoryFn = Callable[..., SamplerFn]
DesignSinkFactory = Callable[[], DesignSink]

# Removed multistate mode routing strings. Only stacked mode is used.

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


SAMPLERS: Registry[SamplerFactoryFn] = Registry[SamplerFactoryFn]("samplers")

OUTPUT_SINKS: Registry[DesignSinkFactory] = Registry[DesignSinkFactory]("output_sinks")

import prxteinmpnn.host.output_sinks as _output_sinks_bootstrap  # noqa: F401 — register default sink keys
