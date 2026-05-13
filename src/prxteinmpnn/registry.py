"""Central registries and frozen dispatch tables for prxteinmpnn.

``MULTISTATE_MODES`` holds immutable :class:`MultistateModeDescriptor` rows for
host-side routing only (Python ``str`` ``multistate_mode`` arguments and
``static_argnames``). ``SAMPLERS`` holds sampler **factories** (``model, …`` →
:class:`~prxteinmpnn.protocols.SamplerFn`). ``OUTPUT_SINKS`` holds **builders**
(``()`` → :class:`~prxteinmpnn.protocols.DesignSink`) for host tensor staging.
JIT-traced code must not depend on registry lookups for
shape/math — descriptors only steer which pre-JIT factory or ``__call__`` branch
runs, matching roadmap §3.3.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from prxteinmpnn.protocols import DesignSink, SamplerFn

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


@dataclass(frozen=True, slots=True)
class MultistateModeDescriptor:
  """Host-only routing flags for one ``multistate_mode`` string."""

  uses_stacked_exact_model_call: bool
  uses_stacked_exact_sample_wave: bool
  uses_stacked_exact_score_factory: bool
  allows_ligand_flat_encoder_path: bool


MULTISTATE_MODES: Registry[MultistateModeDescriptor] = Registry[MultistateModeDescriptor](
  "multistate_modes",
)

def register_multistate_mode(key: str, descriptor: MultistateModeDescriptor) -> None:
  """Register or replace a multistate mode descriptor (tests / extensions)."""
  MULTISTATE_MODES.register(key)(descriptor)

def _register_default_multistate_modes() -> None:
  register_multistate_mode(
    MULTISTATE_MODE_DEFAULT,
    MultistateModeDescriptor(
      uses_stacked_exact_model_call=True,
      uses_stacked_exact_sample_wave=True,
      uses_stacked_exact_score_factory=True,
      allows_ligand_flat_encoder_path=False,
    ),
  )

_register_default_multistate_modes()

def multistate_mode_descriptor(mode: str) -> MultistateModeDescriptor:
  """Return the descriptor for ``mode`` after validating membership."""
  assert_known_multistate_mode(mode)
  return MULTISTATE_MODES.get(mode)

SAMPLERS: Registry[SamplerFactoryFn] = Registry[SamplerFactoryFn]("samplers")

OUTPUT_SINKS: Registry[DesignSinkFactory] = Registry[DesignSinkFactory]("output_sinks")

import prxteinmpnn.host.streaming as _output_sinks_bootstrap  # noqa: F401 — register default sink keys
