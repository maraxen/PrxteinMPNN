"""Base class for host-side model orchestration.

Centralizes common host-side behaviors like barriers, logging, and spec management
shared between SamplingDriver and ScoringDriver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from prxteinmpnn.run.streaming_host import StreamingBatchHost

if TYPE_CHECKING:
  from prxteinmpnn.run.specs import SamplingSpecification, ScoringSpecification

T_spec = TypeVar("T_spec", bound="SamplingSpecification | ScoringSpecification")


class BaseDriver(Generic[T_spec]):
  """Base class for host-side model orchestration."""

  @staticmethod
  def host_effects_barrier() -> None:
    """Barrier at host/sink boundaries for ``ordered=False`` ``io_callback``.

    Delegates to :class:`~prxteinmpnn.run.streaming_host.StreamingBatchHost`.
    """
    StreamingBatchHost.sink_barrier()

  def __init__(self, spec: T_spec) -> None:
    """Initialize driver with its configuration spec."""
    self.spec = spec
