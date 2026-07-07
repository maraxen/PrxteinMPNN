"""aminx.tiling — axis strategy vocabulary (planning/carry/dedup declaration moved to xtrax.tiling, EPIC #1541)."""

from aminx.tiling.strategy import (
  AxisStrategy,
  DedupGather,
  SafeMap,
  Scan,
  Vmap,
)
from aminx.types.tiling_protocols import DedupFn, GatherFn, ScanTransition

__all__ = [
  "AxisStrategy",
  "DedupFn",
  "DedupGather",
  "GatherFn",
  "SafeMap",
  "Scan",
  "ScanTransition",
  "Vmap",
]
