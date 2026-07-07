"""aminx.tiling — axis strategy vocabulary (planning/carry/dedup declaration moved to xtrax.tiling, EPIC #1541)."""

from aminx.tiling.strategy import (
    AxisStrategy,
    DedupFn,
    DedupGather,
    GatherFn,
    SafeMap,
    Scan,
    ScanTransition,
    Vmap,
)

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
