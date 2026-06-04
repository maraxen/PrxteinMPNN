"""aminx.tiling — axis strategy, planning, and dedup-gather."""

from aminx.tiling.dedup import DedupSpec, get_k_bucket
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
    "DedupSpec",
    "GatherFn",
    "SafeMap",
    "Scan",
    "ScanTransition",
    "Vmap",
    "get_k_bucket",
]
