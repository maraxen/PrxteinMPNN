"""prxteinmpnn.tiling — axis strategy, planning, and dedup-gather."""

from prxteinmpnn.tiling.dedup import DedupSpec, K_DEDUP_BUCKETS, get_k_bucket
from prxteinmpnn.tiling.strategy import (
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
    "K_DEDUP_BUCKETS",
    "SafeMap",
    "Scan",
    "ScanTransition",
    "Vmap",
    "get_k_bucket",
]
