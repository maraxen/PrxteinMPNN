"""DedupSpec — declare dedup-gather on a named axis.

Used by BatchPlanner callers to fold K unique physical elements from an N-element
heterogeneous axis, run the body K times, and scatter results back to N positions.
Mirrors CarrySpec / Phase 0 pattern: caller declares, planner pre-demotes in Phase 0b.

K-bucketing: raw k varies per batch (not a static constant), so we pad unique_indices
to the next power of 2. XLA compiles one program per k_bucket value, bounding the
number of compiled variants to O(log k).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aminx.tiling.strategy import DedupFn, DedupGather, GatherFn


def get_k_bucket(k: int) -> int:
    """Round k up to the next power of 2.

    XLA recompiles for each distinct k_bucket; using powers of 2 bounds the number
    of compiled programs to O(log k) while keeping padding waste ≤ 2×.

    # TODO(high-k regime): above ~256 unique elements the 2× worst-case waste
    # becomes exponentially costly in wall-clock (e.g. k=257 pads to 512 and
    # runs 255 wasted iterations). At that scale, add intermediate buckets —
    # e.g. 384, 768 — so that the step ratio stays closer to 1.5× rather than
    # 2×. A mixed strategy (pure powers-of-2 below 256, finer geometric steps
    # above) keeps the compiled-program count manageable while cutting wasted
    # compute in the large-batch regime.

    Raises:
        ValueError: if k <= 0.

    """
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")
    return 1 << (k - 1).bit_length()


@dataclass(frozen=True)
class DedupSpec:
    """Declare dedup-gather on a named axis.

    Attributes:
        axis_name: Must match AxisSpec.name of a dedup_eligible axis.
        unique_indices: (K,) int32 array — indices of unique elements in the N-batch.
            Will be padded to k_bucket in to_dedup_gather().
        index_map: (N,) int32 array — inverse map; index_map[i]=k means position i
            uses the result from unique slot k. Values in [0, k).
        k: int — len(unique_indices); number of distinct elements.
        dedup_fn: Optional custom DedupFn. None → use _default_dedup_fn.
        gather_fn: Optional custom GatherFn. None → use _default_gather_fn.

    Raises:
        ValueError: If invariants k == len(unique_indices) or
            len(np.unique(index_map)) == k are violated.

    """

    axis_name: str
    unique_indices: np.ndarray
    index_map: np.ndarray
    k: int
    dedup_fn: DedupFn | None = None
    gather_fn: GatherFn | None = None

    def __post_init__(self) -> None:
        if len(self.unique_indices) != self.k:
            raise ValueError(
                f"DedupSpec.k={self.k} but len(unique_indices)={len(self.unique_indices)}",
            )
        n_unique_in_map = len(np.unique(self.index_map))
        if n_unique_in_map != self.k:
            raise ValueError(
                f"DedupSpec index_map references {n_unique_in_map} distinct slots "
                f"but k={self.k}. They must match.",
            )

    def to_dedup_gather(self) -> DedupGather:
        """Pad unique_indices to k_bucket and return a DedupGather strategy.

        unique_indices is padded with mode='edge' (last valid index repeated) so
        padded slots produce valid results that no index_map position selects.
        index_map is NOT padded — it always has N entries referencing [0, k).
        """
        from aminx.tiling.strategy import DedupGather, _default_dedup_fn, _default_gather_fn

        k_bucket = get_k_bucket(self.k)
        padded = np.pad(
            self.unique_indices,
            (0, k_bucket - self.k),
            mode="edge",
        )
        return DedupGather(
            unique_indices=padded,
            index_map=self.index_map,
            k=self.k,
            k_bucket=k_bucket,
            dedup_fn=self.dedup_fn if self.dedup_fn is not None else _default_dedup_fn,
            gather_fn=self.gather_fn if self.gather_fn is not None else _default_gather_fn,
        )


__all__ = ["DedupSpec", "get_k_bucket"]
