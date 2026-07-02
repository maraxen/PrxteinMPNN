#!/usr/bin/env python3
"""Decision-parity + recompile/throughput micro-benchmark: aminx.tiling vs xtrax.tiling.

aminx.tiling is a deliberate parallel reimplementation of xtrax.tiling (see
backlog EPIC #1541, "aminx->xtrax refactor", T2-T5 — not yet executed). This
script feeds direct evidence into that epic's T2.GATE (#1556) DoD: "upgraded
xtrax tiling reproduces aminx bit-for-bit + identical recompile-count +
throughput<=X%". It compares, on equivalent toy AxisSpecs, both libraries':

1. Strategy decision parity (does BatchPlanner.plan() choose the same
   strategy TYPE for the same (cardinality, default_batch_size) pair?)
2. Recompile count under repeated calls with identical shapes (should be 1
   for both, since both libraries' dispatch returns a stable iterator
   wrapped in a single eqx.filter_jit boundary)
3. Wall-clock dispatch overhead (post-warmup median latency ratio)

This is NOT a benchmark of aminx's production inference path -- it is a
synthetic, toy-function test of the tiling *primitive* in isolation, since
the two BatchPlanner/dispatch APIs are call-compatible (same
fn(xs, *, in_axes=0) -> ys iterator signature on both sides) but aminx's
production code does not yet route through xtrax.tiling at all.

Usage:
    uv run python scripts/benchmarks/bench_xtrax_vs_aminx_tiling.py \
        --n-warmup 3 --n-timed 10 --out outputs/results/xtrax_vs_aminx_tiling.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from jax import random

logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

DATA_WIDTH = 8  # per-row feature width for the synthetic (cardinality, DATA_WIDTH) array

# (cardinality, default_batch_size) pairs spanning Vmap and SafeMap regimes.
# SafeMap cases use cardinality evenly divisible by batch_size: both aminx's
# safe_map and xtrax's safe_map raise ValueError on non-divisible chunking
# rather than silently padding (verified empirically — not a documented
# behavior difference, both libraries agree here).
AXIS_CASES: list[tuple[int, int]] = [
    (16, 32),   # cardinality <= batch_size -> Vmap
    (32, 32),   # boundary case -> Vmap
    (96, 32),   # cardinality > batch_size, evenly divisible -> SafeMap
    (256, 64),  # larger chunk count, evenly divisible -> SafeMap
]


def _make_compile_counter() -> tuple[dict[str, int], object]:
    """Build a toy fn that increments a counter once per JAX trace."""
    counter = {"n": 0}

    def toy_fn(x):
        counter["n"] += 1  # executes once per Python-level trace under jit
        return x * 2.0 + 1.0

    return counter, toy_fn


def _plan_aminx(cardinality: int, default_batch_size: int, data_width: int = DATA_WIDTH):
    from aminx.tiling.planner import AxisSpec as AminxAxisSpec
    from aminx.tiling.planner import BatchPlanner as AminxBatchPlanner

    spec = AminxAxisSpec(
        name="batch",
        axis_index=0,
        cardinality=cardinality,
        default_batch_size=default_batch_size,
        # NOTE: deliberately set equal to default_batch_size, not the AxisSpec-doc default
        # of 1. Works around a real bug found by this benchmark (#2895): aminx's SafeMap
        # demotion (src/aminx/tiling/planner.py:170,196) computes
        # `tile = max(1, ax.tile_granularity)`, ignoring default_batch_size entirely,
        # contradicting both fields' own docstrings. Setting tile_granularity ==
        # default_batch_size here routes around #2895 so this benchmark measures
        # dispatch/wrapping overhead, not that bug's tile-size divergence.
        tile_granularity=default_batch_size,
        heterogeneous=False,
        doc="synthetic benchmark axis",
    )

    # aminx's BatchPlanner decision rule is memory-budget-driven (estimate_memory(decisions)
    # <= budget_bytes), NOT cardinality-vs-default_batch_size-driven like xtrax's (a real,
    # documented architecture difference — see audit notes). To make decisions comparable,
    # set budget_bytes = default_batch_size * ELEMENT_BYTES so aminx's threshold reduces to
    # the same "cardinality > default_batch_size -> demote" rule xtrax applies directly.
    elements_per_row = data_width
    estimate_memory = lambda decisions: math.prod(  # noqa: E731
        d.spec.cardinality if d.batch_size == 0 else d.batch_size for d in decisions
    ) * elements_per_row
    planner = AminxBatchPlanner(
        axes=[spec],
        budget_bytes=default_batch_size * elements_per_row,
        estimate_memory=estimate_memory,
    )
    plan = planner.plan()
    return plan.decisions[0]


def _plan_xtrax(cardinality: int, default_batch_size: int):
    from xtrax.tiling.plan import AxisSpec as XtraxAxisSpec
    from xtrax.tiling.plan import BatchPlanner as XtraxBatchPlanner

    spec = XtraxAxisSpec(
        name="batch",
        cardinality=cardinality,
        default_batch_size=default_batch_size,
    )
    planner = XtraxBatchPlanner()
    plan = planner.plan([spec])
    return plan.decisions[0]


def _dispatch_aminx(decision):
    from aminx.tiling.dispatch import make_axis_dispatch

    return make_axis_dispatch(decision.strategy, axis="batch")


def _dispatch_xtrax(decision):
    from xtrax.tiling.dispatch import make_axis_dispatch

    return make_axis_dispatch(decision.strategy, axis="batch", heterogeneous_axes=set())


def _time_iterator(
    iterator,
    toy_fn,
    data,
    n_warmup: int,
    n_timed: int,
) -> list[float]:
    @eqx.filter_jit
    def run(xs):
        return iterator(toy_fn, xs)

    for _ in range(n_warmup):
        out = run(data)
        out.block_until_ready()

    times: list[float] = []
    for _ in range(n_timed):
        start = time.perf_counter()
        out = run(data)
        out.block_until_ready()
        times.append(time.perf_counter() - start)
    return times


def run_case(cardinality: int, default_batch_size: int, n_warmup: int, n_timed: int) -> dict:
    key = random.PRNGKey(0)
    data = random.normal(key, (cardinality, 8))

    aminx_decision = _plan_aminx(cardinality, default_batch_size)
    xtrax_decision = _plan_xtrax(cardinality, default_batch_size)

    aminx_strategy = type(aminx_decision.strategy).__name__
    xtrax_strategy = type(xtrax_decision.strategy).__name__
    decision_parity = aminx_strategy == xtrax_strategy

    aminx_iterator = _dispatch_aminx(aminx_decision)
    xtrax_iterator = _dispatch_xtrax(xtrax_decision)

    aminx_counter, aminx_fn = _make_compile_counter()
    aminx_times = _time_iterator(aminx_iterator, aminx_fn, data, n_warmup, n_timed)
    aminx_recompiles = aminx_counter["n"]

    xtrax_counter, xtrax_fn = _make_compile_counter()
    xtrax_times = _time_iterator(xtrax_iterator, xtrax_fn, data, n_warmup, n_timed)
    xtrax_recompiles = xtrax_counter["n"]

    aminx_median = float(jnp.median(jnp.array(aminx_times)))
    xtrax_median = float(jnp.median(jnp.array(xtrax_times)))
    throughput_ratio = xtrax_median / aminx_median if aminx_median > 0 else float("nan")

    return {
        "cardinality": cardinality,
        "default_batch_size": default_batch_size,
        "aminx_strategy": aminx_strategy,
        "xtrax_strategy": xtrax_strategy,
        "decision_parity": decision_parity,
        "aminx_recompiles": aminx_recompiles,
        "xtrax_recompiles": xtrax_recompiles,
        "recompile_parity": aminx_recompiles == xtrax_recompiles,
        "aminx_median_s": aminx_median,
        "xtrax_median_s": xtrax_median,
        "xtrax_vs_aminx_throughput_ratio": throughput_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-timed", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None, help="Path to write result JSON.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)

    cases = []
    for cardinality, default_batch_size in AXIS_CASES:
        logger.info(f"Running case cardinality={cardinality} batch_size={default_batch_size}")
        cases.append(run_case(cardinality, default_batch_size, args.n_warmup, args.n_timed))

    all_decision_parity = all(c["decision_parity"] for c in cases)
    all_recompile_parity = all(c["recompile_parity"] for c in cases)
    max_throughput_ratio = max(c["xtrax_vs_aminx_throughput_ratio"] for c in cases)

    result = {
        "cases": cases,
        "all_decision_parity": all_decision_parity,
        "all_recompile_parity": all_recompile_parity,
        "max_xtrax_vs_aminx_throughput_ratio": max_throughput_ratio,
    }

    print(json.dumps(result, indent=2))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        logger.info(f"Wrote result JSON to {args.out}")

    if not all_decision_parity or not all_recompile_parity:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
