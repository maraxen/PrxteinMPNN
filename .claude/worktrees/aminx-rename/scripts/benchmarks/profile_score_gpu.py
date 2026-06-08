#!/usr/bin/env python3
"""GPU bottleneck profiler for score_conditional.

Answers: where do the ~60ms (H200, L=76/B=1) go? Compute is only ~0.2ms
(11.9 GFLOP / ~67 TFLOP/s), so the time is host-dispatch, launch overhead,
memory, or sync. This script separates HOST-dispatch time from GPU-execution
time using JAX's async dispatch (timing the call without blocking captures
host work; block_until_ready after captures GPU wait), times encode vs decode
on-device, reports the compute-efficiency ratio, and emits a jax.profiler trace.

Run on GPU:
    PYTHONPATH=src python scripts/benchmarks/profile_score_gpu.py \
        --pdb-dir tests/data --seq-len 76 --batch-size 1 --n 30
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_aminx_jax import create_inference_plan, load_model  # noqa: E402
from micro_latency_score import _build_bundle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PEAK_TFLOPS_FP32_H200 = 67.0  # approx dense fp32; for an efficiency lower bound


def _median_ms(times_s: list[float]) -> float:
    return statistics.median(times_s) * 1000.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb-dir", type=Path, default=Path("tests/data"))
    p.add_argument("--seq-len", type=int, default=76)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--trace-dir", type=str, default="/tmp/jax_score_gpu_trace")
    args = p.parse_args()

    logger.info(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    model = load_model(checkpoint_id=None)
    bundle, config, actual_len = _build_bundle(args.pdb_dir, args.seq_len, args.batch_size)
    plan = create_inference_plan(model, "score_conditional")
    key = random.PRNGKey(42)

    logger.info(f"L={actual_len} batch={args.batch_size}")

    # ---- warm up (compile) ----
    r = plan.score(bundle, key, config)
    jax.block_until_ready(r)
    enc = plan.encode(bundle, key, config)
    jax.block_until_ready(enc)
    dec = plan.decode(enc, bundle, key, config)
    jax.block_until_ready(dec)

    # ---- HOST-DISPATCH vs GPU-WAIT split (the key diagnostic) ----
    # Async dispatch: timing the call (no block) ~= host-side work (Python +
    # eqx partition + XLA dispatch). block_until_ready after ~= GPU execution.
    host_disp, gpu_wait, full_sync = [], [], []
    for _ in range(args.n):
        t0 = time.perf_counter()
        r = plan.score(bundle, key, config)
        t1 = time.perf_counter()
        jax.block_until_ready(r)
        t2 = time.perf_counter()
        host_disp.append(t1 - t0)
        gpu_wait.append(t2 - t1)
        full_sync.append(t2 - t0)

    logger.info("=== HOST-DISPATCH vs GPU-WAIT (score, median ms) ===")
    logger.info(f"  host dispatch (call, no block): {_median_ms(host_disp):.2f} ms")
    logger.info(f"  gpu wait (block_until_ready):   {_median_ms(gpu_wait):.2f} ms")
    logger.info(f"  full sync (call + block):       {_median_ms(full_sync):.2f} ms")
    frac_host = statistics.median(host_disp) / statistics.median(full_sync) * 100
    logger.info(f"  --> host-dispatch is {frac_host:.0f}% of wall time "
                f"({'HOST-BOUND' if frac_host > 50 else 'GPU/exec-bound'})")

    # ---- stage split: encode vs decode (on-device, blocked) ----
    def timed(fn, *a):
        ts = []
        for _ in range(args.n):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*a))
            ts.append(time.perf_counter() - t0)
        return _median_ms(ts)

    score_ms = timed(plan.score, bundle, key, config)
    encode_ms = timed(plan.encode, bundle, key, config)
    decode_ms = timed(plan.decode, enc, bundle, key, config)
    logger.info("=== STAGE SPLIT (full-sync median ms) ===")
    logger.info(f"  score:  {score_ms:.2f}  encode: {encode_ms:.2f} "
                f"({encode_ms / score_ms * 100:.0f}%)  decode: {decode_ms:.2f} "
                f"({decode_ms / score_ms * 100:.0f}%)")

    # ---- compute efficiency ----
    try:
        compiled = jax.jit(plan.score).lower(bundle, key, config).compile()
        ca = compiled.cost_analysis()
        flops = ca.get("flops") if isinstance(ca, dict) else None
        if flops:
            ideal_ms = (flops / (PEAK_TFLOPS_FP32_H200 * 1e12)) * 1000.0
            logger.info("=== COMPUTE EFFICIENCY ===")
            logger.info(f"  flops={flops:.3e}  ideal@{PEAK_TFLOPS_FP32_H200}TFLOP/s="
                        f"{ideal_ms:.3f}ms  measured={score_ms:.2f}ms  "
                        f"efficiency={ideal_ms / score_ms * 100:.2f}%")
    except Exception as e:  # noqa: BLE001
        logger.info(f"  (cost_analysis unavailable: {type(e).__name__}: {e})")

    # ---- recompile check ----
    with jax.log_compiles():
        plan.score(bundle, key, config)
        plan.score(bundle, key, config)
    logger.info("  (recompile check: see 'Compiling' lines above — expect none on 2nd call)")

    # ---- profiler trace (pull & inspect device timeline) ----
    Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(args.trace_dir):
        for _ in range(5):
            jax.block_until_ready(plan.score(bundle, key, config))
    logger.info(f"  trace written to {args.trace_dir}")


if __name__ == "__main__":
    main()
