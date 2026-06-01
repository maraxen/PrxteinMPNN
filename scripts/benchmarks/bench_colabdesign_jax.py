#!/usr/bin/env python3
"""GPU benchmark adapter for ColabDesign ProteinMPNN (JAX) inference.

Measures cold-compile time, warm latency, and GPU memory usage for the
no-ligand path (ColabDesign has no ligand conditioning). Produces JSON output
conforming to the benchmark suite specification.

Usage:
    uv run python scripts/benchmarks/bench_colabdesign_jax.py --dry-run
    uv run python scripts/benchmarks/bench_colabdesign_jax.py --smoke
    uv run python scripts/benchmarks/bench_colabdesign_jax.py \
        --task score_conditional \
        --seq-lens 76 500 \
        --batch-sizes 1 4 16 \
        --precision fp32 \
        --hardware A100 \
        --n-warmup 10 \
        --n-timed 20 \
        --pdb-dir tests/data \
        --output-json results.json

Exit codes:
    0: SUCCESS
    1: FAILURE (missing PDB files, model load error, or benchmark error)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from jaxtyping import PRNGKeyArray

# Suppress JAX warnings
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Canonical PDB map: nominal seq_len -> pdb filename
_PDB_MAP: dict[int, str] = {
    76: "1ubq.pdb",
    500: "1SMD.pdb",
}


# ============================================================================
# Configuration & Utilities
# ============================================================================


def _set_jax_defaults():
    """Set JAX configuration before importing models."""
    # Disable XLA compilation cache for cold-run measurement
    jax.config.update("jax_enable_compilation_cache", False)
    # Blackwell workaround: set XLA flags before any compilation
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_shard_autotuning=false")


def _get_cuda_version() -> str | None:
    """Try to get CUDA version, return None if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "unknown"  # Can't easily extract full version; caller can query nvidia-smi
    except Exception:
        pass
    return None


def _query_gpu_memory_gb() -> float:
    """Query current GPU memory usage via nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            mb = int(result.stdout.strip().split("\n")[0].strip())
            return mb / 1024.0
    except Exception:
        pass
    return 0.0


# ============================================================================
# Model Setup
# ============================================================================


def load_model(verbose: bool = False) -> Any:
    """Load ColabDesign ProteinMPNN model."""
    try:
        from colabdesign.mpnn.model import mk_mpnn_model
    except ImportError:
        raise ImportError(
            "ColabDesign not installed. Run: uv sync --group benchmark"
        ) from None

    model = mk_mpnn_model(
        model_name="v_48_020",
        backbone_noise=0.0,
        dropout=0.0,
        seed=42,
        verbose=verbose,
        weights="original",
    )
    return model


# ============================================================================
# PDB Input Preparation
# ============================================================================


def prepare_pdb_input(
    model: Any,
    pdb_file: Path,
) -> tuple[int, bool]:
    """Prepare model inputs from PDB file via model.prep_inputs().

    Returns
    -------
    tuple
        (actual_seq_len, success: bool)
    """
    if not pdb_file.exists():
        logger.warning(f"PDB file not found: {pdb_file}")
        return -1, False

    try:
        model.prep_inputs(pdb_filename=str(pdb_file))
        actual_len = len(model._inputs.get("residue_idx", []))
        return actual_len, True

    except Exception as e:
        logger.warning(f"Failed to load PDB {pdb_file}: {e}")
        return -1, False


# ============================================================================
# Timing
# ============================================================================


def measure_cold_compile_sample(
    model: Any,
    batch_size: int,
    fixture_name: str,
) -> tuple[float, str]:
    """Measure cold XLA compilation time for ar_sample."""
    jax.config.update("jax_enable_compilation_cache", False)
    jax.clear_caches()

    t0 = time.perf_counter()
    result = model.sample(num=batch_size, temperature=1.0)
    jax.block_until_ready(jax.tree_util.tree_leaves(result))
    compile_time_cold_s = time.perf_counter() - t0

    note = "JAX: XLA compilation; cache disabled for cold run (jax_enable_compilation_cache=False)"
    return compile_time_cold_s, note


def measure_cold_compile_score(
    model: Any,
    fixture_name: str,
) -> tuple[float, str]:
    """Measure cold XLA compilation time for score_conditional.

    ColabDesign.score() scores one sequence at a time.
    """
    jax.config.update("jax_enable_compilation_cache", False)
    jax.clear_caches()

    t0 = time.perf_counter()
    if hasattr(model, "score"):
        result = model.score(seq=None)
    elif hasattr(model, "get_logits"):
        result = {"logits": model.get_logits()}
    else:
        raise RuntimeError("ColabDesign model has no score() or get_logits() method")
    jax.block_until_ready(jax.tree_util.tree_leaves(result))
    compile_time_cold_s = time.perf_counter() - t0

    note = "JAX: XLA compilation; cache disabled for cold run; score_conditional uses model.score()"
    return compile_time_cold_s, note


def measure_warm_latency_sample(
    model: Any,
    batch_size: int,
    n_warmup: int,
    n_timed: int,
) -> tuple[float, float, list[float]]:
    """Measure warm ar_sample latency. ColabDesign natively supports batched sampling."""
    for _ in range(n_warmup):
        result = model.sample(num=batch_size, temperature=1.0)
        jax.block_until_ready(jax.tree_util.tree_leaves(result))

    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        result = model.sample(num=batch_size, temperature=1.0)
        jax.block_until_ready(jax.tree_util.tree_leaves(result))
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return float(np.median(times_arr)), float(np.percentile(times_arr, 95)), times


def measure_warm_latency_score(
    model: Any,
    batch_size: int,
    n_warmup: int,
    n_timed: int,
) -> tuple[float, float, list[float], int]:
    """Measure warm score_conditional latency.

    model.score() does not support batch_size > 1 natively.
    For batch_size > 1, runs sequentially and reports per-sequence latency.

    Returns
    -------
    tuple
        (median_s, p95_s, times, reported_batch_size)
        reported_batch_size=1 when sequential workaround was used.
    """
    def _score_once():
        if hasattr(model, "score"):
            return model.score(seq=None)
        elif hasattr(model, "get_logits"):
            return {"logits": model.get_logits()}
        else:
            raise RuntimeError("ColabDesign model has no score() or get_logits() method")

    for _ in range(n_warmup):
        r = _score_once()
        jax.block_until_ready(jax.tree_util.tree_leaves(r))

    times = []
    if batch_size > 1:
        # Sequential workaround: run batch_size times, normalize per-sequence
        for _ in range(n_timed):
            t0 = time.perf_counter()
            for _ in range(batch_size):
                r = _score_once()
                jax.block_until_ready(jax.tree_util.tree_leaves(r))
            times.append((time.perf_counter() - t0) / batch_size)
        reported_batch_size = 1
    else:
        for _ in range(n_timed):
            t0 = time.perf_counter()
            r = _score_once()
            jax.block_until_ready(jax.tree_util.tree_leaves(r))
            times.append(time.perf_counter() - t0)
        reported_batch_size = 1

    times_arr = np.array(times)
    return float(np.median(times_arr)), float(np.percentile(times_arr, 95)), times, reported_batch_size


# ============================================================================
# Benchmark Single Cell
# ============================================================================


def benchmark_cell(
    model: Any,
    pdb_dir: Path,
    seq_len: int,
    batch_size: int,
    precision: str,
    task: str,
    n_warmup: int = 10,
    n_timed: int = 20,
) -> dict[str, Any] | None:
    """Benchmark a single (seq_len, batch_size, precision, task) cell."""
    try:
        if seq_len not in _PDB_MAP:
            logger.info(
                f"  Skipping L={seq_len} (no PDB fixture; have L={list(_PDB_MAP.keys())})"
            )
            return None

        pdb_file = pdb_dir / _PDB_MAP[seq_len]

        actual_len, success = prepare_pdb_input(model, pdb_file)
        if not success:
            return None
        if seq_len != actual_len:
            logger.info(
                f"  Note: nominal L={seq_len}, loaded L={actual_len} from {pdb_file.name}"
            )

        if precision == "bf16":
            logger.info(
                f"    Note: precision={precision} requested but not configurable in ColabDesign"
            )

        # Check score API availability for score_conditional
        if task == "score_conditional":
            if not hasattr(model, "score") and not hasattr(model, "get_logits"):
                logger.warning(
                    "  ColabDesign model has no score() or get_logits() method; "
                    "skipping score_conditional cell"
                )
                return {
                    "schema_version": "1",
                    "model": "colabdesign_jax",
                    "hardware": "unknown",
                    "seq_len": actual_len,
                    "batch_size": batch_size,
                    "task": task,
                    "precision": "fp32",
                    "ligand_conditioning": False,
                    "axis_strategy": None,
                    "average_encoding_mode": None,
                    "compile_time_cold_s": None,
                    "compile_time_warm_s": None,
                    "compile_time_note": "score_api_unavailable",
                    "latency_median_ms": None,
                    "latency_p95_ms": None,
                    "latency_per_residue_us": None,
                    "throughput_seq_per_s": None,
                    "peak_gpu_memory_gb": None,
                    "n_warmup": n_warmup,
                    "n_timed": n_timed,
                    "jax_version": jax.__version__,
                    "torch_version": None,
                    "cuda_version": _get_cuda_version(),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "skipped_reason": "score_api_unavailable",
                }

        # Cold compile
        if task == "ar_sample":
            compile_time_s, compile_note = measure_cold_compile_sample(
                model, batch_size, pdb_file.name
            )
        else:  # score_conditional
            compile_time_s, compile_note = measure_cold_compile_score(
                model, pdb_file.name
            )

        # Warm latency
        if task == "ar_sample":
            median_s, p95_s, times = measure_warm_latency_sample(
                model,
                batch_size=batch_size,
                n_warmup=n_warmup,
                n_timed=n_timed,
            )
            reported_batch_size = batch_size
        else:  # score_conditional
            median_s, p95_s, times, reported_batch_size = measure_warm_latency_score(
                model,
                batch_size=batch_size,
                n_warmup=n_warmup,
                n_timed=n_timed,
            )
            if batch_size > 1 and reported_batch_size == 1:
                extra_note = (
                    f"ColabDesign score() does not support batch_size>1; "
                    f"ran {batch_size}x sequentially, reporting per-sequence latency; "
                    f"reported batch_size=1"
                )
                compile_note = compile_note + "; " + extra_note

        latency_median_ms = median_s * 1000.0
        latency_p95_ms = p95_s * 1000.0
        latency_per_residue_us = (median_s * 1e6) / (actual_len * max(reported_batch_size, 1))
        throughput_seq_per_s = max(reported_batch_size, 1) / median_s

        peak_gpu_memory_gb = _query_gpu_memory_gb()

        result = {
            "schema_version": "1",
            "model": "colabdesign_jax",
            "hardware": "unknown",  # Set by caller
            "seq_len": actual_len,
            "batch_size": reported_batch_size,
            "task": task,
            "precision": "fp32",  # ColabDesign does not expose dtype control; always fp32
            "ligand_conditioning": False,
            "axis_strategy": None,
            "average_encoding_mode": None,
            "compile_time_cold_s": float(compile_time_s),
            "compile_time_warm_s": 0.0,
            "compile_time_note": compile_note,
            "latency_median_ms": float(latency_median_ms),
            "latency_p95_ms": float(latency_p95_ms),
            "latency_per_residue_us": float(latency_per_residue_us),
            "throughput_seq_per_s": float(throughput_seq_per_s),
            "peak_gpu_memory_gb": float(peak_gpu_memory_gb),
            "n_warmup": n_warmup,
            "n_timed": n_timed,
            "jax_version": jax.__version__,
            "torch_version": None,
            "cuda_version": _get_cuda_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"  ✓ seq_len={actual_len}, batch_size={reported_batch_size}, task={task}: "
            f"cold={compile_time_s:.3f}s, warm_median={latency_median_ms:.2f}ms"
        )

        return result

    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return None


# ============================================================================
# Main
# ============================================================================


def main():
    """Run benchmark suite."""
    parser = argparse.ArgumentParser(
        description="GPU benchmark for ColabDesign JAX ProteinMPNN",
    )
    parser.add_argument(
        "--task",
        choices=["score_conditional", "ar_sample"],
        default="score_conditional",
        help="Task to benchmark (default: score_conditional)",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[76],
        help="Sequence lengths to benchmark (default: [76])",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes to benchmark (default: [1])",
    )
    parser.add_argument(
        "--precision",
        type=str,
        nargs="+",
        default=["fp32"],
        choices=["fp32"],
        help="Precision to benchmark (ColabDesign always runs fp32)",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="unknown",
        help="Hardware identifier (default: unknown)",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--n-timed",
        type=int,
        default=20,
        help="Timed iterations (default: 20)",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=Path("prxteinmpnn/tests/data"),
        help="Directory containing PDB files (default: prxteinmpnn/tests/data)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without running benchmarks",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run minimal benchmark: seq_lens=[76], batch_sizes=[1], n_warmup=1, n_timed=3",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.smoke:
        args.seq_lens = [76]
        args.batch_sizes = [1]
        args.n_warmup = 1
        args.n_timed = 3
        logger.info("Smoke test mode: minimal iterations")

    if args.dry_run:
        config = {
            "task": args.task,
            "seq_lens": args.seq_lens,
            "batch_sizes": args.batch_sizes,
            "precisions": args.precision,
            "hardware": args.hardware,
            "pdb_dir": str(args.pdb_dir),
            "n_warmup": args.n_warmup,
            "n_timed": args.n_timed,
            "jax_version": jax.__version__,
            "cuda_available": jax.devices()[0].platform == "gpu" if jax.devices() else False,
        }
        logger.info("DRY RUN - Configuration:")
        logger.info(json.dumps(config, indent=2))
        return 0

    if not args.pdb_dir.exists():
        logger.error(f"PDB directory not found: {args.pdb_dir}")
        return 1

    pdb_files = list(args.pdb_dir.glob("*.pdb"))
    if not pdb_files:
        logger.error(f"No PDB files found in {args.pdb_dir}")
        return 1

    _set_jax_defaults()

    logger.info("Loading ColabDesign ProteinMPNN model...")
    try:
        model = load_model(verbose=False)
        logger.info("Model loaded successfully")
    except ImportError as e:
        logger.error(f"Failed to import ColabDesign: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    all_results = []
    total_cells = len(args.seq_lens) * len(args.batch_sizes) * len(args.precision)
    current_cell = 0

    for seq_len in args.seq_lens:
        for batch_size in args.batch_sizes:
            for precision in args.precision:
                current_cell += 1
                logger.info(
                    f"Benchmarking cell {current_cell}/{total_cells}: "
                    f"seq_len={seq_len}, batch_size={batch_size}, "
                    f"precision={precision}, task={args.task}"
                )

                result = benchmark_cell(
                    model,
                    args.pdb_dir,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    precision=precision,
                    task=args.task,
                    n_warmup=args.n_warmup,
                    n_timed=args.n_timed,
                )

                if result is not None:
                    result["hardware"] = args.hardware
                    all_results.append(result)
                else:
                    logger.warning(f"Skipped cell {current_cell}")

    if not all_results:
        logger.error("No benchmarks completed successfully")
        return 1

    output = {
        "schema_version": "1",
        "results": all_results,
        "total_cells": len(all_results),
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results written to {args.output_json}")
    else:
        print(json.dumps(output, indent=2))

    logger.info(f"Benchmark complete: {len(all_results)}/{total_cells} cells successful")
    return 0


if __name__ == "__main__":
    sys.exit(main())
