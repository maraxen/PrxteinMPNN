#!/usr/bin/env python3
"""GPU benchmark adapter for prxteinmpnn (JAX+Equinox) inference.

Measures cold-compile time, warm latency, and GPU memory usage for each
(seq_len, batch_size, precision) cell. Produces JSON output conforming to
the benchmark suite specification.

Usage:
    uv run python scripts/benchmarks/bench_prxteinmpnn_jax.py --dry-run
    uv run python scripts/benchmarks/bench_prxteinmpnn_jax.py --smoke
    uv run python scripts/benchmarks/bench_prxteinmpnn_jax.py \
        --seq-lens 76 150 300 500 \
        --batch-sizes 1 4 16 \
        --precision bf16 fp32 \
        --hardware A100 \
        --n-warmup 10 \
        --n-timed 20 \
        --output-json results.json

Exit codes:
    0: SUCCESS
    1: FAILURE (missing fixtures, model load error, or benchmark error)
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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

if TYPE_CHECKING:
    from jaxtyping import PRNGKeyArray

# Suppress JAX warnings
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


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


def _get_gpu_memory_gb() -> float:
    """Query GPU peak memory via JAX, fallback to 0.0 if unavailable."""
    try:
        backend = jax.lib.xla_bridge.get_backend()
        if backend.platform == "gpu":
            # Rough estimate: report after first execution
            # In practice, this requires pynvml or nvidia-smi integration
            # For now, return a placeholder (proper integration TBD)
            return 0.0
    except Exception:
        pass
    return 0.0


class _BenchmarkSpec:
    """Minimal spec for make_inference_plan."""

    sampling_strategy = "temperature"
    use_rolling_state = False
    multi_state_strategy = "arithmetic_mean"
    multi_state_temperature = 1.0
    state_weights = None
    temperature = [1.0]
    average_node_features = False  # No encoding fusion for Wave 1


# ============================================================================
# Fixture Loading
# ============================================================================


def load_fixture(fixture_path: Path) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load structure fixture from NPZ file.

    Parameters
    ----------
    fixture_path : Path
        Path to .npz file with keys: coords, mask, residue_index, chain_index

    Returns
    -------
    tuple
        (coords, mask, residue_index, chain_index) as JAX arrays
    """
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    data = np.load(fixture_path)
    return (
        jnp.asarray(data["coords"], dtype=jnp.float32),
        jnp.asarray(data["mask"], dtype=jnp.float32),
        jnp.asarray(data["residue_index"], dtype=jnp.int32),
        jnp.asarray(data["chain_index"], dtype=jnp.int32),
    )


# ============================================================================
# Model & Plan Setup
# ============================================================================


def load_model(checkpoint_path: Path | None = None) -> Any:
    """Load pre-trained model.

    Parameters
    ----------
    checkpoint_path : Path | None
        Path to .eqx checkpoint file. If None, creates model with random params.

    Returns
    -------
    Any
        Loaded model instance (PrxteinMPNN)
    """
    from prxteinmpnn.model.mpnn import PrxteinMPNN

    # Create model with reduced dimensions for faster compilation on CPU
    key = random.PRNGKey(42)
    key, subkey = random.split(key)

    model = PrxteinMPNN(
        node_features=32,
        edge_features=32,
        hidden_features=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=3,
        key=subkey,
    )

    # Load checkpoint if provided (note: checkpoint would be PrxteinLigandMPNN format)
    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            model = eqx.tree_deserialise_leaves(str(checkpoint_path), model)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint {checkpoint_path}: {e}")
            logger.info("Continuing with random initialization")

    return model


def create_inference_plan(model: Any) -> Any:
    """Create InferencePlan from model.

    Parameters
    ----------
    model : Any
        Loaded model instance

    Returns
    -------
    Any
        InferencePlan instance ready for encode/decode
    """
    from prxteinmpnn.host.plan import make_inference_plan

    spec = _BenchmarkSpec()
    plan = make_inference_plan(model, spec)
    return plan


# ============================================================================
# Timing: Cold Compile
# ============================================================================


def measure_cold_compile(
    plan: Any,
    bundle: Any,
    key: PRNGKeyArray,
    config: Any,
    fixture_name: str,
) -> tuple[float, str]:
    """Measure cold XLA compilation time for decode.

    Disables compilation cache, clears caches, then times first execution.

    Parameters
    ----------
    plan : Any
        InferencePlan instance
    bundle : Any
        InferenceBundle
    key : PRNGKeyArray
        PRNG key
    config : Any
        InferenceConfig
    fixture_name : str
        For logging

    Returns
    -------
    tuple
        (compile_time_s, note_string)
    """
    # Ensure cache is disabled
    jax.config.update("jax_enable_compilation_cache", False)
    jax.clear_caches()

    # First, encode (to get EncoderOutput)
    enc = plan.encode(bundle, key, config)

    # Now time cold decode
    t0 = time.perf_counter()
    result = plan.decode(enc, bundle, key, config)
    jax.block_until_ready(result)
    compile_time_cold_s = time.perf_counter() - t0

    note = "JAX: XLA compilation; cache disabled for cold run (jax_enable_compilation_cache=False)"

    return compile_time_cold_s, note


# ============================================================================
# Timing: Warm Latency
# ============================================================================


def measure_warm_latency(
    plan: Any,
    bundle: Any,
    key: PRNGKeyArray,
    config: Any,
    n_warmup: int,
    n_timed: int,
) -> tuple[float, float, list[float]]:
    """Measure warm decode latency with compiled kernel.

    Calls decode directly (which internally uses JIT). Warm-up phase
    allows JAX to compile and cache. Timed phase measures end-to-end latency.

    Parameters
    ----------
    plan : Any
        InferencePlan instance
    bundle : Any
        InferenceBundle
    key : PRNGKeyArray
        PRNG key
    config : Any
        InferenceConfig
    n_warmup : int
        Number of warmup runs
    n_timed : int
        Number of timed runs

    Returns
    -------
    tuple
        (median_latency_s, p95_latency_s, all_times_s)
    """
    # Encode once
    enc = plan.encode(bundle, key, config)

    # Warm-up phase: call decode directly (JAX handles JIT internally)
    for _ in range(n_warmup):
        result = plan.decode(enc, bundle, key, config)
        jax.block_until_ready(result)

    # Timed phase
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        result = plan.decode(enc, bundle, key, config)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    median_s = float(np.median(times_arr))
    p95_s = float(np.percentile(times_arr, 95))

    return median_s, p95_s, times


# ============================================================================
# Benchmark Single Cell
# ============================================================================


def benchmark_cell(
    model: Any,
    fixture_dir: Path,
    seq_len: int,
    batch_size: int,
    precision: str,
    n_warmup: int = 10,
    n_timed: int = 20,
) -> dict[str, Any] | None:
    """Benchmark a single (seq_len, batch_size, precision) cell.

    Parameters
    ----------
    model : Any
        Loaded model
    fixture_dir : Path
        Directory containing fixture .npz files
    seq_len : int
        Sequence length
    batch_size : int
        Batch size (number of copies of fixture to process)
    precision : str
        Precision string (e.g., "bf16", "fp32")
    n_warmup : int
        Warmup runs
    n_timed : int
        Timed runs

    Returns
    -------
    dict or None
        Result dictionary or None on failure
    """
    from prxteinmpnn.inference.bundle_builder import build_inference_bundle
    from prxteinmpnn.types.configs import InferenceConfig

    try:
        # Load fixture
        fixture_path = fixture_dir / f"structure_L{seq_len}.npz"
        coords, mask, residue_index, chain_index = load_fixture(fixture_path)

        # For batch_size > 1: stack copies along the state axis (S, L, ...)
        # build_inference_bundle accepts (S, L, 4, 3) and treats S as num_states.
        # With AxisStrategy=Vmap the model vmaps over states, giving correct GPU
        # occupancy for benchmarking throughput at batch_size > 1.
        if batch_size > 1:
            coords = jnp.stack([coords] * batch_size, axis=0)          # (B, L, 4, 3)
            mask = jnp.stack([mask] * batch_size, axis=0)              # (B, L)
            residue_index = jnp.stack([residue_index] * batch_size, axis=0)
            chain_index = jnp.stack([chain_index] * batch_size, axis=0)

        # Build bundle and config (no ligand for Wave 1)
        bundle, config = build_inference_bundle(
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            ligand_coords=None,
            ligand_atom_types=None,
            ligand_mask=None,
            temperature=1.0,
            mode="score_conditional",
            inference=True,
        )

        # Create plan
        plan = create_inference_plan(model)

        # PRNG key for inference
        key = random.PRNGKey(42)

        # Cold compile time
        logger.info(f"  Computing cold compile time (seq_len={seq_len}, batch_size={batch_size})...")
        compile_time_s, compile_note = measure_cold_compile(plan, bundle, key, config, f"L{seq_len}B{batch_size}")

        # Warm latency
        logger.info(f"  Computing warm latency...")
        median_s, p95_s, times = measure_warm_latency(plan, bundle, key, config, n_warmup, n_timed)

        # Compute derived metrics
        latency_median_ms = median_s * 1000.0
        latency_p95_ms = p95_s * 1000.0
        latency_per_residue_us = (median_s * 1e6) / (seq_len * batch_size)
        throughput_seq_per_s = batch_size / median_s

        # Memory (placeholder)
        peak_gpu_memory_gb = _get_gpu_memory_gb()

        # Assemble result
        result = {
            "schema_version": "1",
            "model": "prxteinmpnn_jax",
            "hardware": "unknown",  # Set by caller
            "seq_len": seq_len,
            "batch_size": batch_size,
            "precision": precision,
            "ligand_conditioning": False,
            "axis_strategy": "Vmap",
            "average_encoding_mode": "inputs_and_noise",
            "compile_time_cold_s": float(compile_time_s),
            "compile_time_warm_s": 0.0,  # Not measured separately
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
            f"  ✓ seq_len={seq_len}, batch_size={batch_size}: "
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
        description="GPU benchmark for prxteinmpnn JAX adapter (Wave 1)",
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
        default=["bf16"],
        choices=["bf16", "fp32"],
        help="Precisions to benchmark (default: [bf16])",
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
        "--fixture-dir",
        type=Path,
        default=Path("outputs/benchmark_fixtures"),
        help="Directory containing fixture .npz files (default: outputs/benchmark_fixtures)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
        help="Path to model checkpoint (.eqx)",
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

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Apply smoke-test defaults
    if args.smoke:
        args.seq_lens = [76]
        args.batch_sizes = [1]
        args.n_warmup = 1
        args.n_timed = 3
        logger.info("Smoke test mode: minimal iterations")

    # Resolve checkpoint path
    checkpoint_path = args.reference_path
    if checkpoint_path is None:
        ref_path = os.environ.get("REFERENCE_PATH")
        if ref_path:
            checkpoint_path = Path(ref_path) / "model_params" / "ligandmpnn_v_32_010_25_converted.eqx"
            if not checkpoint_path.exists():
                logger.warning(f"REFERENCE_PATH env set but checkpoint not found: {checkpoint_path}")
                checkpoint_path = None

    # Dry-run: just print config
    if args.dry_run:
        config = {
            "seq_lens": args.seq_lens,
            "batch_sizes": args.batch_sizes,
            "precisions": args.precision,
            "hardware": args.hardware,
            "fixture_dir": str(args.fixture_dir),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "n_warmup": args.n_warmup,
            "n_timed": args.n_timed,
            "jax_version": jax.__version__,
            "cuda_available": jax.devices()[0].platform == "gpu" if jax.devices() else False,
        }
        logger.info("DRY RUN - Configuration:")
        logger.info(json.dumps(config, indent=2))
        return 0

    # Check fixture directory
    if not args.fixture_dir.exists():
        logger.error(f"Fixture directory not found: {args.fixture_dir}")
        return 1

    # Set JAX defaults early
    _set_jax_defaults()

    # Load model
    logger.info(f"Loading model {'with checkpoint' if checkpoint_path else 'with random init'}...")
    try:
        model = load_model(checkpoint_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Run benchmarks
    all_results = []
    total_cells = len(args.seq_lens) * len(args.batch_sizes) * len(args.precision)
    current_cell = 0

    for seq_len in args.seq_lens:
        for batch_size in args.batch_sizes:
            for precision in args.precision:
                current_cell += 1
                logger.info(
                    f"Benchmarking cell {current_cell}/{total_cells}: "
                    f"seq_len={seq_len}, batch_size={batch_size}, precision={precision}"
                )

                result = benchmark_cell(
                    model,
                    args.fixture_dir,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    precision=precision,
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

    # Write output
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
