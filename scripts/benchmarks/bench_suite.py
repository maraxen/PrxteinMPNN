#!/usr/bin/env python3
"""Benchmark suite dispatcher for prxteinmpnn, LigandMPNN, and ColabDesign.

This script coordinates three independent GPU benchmark adapters, each running
in a fresh subprocess to ensure GPU memory isolation between frameworks.

Usage:
    uv run python scripts/benchmarks/bench_suite.py \\
        --hardware A100 \\
        --output-dir outputs/results/benchmarks \\
        --tasks score_conditional ar_sample \\
        --seq-lens 76 500 \\
        --batch-sizes 1 4 16 \\
        --precision fp32 \\
        --n-warmup 10 \\
        --n-timed 20 \\
        --pdb-dir tests/data \\
        --reference-path /path/to/LigandMPNN \\
        --dry-run

Optional skip flags: --skip-pytorch, --skip-colabdesign, --skip-prxteinmpnn
Capability skip flags: --skip-temperature, --skip-dedup, --skip-mixed-length
  (only active if scripts are present on disk)
--smoke: pass through and set seq-lens to 76, batch-sizes to 1, tasks to both
--dry-run: print resolved adapter commands without executing

Exit codes:
    0: SUCCESS (at least one adapter succeeded)
    1: FAILURE (all adapters failed or were skipped)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


# ============================================================================
# Adapter Dispatch
# ============================================================================


_BENCH_DIR = Path(__file__).parent


def build_prxteinmpnn_argv(
    args: argparse.Namespace,
    output_json: Path,
    task: str,
) -> list[str]:
    """Build argv for bench_prxteinmpnn_jax.py."""
    argv = [
        sys.executable,
        str(_BENCH_DIR / "bench_prxteinmpnn_jax.py"),
        "--task",
        task,
        "--hardware",
        args.hardware,
        "--output-json",
        str(output_json),
        "--pdb-dir",
        str(args.pdb_dir),
        "--seq-lens",
        *map(str, args.seq_lens),
        "--batch-sizes",
        *map(str, args.batch_sizes),
        "--precision",
        *args.precision,
        "--n-warmup",
        str(args.n_warmup),
        "--n-timed",
        str(args.n_timed),
    ]

    if args.reference_path:
        argv.extend(["--reference-path", str(args.reference_path)])

    if args.ligand:
        argv.append("--ligand")

    if args.smoke:
        argv.append("--smoke")

    if args.dry_run:
        argv.append("--dry-run")

    return argv


def build_pytorch_argv(
    args: argparse.Namespace,
    output_json: Path,
    task: str,
) -> list[str]:
    """Build argv for bench_ligandmpnn_pytorch.py."""
    argv = [
        sys.executable,
        str(_BENCH_DIR / "bench_ligandmpnn_pytorch.py"),
        "--task",
        task,
        "--hardware",
        args.hardware,
        "--output-json",
        str(output_json),
        "--pdb-dir",
        str(args.pdb_dir),
        "--seq-lens",
        *map(str, args.seq_lens),
        "--batch-sizes",
        *map(str, args.batch_sizes),
        "--precision",
        *args.precision,
        "--n-warmup",
        str(args.n_warmup),
        "--n-timed",
        str(args.n_timed),
    ]

    if args.reference_path:
        argv.extend(["--reference-path", str(args.reference_path)])

    if args.ligand:
        argv.append("--ligand-conditioning")

    if args.smoke:
        argv.append("--smoke")

    if args.dry_run:
        argv.append("--dry-run")

    return argv


def build_colabdesign_argv(
    args: argparse.Namespace,
    output_json: Path,
    task: str,
) -> list[str]:
    """Build argv for bench_colabdesign_jax.py.

    Note: ColabDesign uses --pdb-dir and respects --seq-lens.
    Does NOT support --precision flag (always fp32).
    """
    argv = [
        sys.executable,
        str(_BENCH_DIR / "bench_colabdesign_jax.py"),
        "--task",
        task,
        "--hardware",
        args.hardware,
        "--output-json",
        str(output_json),
        "--pdb-dir",
        str(args.pdb_dir),
        "--seq-lens",
        *map(str, args.seq_lens),
        "--batch-sizes",
        *map(str, args.batch_sizes),
        "--n-warmup",
        str(args.n_warmup),
        "--n-timed",
        str(args.n_timed),
    ]

    if args.smoke:
        argv.append("--smoke")

    if args.dry_run:
        argv.append("--dry-run")

    return argv


def build_temperature_array_argv(
    args: argparse.Namespace,
    output_json: Path,
) -> list[str]:
    """Build argv for bench_temperature_array.py.

    Temperature array benchmark measures latency for M-temperature JIT-native
    sweep (prxteinmpnn) vs sequential baselines (ColabDesign, PyTorch).
    """
    argv = [
        sys.executable,
        str(_BENCH_DIR / "bench_temperature_array.py"),
        "--hardware",
        args.hardware,
        "--output-json",
        str(output_json),
        "--pdb-dir",
        str(args.pdb_dir),
        "--seq-len",
        str(args.seq_lens[0]) if args.seq_lens else "76",
        "--n-warmup",
        str(args.n_warmup),
        "--n-timed",
        str(args.n_timed),
    ]

    if args.smoke:
        argv.append("--smoke")

    if args.dry_run:
        argv.append("--dry-run")

    return argv


def run_adapter(
    adapter_name: str,
    argv: list[str],
    dry_run: bool = False,
    timeout: int = 3600,
) -> tuple[str, dict[str, Any] | None]:
    """Run a single adapter subprocess.

    stderr is streamed live to the parent process (SLURM .err file) so per-cell
    progress is visible in real time. stdout is captured for return-code checking.

    Returns
    -------
    tuple[str, dict | None]
        (status_string, results_dict_or_none)
        status_string in {"ok", "failed", "skipped", "dry_run"}
    """
    logger.info(f"Adapter: {adapter_name}")
    logger.info(f"Command: {' '.join(argv)}")

    if dry_run:
        return "dry_run", None

    try:
        result = subprocess.run(
            argv,
            stdout=subprocess.PIPE,  # capture stdout (adapters write JSON to file, not stdout)
            stderr=None,             # stream stderr live → SLURM .err for incremental progress
            text=True,
            timeout=timeout,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"{adapter_name} failed with exit code {result.returncode}")
            return "failed", None

        logger.info(f"{adapter_name} completed successfully")
        return "ok", None

    except subprocess.TimeoutExpired:
        logger.error(f"{adapter_name} timed out (>{timeout}s)")
        return "failed", None
    except Exception as e:
        logger.error(f"{adapter_name} error: {e}")
        return "failed", None


def load_adapter_results(output_json: Path) -> dict[str, Any] | None:
    """Load results from adapter JSON file."""
    if not output_json.exists():
        logger.warning(f"Output file not found: {output_json}")
        return None

    try:
        with open(output_json) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {output_json}: {e}")
        return None


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    """Main entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Benchmark suite dispatcher for prxteinmpnn, LigandMPNN, and ColabDesign.",
    )

    # Hardware and output
    parser.add_argument(
        "--hardware",
        type=str,
        required=True,
        help="Hardware name (e.g., A100, H100)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for benchmark results",
    )

    # Benchmark parameters
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["score_conditional", "ar_sample"],
        default=["score_conditional", "ar_sample"],
        help="Tasks to benchmark (default: both)",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[76, 500],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--precision",
        type=str,
        nargs="+",
        default=["fp32"],
        choices=["bf16", "fp32", "fp16"],
        help="Precisions to benchmark",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--n-timed",
        type=int,
        default=20,
        help="Number of timed iterations",
    )

    # Fixture paths
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Directory containing PDB fixture files (1ubq.pdb, 1SMD.pdb). "
             "Defaults to tests/data relative to the prxteinmpnn package root.",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=None,
        help="[DEPRECATED] Use --pdb-dir instead.",
    )

    # Reference paths
    parser.add_argument(
        "--reference-path",
        type=str,
        default=None,
        help="Path to LigandMPNN reference (for PyTorch adapter)",
    )

    # Mode flags
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke-test mode (seq-lens 76, batch-sizes 1, both tasks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print adapter commands without executing",
    )

    # Ligand conditioning
    parser.add_argument(
        "--ligand",
        action="store_true",
        default=False,
        help="Enable ligand conditioning (passes --ligand to prxteinmpnn_jax, "
             "--ligand-conditioning to ligandmpnn_pytorch). Requires 1BC8.cif fixture.",
    )

    # Skip flags
    parser.add_argument(
        "--subprocess-timeout",
        type=int,
        default=3600,
        help="Seconds before a single adapter subprocess is killed (default: 3600). "
             "Increase for ar_sample on JAX (JIT compilation can take >1 hr).",
    )

    # Skip flags
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="Skip PyTorch (LigandMPNN) adapter",
    )
    parser.add_argument(
        "--skip-colabdesign",
        action="store_true",
        help="Skip ColabDesign adapter",
    )
    parser.add_argument(
        "--skip-prxteinmpnn",
        action="store_true",
        help="Skip prxteinmpnn adapter",
    )

    # Capability skip flags
    parser.add_argument(
        "--skip-temperature",
        action="store_true",
        help="Skip temperature array benchmark (if present on disk)",
    )
    parser.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Skip DedupGather heterogeneous batch benchmark (if present on disk)",
    )
    parser.add_argument(
        "--skip-mixed-length",
        action="store_true",
        help="Skip mixed-length heterogeneous batch benchmark (if present on disk)",
    )

    args = parser.parse_args()

    # Handle deprecated --fixture-dir
    if args.fixture_dir is not None:
        import warnings
        warnings.warn(
            "--fixture-dir is deprecated; use --pdb-dir instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        if args.pdb_dir is None:
            args.pdb_dir = args.fixture_dir

    # Resolve pdb_dir default
    if args.pdb_dir is None:
        args.pdb_dir = Path(__file__).parents[2] / "tests" / "data"

    # Handle --smoke: override grid
    if args.smoke:
        args.seq_lens = [76]
        args.batch_sizes = [1]
        args.n_warmup = 1
        args.n_timed = 3
        # Keep both tasks in smoke mode
        logger.info("Smoke mode: seq-lens=[76], batch-sizes=[1], n_warmup=1, n_timed=3")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve reference path
    reference_path = args.reference_path
    if reference_path is None:
        reference_path = os.environ.get("REFERENCE_PATH")
    if reference_path:
        logger.info(f"Using REFERENCE_PATH: {reference_path}")

    # Dispatch adapters: outer loop over tasks
    adapter_status: dict[str, str] = {}
    combined_results: list[dict[str, Any]] = []

    for task in args.tasks:
        logger.info("=" * 70)
        logger.info(f"Task: {task}")

        # === PRXTEINMPNN ===
        if not args.skip_prxteinmpnn:
            logger.info(f"Dispatching prxteinmpnn_jax adapter (task={task})...")
            output_json = args.output_dir / f"{args.hardware}_prxteinmpnn_jax_{task}_bench.json"
            argv = build_prxteinmpnn_argv(args, output_json, task)

            key = f"prxteinmpnn_jax_{task}"
            status, _ = run_adapter(key, argv, dry_run=args.dry_run, timeout=args.subprocess_timeout)
            adapter_status[key] = status

            if status == "ok":
                results = load_adapter_results(output_json)
                if results and "results" in results:
                    combined_results.extend(results["results"])
                    logger.info(
                        f"Loaded {len(results['results'])} cells from prxteinmpnn_jax/{task}"
                    )

        # === LIGANDMPNN (PYTORCH) ===
        if not args.skip_pytorch:
            if not reference_path:
                logger.warning(
                    "REFERENCE_PATH not set and --reference-path not provided. "
                    "Skipping PyTorch adapter."
                )
                adapter_status[f"ligandmpnn_pytorch_{task}"] = "skipped"
            else:
                logger.info(f"Dispatching ligandmpnn_pytorch adapter (task={task})...")
                output_json = (
                    args.output_dir / f"{args.hardware}_ligandmpnn_pytorch_{task}_bench.json"
                )
                argv = build_pytorch_argv(args, output_json, task)

                key = f"ligandmpnn_pytorch_{task}"
                status, _ = run_adapter(key, argv, dry_run=args.dry_run, timeout=args.subprocess_timeout)
                adapter_status[key] = status

                if status == "ok":
                    results = load_adapter_results(output_json)
                    if results and "results" in results:
                        combined_results.extend(results["results"])
                        logger.info(
                            f"Loaded {len(results['results'])} cells from ligandmpnn_pytorch/{task}"
                        )

        # === COLABDESIGN ===
        if not args.skip_colabdesign:
            logger.info(f"Dispatching colabdesign_jax adapter (task={task})...")
            output_json = args.output_dir / f"{args.hardware}_colabdesign_jax_{task}_bench.json"
            argv = build_colabdesign_argv(args, output_json, task)

            key = f"colabdesign_jax_{task}"
            status, _ = run_adapter(key, argv, dry_run=args.dry_run, timeout=args.subprocess_timeout)
            adapter_status[key] = status

            if status == "ok":
                results = load_adapter_results(output_json)
                if results and "results" in results:
                    combined_results.extend(results["results"])
                    logger.info(
                        f"Loaded {len(results['results'])} cells from colabdesign_jax/{task}"
                    )

    # === CAPABILITY BENCHMARKS (non-task-specific) ===
    capability_results: dict[str, Any] = {}

    # Temperature array benchmark
    temperature_script = _BENCH_DIR / "bench_temperature_array.py"
    if temperature_script.exists() and not args.skip_temperature:
        logger.info("=" * 70)
        logger.info("Dispatching temperature array benchmark...")
        output_json = args.output_dir / f"{args.hardware}_bench_temperature_array.json"
        argv = build_temperature_array_argv(args, output_json)

        key = "temperature_array"
        status, _ = run_adapter(key, argv, dry_run=args.dry_run, timeout=args.subprocess_timeout)
        adapter_status[key] = status

        if status == "ok":
            results = load_adapter_results(output_json)
            if results:
                capability_results["temperature"] = results
                logger.info(f"Loaded temperature array results")

    # DedupGather heterogeneous batch benchmark
    dedup_script = _BENCH_DIR / "bench_dedup_hetero.py"
    if dedup_script.exists() and not args.skip_dedup:
        logger.info("=" * 70)
        logger.info("Dispatching DedupGather heterogeneous batch benchmark...")
        output_json = args.output_dir / f"{args.hardware}_bench_dedup_hetero.json"
        argv = [
            sys.executable,
            str(dedup_script),
            "--hardware",
            args.hardware,
            "--output-json",
            str(output_json),
            "--pdb-dir",
            str(args.pdb_dir),
            "--n-warmup",
            str(args.n_warmup),
            "--n-timed",
            str(args.n_timed),
        ]

        if args.smoke:
            argv.append("--smoke")

        if args.dry_run:
            argv.append("--dry-run")

        key = "dedup_hetero"
        status, _ = run_adapter(key, argv, dry_run=args.dry_run, timeout=args.subprocess_timeout)
        adapter_status[key] = status

        if status == "ok":
            results = load_adapter_results(output_json)
            if results:
                capability_results["dedup"] = results
                logger.info(f"Loaded DedupGather results")

    # Mixed-length heterogeneous batch benchmark
    mixed_length_script = _BENCH_DIR / "bench_mixed_length.py"
    if mixed_length_script.exists() and not args.skip_mixed_length:
        logger.info("=" * 70)
        logger.info("Dispatching mixed-length heterogeneous batch benchmark...")
        output_json = args.output_dir / f"{args.hardware}_bench_mixed_length.json"
        argv = [
            sys.executable,
            str(mixed_length_script),
            "--hardware",
            args.hardware,
            "--output-json",
            str(output_json),
            "--pdb-dir",
            str(args.pdb_dir),
            "--n-warmup",
            str(args.n_warmup),
            "--n-timed",
            str(args.n_timed),
        ]

        if args.smoke:
            argv.append("--smoke")

        if args.dry_run:
            argv.append("--dry-run")

        key = "mixed_length"
        status, _ = run_adapter(key, argv, dry_run=args.dry_run, timeout=args.subprocess_timeout)
        adapter_status[key] = status

        if status == "ok":
            results = load_adapter_results(output_json)
            if results:
                capability_results["mixed_length"] = results
                logger.info(f"Loaded mixed-length results")

    # === Summary ===
    logger.info("=" * 70)
    logger.info("Adapter status:")
    for adapter, status in adapter_status.items():
        logger.info(f"  {adapter}: {status}")

    if args.dry_run:
        logger.info("Dry-run complete")
        return 0

    successes = sum(1 for status in adapter_status.values() if status == "ok")
    if successes == 0:
        logger.error("All adapters failed or were skipped")
        return 1

    # Write combined output
    combined_output = {
        "schema_version": "1",
        "hardware": args.hardware,
        "tasks": args.tasks,
        "adapter_status": adapter_status,
        "results": combined_results,
        "total_cells": len(combined_results),
        "capability_results": capability_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    combined_json = args.output_dir / f"{args.hardware}_combined.json"
    with open(combined_json, "w") as f:
        json.dump(combined_output, f, indent=2)

    logger.info(f"Combined results written to {combined_json}")
    logger.info(f"Total cells from all adapters: {len(combined_results)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
