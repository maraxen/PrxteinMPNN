#!/usr/bin/env python3
"""Benchmark suite dispatcher for prxteinmpnn, LigandMPNN, and ColabDesign.

This script coordinates three independent GPU benchmark adapters, each running
in a fresh subprocess to ensure GPU memory isolation between frameworks.

Usage:
    uv run python scripts/benchmarks/bench_suite.py \\
        --hardware A100 \\
        --output-dir outputs/results/benchmarks \\
        --seq-lens 76 150 300 500 \\
        --batch-sizes 1 4 16 \\
        --precision bf16 \\
        --n-warmup 10 \\
        --n-timed 20 \\
        --fixture-dir outputs/benchmark_fixtures \\
        --pdb-dir tests/data \\
        --reference-path /path/to/LigandMPNN \\
        --dry-run

Optional skip flags: --skip-pytorch, --skip-colabdesign, --skip-prxteinmpnn
--smoke: pass through and set seq-lens to 76, batch-sizes to 1
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
) -> list[str]:
    """Build argv for bench_prxteinmpnn_jax.py."""
    argv = [
        sys.executable,
        str(_BENCH_DIR / "bench_prxteinmpnn_jax.py"),
        "--hardware",
        args.hardware,
        "--output-json",
        str(output_json),
        "--fixture-dir",
        str(args.fixture_dir),
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

    if args.smoke:
        argv.append("--smoke")

    if args.dry_run:
        argv.append("--dry-run")

    return argv


def build_pytorch_argv(
    args: argparse.Namespace,
    output_json: Path,
) -> list[str]:
    """Build argv for bench_ligandmpnn_pytorch.py."""
    argv = [
        sys.executable,
        str(_BENCH_DIR / "bench_ligandmpnn_pytorch.py"),
        "--hardware",
        args.hardware,
        "--output-json",
        str(output_json),
        "--fixture-dir",
        str(args.fixture_dir),
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

    if args.smoke:
        argv.append("--smoke")

    if args.dry_run:
        argv.append("--dry-run")

    return argv


def build_colabdesign_argv(
    args: argparse.Namespace,
    output_json: Path,
) -> list[str]:
    """Build argv for bench_colabdesign_jax.py.

    Note: ColabDesign uses --pdb-dir instead of --fixture-dir,
    and always uses seq-lens 76 and 500 (not 150, 300).
    Does NOT support --precision flag.
    """
    argv = [
        sys.executable,
        str(_BENCH_DIR / "bench_colabdesign_jax.py"),
        "--hardware",
        args.hardware,
        "--output-json",
        str(output_json),
        "--pdb-dir",
        str(args.pdb_dir),
        "--seq-lens",
        "76",
        "500",
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


def run_adapter(
    adapter_name: str,
    argv: list[str],
    dry_run: bool = False,
) -> tuple[str, dict[str, Any] | None]:
    """Run a single adapter subprocess.

    Parameters
    ----------
    adapter_name : str
        Name of adapter (e.g., "prxteinmpnn_jax")
    argv : list[str]
        Full argv list (starting with sys.executable)
    dry_run : bool
        If True, print command and return without executing

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
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"{adapter_name} failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"stderr: {result.stderr[-500:]}")  # Last 500 chars
            if result.stdout:
                logger.error(f"stdout: {result.stdout[-500:]}")
            return "failed", None

        # Try to parse results from stdout if --output-json was passed
        # The adapter writes to file, so we'll just check that it succeeded
        logger.info(f"{adapter_name} completed successfully")
        return "ok", None

    except subprocess.TimeoutExpired:
        logger.error(f"{adapter_name} timed out (>1 hour)")
        return "failed", None
    except Exception as e:
        logger.error(f"{adapter_name} error: {e}")
        return "failed", None


def load_adapter_results(output_json: Path) -> dict[str, Any] | None:
    """Load results from adapter JSON file.

    Parameters
    ----------
    output_json : Path
        Path to adapter output JSON

    Returns
    -------
    dict | None
        Parsed JSON, or None if file not found
    """
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
        "--seq-lens",
        type=int,
        nargs="+",
        default=[76, 150, 300, 500],
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
        default=["bf16", "fp32"],
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
        "--fixture-dir",
        type=Path,
        default=Path("outputs/benchmark_fixtures"),
        help="Directory containing benchmark fixtures (for JAX adapters)",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=Path("tests/data"),
        help="Directory containing PDB files (for ColabDesign)",
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
        help="Run in smoke-test mode (seq-lens 76, batch-sizes 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print adapter commands without executing",
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

    args = parser.parse_args()

    # Handle --smoke: override seq-lens and batch-sizes
    if args.smoke:
        args.seq_lens = [76]
        args.batch_sizes = [1]
        logger.info("Smoke mode: seq-lens=[76], batch-sizes=[1]")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve reference path
    reference_path = args.reference_path
    if reference_path is None:
        reference_path = os.environ.get("REFERENCE_PATH")
    if reference_path:
        logger.info(f"Using REFERENCE_PATH: {reference_path}")

    # Dispatch adapters
    adapter_status: dict[str, str] = {}
    combined_results: list[dict[str, Any]] = []

    # === PRXTEINMPNN ===
    if not args.skip_prxteinmpnn:
        logger.info("=" * 70)
        logger.info("Dispatching prxteinmpnn_jax adapter...")
        output_json = args.output_dir / f"{args.hardware}_prxteinmpnn_jax_bench.json"
        argv = build_prxteinmpnn_argv(args, output_json)

        status, _ = run_adapter("prxteinmpnn_jax", argv, dry_run=args.dry_run)
        adapter_status["prxteinmpnn_jax"] = status

        if status == "ok":
            results = load_adapter_results(output_json)
            if results and "results" in results:
                combined_results.extend(results["results"])
                logger.info(
                    f"Loaded {len(results['results'])} cells from prxteinmpnn_jax"
                )

    # === LIGANDMPNN (PYTORCH) ===
    if not args.skip_pytorch:
        logger.info("=" * 70)
        logger.info("Dispatching ligandmpnn_pytorch adapter...")

        # Check reference path
        if not reference_path:
            logger.warning(
                "REFERENCE_PATH not set and --reference-path not provided. "
                "Skipping PyTorch adapter."
            )
            adapter_status["ligandmpnn_pytorch"] = "skipped"
        else:
            output_json = (
                args.output_dir / f"{args.hardware}_ligandmpnn_pytorch_bench.json"
            )
            argv = build_pytorch_argv(args, output_json)

            status, _ = run_adapter("ligandmpnn_pytorch", argv, dry_run=args.dry_run)
            adapter_status["ligandmpnn_pytorch"] = status

            if status == "ok":
                results = load_adapter_results(output_json)
                if results and "results" in results:
                    combined_results.extend(results["results"])
                    logger.info(
                        f"Loaded {len(results['results'])} cells from ligandmpnn_pytorch"
                    )

    # === COLABDESIGN ===
    if not args.skip_colabdesign:
        logger.info("=" * 70)
        logger.info("Dispatching colabdesign_jax adapter...")
        output_json = args.output_dir / f"{args.hardware}_colabdesign_jax_bench.json"
        argv = build_colabdesign_argv(args, output_json)

        status, _ = run_adapter("colabdesign_jax", argv, dry_run=args.dry_run)
        adapter_status["colabdesign_jax"] = status

        if status == "ok":
            results = load_adapter_results(output_json)
            if results and "results" in results:
                combined_results.extend(results["results"])
                logger.info(
                    f"Loaded {len(results['results'])} cells from colabdesign_jax"
                )

    # === Summary ===
    logger.info("=" * 70)
    logger.info("Adapter status:")
    for adapter, status in adapter_status.items():
        logger.info(f"  {adapter}: {status}")

    # Check if at least one succeeded
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
        "adapter_status": adapter_status,
        "results": combined_results,
        "total_cells": len(combined_results),
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
