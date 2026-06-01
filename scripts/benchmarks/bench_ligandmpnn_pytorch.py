#!/usr/bin/env python3
"""GPU benchmark adapter for LigandMPNN (PyTorch) inference.

Measures cold first-call overhead, warm latency, and GPU memory usage for each
(seq_len, batch_size, precision) cell. Produces JSON output conforming to
the benchmark suite specification.

Usage:
    uv run python scripts/benchmarks/bench_ligandmpnn_pytorch.py --dry-run
    uv run python scripts/benchmarks/bench_ligandmpnn_pytorch.py --smoke
    uv run python scripts/benchmarks/bench_ligandmpnn_pytorch.py \
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
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration & Utilities
# ============================================================================


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


def _get_torch_version() -> str:
    """Get PyTorch version."""
    try:
        import torch

        return torch.__version__
    except Exception:
        return "unknown"


def _check_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# ============================================================================
# Reference Path Resolution
# ============================================================================


def resolve_reference_path(reference_path: str | None = None) -> Path:
    """Resolve reference path from argument or environment.

    Parameters
    ----------
    reference_path : str | None
        Explicit reference path from CLI, or None to use env var

    Returns
    -------
    Path
        Resolved path to LigandMPNN reference repository

    Raises
    ------
    ValueError
        If path cannot be resolved or doesn't exist
    """
    if reference_path:
        path = Path(reference_path)
    else:
        env_path = os.environ.get("REFERENCE_PATH", "/home/marielle/repos/LigandMPNN")
        path = Path(env_path)

    if not path.exists():
        raise ValueError(f"REFERENCE_PATH does not exist: {path}")

    return path


# ============================================================================
# Fixture Loading
# ============================================================================


def load_fixture(fixture_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load structure fixture from NPZ file.

    Parameters
    ----------
    fixture_path : Path
        Path to .npz file with keys: coords, mask, residue_index, chain_index

    Returns
    -------
    tuple
        (coords, mask, residue_index, chain_index) as numpy arrays
    """
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    data = np.load(fixture_path)
    return (
        np.asarray(data["coords"], dtype=np.float32),
        np.asarray(data["mask"], dtype=np.float32),
        np.asarray(data["residue_index"], dtype=np.int64),
        np.asarray(data["chain_index"], dtype=np.int64),
    )


# ============================================================================
# Model & Feature Setup (PyTorch)
# ============================================================================


def load_model(
    reference_path: Path,
    device: Any,
    model_type: str = "protein_mpnn",
) -> Any:
    """Load pre-trained LigandMPNN model.

    Parameters
    ----------
    reference_path : Path
        Path to LigandMPNN reference repository
    device : torch.device
        Device to load model on (cuda or cpu)
    model_type : str
        Model type: "protein_mpnn" or "ligand_mpnn"

    Returns
    -------
    Any
        Loaded ProteinMPNN model instance
    """
    import sys

    import torch

    # Add reference to path
    sys.path.insert(0, str(reference_path))
    from model_utils import ProteinMPNN

    # Load checkpoint
    checkpoint_path = reference_path / "model_params" / "ligandmpnn_v_32_010_25.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    k_neighbors = checkpoint["num_edges"]
    atom_context_num = checkpoint.get("atom_context_num", 1)

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=model_type,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return model, atom_context_num


def create_protein_dict(
    coords: np.ndarray,
    mask: np.ndarray,
    residue_index: np.ndarray,
    chain_index: np.ndarray,
    device: Any,
) -> dict[str, Any]:
    """Create protein_dict for featurize() call.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates (L, 4, 3)
    mask : np.ndarray
        Residue mask (L,)
    residue_index : np.ndarray
        Residue indices (L,)
    chain_index : np.ndarray
        Chain indices (L,)
    device : torch.device
        Device to create tensors on

    Returns
    -------
    dict
        protein_dict ready for featurize()
    """
    import torch

    L = coords.shape[0]
    return {
        "X": torch.tensor(coords, dtype=torch.float32, device=device),  # (L, 4, 3)
        "mask": torch.tensor(mask, dtype=torch.float32, device=device),  # (L,)
        "R_idx": torch.tensor(residue_index, dtype=torch.long, device=device),  # (L,)
        "chain_labels": torch.tensor(chain_index, dtype=torch.long, device=device),  # (L,)
        "S": torch.zeros(L, dtype=torch.long, device=device),  # (L,) dummy seq
        "chain_mask": torch.ones(L, dtype=torch.float32, device=device),  # (L,) design all
    }


def prepare_feature_dict(
    protein_dict: dict[str, Any],
    reference_path: Path,
    model_type: str,
    atom_context_num: int,
    device: Any,
) -> dict[str, Any]:
    """Prepare feature_dict for model.sample().

    Parameters
    ----------
    protein_dict : dict
        Output from create_protein_dict()
    reference_path : Path
        Path to LigandMPNN reference repo (for parse_PDB if needed)
    model_type : str
        "protein_mpnn" or "ligand_mpnn"
    atom_context_num : int
        Number of ligand atoms (for ligand_mpnn)
    device : torch.device
        Device

    Returns
    -------
    dict
        feature_dict ready for model.sample()
    """
    import sys

    import torch

    sys.path.insert(0, str(reference_path))
    from data_utils import featurize

    # Featurize
    feature_dict = featurize(protein_dict, model_type=model_type, number_of_ligand_atoms=atom_context_num)

    # Add required keys for model.sample()
    L = feature_dict["X"].shape[1]  # Sequence length from batch dim
    feature_dict["temperature"] = 0.1
    feature_dict["bias"] = torch.zeros([1, L, 21], device=device)
    feature_dict["symmetry_residues"] = []
    feature_dict["symmetry_weights"] = []

    return feature_dict


# ============================================================================
# Timing: Cold First-Call Overhead
# ============================================================================


def measure_cold_overhead(
    model: Any,
    feature_dict: dict[str, Any],
    batch_size: int,
    device: Any,
) -> tuple[float, str]:
    """Measure cold first-call overhead (CUDA kernel warmup).

    Parameters
    ----------
    model : Any
        Loaded ProteinMPNN model
    feature_dict : dict
        Feature dict from prepare_feature_dict()
    batch_size : int
        Batch size (number of sequences to sample)
    device : torch.device
        Device

    Returns
    -------
    tuple
        (overhead_time_s, note_string)
    """
    import torch

    # Set up feature dict with batch_size and randn
    L = feature_dict["X"].shape[1]
    feature_dict["batch_size"] = batch_size
    feature_dict["randn"] = torch.randn([batch_size, L], device=device)

    # Cold first call
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.sample(feature_dict)
    torch.cuda.synchronize()
    overhead_s = time.perf_counter() - t0

    note = "PyTorch eager mode: CUDA kernel warmup only; no torch.compile; value is first_call_overhead_s, not compiler overhead"

    return overhead_s, note


# ============================================================================
# Timing: Warm Latency
# ============================================================================


def measure_warm_latency(
    model: Any,
    feature_dict: dict[str, Any],
    batch_size: int,
    device: Any,
) -> tuple[float, float, list[float], int]:
    """Measure warm decode latency using torch.utils.benchmark.

    Parameters
    ----------
    model : Any
        Loaded ProteinMPNN model
    feature_dict : dict
        Feature dict from prepare_feature_dict()
    batch_size : int
        Batch size
    device : torch.device
        Device

    Returns
    -------
    tuple
        (median_latency_s, p95_latency_s, all_times_s, n_runs)
    """
    import torch
    import torch.utils.benchmark as benchmark

    # Set up feature dict with batch_size and randn
    L = feature_dict["X"].shape[1]
    feature_dict["batch_size"] = batch_size
    feature_dict["randn"] = torch.randn([batch_size, L], device=device)

    # Use torch.utils.benchmark.Timer for warm latency
    timer = benchmark.Timer(
        stmt="with torch.no_grad(): model.sample(fd)",
        globals={"model": model, "fd": feature_dict},
        num_threads=1,
    )

    # blocked_autorange handles timing multiple runs automatically
    measurement = timer.blocked_autorange(min_run_time=2.0)

    # measurement.times is total time per measurement group; divide by number_per_run for per-run times
    per_run_s = np.array(measurement.times) / measurement.number_per_run
    median_s = float(measurement.median)  # already per-run
    p95_s = float(np.percentile(per_run_s, 95))

    # Number of actual runs
    n_runs = measurement.number_per_run * measurement.num_measurements

    return median_s, p95_s, per_run_s.tolist(), n_runs


# ============================================================================
# Benchmark Single Cell
# ============================================================================


def benchmark_cell(
    model: Any,
    fixture_dir: Path,
    seq_len: int,
    batch_size: int,
    precision: str,
    device: Any,
    reference_path: Path,
    atom_context_num: int,
    model_type: str = "protein_mpnn",
) -> dict[str, Any] | None:
    """Benchmark a single (seq_len, batch_size, precision) cell.

    Parameters
    ----------
    model : Any
        Loaded ProteinMPNN model
    feature_dict_template : dict
        Template feature dict (will be copied for this cell)
    fixture_dir : Path
        Directory containing fixture .npz files
    seq_len : int
        Sequence length
    batch_size : int
        Batch size
    precision : str
        Precision string ("bf16", "fp32")
    device : torch.device
        Device

    Returns
    -------
    dict or None
        Result dictionary or None on failure
    """
    import torch

    try:
        # Load fixture
        fixture_path = fixture_dir / f"structure_L{seq_len}.npz"
        coords, mask, residue_index, chain_index = load_fixture(fixture_path)

        # Create and prepare feature dict for this cell
        protein_dict = create_protein_dict(coords, mask, residue_index, chain_index, device)

        # Prepare feature dict (will be modified in-place by timing functions)
        feature_dict = prepare_feature_dict(
            protein_dict,
            reference_path=reference_path,
            model_type=model_type,
            atom_context_num=atom_context_num,
            device=device,
        )

        # LigandMPNN reference model runs fp32 natively; this is its user-facing default.
        # bf16 is not supported by the reference implementation.
        if precision != "fp32":
            logger.info(f"  Skipping precision={precision}: LigandMPNN reference only supports fp32")
            return None

        # Cold first-call overhead
        logger.info(f"  Computing cold first-call overhead (seq_len={seq_len}, batch_size={batch_size})...")
        overhead_s, overhead_note = measure_cold_overhead(model, feature_dict, batch_size, device)

        # Warm latency
        logger.info(f"  Computing warm latency...")
        median_s, p95_s, times, n_runs = measure_warm_latency(model, feature_dict, batch_size, device)

        # Compute derived metrics
        latency_median_ms = median_s * 1000.0
        latency_p95_ms = p95_s * 1000.0
        latency_per_residue_us = (median_s * 1e6) / (seq_len * batch_size)
        throughput_seq_per_s = batch_size / median_s

        # GPU memory: query via nvidia-smi after warmup kernels are resident
        peak_gpu_memory_gb = 0.0
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                mb = int(result.stdout.strip().split("\n")[0].strip())
                peak_gpu_memory_gb = mb / 1024.0
        except Exception:
            pass

        # Assemble result
        result = {
            "schema_version": "1",
            "model": "ligandmpnn_pytorch",
            "hardware": "unknown",  # Set by caller
            "seq_len": seq_len,
            "batch_size": batch_size,
            "precision": precision,
            "ligand_conditioning": False,
            "axis_strategy": None,  # PyTorch eager; no JAX axis strategy
            "average_encoding_mode": None,
            "compile_time_cold_s": float(overhead_s),
            "compile_time_warm_s": 0.0,
            "compile_time_note": overhead_note,
            "latency_median_ms": float(latency_median_ms),
            "latency_p95_ms": float(latency_p95_ms),
            "latency_per_residue_us": float(latency_per_residue_us),
            "throughput_seq_per_s": float(throughput_seq_per_s),
            "peak_gpu_memory_gb": float(peak_gpu_memory_gb),
            "n_warmup": 0,  # torch.utils.benchmark handles warmup internally
            "n_timed": n_runs,
            "jax_version": None,
            "torch_version": _get_torch_version(),
            "cuda_version": _get_cuda_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"  ✓ seq_len={seq_len}, batch_size={batch_size}: "
            f"cold={overhead_s:.3f}s, warm_median={latency_median_ms:.2f}ms"
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
        description="GPU benchmark for LigandMPNN PyTorch adapter (Wave 2)",
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
        choices=["bf16", "fp32"],
        help="Precisions to benchmark (default: [fp32])",
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
        help="Warmup iterations (default: 10; unused for PyTorch Timer)",
    )
    parser.add_argument(
        "--n-timed",
        type=int,
        default=20,
        help="Timed iterations (default: 20; unused for PyTorch Timer)",
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
        type=str,
        default=None,
        help="Path to LigandMPNN reference repo (default: $REFERENCE_PATH env, /home/marielle/repos/LigandMPNN)",
    )
    parser.add_argument(
        "--ligand",
        action="store_true",
        help="Use ligand_mpnn model type and 1BC8.pdb (default: protein_mpnn with fixture)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without running benchmarks",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run minimal benchmark: seq_lens=[76], batch_sizes=[1]",
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
        logger.info("Smoke test mode: minimal configuration")

    # Resolve reference path
    try:
        reference_path = resolve_reference_path(args.reference_path)
    except ValueError as e:
        if args.dry_run:
            reference_path = Path(args.reference_path or "/home/marielle/repos/LigandMPNN")
            logger.warning(f"Reference path warning (dry-run): {e}")
        else:
            logger.error(f"Reference path error: {e}")
            return 1

    # Dry-run: just print config
    if args.dry_run:
        config = {
            "seq_lens": args.seq_lens,
            "batch_sizes": args.batch_sizes,
            "precisions": args.precision,
            "hardware": args.hardware,
            "fixture_dir": str(args.fixture_dir),
            "reference_path": str(reference_path),
            "model_type": "ligand_mpnn" if args.ligand else "protein_mpnn",
            "ligand": args.ligand,
            "cuda_available": _check_cuda_available(),
            "torch_version": _get_torch_version(),
        }
        logger.info("DRY RUN - Configuration:")
        logger.info(json.dumps(config, indent=2))
        return 0

    # Check fixture directory
    if not args.fixture_dir.exists():
        logger.error(f"Fixture directory not found: {args.fixture_dir}")
        return 1

    # Import torch (not done during dry-run)
    try:
        import torch

        logger.info(f"PyTorch {_get_torch_version()} loaded")
    except ImportError as e:
        logger.error(f"Failed to import PyTorch: {e}")
        return 1

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA not available, falling back to CPU (benchmark will be slow)")
    logger.info(f"Using device: {device}")

    # Load model
    model_type = "ligand_mpnn" if args.ligand else "protein_mpnn"
    logger.info(f"Loading LigandMPNN model (model_type={model_type})...")
    try:
        model, atom_context_num = load_model(reference_path, device, model_type=model_type)
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
                    device=device,
                    reference_path=reference_path,
                    atom_context_num=atom_context_num,
                    model_type=model_type,
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
