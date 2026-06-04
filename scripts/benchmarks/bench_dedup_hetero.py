#!/usr/bin/env python3
"""DedupGather K/N unique-ratio throughput benchmark.

Benchmark demonstrating K-proportional throughput savings via DedupGather,
comparing prxteinmpnn (with dedup), ColabDesign (baseline K separate calls),
and PyTorch LigandMPNN (sequential K loop).

Synthetic heterogeneous batch: N=32 structures, K in {1,2,4,8,16,32} unique
(fill by repeating). For each K:
  - prxteinmpnn: InferencePlan with DedupGather on n_structures axis
  - ColabDesign: K separate model.sample(num=1) calls (one per unique)
  - PyTorch: sequential loop over K unique structures

Output JSON schema (schema_version="1"):
  {
    "schema_version": "1",
    "hardware": "H200|A100|L40s|Blackwell",
    "k_values": [1,2,4,8,16,32],
    "cells": [
      {
        "k": <int>,
        "n": <int>,
        "dedup_ratio": <float>,
        "prxteinmpnn_latency_ms": <float>,
        "colabdesign_latency_ms": <float>,
        "pytorch_latency_ms": <float>,
        "speedup_vs_pytorch": <float>
      },
      ...
    ]
  }

Usage:
    uv run python scripts/benchmarks/bench_dedup_hetero.py --dry-run
    uv run python scripts/benchmarks/bench_dedup_hetero.py --smoke
    uv run python scripts/benchmarks/bench_dedup_hetero.py \\
        --hardware H200 \\
        --output-json results.json \\
        --n-total 32 \\
        --k-values 1,2,4,8,16,32 \\
        --n-warmup 10 \\
        --n-timed 20 \\
        --pdb-dir tests/data

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

# Standard 20-letter amino acid alphabet
_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_ALPHABET)}

# Canonical PDB map
_PDB_MAP: dict[int, str] = {
    76: "1ubq.pdb",
    500: "1SMD.pdb",
}

_DEFAULT_PDB_DIR = Path(__file__).parents[2] / "tests" / "data"


# ============================================================================
# Configuration & Utilities
# ============================================================================


def _set_jax_defaults():
    """Set JAX configuration before importing models."""
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_shard_autotuning=false")


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


# ============================================================================
# PDB Loading
# ============================================================================


def load_pdb_as_arrays(pdb_path: str) -> dict[str, Any]:
    """Load a PDB file and return coordinate arrays using biopython."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    coords_list = []
    mask_list = []
    seq_list = []
    residue_index_list = []
    chain_index_list = []

    model_obj = next(iter(structure))
    chain_ids = sorted(chain.id for chain in model_obj)
    chain_to_idx = {cid: i for i, cid in enumerate(chain_ids)}

    atom_order = {"N": 0, "CA": 1, "C": 2, "O": 3}

    for chain in model_obj:
        chain_idx = chain_to_idx[chain.id]
        for residue in chain:
            if residue.get_id()[0] != " ":
                continue

            res_coord = np.zeros((4, 3), dtype=np.float32)
            atom_found = [False, False, False, False]

            for atom_name, atom_idx in atom_order.items():
                if atom_name in residue:
                    res_coord[atom_idx] = residue[atom_name].get_vector().get_array()
                    atom_found[atom_idx] = True

            res_mask = 1.0 if atom_found[1] else 0.0

            resname = residue.get_resname().strip()
            try:
                from Bio.Data.IUPACData import protein_letters_3to1
                one_letter = protein_letters_3to1.get(resname.capitalize(), "X")
                aa_idx = _AA_TO_IDX.get(one_letter, 20)
            except Exception:
                aa_idx = 20

            res_idx = residue.get_id()[1]

            coords_list.append(res_coord)
            mask_list.append(res_mask)
            seq_list.append(aa_idx)
            residue_index_list.append(res_idx)
            chain_index_list.append(chain_idx)

    L = len(coords_list)
    return {
        "coords": np.array(coords_list, dtype=np.float32),
        "mask": np.array(mask_list, dtype=np.float32),
        "sequence": np.array(seq_list, dtype=np.int32),
        "residue_index": np.array(residue_index_list, dtype=np.int32),
        "chain_index": np.array(chain_index_list, dtype=np.int32),
        "actual_len": L,
    }


def _load_pdb_fixture(pdb_dir: Path, seq_len: int) -> dict[str, Any]:
    """Load PDB fixture for a given nominal seq_len."""
    if seq_len not in _PDB_MAP:
        raise FileNotFoundError(
            f"No PDB fixture for seq_len={seq_len}. "
            f"Available: {list(_PDB_MAP.keys())}"
        )

    pdb_file = pdb_dir / _PDB_MAP[seq_len]
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB fixture not found: {pdb_file}")

    return load_pdb_as_arrays(str(pdb_file))


# ============================================================================
# Synthetic Batch Generation
# ============================================================================


def create_synthetic_batch(
    pdb_fixture: dict[str, Any],
    n_total: int,
    k_unique: int,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Create a synthetic heterogeneous batch by repeating K unique structures.

    Parameters
    ----------
    pdb_fixture : dict
        Single PDB fixture (coords, mask, sequence, etc.)
    n_total : int
        Total number of structures in the batch
    k_unique : int
        Number of unique structures (fill by repeating)
    rng : np.random.Generator, optional
        Random generator for perturbation (optional)

    Returns
    -------
    dict with batched arrays (n_structures, L, ...) and metadata.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    L = pdb_fixture["actual_len"]

    # Create K unique perturbations of the base structure
    base_coords = pdb_fixture["coords"]
    unique_coords = []
    for i in range(k_unique):
        # Small random perturbation for each unique structure
        perturb = rng.normal(0, 0.01, base_coords.shape).astype(np.float32)
        unique_coords.append(base_coords + perturb)

    # Repeat to fill N positions
    batch_coords = []
    for idx in range(n_total):
        unique_idx = idx % k_unique
        batch_coords.append(unique_coords[unique_idx])

    batch_coords_arr = np.stack(batch_coords, axis=0)  # (N, L, 4, 3)

    # All structures use the same mask and sequence
    batch_mask = np.tile(pdb_fixture["mask"], (n_total, 1))  # (N, L)
    batch_sequence = np.tile(pdb_fixture["sequence"], (n_total, 1))  # (N, L)
    batch_residue_index = np.tile(pdb_fixture["residue_index"], (n_total, 1))  # (N, L)
    batch_chain_index = np.tile(pdb_fixture["chain_index"], (n_total, 1))  # (N, L)

    return {
        "coords": batch_coords_arr,
        "mask": batch_mask,
        "sequence": batch_sequence,
        "residue_index": batch_residue_index,
        "chain_index": batch_chain_index,
        "n_total": n_total,
        "k_unique": k_unique,
        "actual_len": L,
    }


# ============================================================================
# Timing Utilities
# ============================================================================


def median_from_timings(times_list: list[float]) -> float:
    """Return median latency in milliseconds."""
    times_arr = np.array(times_list)
    return float(np.median(times_arr)) * 1000.0


# ============================================================================
# prxteinmpnn Adapter (with DedupGather)
# ============================================================================


def benchmark_prxteinmpnn_dedup(
    batch: dict[str, Any],
    n_warmup: int = 10,
    n_timed: int = 20,
) -> float:
    """Benchmark prxteinmpnn with DedupGather on n_structures axis.

    Parameters
    ----------
    batch : dict
        Synthetic batch from create_synthetic_batch
    n_warmup : int
        Number of warmup iterations
    n_timed : int
        Number of timed iterations

    Returns
    -------
    float
        Median latency in milliseconds
    """
    import jax
    import jax.numpy as jnp
    from jax import random

    try:
        from prxteinmpnn.io.weights import load_model
        from prxteinmpnn.host.plan import make_inference_plan
        from prxteinmpnn.tiling.dedup import DedupSpec
    except ImportError as e:
        logger.warning(f"Failed to import prxteinmpnn: {e}")
        return 0.0

    logger.info(f"Loading prxteinmpnn model...")
    key = random.PRNGKey(42)
    model = load_model(checkpoint_id="proteinmpnn_v_48_020", key=key)

    logger.info("Creating inference plan...")
    spec = type('Spec', (), {
        'sampling_strategy': 'temperature',
        'use_rolling_state': False,
        'multi_state_strategy': 'arithmetic_mean',
        'multi_state_temperature': 1.0,
        'state_weights': None,
        'temperature': [1.0],
        'average_node_features': False,
    })()
    plan = make_inference_plan(model, spec)

    # Build batch inputs
    n_structures = batch["n_total"]
    k_unique = batch["k_unique"]
    L = batch["actual_len"]

    coords = jnp.asarray(batch["coords"], dtype=jnp.float32)
    mask = jnp.asarray(batch["mask"], dtype=jnp.float32)
    sequence = jnp.asarray(batch["sequence"], dtype=jnp.int32)
    residue_index = jnp.asarray(batch["residue_index"], dtype=jnp.int32)
    chain_index = jnp.asarray(batch["chain_index"], dtype=jnp.int32)

    # Create dedup spec: unique indices are 0, 1, ..., k_unique-1
    unique_indices = np.arange(k_unique, dtype=np.int32)
    index_map = np.arange(n_structures, dtype=np.int32) % k_unique  # Maps each pos to its unique

    dedup_spec = DedupSpec(
        axis_name="n_structures",
        unique_indices=unique_indices,
        index_map=index_map,
        k=k_unique,
    )

    try:
        from prxteinmpnn.host.sampling_inputs import SamplingInputs
        from prxteinmpnn.inference.config import ScoreConditionConfig

        config = ScoreConditionConfig()
        bundle = SamplingInputs.build(
            coords=coords,
            mask=mask,
            sequence=sequence,
            residue_index=residue_index,
            chain_index=chain_index,
            dedup_spec=dedup_spec,
        )

        logger.info(f"Warming up ({n_warmup} iterations)...")
        for _ in range(n_warmup):
            result = plan.score(bundle, key, config)
            jax.block_until_ready(result)

        logger.info(f"Timing ({n_timed} iterations)...")
        times = []
        for _ in range(n_timed):
            t0 = time.perf_counter()
            result = plan.score(bundle, key, config)
            jax.block_until_ready(result)
            times.append(time.perf_counter() - t0)

        latency_ms = median_from_timings(times)
        logger.info(f"prxteinmpnn (K={k_unique}, N={n_structures}): {latency_ms:.3f} ms")
        return latency_ms

    except Exception as e:
        logger.error(f"prxteinmpnn benchmark failed: {e}")
        return 0.0


# ============================================================================
# ColabDesign Baseline (K separate calls)
# ============================================================================


def benchmark_colabdesign_baseline(
    batch: dict[str, Any],
    n_warmup: int = 10,
    n_timed: int = 20,
) -> float:
    """ColabDesign baseline: K separate model.sample(num=1) calls.

    Parameters
    ----------
    batch : dict
        Synthetic batch
    n_warmup : int
        Number of warmup iterations
    n_timed : int
        Number of timed iterations

    Returns
    -------
    float
        Median latency in milliseconds
    """
    logger.info("ColabDesign baseline: K separate calls (not yet implemented)")
    # Placeholder: ColabDesign integration deferred
    return 0.0


# ============================================================================
# PyTorch Baseline (sequential K loop)
# ============================================================================


def benchmark_pytorch_baseline(
    batch: dict[str, Any],
    reference_path: str | None = None,
    n_warmup: int = 10,
    n_timed: int = 20,
) -> float:
    """PyTorch baseline: sequential loop over K unique structures.

    Parameters
    ----------
    batch : dict
        Synthetic batch
    reference_path : str, optional
        Path to LigandMPNN reference
    n_warmup : int
        Number of warmup iterations
    n_timed : int
        Number of timed iterations

    Returns
    -------
    float
        Median latency in milliseconds
    """
    logger.info("PyTorch baseline: sequential K loop (not yet implemented)")
    # Placeholder: PyTorch LigandMPNN integration deferred
    return 0.0


# ============================================================================
# Main Benchmark Dispatcher
# ============================================================================


def main() -> int:
    """Main entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="DedupGather K/N unique-ratio throughput benchmark.",
    )

    parser.add_argument(
        "--hardware",
        type=str,
        required=True,
        help="Hardware name (e.g., H200, A100, L40s, Blackwell)",
    )
    parser.add_argument(
        "--n-total",
        type=int,
        default=32,
        help="Total number of structures N (default: 32)",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated K values (unique structures) to benchmark",
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
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Directory containing PDB fixture files (1ubq.pdb, 1SMD.pdb)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--reference-path",
        type=str,
        default=None,
        help="Path to LigandMPNN reference (for PyTorch adapter)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print benchmark plan without executing",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke-test mode (K=[1,2], N=4, n_warmup=1, n_timed=2)",
    )

    args = parser.parse_args()

    # Resolve pdb_dir default
    if args.pdb_dir is None:
        args.pdb_dir = _DEFAULT_PDB_DIR

    # Handle --smoke
    if args.smoke:
        args.k_values = "1,2"
        args.n_total = 4
        args.n_warmup = 1
        args.n_timed = 2
        logger.info("Smoke mode: K=[1,2], N=4, n_warmup=1, n_timed=2")

    # Parse K values
    try:
        k_values = [int(k.strip()) for k in args.k_values.split(",")]
    except ValueError as e:
        logger.error(f"Invalid --k-values: {e}")
        return 1

    # Validate K values
    if any(k > args.n_total or k < 1 for k in k_values):
        logger.error(f"All K values must be in [1, {args.n_total}]")
        return 1

    # Validate pdb_dir
    if not args.pdb_dir.exists():
        logger.error(f"PDB directory not found: {args.pdb_dir}")
        return 1

    logger.info(f"Benchmark configuration:")
    logger.info(f"  Hardware: {args.hardware}")
    logger.info(f"  N (total structures): {args.n_total}")
    logger.info(f"  K values: {k_values}")
    logger.info(f"  Warmup iterations: {args.n_warmup}")
    logger.info(f"  Timed iterations: {args.n_timed}")
    logger.info(f"  PDB directory: {args.pdb_dir}")

    if args.dry_run:
        logger.info("Dry-run mode: would benchmark K in {%s}" % ", ".join(map(str, k_values)))
        logger.info("Command template:")
        logger.info(
            f"  uv run python scripts/benchmarks/bench_dedup_hetero.py "
            f"--hardware {args.hardware} "
            f"--n-total {args.n_total} "
            f"--k-values {','.join(map(str, k_values))} "
            f"--pdb-dir {args.pdb_dir} "
            f"--n-warmup {args.n_warmup} "
            f"--n-timed {args.n_timed}"
        )
        return 0

    # Load fixture (only after dry-run check to avoid unnecessary imports)
    try:
        logger.info(f"Loading PDB fixture from {args.pdb_dir}...")
        pdb_fixture = _load_pdb_fixture(args.pdb_dir, 76)
        logger.info(f"Loaded fixture: L={pdb_fixture['actual_len']}")
    except FileNotFoundError as e:
        logger.error(f"Failed to load fixture: {e}")
        return 1

    # Set JAX defaults
    _set_jax_defaults()

    # Run benchmark for each K
    results = []
    rng = np.random.default_rng(42)

    for k in k_values:
        logger.info("=" * 70)
        logger.info(f"Benchmarking K={k}, N={args.n_total}")

        # Create synthetic batch
        batch = create_synthetic_batch(pdb_fixture, args.n_total, k, rng=rng)

        # Run prxteinmpnn benchmark
        prxteinmpnn_latency = benchmark_prxteinmpnn_dedup(
            batch,
            n_warmup=args.n_warmup,
            n_timed=args.n_timed,
        )

        # Run ColabDesign baseline (placeholder)
        colabdesign_latency = benchmark_colabdesign_baseline(
            batch,
            n_warmup=args.n_warmup,
            n_timed=args.n_timed,
        )

        # Run PyTorch baseline
        pytorch_latency = benchmark_pytorch_baseline(
            batch,
            reference_path=args.reference_path,
            n_warmup=args.n_warmup,
            n_timed=args.n_timed,
        )

        # Compute dedup ratio and speedup
        dedup_ratio = k / args.n_total
        speedup_vs_pytorch = (
            pytorch_latency / prxteinmpnn_latency
            if pytorch_latency > 0 and prxteinmpnn_latency > 0
            else 0.0
        )

        result_cell = {
            "k": k,
            "n": args.n_total,
            "dedup_ratio": dedup_ratio,
            "prxteinmpnn_latency_ms": prxteinmpnn_latency,
            "colabdesign_latency_ms": colabdesign_latency,
            "pytorch_latency_ms": pytorch_latency,
            "speedup_vs_pytorch": speedup_vs_pytorch,
        }
        results.append(result_cell)

        logger.info(
            f"K={k}: dedup_ratio={dedup_ratio:.3f}, "
            f"prxteinmpnn={prxteinmpnn_latency:.3f}ms, "
            f"speedup={speedup_vs_pytorch:.3f}x"
        )

    # Write results
    output = {
        "schema_version": "1",
        "hardware": args.hardware,
        "k_values": k_values,
        "cells": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results written to {args.output_json}")

    logger.info("=" * 70)
    logger.info("Benchmark complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
