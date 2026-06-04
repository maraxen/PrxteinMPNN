#!/usr/bin/env python3
"""GPU benchmark for SafeMap heterogeneous batch (mixed-length) inference.

Measures the performance of prxteinmpnn's SafeMap dispatcher on a heterogeneous batch
containing structures with variable sequence lengths, compared to PyTorch baselines:
  - Padded to max length (batch_size=4, all sequences padded to L=max)
  - Sequential per-length (4 separate model calls, one per length)

Hypothesis: SafeMap heterogeneous batch avoids padding-induced computation overhead,
delivering higher per-residue throughput than max-length-padded baseline.

Usage:
    uv run python scripts/benchmarks/bench_mixed_length.py --dry-run
    uv run python scripts/benchmarks/bench_mixed_length.py --smoke
    uv run python scripts/benchmarks/bench_mixed_length.py \\
        --hardware A100 \\
        --lengths 76 150 300 500 \\
        --n-warmup 10 \\
        --n-timed 20 \\
        --pdb-dir tests/data \\
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

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import random

# Suppress JAX warnings
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Standard 20-letter amino acid alphabet (index 0-19), unknown=20
_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_ALPHABET)}

# Canonical PDB map: nominal seq_len -> pdb filename
_PDB_MAP: dict[int, str] = {
    76: "1ubq.pdb",
    150: "1mbn.pdb",
    300: "3pgk.pdb",
    500: "1SMD.pdb",
}

# Default PDB directory relative to the prxteinmpnn package root
_DEFAULT_PDB_DIR = Path(__file__).parents[2] / "tests" / "data"


# ============================================================================
# Configuration & Utilities
# ============================================================================


def _set_jax_defaults() -> None:
    """Set JAX configuration before importing models."""
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_shard_autotuning=false")


class _BenchmarkSpec:
    """Minimal spec for make_inference_plan."""
    sampling_strategy = "temperature"
    use_rolling_state = False
    multi_state_strategy = "arithmetic_mean"
    multi_state_temperature = 1.0
    state_weights = None
    temperature = [1.0]
    average_node_features = False


def _make_benchmark_spec_with_temperatures(temperature_list: list[float] | None = None) -> _BenchmarkSpec:
    """Create a benchmark spec with custom temperature list."""
    spec = _BenchmarkSpec()
    if temperature_list is not None:
        spec.temperature = temperature_list
    return spec


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


def _load_pdb_fixture(pdb_dir: Path, seq_len: int) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int
]:
    """Load PDB fixture for a given nominal seq_len."""
    if seq_len not in _PDB_MAP:
        raise FileNotFoundError(
            f"No PDB fixture for seq_len={seq_len}. "
            f"Available: {list(_PDB_MAP.keys())}"
        )

    pdb_file = pdb_dir / _PDB_MAP[seq_len]
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB fixture not found: {pdb_file}")

    data = load_pdb_as_arrays(str(pdb_file))

    return (
        jnp.asarray(data["coords"], dtype=jnp.float32),
        jnp.asarray(data["mask"], dtype=jnp.float32),
        jnp.asarray(data["sequence"], dtype=jnp.int32),
        jnp.asarray(data["residue_index"], dtype=jnp.int32),
        jnp.asarray(data["chain_index"], dtype=jnp.int32),
        data["actual_len"],
    )


# ============================================================================
# Model & Plan Setup
# ============================================================================


_DEFAULT_CHECKPOINT_ID = "proteinmpnn_v_48_020"


def load_model(
    checkpoint_path: Path | None = None,
    checkpoint_id: str | None = None,
) -> Any:
    """Load pre-trained model via io.weights.load_model."""
    from prxteinmpnn.io.weights import load_model as _load

    effective_id = checkpoint_id or _DEFAULT_CHECKPOINT_ID
    key = random.PRNGKey(42)
    local_path = str(checkpoint_path) if checkpoint_path is not None else None

    model = _load(
        checkpoint_id=effective_id,
        local_path=local_path,
        key=key,
    )
    if local_path:
        logger.info(f"Loaded checkpoint: {local_path} (id={effective_id})")
    else:
        logger.info(f"Loaded bundled checkpoint: {effective_id}")
    return model


def create_inference_plan(model: Any, spec: _BenchmarkSpec | None = None) -> Any:
    """Create InferencePlan from model for score_conditional task."""
    from prxteinmpnn.host.plan import make_inference_plan

    if spec is None:
        spec = _BenchmarkSpec()

    # score_conditional uses ConditionalMode (make_inference_plan default)
    return make_inference_plan(model, spec)


# ============================================================================
# SafeMap Heterogeneous Batch Benchmark
# ============================================================================


def benchmark_safe_map_mixed_batch(
    model: Any,
    plan: Any,
    pdb_dir: Path,
    lengths: list[int],
    n_warmup: int = 10,
    n_timed: int = 20,
) -> dict[str, Any]:
    """Benchmark SafeMap on heterogeneous batch with mixed sequence lengths.

    Loads one structure per length, measures latency for processing all structures
    together in a single timing run.

    Returns metrics:
      - mixed_batch_latency_ms: total latency
      - per_residue_throughput: residues/second
    """
    from prxteinmpnn.inference.bundle_builder import build_inference_bundle
    from prxteinmpnn.tiling.bucketing import BucketingConfig

    _BUCKET_CFG = BucketingConfig()

    # Load one structure per length and build bundles
    bundles = []
    configs = []
    total_residues = 0

    for length in lengths:
        coords, mask, sequence, residue_index, chain_index, actual_len = _load_pdb_fixture(
            pdb_dir, length
        )
        total_residues += actual_len
        logger.info(f"Loaded {length}: actual_len={actual_len}")

        # Build bundle for score_conditional
        bundle, config = build_inference_bundle(
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            sequence=sequence,
            ligand_coords=None,
            ligand_atom_types=None,
            ligand_mask=None,
            temperature=1.0,
            mode="score_conditional",
            inference=True,
            bucket_config=_BUCKET_CFG,
        )
        bundles.append(bundle)
        configs.append(config)

    logger.info(f"Total residues in heterogeneous batch: {total_residues}")

    key = random.PRNGKey(42)

    # Create JIT function for scoring a single bundle/config pair
    @eqx.filter_jit
    def score_one(bundle: Any, config: Any) -> Any:
        """Score a single structure."""
        return plan.score(bundle, key, config)

    # Warmup
    logger.info(f"Warmup ({n_warmup} iterations)...")
    for _ in range(n_warmup):
        for bundle, config in zip(bundles, configs):
            _ = score_one(bundle, config)

    # Timed runs (measure total time for all structures)
    logger.info(f"Timed runs ({n_timed} iterations)...")
    times = []
    for _ in range(n_timed):
        start = time.perf_counter()
        for bundle, config in zip(bundles, configs):
            _ = score_one(bundle, config)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    latency_ms = np.mean(times)
    latency_s = latency_ms / 1000.0
    per_residue_throughput = total_residues / latency_s

    return {
        "mixed_batch_latency_ms": float(latency_ms),
        "per_residue_throughput": float(per_residue_throughput),
        "total_residues": int(total_residues),
        "latency_samples_ms": [float(t) for t in times],
    }


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    """Main entry point."""
    _set_jax_defaults()
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Benchmark SafeMap heterogeneous batch (mixed-length) inference."
    )

    parser.add_argument(
        "--hardware",
        type=str,
        required=True,
        help="Hardware name (e.g., A100, H100, H200, blackwell)",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[76, 150, 300, 500],
        help="Sequence lengths in heterogeneous batch (default: 76 150 300 500)",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--n-timed",
        type=int,
        default=20,
        help="Number of timed iterations (default: 20)",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Directory containing PDB fixture files (defaults to tests/data)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke-test mode (1 warmup, 3 timed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved parameters without executing",
    )
    # Unused args for compatibility
    parser.add_argument(
        "--reference-path",
        type=str,
        default=None,
        help="[UNUSED] For CLI compatibility",
    )

    args = parser.parse_args()

    # Resolve pdb_dir
    if args.pdb_dir is None:
        args.pdb_dir = _DEFAULT_PDB_DIR

    # Handle --smoke
    if args.smoke:
        args.n_warmup = 1
        args.n_timed = 3
        logger.info("Smoke mode: n_warmup=1, n_timed=3")

    logger.info(f"Hardware: {args.hardware}")
    logger.info(f"Batch lengths: {args.lengths}")
    logger.info(f"PDB directory: {args.pdb_dir}")

    if args.dry_run:
        logger.info("Dry-run complete")
        return 0

    # Load model
    logger.info("Loading prxteinmpnn model...")
    try:
        model = load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Create inference plan
    logger.info("Creating inference plan...")
    try:
        spec = _make_benchmark_spec_with_temperatures()
        plan = create_inference_plan(model, spec)
    except Exception as e:
        logger.error(f"Failed to create inference plan: {e}")
        return 1

    # Run SafeMap mixed-batch benchmark
    logger.info("=" * 70)
    logger.info("SafeMap Mixed-Length Batch Benchmark")
    logger.info("=" * 70)
    try:
        safemap_results = benchmark_safe_map_mixed_batch(
            model=model,
            plan=plan,
            pdb_dir=args.pdb_dir,
            lengths=args.lengths,
            n_warmup=args.n_warmup,
            n_timed=args.n_timed,
        )
        logger.info(f"SafeMap results: {safemap_results}")
    except Exception as e:
        logger.error(f"SafeMap benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Assemble output (minimal for now - PyTorch baselines skipped in Sprint 23)
    output = {
        "schema_version": "1",
        "hardware": args.hardware,
        "batch_lengths": args.lengths,
        "mixed_latency_ms": safemap_results["mixed_batch_latency_ms"],
        "per_residue_throughput": safemap_results["per_residue_throughput"],
        "pytorch_padded_latency_ms": None,
        "pytorch_sequential_latency_ms": None,
        "per_residue_throughput_improvement_vs_padded": None,
        "colabdesign": "not_applicable",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Write output
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results written to {args.output_json}")
    return 0


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


if __name__ == "__main__":
    sys.exit(main())
