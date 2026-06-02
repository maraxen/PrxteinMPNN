#!/usr/bin/env python3
"""GPU benchmark adapter for prxteinmpnn (JAX+Equinox) inference.

Measures cold-compile time, warm latency, and GPU memory usage for each
(seq_len, batch_size, precision, task) cell. Produces JSON output conforming to
the benchmark suite specification.

Usage:
    uv run python scripts/benchmarks/bench_prxteinmpnn_jax.py --dry-run
    uv run python scripts/benchmarks/bench_prxteinmpnn_jax.py --smoke
    uv run python scripts/benchmarks/bench_prxteinmpnn_jax.py \
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
    1: FAILURE (missing fixtures, model load error, or benchmark error)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
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

# Canonical PDB map: nominal seq_len -> pdb filename
_PDB_MAP: dict[int, str] = {
    76: "1ubq.pdb",
    500: "1SMD.pdb",
}

# Default PDB directory relative to the prxteinmpnn package root (this file is scripts/benchmarks/)
_DEFAULT_PDB_DIR = Path(__file__).parents[2] / "tests" / "data"

# Standard 20-letter amino acid alphabet (index 0-19), unknown=20
_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_ALPHABET)}


# ============================================================================
# Configuration & Utilities
# ============================================================================


def _set_jax_defaults():
    """Set JAX configuration before importing models."""
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
# PDB Loading
# ============================================================================


def load_pdb_as_arrays(pdb_path: str) -> dict[str, Any]:
    """Load a PDB file and return coordinate arrays using biopython.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file.

    Returns
    -------
    dict with keys:
        coords:         float32 (L, 4, 3)   - N, CA, C, O in angstroms
        mask:           float32 (L,)         - 1.0 where residue is present
        sequence:       int32   (L,)         - amino acid indices 0-19 (unk=20)
        residue_index:  int32   (L,)         - per-residue sequence number
        chain_index:    int32   (L,)         - per-chain integer index (0-based)
        actual_len:     int                  - L
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    coords_list = []
    mask_list = []
    seq_list = []
    residue_index_list = []
    chain_index_list = []

    # Sort chains alphabetically and assign 0-based integer indices
    model_obj = next(iter(structure))
    chain_ids = sorted(chain.id for chain in model_obj)
    chain_to_idx = {cid: i for i, cid in enumerate(chain_ids)}

    atom_order = {"N": 0, "CA": 1, "C": 2, "O": 3}

    for chain in model_obj:
        chain_idx = chain_to_idx[chain.id]
        for residue in chain:
            # Skip HETATM residues (waters, ligands)
            if residue.get_id()[0] != " ":
                continue

            res_coord = np.zeros((4, 3), dtype=np.float32)
            atom_found = [False, False, False, False]

            for atom_name, atom_idx in atom_order.items():
                if atom_name in residue:
                    res_coord[atom_idx] = residue[atom_name].get_vector().get_array()
                    atom_found[atom_idx] = True

            # Mark present if at least CA exists
            res_mask = 1.0 if atom_found[1] else 0.0

            # Encode residue name: 3-letter -> 1-letter -> integer index
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
        "coords": np.array(coords_list, dtype=np.float32),              # (L, 4, 3)
        "mask": np.array(mask_list, dtype=np.float32),                   # (L,)
        "sequence": np.array(seq_list, dtype=np.int32),                  # (L,)
        "residue_index": np.array(residue_index_list, dtype=np.int32),  # (L,)
        "chain_index": np.array(chain_index_list, dtype=np.int32),      # (L,)
        "actual_len": L,
    }


def _load_pdb_fixture(pdb_dir: Path, seq_len: int) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int
]:
    """Load PDB fixture for a given nominal seq_len.

    Returns
    -------
    tuple
        (coords, mask, sequence, residue_index, chain_index, actual_len)
    """
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


def load_model(checkpoint_path: Path | None = None) -> Any:
    """Load pre-trained model via io.weights.load_model.

    Uses the bundled .eqx.zst checkpoint (always architecture-compatible).
    Pass checkpoint_path to override with a local .eqx file.
    """
    from prxteinmpnn.io.weights import load_model as _load

    key = random.PRNGKey(42)
    local_path = str(checkpoint_path) if checkpoint_path is not None else None
    model = _load(
        checkpoint_id=_DEFAULT_CHECKPOINT_ID,
        local_path=local_path,
        key=key,
    )
    if local_path:
        logger.info(f"Loaded checkpoint: {local_path}")
    else:
        logger.info(f"Loaded bundled checkpoint: {_DEFAULT_CHECKPOINT_ID}")
    return model


def create_inference_plan(model: Any, task: str) -> Any:
    """Create InferencePlan from model, configured for the given task.

    Parameters
    ----------
    task : str
        "score_conditional" uses ConditionalMode (default make_inference_plan).
        "ar_sample" builds InferencePlan with AutoregressiveMode.
    """
    from prxteinmpnn.host.plan import make_inference_plan

    if task == "ar_sample":
        from prxteinmpnn.host.plan import InferenceComponents, InferencePlan
        from prxteinmpnn.inference.decode.factory import make_decode_fn
        from prxteinmpnn.inference.decode.mode import AutoregressiveMode
        from prxteinmpnn.inference.encode import make_encode_fn
        from prxteinmpnn.inference.logits import make_stage_set
        from prxteinmpnn.tiling.strategy import Vmap

        spec = _BenchmarkSpec()
        stage_set = make_stage_set(
            spec.multi_state_strategy,
            spec.multi_state_temperature,
            spec.state_weights,
        )
        encode_fn = make_encode_fn(model, use_rolling_state=spec.use_rolling_state)
        decode_fn = make_decode_fn(model, mode=AutoregressiveMode(inference_only=True), strategy=Vmap())
        components = InferenceComponents(encode_fn=encode_fn, stage_set=stage_set)
        return InferencePlan(model=model, components=components, decode_fn=decode_fn)
    else:
        # score_conditional uses ConditionalMode (make_inference_plan default)
        spec = _BenchmarkSpec()
        return make_inference_plan(model, spec)


# ============================================================================
# Timing
# ============================================================================


def measure_cold_compile_score(
    plan: Any,
    bundle: Any,
    key: "PRNGKeyArray",
    config: Any,
    fixture_name: str,
) -> tuple[float, str]:
    """Measure cold XLA compilation time for score_conditional (full encode+decode)."""
    jax.clear_caches()

    t0 = time.perf_counter()
    result = plan.score(bundle, key, config)
    jax.block_until_ready(result)
    compile_time_cold_s = time.perf_counter() - t0

    note = "JAX: XLA compilation; cold run via plan.score() (encode+decode end-to-end)"
    return compile_time_cold_s, note


def measure_cold_compile_sample(
    plan: Any,
    bundle: Any,
    key: "PRNGKeyArray",
    config: Any,
    fixture_name: str,
) -> tuple[float, str]:
    """Measure cold XLA compilation time for ar_sample."""
    jax.clear_caches()

    t0 = time.perf_counter()
    result = plan.sample(bundle, jax.random.fold_in(key, 0), config)
    jax.block_until_ready(result)
    compile_time_cold_s = time.perf_counter() - t0

    note = "JAX: cold run via plan.sample(); key derived with fold_in"
    return compile_time_cold_s, note


def measure_warm_latency_score(
    plan: Any,
    bundle: Any,
    key: "PRNGKeyArray",
    config: Any,
    n_warmup: int,
    n_timed: int,
) -> tuple[float, float, list[float]]:
    """Measure warm latency for score_conditional (full encode+decode per call)."""
    for _ in range(n_warmup):
        result = plan.score(bundle, key, config)
        jax.block_until_ready(result)

    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        result = plan.score(bundle, key, config)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return float(np.median(times_arr)), float(np.percentile(times_arr, 95)), times


def measure_warm_latency_sample(
    plan: Any,
    bundle: Any,
    key: "PRNGKeyArray",
    config: Any,
    n_warmup: int,
    n_timed: int,
) -> tuple[float, float, list[float]]:
    """Measure warm latency for ar_sample (plan.sample end-to-end).

    Uses fold_in(key, i) per iteration — distinct subkey each call,
    no key array allocation per step.
    """
    for i in range(n_warmup):
        result = plan.sample(bundle, jax.random.fold_in(key, i), config)
        jax.block_until_ready(result)

    times = []
    for i in range(n_timed):
        subkey = jax.random.fold_in(key, n_warmup + i)
        t0 = time.perf_counter()
        result = plan.sample(bundle, subkey, config)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return float(np.median(times_arr)), float(np.percentile(times_arr, 95)), times


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
    from prxteinmpnn.inference.bundle_builder import build_inference_bundle
    from prxteinmpnn.tiling.bucketing import BucketingConfig
    _BUCKET_CFG = BucketingConfig()

    try:
        if seq_len not in _PDB_MAP:
            logger.info(
                f"  Skipping L={seq_len} (no PDB fixture; have L={list(_PDB_MAP.keys())})"
            )
            return None

        coords, mask, sequence, residue_index, chain_index, actual_len = _load_pdb_fixture(
            pdb_dir, seq_len
        )

        if seq_len != actual_len:
            logger.info(
                f"  Note: nominal L={seq_len}, loaded L={actual_len} from {_PDB_MAP[seq_len]}"
            )

        # For batch_size > 1: stack geometry along the state axis.
        # sequence is NOT stacked — build_inference_bundle expects (L,) and
        # broadcasts internally across states at decode time.
        if batch_size > 1:
            coords = jnp.stack([coords] * batch_size, axis=0)
            mask = jnp.stack([mask] * batch_size, axis=0)
            residue_index = jnp.stack([residue_index] * batch_size, axis=0)
            chain_index = jnp.stack([chain_index] * batch_size, axis=0)
        sequence_stacked = sequence

        if task == "score_conditional":
            bundle, config = build_inference_bundle(
                coords=coords,
                mask=mask,
                residue_index=residue_index,
                chain_index=chain_index,
                sequence=sequence_stacked,    # native sequence from PDB
                ligand_coords=None,
                ligand_atom_types=None,
                ligand_mask=None,
                temperature=1.0,
                mode="score_conditional",
                inference=True,
                bucket_config=_BUCKET_CFG,
            )
        else:  # ar_sample
            bundle, config = build_inference_bundle(
                coords=coords,
                mask=mask,
                residue_index=residue_index,
                chain_index=chain_index,
                sequence=None,               # no fixed sequence for AR sampling
                ligand_coords=None,
                ligand_atom_types=None,
                ligand_mask=None,
                temperature=1.0,
                mode="sample",
                inference=True,
                bucket_config=_BUCKET_CFG,
            )

        plan = create_inference_plan(model, task)
        key = random.PRNGKey(42)

        logger.info(
            f"  Computing cold compile time "
            f"(seq_len={actual_len}, batch_size={batch_size}, task={task})..."
        )
        if task == "score_conditional":
            compile_time_s, compile_note = measure_cold_compile_score(
                plan, bundle, key, config, f"L{actual_len}B{batch_size}"
            )
        else:
            compile_time_s, compile_note = measure_cold_compile_sample(
                plan, bundle, key, config, f"L{actual_len}B{batch_size}"
            )

        logger.info("  Computing warm latency...")
        if task == "score_conditional":
            median_s, p95_s, times = measure_warm_latency_score(
                plan, bundle, key, config, n_warmup, n_timed
            )
        else:
            median_s, p95_s, times = measure_warm_latency_sample(
                plan, bundle, key, config, n_warmup, n_timed
            )

        latency_median_ms = median_s * 1000.0
        latency_p95_ms = p95_s * 1000.0
        latency_per_residue_us = (median_s * 1e6) / (actual_len * batch_size)
        throughput_seq_per_s = batch_size / median_s
        peak_gpu_memory_gb = _query_gpu_memory_gb()

        result = {
            "schema_version": "1",
            "model": "prxteinmpnn_jax",
            "hardware": "unknown",  # Set by caller
            "seq_len": actual_len,
            "batch_size": batch_size,
            "task": task,
            "precision": precision,
            "ligand_conditioning": False,
            "axis_strategy": "Vmap",
            "average_encoding_mode": "inputs_and_noise",
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
            f"  ✓ seq_len={actual_len}, batch_size={batch_size}, task={task}: "
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
        description="GPU benchmark for prxteinmpnn JAX adapter",
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
        default=_DEFAULT_PDB_DIR,
        help="Directory containing PDB fixture files 1ubq.pdb and 1SMD.pdb (default: tests/data)",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=None,
        help="[DEPRECATED] Use --pdb-dir instead.",
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Handle deprecated --fixture-dir
    if args.fixture_dir is not None:
        warnings.warn(
            "--fixture-dir is deprecated; use --pdb-dir instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        if args.pdb_dir == _DEFAULT_PDB_DIR:
            args.pdb_dir = args.fixture_dir

    if args.smoke:
        args.seq_lens = [76]
        args.batch_sizes = [1]
        args.n_warmup = 1
        args.n_timed = 3
        logger.info("Smoke test mode: minimal iterations")

    # Resolve checkpoint path (optional — None means use bundled .eqx.zst via io.weights)
    checkpoint_path = args.reference_path

    if args.dry_run:
        config = {
            "task": args.task,
            "seq_lens": args.seq_lens,
            "batch_sizes": args.batch_sizes,
            "precisions": args.precision,
            "hardware": args.hardware,
            "pdb_dir": str(args.pdb_dir),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
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

    _set_jax_defaults()

    logger.info(f"Loading model {'with checkpoint' if checkpoint_path else 'with random init'}...")
    try:
        model = load_model(checkpoint_path)
        logger.info("Model loaded successfully")
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
