#!/usr/bin/env python3
"""Temperature array sweep benchmark — JIT-native M-temperature comparison.

Measures latency and throughput for batched temperature sampling in prxteinmpnn
vs sequential temperature samples in ColabDesign and PyTorch baselines.

Temperature sets by M: M=1:[1.0]; M=2:[0.1,1.0]; M=4:[0.1,0.5,1.0,2.0];
M=8:[0.1,0.3,0.5,0.7,1.0,1.5,2.0,5.0]

Usage:
    uv run python scripts/benchmarks/bench_temperature_array.py --dry-run
    uv run python scripts/benchmarks/bench_temperature_array.py --smoke
    uv run python scripts/benchmarks/bench_temperature_array.py \
        --hardware H200 \
        --m-values 1 2 4 8 \
        --seq-len 76 \
        --n-warmup 10 \
        --n-timed 20 \
        --pdb-dir tests/data \
        --output-json results.json

Exit codes:
    0: SUCCESS
    1: FAILURE
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

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

# Add scripts/benchmarks to sys.path for relative imports
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

_DEFAULT_PDB_DIR = Path(__file__).parents[2] / "tests" / "data"

# Standard 20-letter amino acid alphabet (index 0-19), unknown=20
_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_ALPHABET)}

# Temperature configurations for each M value
_TEMPERATURE_CONFIGS = {
    1: [1.0],
    2: [0.1, 1.0],
    4: [0.1, 0.5, 1.0, 2.0],
    8: [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0],
}

def _set_jax_defaults():
    """Set JAX configuration before importing models."""
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_shard_autotuning=false")

def main():
    """Run temperature array sweep benchmark."""
    parser = argparse.ArgumentParser(
        description="Temperature array sweep benchmark (prxteinmpnn vs baselines)",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="unknown",
        help="Hardware identifier (default: unknown)",
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Temperature set sizes to benchmark (default: [1, 2, 4, 8])",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=76,
        help="Sequence length (default: 76)",
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
        help="Directory containing PDB fixture files (default: tests/data)",
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
        help="Run minimal benchmark: m_values=[1], n_warmup=1, n_timed=3",
    )

    args = parser.parse_args()

    # Consolidate smoke mode overrides
    if args.smoke:
        args.m_values = [1]
        args.n_warmup = 1
        args.n_timed = 3
        _smoke_seq_lens = [76]
        _smoke_batch_sizes = [1]
    else:
        _smoke_seq_lens = [76, 150, 300, 500]
        _smoke_batch_sizes = [1, 4]

    # Display configuration
    config = {
        "hardware": args.hardware,
        "m_values": args.m_values,
        "seq_len": args.seq_len,
        "n_warmup": args.n_warmup,
        "n_timed": args.n_timed,
        "pdb_dir": str(args.pdb_dir),
        "dry_run": args.dry_run,
        "smoke": args.smoke,
    }

    logger.info("Temperature Array Benchmark Configuration:")
    logger.info(json.dumps(config, indent=2))

    if args.dry_run:
        logger.info("--dry-run: exiting without running benchmarks")
        return 0

    logger.info("Starting benchmarks...")
    logger.info("About to load adapters...")

    # ========================================================================
    # Load models once (reuse across all cells)
    # ========================================================================
    logger.debug("Attempting prxteinmpnn adapter import...")
    try:
        from bench_prxteinmpnn_jax import (
            load_model,
            create_inference_plan,
            _make_benchmark_spec_with_temperatures,
            _load_pdb_fixture,
            _PDB_MAP as PRXTA_PDB_MAP,
        )
        from prxteinmpnn.inference.bundle_builder import build_inference_bundle
        from prxteinmpnn.tiling.bucketing import BucketingConfig
        prxta_available = True
        logger.info("prxteinmpnn adapter loaded successfully")
    except Exception as e:
        import traceback
        logger.warning(f"prxteinmpnn adapter unavailable: {e}")
        logger.debug(traceback.format_exc())
        prxta_available = False

    try:
        from bench_colabdesign_jax import (
            load_model as cd_load_model,
            measure_warm_latency_sample as cd_time_sample,
            measure_warm_latency_score as cd_time_score,
        )
        cd_available = True
        logger.info("ColabDesign adapter loaded successfully")
    except Exception as e:
        logger.warning(f"ColabDesign adapter unavailable: {e}")
        cd_available = False

    # Check REFERENCE_PATH for PyTorch availability
    pt_available = False
    if os.environ.get("REFERENCE_PATH"):
        try:
            from bench_ligandmpnn_pytorch import (
                load_model as pt_load_model,
                prepare_feature_dict as pt_prepare_feature_dict,
                measure_warm_latency_sample as pt_time_sample,
                measure_warm_latency_score as pt_time_score,
                _load_pdb_fixture as pt_load_pdb_fixture,
            )
            pt_available = True
            logger.info("PyTorch adapter loaded successfully")
        except Exception as e:
            logger.warning(f"PyTorch adapter unavailable: {e}")
    else:
        logger.warning("REFERENCE_PATH not set; PyTorch baselines skipped")

    if not prxta_available:
        logger.error("prxteinmpnn adapter is required; cannot proceed")
        return 1

    # Load prxteinmpnn model once
    model = load_model()
    bucket_cfg = BucketingConfig()

    # ========================================================================
    # Benchmark dimensions
    # ========================================================================
    seq_lens = _smoke_seq_lens
    batch_sizes = _smoke_batch_sizes
    tasks = ["ar_sample", "score_conditional"]

    cells = []

    # ========================================================================
    # Main benchmark loop
    # ========================================================================
    for m in args.m_values:
        temperatures = _TEMPERATURE_CONFIGS[m]
        logger.info(f"\nBenchmarking M={m} temperatures: {temperatures}")

        for seq_len in seq_lens:
            for batch_size in batch_sizes:
                for task in tasks:
                    cell_start = time.time()
                    cell_key = random.PRNGKey(42)

                    logger.info(
                        f"  M={m}, L={seq_len}, B={batch_size}, task={task} ... ",
                    )

                    # ====================================================================
                    # 1. prxteinmpnn: single vmapped call with all M temperatures
                    # ====================================================================
                    prxta_median_ms = None
                    prxta_p95_ms = None
                    prxta_per_temp_ms = None

                    if prxta_available:
                        try:
                            # Load fixture once
                            coords, mask, sequence, residue_index, chain_index, actual_len = \
                                _load_pdb_fixture(args.pdb_dir, seq_len)

                            # Stack for batch_size > 1
                            if batch_size > 1:
                                coords = jnp.stack([coords] * batch_size, axis=0)
                                mask = jnp.stack([mask] * batch_size, axis=0)
                                residue_index = jnp.stack([residue_index] * batch_size, axis=0)
                                chain_index = jnp.stack([chain_index] * batch_size, axis=0)

                            # Create spec with all M temperatures
                            spec = _make_benchmark_spec_with_temperatures(temperatures)

                            # Create plan
                            plan = create_inference_plan(model, task, spec)

                            # Build bundle
                            if task == "score_conditional":
                                bundle, config = build_inference_bundle(
                                    coords=coords,
                                    mask=mask,
                                    residue_index=residue_index,
                                    chain_index=chain_index,
                                    sequence=sequence,
                                    temperature=1.0,
                                    mode="score_conditional",
                                    inference=True,
                                    bucket_config=bucket_cfg,
                                )
                                fn = plan.score
                            else:  # ar_sample
                                bundle, config = build_inference_bundle(
                                    coords=coords,
                                    mask=mask,
                                    residue_index=residue_index,
                                    chain_index=chain_index,
                                    sequence=None,
                                    temperature=1.0,
                                    mode="sample",
                                    inference=True,
                                    bucket_config=bucket_cfg,
                                )
                                fn = plan.sample

                            # Warmup
                            for _ in range(args.n_warmup):
                                result = fn(bundle, cell_key, config)
                                jax.block_until_ready(result)

                            # Timed
                            times_ms = []
                            for i in range(args.n_timed):
                                t0 = time.perf_counter()
                                result = fn(bundle, jax.random.fold_in(cell_key, i), config)
                                jax.block_until_ready(result)
                                t1 = time.perf_counter()
                                times_ms.append((t1 - t0) * 1000.0)

                            times_ms.sort()
                            prxta_median_ms = times_ms[len(times_ms) // 2]
                            prxta_p95_ms = times_ms[int(0.95 * len(times_ms))]
                            prxta_per_temp_ms = prxta_median_ms / m

                        except Exception as e:
                            logger.warning(
                                f"    prxteinmpnn failed: {e.__class__.__name__}: {e}"
                            )

                    # ====================================================================
                    # 2. ColabDesign: sum of M sequential calls
                    # ====================================================================
                    cd_total_ms = None
                    cd_per_temp_ms = None

                    if cd_available:
                        try:
                            # Build ColabDesign model once per seq_len
                            cd_model = cd_load_model()

                            # For ar_sample task: loop M times and sum medians
                            if task == "ar_sample":
                                cd_total = 0.0
                                for _ in range(m):
                                    median_s, _, _ = cd_time_sample(
                                        cd_model,
                                        batch_size=batch_size,
                                        n_warmup=1,
                                        n_timed=args.n_timed,
                                    )
                                    cd_total += median_s * 1000.0  # Convert to ms
                                cd_total_ms = cd_total
                            else:
                                # For score_conditional: need native sequence
                                # Load PDB fixture to get native sequence
                                from bench_prxteinmpnn_jax import _load_pdb_fixture as prxta_load_pdb
                                coords, mask, sequence, residue_index, chain_index, actual_len = \
                                    prxta_load_pdb(args.pdb_dir, seq_len)
                                # Convert to amino acid string (sequence is int32 indices 0-19)
                                aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
                                native_seq = "".join(aa_alphabet[int(s)] for s in sequence)

                                cd_total = 0.0
                                for _ in range(m):
                                    median_s, _, _, _ = cd_time_score(
                                        cd_model,
                                        native_seq=native_seq,
                                        batch_size=batch_size,
                                        n_warmup=1,
                                        n_timed=args.n_timed,
                                    )
                                    cd_total += median_s * 1000.0  # Convert to ms
                                cd_total_ms = cd_total

                            cd_per_temp_ms = cd_total_ms / m

                        except Exception as e:
                            logger.warning(
                                f"    ColabDesign failed: {e.__class__.__name__}: {e}"
                            )

                    # ====================================================================
                    # 3. PyTorch: sum of M sequential calls (if available)
                    # ====================================================================
                    pt_total_ms = None
                    pt_per_temp_ms = None

                    if pt_available:
                        try:
                            import torch
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                            # Load PyTorch model once per seq_len
                            pt_model = pt_load_model(device=device)

                            # Load fixture
                            coords, mask, sequence, residue_index, chain_index, actual_len = \
                                pt_load_pdb_fixture(args.pdb_dir, seq_len)

                            # Build feature dict
                            coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
                            mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
                            seq_t = torch.tensor(sequence, dtype=torch.long, device=device)
                            residue_index_t = torch.tensor(residue_index, dtype=torch.long, device=device)
                            chain_index_t = torch.tensor(chain_index, dtype=torch.long, device=device)

                            protein_dict = {
                                "X": coords_t,
                                "mask": mask_t,
                                "R_idx": residue_index_t,
                                "chain_labels": chain_index_t,
                                "S": seq_t,
                                "chain_mask": torch.ones(actual_len, dtype=torch.float32, device=device),
                            }

                            reference_path = Path(os.environ.get("REFERENCE_PATH", str(Path.home() / "repos" / "LigandMPNN")))
                            feature_dict = pt_prepare_feature_dict(
                                protein_dict,
                                reference_path=reference_path,
                                model_type="ligand_mpnn",
                                atom_context_num=20,
                                device=device,
                            )

                            # Native sequence for scoring
                            aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
                            native_seq = "".join(aa_alphabet[int(s)] for s in sequence)
                            native_sequence = torch.tensor(
                                [_AA_TO_IDX.get(aa, 20) for aa in native_seq],
                                dtype=torch.long,
                                device=device
                            )

                            # For ar_sample task
                            if task == "ar_sample":
                                pt_total = 0.0
                                for _ in range(m):
                                    median_s, _, _ = pt_time_sample(
                                        pt_model,
                                        feature_dict=feature_dict,
                                        batch_size=batch_size,
                                        device=device,
                                        n_warmup=1,
                                        n_timed=args.n_timed,
                                    )
                                    pt_total += median_s * 1000.0  # Convert to ms
                                pt_total_ms = pt_total
                            else:
                                # For score_conditional
                                pt_total = 0.0
                                for _ in range(m):
                                    median_s, _, _ = pt_time_score(
                                        pt_model,
                                        feature_dict=feature_dict,
                                        batch_size=batch_size,
                                        device=device,
                                        native_sequence=native_sequence,
                                        n_warmup=1,
                                        n_timed=args.n_timed,
                                    )
                                    pt_total += median_s * 1000.0  # Convert to ms
                                pt_total_ms = pt_total

                            pt_per_temp_ms = pt_total_ms / m

                        except Exception as e:
                            logger.warning(
                                f"    PyTorch failed: {e.__class__.__name__}: {e}"
                            )

                    # ====================================================================
                    # 4. Compute speedups and record cell
                    # ====================================================================
                    speedup_vs_cd = None
                    speedup_vs_pt = None

                    if prxta_median_ms is not None and cd_total_ms is not None:
                        speedup_vs_cd = cd_total_ms / prxta_median_ms

                    if prxta_median_ms is not None and pt_total_ms is not None:
                        speedup_vs_pt = pt_total_ms / prxta_median_ms

                    cell = {
                        "m_value": m,
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "task": task,
                        "temperatures": temperatures,
                        "prxteinmpnn_latency_ms": prxta_median_ms,
                        "prxteinmpnn_latency_p95_ms": prxta_p95_ms,
                        "prxteinmpnn_latency_per_temp_ms": prxta_per_temp_ms,
                        "colabdesign_latency_ms": cd_total_ms,
                        "pytorch_latency_ms": pt_total_ms,
                        "speedup_vs_colabdesign": speedup_vs_cd,
                        "speedup_vs_pytorch": speedup_vs_pt,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    cells.append(cell)

                    cell_elapsed = time.time() - cell_start
                    logger.info(f"    {cell_elapsed:.1f}s")

    logger.info(f"\nTemperature array benchmark complete — {len(cells)} cells")

    output = {
        "schema_version": "1",
        "hardware": args.hardware,
        "m_values": args.m_values,
        "cells": cells,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results written to {args.output_json}")
    else:
        print(json.dumps(output, indent=2))

    return 0

if __name__ == "__main__":
    _set_jax_defaults()
    sys.exit(main())
