#!/usr/bin/env python3
"""Micro latency probe for score_conditional warm latency.

Purpose
-------
Observe warm ``plan.score()`` latency for a single sequence length across batch
sizes, reusing the benchmark's own measurement function so the methodology is
identical to the full suite. Intended for fast local (CPU) before/after
comparisons of the static-vs-dynamic ``model`` field change — the absolute
numbers are not comparable to GPU, but the same-machine before/after ratio is.

Run before (static model) vs after (dynamic model) by checking out the
respective source revisions of ``src/prxteinmpnn/inference/`` and re-running.

Example
-------
    PYTHONPATH=src python scripts/benchmarks/micro_latency_score.py \
        --pdb-dir tests/data --seq-len 76 --batch-sizes 1 16 \
        --n-warmup 2 --n-timed 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import jax.numpy as jnp
from jax import random

# Import the benchmark's own machinery so timing methodology matches exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_prxteinmpnn_jax import (  # noqa: E402
    _load_pdb_fixture,
    create_inference_plan,
    load_model,
    measure_warm_latency_score,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _build_bundle(pdb_dir: Path, seq_len: int, batch_size: int):
    from prxteinmpnn.inference.bundle_builder import build_inference_bundle
    from prxteinmpnn.tiling.bucketing import BucketingConfig

    coords, mask, sequence, residue_index, chain_index, actual_len = _load_pdb_fixture(
        pdb_dir, seq_len
    )
    if batch_size > 1:
        coords = jnp.stack([coords] * batch_size, axis=0)
        mask = jnp.stack([mask] * batch_size, axis=0)
        residue_index = jnp.stack([residue_index] * batch_size, axis=0)
        chain_index = jnp.stack([chain_index] * batch_size, axis=0)
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
        bucket_config=BucketingConfig(),
    )
    return bundle, config, actual_len


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-dir", type=Path, default=Path("tests/data"))
    parser.add_argument("--seq-len", type=int, default=76)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-timed", type=int, default=5)
    parser.add_argument("--checkpoint-id", type=str, default=None)
    args = parser.parse_args()

    logger.info("Loading model...")
    model = load_model(checkpoint_id=args.checkpoint_id)
    key = random.PRNGKey(42)

    logger.info("=== warm latency (median ms) ===")
    for batch_size in args.batch_sizes:
        try:
            bundle, config, actual_len = _build_bundle(args.pdb_dir, args.seq_len, batch_size)
            plan = create_inference_plan(model, "score_conditional")
            median_s, p95_s, _ = measure_warm_latency_score(
                plan, bundle, key, config, args.n_warmup, args.n_timed
            )
            logger.info(
                f"  L={actual_len} batch={batch_size}: "
                f"median={median_s * 1000:.2f}ms  p95={p95_s * 1000:.2f}ms"
            )
        except Exception as e:  # noqa: BLE001 - probe reports crash, does not raise
            logger.info(f"  L={args.seq_len} batch={batch_size}: CRASHED -> {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
