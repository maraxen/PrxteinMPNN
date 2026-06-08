#!/usr/bin/env python3
"""Performance parity benchmark for InferencePlan decode dispatch.

MR-11: Verifies that InferencePlan.decode() overhead remains small by comparing
it against a minimal baseline that calls the same underlying decode_fn.

This benchmark uses a REAL protein structure fixture to produce valid
EncoderOutput, then measures the latency of InferencePlan.decode() against
an equivalent low-overhead baseline.

Usage:
    uv run python scripts/benchmarks/bench_inference_plan_latency.py \
        --n-warmup 3 --n-timed 5 --verbose

Exit codes:
    0: PASS (overhead <= 1.10x)
    1: FAIL (overhead > 1.10x)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

if TYPE_CHECKING:
    from jaxtyping import PRNGKeyArray

# Suppress JAX warnings for cleaner output
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def _parse_first_structure(path: Path):
    """Parse a single structure and skip if parser backend is unavailable."""
    try:
        from prxteinmpnn.io.parsing import parse_input
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Cannot parse structure: {exc}") from exc
    try:
        return next(parse_input(str(path)))
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Cannot parse structure: {exc}") from exc


def create_bundle_and_plan_from_real_structure(seq_len: int | None = None) -> tuple:
    """Create InferenceBundle and InferencePlan from a real protein structure fixture.

    Uses the 1ubq.pdb fixture from tests/data/ to create a valid geometry
    with real protein coordinates. The bundle is truncated/padded to seq_len if specified.

    Parameters
    ----------
    seq_len : int | None
        Target sequence length. If None, use structure as-is. If < structure length,
        truncate to first seq_len residues. If > structure length, pad with zeros.

    Returns
    -------
    tuple
        (bundle, plan, spec, config, encoder_output) where encoder_output is
        a pre-computed EncoderOutput for decode benchmarking.
    """
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.types.configs import InferenceConfig
    from prxteinmpnn.host.plan import make_inference_plan
    from prxteinmpnn.inference.bundle_builder import build_inference_bundle

    # Load real structure from test fixture
    fixture_path = (
        Path(__file__).parent.parent.parent / "tests" / "data" / "1ubq.pdb"
    )
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"Test fixture not found at {fixture_path}. "
            "Ensure tests/data/1ubq.pdb exists."
        )

    protein = _parse_first_structure(fixture_path)

    # Extract coordinates and metadata
    coords = protein.coordinates  # (L, 4, 3)
    mask = protein.mask  # (L,)
    residue_index = protein.residue_index  # (L,)
    chain_index = protein.chain_index  # (L,)

    # Optionally truncate or pad to target seq_len
    if seq_len is not None:
        L = coords.shape[0]
        if seq_len < L:
            # Truncate
            coords = coords[:seq_len]
            mask = mask[:seq_len]
            residue_index = residue_index[:seq_len]
            chain_index = chain_index[:seq_len]
        elif seq_len > L:
            # Pad with zeros
            pad_len = seq_len - L
            coords = jnp.pad(coords, ((0, pad_len), (0, 0), (0, 0)))
            mask = jnp.pad(mask, (0, pad_len))
            residue_index = jnp.pad(
                residue_index, (0, pad_len), constant_values=-1
            )
            chain_index = jnp.pad(chain_index, (0, pad_len))

    # Build bundle from real structure (OUTSIDE timed region)
    bundle, config = build_inference_bundle(
        coords=coords,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        temperature=1.0,
        mode="score_conditional",
        inference=True,
    )

    # Create minimal model using PrxteinMPNN (no ligand features)
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

    # Create plan
    class DummySpec:
        sampling_strategy = "temperature"
        use_rolling_state = False
        multi_state_strategy = "arithmetic_mean"
        multi_state_temperature = 1.0
        state_weights = None
        temperature = [1.0]

    spec = DummySpec()
    plan = make_inference_plan(model, spec)

    # Pre-encode ONCE, outside timed region (CRITICAL)
    key, subkey = random.split(key)
    encoder_output = plan.encode(bundle, subkey, config)

    logger.info(f"Bundle created from real structure (L={bundle.geometry.coords.shape[1]})")
    logger.info(f"EncoderOutput: node_f={encoder_output.node_features.shape}, "
                f"edge_f={encoder_output.edge_features.shape}")

    return bundle, plan, spec, config, encoder_output


def benchmark_subject(
    plan,
    bundle,
    encoder_output,
    config,
    key: PRNGKeyArray,
    n_warmup: int = 10,
    n_timed: int = 20,
) -> list[float]:
    """Benchmark InferencePlan.decode() latency.

    This measures the decode wrapper by calling plan.decode(encoder_output, ...),
    which dispatches to the underlying decode_fn.

    Parameters
    ----------
    plan : InferencePlan
        Inference plan instance.
    bundle : InferenceBundle
        Input bundle (passed to decode).
    encoder_output : EncoderOutput
        Pre-computed encoder output (built outside timing region).
    config : InferenceConfig
        Inference config.
    key : PRNGKeyArray
        PRNG key for decoding.
    n_warmup : int
        Number of warmup iterations. Default 10.
    n_timed : int
        Number of timed iterations. Default 20.

    Returns
    -------
    list[float]
        Wall-clock times (seconds) for each timed iteration.
    """
    # Warmup
    for _ in range(n_warmup):
        key, subkey = random.split(key)
        result = plan.decode(encoder_output, bundle, subkey, config)
        jax.block_until_ready(result)

    # Timed runs
    times = []
    for _ in range(n_timed):
        key, subkey = random.split(key)
        start = time.perf_counter()
        result = plan.decode(encoder_output, bundle, subkey, config)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times


def benchmark_baseline(
    plan,
    encoder_output,
    config,
    bundle,
    key: PRNGKeyArray,
    n_warmup: int = 10,
    n_timed: int = 20,
) -> list[float]:
    """Benchmark minimal decode_fn call as baseline.

    This measures the underlying decode_fn directly with minimal wrapping.
    Both subject and baseline call the same decode_fn, so we measure the
    overhead (if any) of the InferencePlan.decode wrapper.

    Parameters
    ----------
    plan : InferencePlan
        Inference plan (contains the decode_fn we measure).
    encoder_output : EncoderOutput
        Pre-computed encoder output.
    config : InferenceConfig
        Inference config.
    bundle : InferenceBundle
        Input bundle.
    key : PRNGKeyArray
        PRNG key for decoding.
    n_warmup : int
        Number of warmup iterations. Default 10.
    n_timed : int
        Number of timed iterations. Default 20.

    Returns
    -------
    list[float]
        Wall-clock times (seconds) for each timed iteration.
    """
    # Extract decode_fn directly from plan
    decode_fn = plan.decode_fn

    # Warmup
    for _ in range(n_warmup):
        key, subkey = random.split(key)
        # Call decode_fn directly with the same arguments as plan.decode does internally
        result = decode_fn(subkey, encoder_output, bundle, config, plan.components.stage_set)
        jax.block_until_ready(result)

    # Timed runs
    times = []
    for _ in range(n_timed):
        key, subkey = random.split(key)
        start = time.perf_counter()
        result = decode_fn(subkey, encoder_output, bundle, config, plan.components.stage_set)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times


def main():
    """Run benchmark and report overhead ratio."""
    parser = argparse.ArgumentParser(
        description="Benchmark InferencePlan.decode() overhead.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default 3).",
    )
    parser.add_argument(
        "--n-timed",
        type=int,
        default=5,
        help="Number of timed iterations (default 5).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Target sequence length (default: use structure as-is).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed timing info.",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)

    logger.info("Loading real protein structure and creating bundle...")
    try:
        bundle, plan, spec, config, encoder_output = (
            create_bundle_and_plan_from_real_structure(seq_len=args.seq_len)
        )
    except Exception as e:
        logger.error(f"Failed to create bundle: {e}")
        sys.exit(1)

    key = random.PRNGKey(42)
    key, subkey = random.split(key)

    logger.info(
        f"Running baseline: decode_fn directly "
        f"({args.n_warmup} warmup + {args.n_timed} timed)..."
    )
    baseline_times = benchmark_baseline(
        plan,
        encoder_output,
        config,
        bundle,
        subkey,
        n_warmup=args.n_warmup,
        n_timed=args.n_timed,
    )

    if baseline_times is None:
        logger.error("Baseline benchmark failed")
        sys.exit(1)

    baseline_median = jnp.median(jnp.array(baseline_times))

    key, subkey = random.split(key)
    logger.info(
        f"Running subject: InferencePlan.decode() "
        f"({args.n_warmup} warmup + {args.n_timed} timed)..."
    )
    subject_times = benchmark_subject(
        plan,
        bundle,
        encoder_output,
        config,
        subkey,
        n_warmup=args.n_warmup,
        n_timed=args.n_timed,
    )

    if subject_times is None:
        logger.error("Subject benchmark failed")
        sys.exit(1)

    subject_median = jnp.median(jnp.array(subject_times))

    overhead_ratio = float(subject_median / baseline_median)
    threshold = 1.10

    if args.verbose:
        logger.info(
            f"Baseline (decode_fn) median: {baseline_median:.6f}s"
        )
        logger.info(
            f"Subject (InferencePlan.decode) median: {subject_median:.6f}s"
        )
        logger.info(f"Overhead ratio: {overhead_ratio:.3f}x")
        logger.info(f"Threshold: {threshold}x")

    if overhead_ratio <= threshold:
        print("PASS")
        print(f"Overhead ratio: {overhead_ratio:.3f}x")
        sys.exit(0)
    else:
        print("FAIL")
        print(
            f"Overhead ratio: {overhead_ratio:.3f}x "
            f"(exceeds {threshold}x threshold)"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
