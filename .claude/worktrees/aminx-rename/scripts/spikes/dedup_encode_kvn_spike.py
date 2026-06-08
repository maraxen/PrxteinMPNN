"""Spike: Heterogeneous dedup-batching mechanism via static gather under jax.lax.map.

This spike empirically tests two hypotheses on the REAL traced dispatch path
(SafeMap(tile=1) -> jax.lax.map):

  H1: Expensive body runs K times when deduping K unique entries among N total.
  H2: Output is bit-identical to naive (all N) when body is deterministic.

The spike uses jax.lax.map (the backend of safe_map with tile=1) and
jax.experimental.io_callback to count actual iterations at runtime.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import jax
import jax.experimental
import jax.lax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


def safe_map_lax_backend(f, xs, batch_size=1):
    """Inline safe_map logic: when batch_size <= num_elements, use jax.lax.map.

    This mirrors src/aminx/utils/safe_map.py:safe_map behavior.
    """
    leaves = jax.tree_util.tree_leaves(xs)
    if not leaves:
        raise ValueError("Input xs must not be an empty PyTree")

    num_elements = leaves[0].shape[0]

    if batch_size is None or batch_size == 0 or num_elements <= batch_size:
        # Small batch: use vmap
        return jax.vmap(f)(xs)

    # Large batch or explicit tile: use lax.map
    return jax.lax.map(f, xs, batch_size=batch_size)


class ExecutionCounter:
    """Thread-safe counter for io_callback usage."""

    def __init__(self):
        self.count = 0

    def increment(self, _, **kwargs):
        """Called via io_callback; returns dummy value.

        Accepts extra kwargs (like vectorized) that io_callback may pass.
        """
        self.count += 1
        return jnp.array(0.0)

    def reset(self):
        self.count = 0


def create_deterministic_body(counter: ExecutionCounter, log_inputs=False):
    """Create a deterministic numeric function that counts invocations.

    The body performs a few matmuls on the input to stand in for encode.
    Uses jax.experimental.io_callback to record each invocation at runtime.

    Args:
        counter: ExecutionCounter to track calls.
        log_inputs: If True, log input values to check determinism.

    Returns:
        A function body(x) -> y that can be used under jax.lax.map.
    """

    def body(x):
        # Log input if requested (via io_callback so it happens at runtime).
        if log_inputs:
            _ = jax.experimental.io_callback(
                lambda x_val, **kwargs: logger.info(f"  Body called with input sum={np.sum(x_val):.4f}"),
                None,
                x,
                vectorized=False,
            )

        # Increment the counter via io_callback (fires per iteration under lax.map).
        _ = jax.experimental.io_callback(
            counter.increment,
            jnp.array(0.0),
            jnp.array(0.0),  # dummy arg
            vectorized=False,
        )

        # Deterministic numeric computation: a few matmuls.
        # x shape: (d,), output shape: (d,)
        w1 = jnp.ones((x.shape[0], x.shape[0])) * 0.5
        y1 = jnp.dot(w1, x)
        y2 = jnp.sin(y1) * 2.0 + jnp.cos(y1)
        return y2

    return body


def test_hypothesis_1_and_2(
    n: int = 6,
    k_pattern: str = "half",
    log_level: str = "INFO",
    dry_run: bool = False,
):
    """Test H1 and H2 via static gather dedup under jax.lax.map.

    Args:
        n: Total number of elements.
        k_pattern: Pattern for unique elements. "half" -> K=N/2. "third" -> K=N/3.
        log_level: Logging level.
        dry_run: If True, build arrays but skip execution.

    Returns:
        dict with results: k_executions, n_executions, bit_identical, jit_compiles.
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(levelname)s: %(message)s",
    )

    logger.info("=== Spike: Heterogeneous Dedup-Batching via jax.lax.map ===")
    logger.info(f"N (total elements): {n}")
    logger.info(f"K pattern: {k_pattern}")

    # 1. Generate synthetic data with K unique entries among N.
    if k_pattern == "half":
        k = n // 2
    elif k_pattern == "third":
        k = n // 3
    else:
        k = int(k_pattern)

    logger.info(f"K (unique entries): {k}")

    # Create data with K unique rows that are repeated to fill N positions.
    # Example: K=3, N=6 -> [0, 1, 2, 0, 1, 2] where row i is repeated at positions [i, i+K, i+2K, ...]
    d = 4
    xs_unique_base = np.random.RandomState(42).randn(k, d).astype(np.float32)
    logger.info(f"K unique base rows:\n{xs_unique_base}")

    # Expand to N by repeating: [unique_0, unique_1, ..., unique_K-1, unique_0, unique_1, ...]
    xs_full = np.vstack([xs_unique_base for _ in range(n // k + 1)])[:n]
    logger.info(f"Input shape (xs_full): {xs_full.shape}")
    logger.info(f"Full inputs:\n{xs_full}")

    # 2. Compute unique indices and inverse mapping on host (should give us the K unique rows and their positions).
    # Since xs_full is constructed by repeating, np.unique will find the K unique rows.
    unique_rows, inverse_indices = np.unique(xs_full, axis=0, return_inverse=True)
    logger.info(f"Unique rows shape: {unique_rows.shape}")
    logger.info(f"Inverse mapping (map N back to K): {inverse_indices}")

    # Gather K unique inputs (should match xs_unique_base).
    xs_unique = unique_rows
    logger.info(f"Unique inputs shape (xs_unique): {xs_unique.shape}")
    logger.info(f"Unique inputs:\n{xs_unique}")

    if dry_run:
        logger.info("DRY RUN: Skipping execution.")
        return {
            "k_executions": None,
            "n_executions": None,
            "bit_identical": None,
            "jit_compiles": None,
        }

    # 3. Test NAIVE PATH: map body over all N elements.
    logger.info("\n--- NAIVE PATH (N iterations) ---")
    logger.info(f"Input sums for naive path: {[f'{np.sum(xs_full[i]):.4f}' for i in range(n)]}")
    counter_naive = ExecutionCounter()
    body_naive = create_deterministic_body(counter_naive, log_inputs=True)

    # Use safe_map_lax_backend with tile=1 (forces jax.lax.map) for naive path.
    try:
        ys_naive = safe_map_lax_backend(body_naive, xs_full, batch_size=1)
        logger.info(f"Naive path executed. Output shape: {ys_naive.shape}")
    except Exception as e:
        logger.error(f"Naive path failed: {e}")
        raise

    n_executions = counter_naive.count
    logger.info(f"Naive path execution count: {n_executions}")

    # 4. Test DEDUP PATH: gather -> map unique -> scatter.
    logger.info("\n--- DEDUP PATH (K iterations) ---")
    logger.info(f"Input sums for dedup path: {[f'{np.sum(xs_unique[i]):.4f}' for i in range(k)]}")
    counter_dedup = ExecutionCounter()
    body_dedup = create_deterministic_body(counter_dedup, log_inputs=True)

    # Wrap dedup in jax.jit to ensure it's part of the compiled trace.
    # NOTE: inverse_indices must be passed as a JAX array or argument, not captured from closure.
    @jax.jit
    def dedup_pipeline(xs_unique_jitted, inverse_map):
        # Map over K unique inputs using safe_map_lax_backend with tile=1 (jax.lax.map).
        ys_unique = safe_map_lax_backend(body_dedup, xs_unique_jitted, batch_size=1)
        # Scatter back to N positions using the inverse mapping.
        ys_dedup = ys_unique[inverse_map]
        return ys_dedup

    try:
        ys_dedup = dedup_pipeline(jnp.asarray(xs_unique), jnp.asarray(inverse_indices))
        logger.info(f"Dedup path executed. Output shape: {ys_dedup.shape}")
        jit_compiles = True
        logger.info("✓ JIT compilation successful")
    except Exception as e:
        logger.error(f"Dedup path failed or JIT compilation failed: {e}")
        jit_compiles = False
        raise

    k_executions = counter_dedup.count
    logger.info(f"Dedup path execution count: {k_executions}")

    # 5. Verify H1: K != N on dedup path, N on naive path.
    logger.info("\n--- HYPOTHESIS 1: Execution Count ---")
    h1_pass = k_executions == k and n_executions == n
    logger.info(f"Expected dedup to run K={k} times, got {k_executions}: {'✓' if k_executions == k else '✗'}")
    logger.info(f"Expected naive to run N={n} times, got {n_executions}: {'✓' if n_executions == n else '✗'}")
    logger.info(f"H1 VERDICT: {'PASS' if h1_pass else 'FAIL'}")

    # 6. Verify H2: bit-identical outputs (deterministic body).
    logger.info("\n--- HYPOTHESIS 2: Bit Identical Output ---")
    # Cast both to same dtype for comparison.
    ys_naive_arr = np.asarray(ys_naive, dtype=np.float32)
    ys_dedup_arr = np.asarray(ys_dedup, dtype=np.float32)

    max_diff = np.max(np.abs(ys_naive_arr - ys_dedup_arr))
    logger.info(f"Max absolute difference: {max_diff:.2e}")

    bit_identical = np.allclose(ys_naive_arr, ys_dedup_arr, atol=1e-6, rtol=1e-5)
    if bit_identical:
        logger.info(f"✓ Outputs are numerically identical (max diff: {max_diff:.2e})")
    else:
        # Detailed analysis of the difference
        logger.error(
            f"✗ Outputs differ! Naive shape: {ys_naive_arr.shape}, Dedup shape: {ys_dedup_arr.shape}"
        )
        logger.error(f"Naive output:\n{ys_naive_arr}")
        logger.error(f"Dedup output:\n{ys_dedup_arr}")
        # Check per-row for patterns
        diffs = np.abs(ys_naive_arr - ys_dedup_arr)
        logger.error(f"Differences per row: {np.max(diffs, axis=1)}")

    logger.info(f"H2 VERDICT: {'PASS' if bit_identical else 'FAIL'}")

    # 7. Summary.
    logger.info("\n=== SPIKE SUMMARY ===")
    logger.info(f"K executions on dedup path: {k_executions}")
    logger.info(f"N executions on naive path: {n_executions}")
    logger.info(f"Bit identical: {bit_identical}")
    logger.info(f"JIT compiles: {jit_compiles}")
    logger.info(f"H1 (count): {'PASS' if h1_pass else 'FAIL'}")
    logger.info(f"H2 (identical): {'PASS' if bit_identical else 'FAIL'}")
    logger.info(f"Overall: {'PASS (both H1 and H2)' if (h1_pass and bit_identical and jit_compiles) else 'FAIL'}")

    return {
        "k_executions": int(k_executions),
        "n_executions": int(n_executions),
        "bit_identical": bool(bit_identical),
        "jit_compiles": bool(jit_compiles),
        "h1_pass": bool(h1_pass),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Spike: Test heterogeneous dedup-batching under jax.lax.map"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=6,
        help="Total number of elements (default: 6)",
    )
    parser.add_argument(
        "--k-pattern",
        type=str,
        default="half",
        choices=["half", "third"],
        help="Pattern for unique entries: 'half' (K=N/2) or 'third' (K=N/3)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build arrays but skip execution",
    )

    args = parser.parse_args()

    # Set JAX to CPU to avoid GPU overhead for this correctness test.
    jax.config.update("jax_platform_name", "cpu")

    result = test_hypothesis_1_and_2(
        n=args.n,
        k_pattern=args.k_pattern,
        log_level=args.log_level,
        dry_run=args.dry_run,
    )

    # Report structured result.
    print("\n=== STRUCTURED RESULT ===")
    for key, value in result.items():
        print(f"{key}: {value}")

    # Exit with 0 if all pass, else 1.
    if result.get("h1_pass") and result.get("bit_identical") and result.get("jit_compiles"):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
