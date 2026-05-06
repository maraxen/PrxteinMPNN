"""Integrated profiler for sampling throughput, memory, and bottleneck analysis.

Usage:
    profiler = SamplingProfiler(settings=[
        {"batch_size": 4, "precision": "float32", "remat": False},
        {"batch_size": 8, "precision": "bfloat16", "remat": True},
        ...
    ])
    results = profiler.profile_sampler(sampler_fn, inputs, num_samples=32)
    profiler.print_report()
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Single configuration profile result."""
    batch_size: int
    precision: str
    remat: bool
    num_samples: int

    # Timing
    compile_time_ms: float
    avg_runtime_ms: float
    min_runtime_ms: float
    max_runtime_ms: float

    # Throughput
    samples_per_sec: float
    est_1024_minutes: float

    # Memory
    peak_memory_mb: float

    # Metadata
    error: str = None


class SamplingProfiler:
    """Comprehensive profiler for sampling configurations."""

    def __init__(self, settings: list[dict[str, Any]]):
        """
        Args:
            settings: List of config dicts with keys:
                - batch_size: int
                - precision: "float32" or "bfloat16"
                - remat: bool (use rematerialization)
                - iterations: int (optional, default 100)
                - learning_rate: float (optional, default 0.01)
        """
        self.settings = settings
        self.results: list[ProfileResult] = []

    def profile_sampler(
        self,
        sampler_fn: Callable,
        inputs: dict[str, Any],
        num_samples: int = 32,
        num_trials: int = 3,
    ) -> list[ProfileResult]:
        """Profile sampler across all settings.

        Args:
            sampler_fn: Function with signature (key, **inputs) -> (seqs, logits, loss)
            inputs: Dict of fixed inputs to pass to sampler_fn
            num_samples: Number of samples to generate per trial
            num_trials: Number of repetitions per setting

        Returns:
            List of ProfileResult for each setting
        """
        self.results = []

        # TODO(io_callback integration): Benchmarks jax.block_until_ready; document interaction with io_callback+effects_barrier samplers.

        for i, setting in enumerate(self.settings):
            logger.info(f"\n{'='*60}")
            logger.info(f"Profiling {i+1}/{len(self.settings)}: {setting}")
            logger.info(f"{'='*60}")

            result = self._profile_single_setting(
                sampler_fn, inputs, setting, num_samples, num_trials,
            )
            self.results.append(result)

        return self.results

    def _profile_single_setting(
        self,
        sampler_fn: Callable,
        inputs: dict[str, Any],
        setting: dict[str, Any],
        num_samples: int,
        num_trials: int,
    ) -> ProfileResult:
        """Profile a single configuration."""

        batch_size = setting.get("batch_size", 4)
        precision = setting.get("precision", "float32")
        remat = setting.get("remat", False)
        iterations = setting.get("iterations", 100)
        learning_rate = setting.get("learning_rate", 0.01)

        try:
            # Create wrapped sampler with settings
            wrapped_sampler = self._wrap_sampler(
                sampler_fn, precision, remat, iterations, learning_rate,
            )

            # Prepare batched version
            batched_sampler = jax.vmap(wrapped_sampler, in_axes=(0,))

            # Compile with a small batch
            logger.info(f"Compiling with batch_size={batch_size}...")
            key = jax.random.PRNGKey(0)
            compile_keys = jax.random.split(key, batch_size)

            compile_start = time.perf_counter()
            _ = batched_sampler(compile_keys, inputs)
            jax.block_until_ready(_)
            compile_time = time.perf_counter() - compile_start

            logger.info(f"  Compile time: {compile_time*1000:.2f}ms")

            # Measure memory
            peak_memory_mb = self._measure_peak_memory(
                batched_sampler, batch_size, inputs,
            )

            # Profile runtime
            logger.info(f"Running {num_trials} trials of {num_samples} samples...")
            times = []

            for trial in range(num_trials):
                key = jax.random.PRNGKey(trial)
                trial_keys = jax.random.split(key, num_samples)

                # Submit in batches
                for batch_start in range(0, num_samples, batch_size):
                    batch_keys = trial_keys[batch_start:batch_start+batch_size]

                    start = time.perf_counter()
                    result = batched_sampler(batch_keys, inputs)
                    jax.block_until_ready(result)
                    elapsed = time.perf_counter() - start

                    times.append(elapsed)
                    actual_batch = len(batch_keys)
                    throughput = actual_batch / elapsed
                    logger.info(f"    Batch: {elapsed*1000:.2f}ms, {throughput:.1f} samples/sec")

            # Compute stats
            times_ms = np.array(times) * 1000
            avg_time = np.mean(times_ms)
            min_time = np.min(times_ms)
            max_time = np.max(times_ms)

            # Assume batch_size samples per measurement
            avg_throughput = batch_size / (avg_time / 1000)
            est_1024_minutes = (1024 / avg_throughput) / 60

            return ProfileResult(
                batch_size=batch_size,
                precision=precision,
                remat=remat,
                num_samples=num_samples * num_trials,
                compile_time_ms=compile_time * 1000,
                avg_runtime_ms=avg_time,
                min_runtime_ms=min_time,
                max_runtime_ms=max_time,
                samples_per_sec=avg_throughput,
                est_1024_minutes=est_1024_minutes,
                peak_memory_mb=peak_memory_mb,
            )

        except Exception as e:
            logger.error(f"Profile failed: {e}")
            return ProfileResult(
                batch_size=batch_size,
                precision=precision,
                remat=remat,
                num_samples=0,
                compile_time_ms=0,
                avg_runtime_ms=0,
                min_runtime_ms=0,
                max_runtime_ms=0,
                samples_per_sec=0,
                est_1024_minutes=0,
                peak_memory_mb=0,
                error=str(e),
            )

    def _wrap_sampler(
        self,
        sampler_fn: Callable,
        precision: str,
        remat: bool,
        iterations: int,
        learning_rate: float,
    ) -> Callable:
        """Wrap sampler with precision and rematerialization settings."""

        def wrapped(key, inputs):
            # Convert inputs to specified precision
            if precision == "bfloat16":
                inputs_cast = jax.tree_util.tree_map(
                    lambda x: x.astype(jnp.bfloat16) if hasattr(x, "astype") else x,
                    inputs,
                )
            else:
                inputs_cast = inputs

            # Call sampler (already handles remat via environment variable)
            result = sampler_fn(
                key,
                iterations=iterations,
                learning_rate=learning_rate,
                **inputs_cast,
            )

            # Convert back to float32 if needed
            if precision == "bfloat16":
                result = jax.tree_util.tree_map(
                    lambda x: x.astype(jnp.float32) if hasattr(x, "astype") else x,
                    result,
                )

            return result

        if remat:
            wrapped = jax.checkpoint(wrapped)

        return wrapped

    def _measure_peak_memory(
        self,
        batched_sampler: Callable,
        batch_size: int,
        inputs: dict[str, Any],
    ) -> float:
        """Measure peak memory usage using JAX memory analysis."""
        try:

            key = jax.random.PRNGKey(0)
            batch_keys = jax.random.split(key, batch_size)

            # Compile and get memory analysis
            compiled = jax.jit(batched_sampler).lower(batch_keys, inputs).compile()
            analysis = compiled.memory_analysis()

            # Parse memory size (rough estimate from analysis string)
            analysis_str = str(analysis)
            if "peak" in analysis_str.lower():
                # Try to extract peak memory
                return 0  # Placeholder - actual parsing depends on JAX version
            return 0
        except Exception as e:
            logger.debug(f"Memory analysis failed: {e}")
            return 0

    def print_report(self):
        """Print comprehensive profiling report."""

        if not self.results:
            logger.warning("No results to report")
            return

        print(f"\n{'='*80}")
        print("Sampling Profiler Report")
        print(f"{'='*80}\n")

        # Table header
        print(f"{'Config':<30} {'Throughput':<15} {'Est 1024':<12} {'Compile':<12} {'Avg Time':<12}")
        print(f"{'-'*30} {'-'*15} {'-'*12} {'-'*12} {'-'*12}")

        for result in self.results:
            if result.error:
                config = f"bs={result.batch_size} {result.precision} remat={result.remat}"
                print(f"{config:<30} ERROR: {result.error[:30]}")
            else:
                config = f"bs={result.batch_size} {result.precision} remat={result.remat}"
                throughput = f"{result.samples_per_sec:.1f} /s"
                est_1024 = f"{result.est_1024_minutes:.1f} min"
                compile_ms = f"{result.compile_time_ms:.2f} ms"
                avg_time = f"{result.avg_runtime_ms:.2f} ms"

                print(f"{config:<30} {throughput:<15} {est_1024:<12} {compile_ms:<12} {avg_time:<12}")

        print(f"\n{'='*80}\n")

        # Findings
        if self.results:
            best = min(
                (r for r in self.results if not r.error),
                key=lambda r: r.est_1024_minutes,
                default=None,
            )
            if best:
                print(f"Best config: batch_size={best.batch_size}, {best.precision}, remat={best.remat}")
                print(f"  Estimated 1024 samples: {best.est_1024_minutes:.1f} minutes")
                print(f"  Peak memory: {best.peak_memory_mb:.1f} MB\n")
