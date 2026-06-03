"""DEFECT 2-B: InferencePlan reuse compiles only on first call (library invariant).

This test guards the benchmark's plan-once change: if model arrays are static
(DEFECT 2), each plan construction triggers a different XLA compilation cache key,
defeating the entire point of plan reuse.

With static model fields: every call to make_inference_plan creates an object
whose static partition includes the model weights as hashed bytes, causing every
plan instance to produce a unique JIT trace.

With dynamic model fields (the fix): the plan is compiled once for a given
input shape, and subsequent calls with the same shape hit the cache.

Mechanism tested: jax.log_compiles captures XLA compilation events.
A second call with identical-shape inputs MUST NOT trigger a new compilation.

Mark: @pytest.mark.slow (compilation + actual model forward pass can take >2s)
"""

from __future__ import annotations

import logging
import pytest
import jax
import jax.numpy as jnp

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.inference.bundle_builder import build_inference_bundle
from prxteinmpnn.host.plan import make_inference_plan
from prxteinmpnn.types.configs import InferenceConfig


def _make_small_model() -> PrxteinMPNN:
    """Minimal model: fast compilation, not from checkpoint."""
    return PrxteinMPNN(
        node_features=16,
        edge_features=16,
        hidden_features=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=8,
        key=jax.random.PRNGKey(42),
    )


class _DummySpec:
    """Minimal spec duck-type for make_inference_plan."""
    sampling_strategy = "temperature"
    use_rolling_state = False
    multi_state_strategy = "arithmetic_mean"
    multi_state_temperature = 1.0
    state_weights = None
    temperature = [1.0]
    average_node_features = False


def _make_bundle_and_config(model, L=10, S=1):
    """Build a minimal InferenceBundle for score_conditional mode."""
    coords = jnp.zeros((S, L, 4, 3))
    mask = jnp.ones((S, L))
    residue_index = jnp.broadcast_to(jnp.arange(L)[None, :], (S, L))
    chain_index = jnp.zeros((S, L), dtype=jnp.int32)
    sequence = jnp.zeros((L, 21))

    bundle, config = build_inference_bundle(
        coords=coords,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        sequence=sequence,
        mode="score_conditional",
    )
    return bundle, config


@pytest.mark.slow
def test_inference_plan_score_second_call_does_not_recompile():
    """plan.score() with identical-shape inputs must NOT trigger XLA recompilation.

    CURRENTLY: With static model fields (DEFECT 2), plan construction may produce
    different XLA compilation keys due to model weight hashes in the static
    partition, causing silent recompilation.

    After fix (dynamic model fields): the plan is compiled once; second call
    with identical inputs hits the XLA cache (zero new compilations).

    Detection method: capture logging output from jax.log_compiles().
    The logger 'jax._src.interpreters.mlir' or 'jax.compilation_cache'
    emits messages when XLA compilation occurs. Zero such messages on 2nd call
    confirms cache hit.
    """
    model = _make_small_model()
    plan = make_inference_plan(model, _DummySpec())
    bundle, config = _make_bundle_and_config(model)
    key = jax.random.PRNGKey(0)

    # Collect JAX compilation log messages during both calls
    # jax.log_compiles() context manager emits tracing messages
    compile_counts = []

    class _CompileCounter(logging.Handler):
        """Count XLA compilation log lines."""
        def __init__(self):
            super().__init__()
            self.count = 0

        def emit(self, record):
            msg = record.getMessage().lower()
            # JAX emits "compiling" or "Traced" messages on XLA compilation
            if "compiling" in msg or "compilation" in msg or "traced" in msg:
                self.count += 1

    jax_logger = logging.getLogger("jax")
    counter = _CompileCounter()
    jax_logger.addHandler(counter)
    old_level = jax_logger.level
    jax_logger.setLevel(logging.DEBUG)

    try:
        # First call — should compile
        with jax.log_compiles():
            result1 = plan.score(bundle, key, config)

        first_call_compile_count = counter.count
        counter.count = 0  # reset

        # Second call with IDENTICAL inputs — must NOT recompile
        with jax.log_compiles():
            result2 = plan.score(bundle, key, config)

        second_call_compile_count = counter.count
    finally:
        jax_logger.removeHandler(counter)
        jax_logger.setLevel(old_level)

    # Results must be numerically identical (deterministic)
    assert jnp.allclose(result1.logits, result2.logits, atol=1e-5), (
        "score() results differ between identical calls — non-deterministic?"
    )

    # The key invariant: second call has fewer or equal compilations than first
    # Ideally 0. We test <= first_call_compile_count as a weak invariant.
    # The strong invariant is second_call_compile_count == 0 after fix.
    assert second_call_compile_count == 0, (
        f"Expected 0 XLA compilations on 2nd call with identical inputs, "
        f"but got {second_call_compile_count}. "
        f"(First call: {first_call_compile_count} compilation events.) "
        f"This suggests model weights are in static partition, causing cache miss."
    )


@pytest.mark.slow
def test_inference_plan_same_plan_two_calls_consistent_results():
    """plan.score() called twice on identical inputs returns identical logits.

    This is a correctness invariant independent of compilation. If model arrays
    were accidentally getting hashed into static, the computation should still
    be correct (same hash -> same result). This test guards against silent
    numerical divergence from any static-field side effects.
    """
    model = _make_small_model()
    plan = make_inference_plan(model, _DummySpec())
    bundle, config = _make_bundle_and_config(model)
    key = jax.random.PRNGKey(0)

    result1 = plan.score(bundle, key, config)
    result2 = plan.score(bundle, key, config)

    assert jnp.allclose(result1.logits, result2.logits, atol=1e-6), (
        f"score() is non-deterministic: max diff = "
        f"{jnp.max(jnp.abs(result1.logits - result2.logits))}"
    )


@pytest.mark.slow
def test_different_model_weights_produce_different_logits_through_plan():
    """Two plans with different model weights must produce different logits.

    This test documents the KEY correctness invariant the static-field bug breaks:
    when model is marked static=True inside _VmapEncode / ConditionalDecode,
    JAX treats the model weights as a hashable Python object for JIT cache lookup.

    CURRENTLY FAILS with:
        ValueError: static arguments should be comparable using __eq__.
        ...arrays cannot be passed as metadata fields!

    This is the core breakage: JAX cannot compare two plans (for cache lookup)
    when model arrays are in the static partition, because JAX prohibits arrays
    as static metadata fields. This means any second plan construction of a
    different model crashes JIT comparison — worse than just a cache miss.

    After fix (model is dynamic): plan_a and plan_b trace independently on first
    call, then both cache. score() for each returns the logits for their
    respective model weights, confirming dynamic-partition correctness.
    """
    import equinox as eqx

    model_a = _make_small_model()
    model_b = _make_small_model()

    # Create model_b with zeroed-out weights (different from model_a)
    # This should make logits very different
    model_b_zeroed = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x,
        model_b,
    )

    plan_a = make_inference_plan(model_a, _DummySpec())
    plan_b = make_inference_plan(model_b_zeroed, _DummySpec())

    bundle, config = _make_bundle_and_config(model_a)
    key = jax.random.PRNGKey(0)

    result_a = plan_a.score(bundle, key, config)
    result_b = plan_b.score(bundle, key, config)

    # With different weights, logits must differ
    max_diff = float(jnp.max(jnp.abs(result_a.logits - result_b.logits)))
    assert max_diff > 0.01, (
        f"Plans with different model weights produced identical logits "
        f"(max diff={max_diff}). "
        f"This may indicate model weights are in the static partition and "
        f"the JIT is using stale cached code."
    )
