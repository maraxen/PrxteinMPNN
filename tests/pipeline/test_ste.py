"""Tests for STEPipeline."""

import jax
import jax.numpy as jnp

from prxteinmpnn.model.mpnn import PrxteinMPNN


def test_ste_pipeline_importable():
    from prxteinmpnn.pipeline.ste import STEPipeline
    assert STEPipeline is not None


def test_ste_pipeline_smoke():
    """STEPipeline constructs and calls make_optimize_sequence_fn without error."""
    from prxteinmpnn.pipeline.ste import STEPipeline, STEInputs
    from prxteinmpnn.pipeline_fns import PipelineFns

    L = 6
    key = jax.random.PRNGKey(9)
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))

    inputs = STEInputs(
        coords=jnp.zeros((L, 4, 3)),
        mask=jnp.ones((L,)),
        residue_index=jnp.arange(L, dtype=jnp.int32),
        chain_index=jnp.zeros((L,), dtype=jnp.int32),
        iterations=3,
        learning_rate=0.1,
        temperature=1.0,
    )

    fns = PipelineFns.default()
    pipeline = STEPipeline()
    result = pipeline(m, key, inputs, fns=fns)
    assert result is not None
