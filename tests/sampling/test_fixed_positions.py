"""Tests: fixed-position constraints enforce correct amino acid in output sequences.

Invariant: when fixed_mask[i] = 1.0 and fixed_tokens[i] = k, the output
sequence at position i must equal k regardless of sampling strategy.

Covers both temperature (autoregressive) and straight-through (STE) sampling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.sampling.sample import make_sample_sequences

N = 16  # small enough for fast CPU tests
_KEY = jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def small_model():
    return PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        k_neighbors=8,
        key=_KEY,
    )


@pytest.fixture(scope="module")
def synthetic_inputs():
    coords = jax.random.normal(_KEY, (N, 4, 3)) * 5.0
    mask = jnp.ones(N, dtype=jnp.float32)
    res_idx = jnp.arange(N, dtype=jnp.int32)
    chain_idx = jnp.zeros(N, dtype=jnp.int32)
    return coords, mask, res_idx, chain_idx


# ── temperature sampling ──────────────────────────────────────────────────────

def test_temperature_all_positions_fixed(small_model, synthetic_inputs):
    """All positions fixed: output must exactly match fixed_tokens."""
    coords, mask, res_idx, chain_idx = synthetic_inputs
    sample_fn = make_sample_sequences(small_model, sampling_strategy="temperature")

    fixed_mask = jnp.ones(N, dtype=jnp.float32)
    fixed_tokens = jnp.arange(N, dtype=jnp.int8)  # tokens 0-15

    seq, _, _ = sample_fn(
        _KEY, coords, mask, res_idx, chain_idx,
        temperature=jnp.array(0.01),  # very sharp — free positions also nearly deterministic
        fixed_mask=fixed_mask,
        fixed_tokens=fixed_tokens,
    )

    np.testing.assert_array_equal(
        np.array(seq), np.array(fixed_tokens),
        err_msg="Temperature sampling: fixed positions must match fixed_tokens exactly",
    )


def test_temperature_partial_fixed(small_model, synthetic_inputs):
    """Partial fix: only even-indexed positions must equal fixed_tokens; odd positions free."""
    coords, mask, res_idx, chain_idx = synthetic_inputs
    sample_fn = make_sample_sequences(small_model, sampling_strategy="temperature")

    fixed_mask = jnp.array([1, 0] * (N // 2), dtype=jnp.float32)
    fixed_tokens = jnp.zeros(N, dtype=jnp.int8)  # force token 0 at fixed positions

    seq, _, _ = sample_fn(
        _KEY, coords, mask, res_idx, chain_idx,
        temperature=jnp.array(0.01),
        fixed_mask=fixed_mask,
        fixed_tokens=fixed_tokens,
    )

    fixed_positions = np.where(np.array(fixed_mask) == 1.0)[0]
    np.testing.assert_array_equal(
        np.array(seq)[fixed_positions],
        np.zeros(len(fixed_positions), dtype=np.int8),
        err_msg=f"Temperature sampling: positions {fixed_positions.tolist()} must all be token 0",
    )


# ── STE (straight-through) sampling ──────────────────────────────────────────

def test_ste_all_positions_fixed(small_model, synthetic_inputs):
    """STE: all positions fixed must produce exactly fixed_tokens in output."""
    coords, mask, res_idx, chain_idx = synthetic_inputs
    sample_fn = make_sample_sequences(small_model, sampling_strategy="straight_through")

    fixed_mask = jnp.ones(N, dtype=jnp.float32)
    fixed_tokens = jnp.arange(N, dtype=jnp.int8)  # tokens 0-15

    seq, _, _ = sample_fn(
        _KEY, coords, mask, res_idx, chain_idx,
        iterations=jnp.array(10),
        learning_rate=jnp.array(0.01),
        temperature=jnp.array(0.1),
        fixed_mask=fixed_mask,
        fixed_tokens=fixed_tokens,
    )

    np.testing.assert_array_equal(
        np.array(seq), np.array(fixed_tokens),
        err_msg="STE sampling: fixed positions must match fixed_tokens exactly",
    )


def test_ste_partial_fixed(small_model, synthetic_inputs):
    """STE: only even-indexed positions must equal token 0; odd positions unconstrained."""
    coords, mask, res_idx, chain_idx = synthetic_inputs
    sample_fn = make_sample_sequences(small_model, sampling_strategy="straight_through")

    fixed_mask = jnp.array([1, 0] * (N // 2), dtype=jnp.float32)
    fixed_tokens = jnp.zeros(N, dtype=jnp.int8)  # force token 0 at even positions

    seq, _, _ = sample_fn(
        _KEY, coords, mask, res_idx, chain_idx,
        iterations=jnp.array(10),
        learning_rate=jnp.array(0.01),
        temperature=jnp.array(0.1),
        fixed_mask=fixed_mask,
        fixed_tokens=fixed_tokens,
    )

    fixed_positions = np.where(np.array(fixed_mask) == 1.0)[0]
    np.testing.assert_array_equal(
        np.array(seq)[fixed_positions],
        np.zeros(len(fixed_positions), dtype=np.int8),
        err_msg=f"STE sampling: positions {fixed_positions.tolist()} must all be token 0",
    )


# ── no-fixed-mask baseline (both strategies accept None) ─────────────────────

@pytest.mark.parametrize("strategy", ["temperature", "straight_through"])
def test_no_fixed_mask_runs_without_error(small_model, synthetic_inputs, strategy):
    """Sanity: passing no fixed args still works for both strategies."""
    coords, mask, res_idx, chain_idx = synthetic_inputs
    sample_fn = make_sample_sequences(small_model, sampling_strategy=strategy)

    kwargs = dict(temperature=jnp.array(0.5))
    if strategy == "straight_through":
        kwargs.update(iterations=jnp.array(5), learning_rate=jnp.array(0.01))

    seq, logits, _ = sample_fn(_KEY, coords, mask, res_idx, chain_idx, **kwargs)

    assert seq.shape == (N,), f"Expected seq shape ({N},), got {seq.shape}"
    assert logits.shape == (N, 21), f"Expected logits shape ({N}, 21), got {logits.shape}"
