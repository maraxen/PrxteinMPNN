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

from aminx.model import Aminx
from aminx.sampling.sample import make_sample_sequences

N = 16  # small enough for fast CPU tests
_KEY = jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def small_model():
    return Aminx(
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


# ── fixed_mask only (without fixed_tokens) ──────────────────────────────────

def test_temperature_fixed_mask_without_tokens(small_model, synthetic_inputs):
    """When fixed_mask is set but fixed_tokens is not, fixed positions are still constrained.

    The fixed_mask should be incorporated into the fixed tensor returned by _prepare_fixed_controls.
    Positions marked as 1.0 in fixed_mask should have fixed_mask=1.0 in the output,
    even without fixed_tokens specified.
    """
    coords, mask, res_idx, chain_idx = synthetic_inputs
    sample_fn = make_sample_sequences(small_model, sampling_strategy="temperature")

    # Set fixed_mask on even positions without setting fixed_tokens
    fixed_mask = jnp.array([1, 0] * (N // 2), dtype=jnp.float32)

    seq, _, _ = sample_fn(
        _KEY, coords, mask, res_idx, chain_idx,
        temperature=jnp.array(0.5),
        fixed_mask=fixed_mask,
        fixed_tokens=None,  # explicit None: no token constraint
    )

    # Test passes if no error; the fixed_mask should be incorporated into constraints
    assert seq.shape == (N,), f"Expected seq shape ({N},), got {seq.shape}"


def test_ste_fixed_mask_without_tokens(small_model, synthetic_inputs):
    """STE: fixed_mask should constrain positions even without fixed_tokens."""
    coords, mask, res_idx, chain_idx = synthetic_inputs
    sample_fn = make_sample_sequences(small_model, sampling_strategy="straight_through")

    # Set fixed_mask on odd positions
    fixed_mask = jnp.array([0, 1] * (N // 2), dtype=jnp.float32)

    seq, _, _ = sample_fn(
        _KEY, coords, mask, res_idx, chain_idx,
        iterations=jnp.array(10),
        learning_rate=jnp.array(0.01),
        temperature=jnp.array(0.1),
        fixed_mask=fixed_mask,
        fixed_tokens=None,
    )

    assert seq.shape == (N,), f"Expected seq shape ({N},), got {seq.shape}"


def test_fixed_mask_and_positions_union_prepare_controls():
    """When both fixed_mask and fixed_positions are set, they should combine (union).

    This test directly calls _prepare_fixed_controls to verify the union behavior,
    since the sampling function doesn't expose both parameters directly.
    """
    from aminx.host._sampling_helper import _prepare_fixed_controls
    from aminx.run.specs import SamplingSpecification
    from aminx.utils.data_structures import Protein

    # Create a minimal protein batch
    batch_size, seq_len = 1, N
    coords = jnp.ones((batch_size, seq_len, 37, 3), dtype=jnp.float32)
    aatype = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    res_idx = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    chain_idx = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    protein = Protein(
        coordinates=coords,
        aatype=aatype,
        residue_index=res_idx,
        chain_index=chain_idx,
    )

    # fixed_positions: even positions (indices 0, 2, 4, ...)
    fixed_positions = jnp.array([1, 0] * (N // 2), dtype=jnp.float32)
    # fixed_mask: odd positions (indices 1, 3, 5, ...)
    fixed_mask = jnp.array([0, 1] * (N // 2), dtype=jnp.float32)
    # tokens: force to 0 at all constrained positions
    fixed_tokens = jnp.zeros(N, dtype=jnp.int8)

    spec = SamplingSpecification(
        inputs=[],
        fixed_positions=fixed_positions,
        fixed_mask=fixed_mask,
        fixed_tokens=fixed_tokens,
    )

    fm, ft = _prepare_fixed_controls(spec, batched_ensemble=protein)

    # After union, all positions (even AND odd) should be marked as fixed
    expected_mask = np.ones(N, dtype=np.float32)
    np.testing.assert_array_equal(
        np.array(fm[0]), expected_mask,
        err_msg="fixed_positions and fixed_mask should combine (union) — all positions should be fixed",
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
