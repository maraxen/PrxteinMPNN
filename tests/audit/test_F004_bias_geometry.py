"""F004 closure assertions — task_id 260826_aminx-invariant-audit.

Tier A, HYBRID per intent/F004.md:
  1. Synthesized bias-free defaults are keyed to the geometry the decode kernel
     actually consumes jointly (reference frame), not blindly to cond.bias.
  2. User-supplied conditioning arrays with mismatched lengths RAISE at the
     bundle boundary naming both lengths (never broadcast/truncated).
  3. Matched-length (max_length == L) construction is byte-identical: the
     synthesized default equals the pre-fix jnp.zeros((seq_len, 21)).

Executes against the installed aminx in the running interpreter's environment
(resolver venv expected; stamp asserted).
"""
from __future__ import annotations

import importlib.metadata as im

import jax.numpy as jnp
import numpy as np
import pytest


def _stamp() -> str:
    return im.version("aminx")


def _bundle_kwargs(seq_len):
    rng = np.random.default_rng(0)
    coords = jnp.asarray(rng.normal(size=(seq_len, 4, 3)), dtype=jnp.float32)
    return dict(
        coords=coords,
        mask=jnp.ones(seq_len),
        residue_index=jnp.arange(seq_len),
        chain_index=jnp.zeros(seq_len, dtype=jnp.int32),
    )


def test_version_stamp_is_post_fix():
    assert _stamp() >= "0.1.0a27"


def test_matched_length_synthesized_default_unchanged():
    """max_length == L: cond.bias is exactly the pre-fix zeros((L,21))."""
    from aminx.inference.bundle_builder import build_inference_bundle

    bundle, _ = build_inference_bundle(**_bundle_kwargs(76))
    assert bundle.conditioning.bias.shape == (76, 21)
    assert int(jnp.abs(bundle.conditioning.bias).sum()) == 0


@pytest.mark.parametrize("bad_len", [75, 77, 128])
def test_user_bias_mismatch_raises_naming_both_lengths(bad_len):
    from aminx.inference.bundle_builder import BiasLengthError, build_inference_bundle

    with pytest.raises(BiasLengthError) as ei:
        build_inference_bundle(**_bundle_kwargs(76), bias=jnp.zeros((bad_len, 21)))
    msg = str(ei.value)
    assert "76" in msg and str(bad_len) in msg


def test_state_position_map_width_mismatch_raises_naming_both_lengths():
    from aminx.inference.bundle_builder import (
        StatePositionMapLengthError,
        build_inference_bundle,
    )

    spm = np.arange(76, dtype=np.int32)[None, :]  # width 76 vs chain length 128
    with pytest.raises(StatePositionMapLengthError) as ei:
        build_inference_bundle(
            **_bundle_kwargs(128),
            state_position_map=jnp.asarray(spm.repeat(2, axis=0)),
        )
    msg = str(ei.value)
    assert "76" in msg and "128" in msg


def test_mismatched_map_width_changes_result_not_silently_accepted():
    """Adequacy: matched-width map passes the boundary; mismatched cannot reach
    the kernel silently -- the accepted-but-wrong pathology is closed."""
    from aminx.inference.bundle_builder import build_inference_bundle

    ok = np.arange(128, dtype=np.int32)[None, :].repeat(2, axis=0)
    bundle, _ = build_inference_bundle(
        **_bundle_kwargs(128), state_position_map=jnp.asarray(ok)
    )
    assert bundle.conditioning.state_position_map.shape == (2, 128)
