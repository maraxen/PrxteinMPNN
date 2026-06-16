"""Test decoding fusion integration in kernel_dispatch."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from aminx.types.bundles import DecodeOutput
from aminx.types.stages import DecodingFusionFn


class IdentityDecodingFusion:
    """Test implementation: pass-through fusion (no aggregation)."""

    def __call__(self, stacked: DecodeOutput) -> DecodeOutput:
        """Return stacked outputs as-is."""
        return stacked


class BestOfKDecodingFusion:
    """Test implementation: select best sequence by logit score."""

    def __call__(self, stacked: DecodeOutput) -> DecodeOutput:
        """Select sequence with highest average logit score."""
        # stacked.sequences: (K, L)
        # stacked.logits: (K, L, V)
        # Average logits per sequence, pick best
        avg_logits = jnp.mean(stacked.logits, axis=(1, 2))  # (K,)
        best_idx = jnp.argmax(avg_logits)
        return DecodeOutput(
            sequences=stacked.sequences[best_idx:best_idx + 1],  # (1, L)
            logits=stacked.logits[best_idx:best_idx + 1],  # (1, L, V)
        )


def test_decoding_fusion_identity():
    """Test that identity fusion preserves stacked outputs."""
    fusion = IdentityDecodingFusion()
    stacked = DecodeOutput(
        sequences=jnp.array([[0, 1, 2], [3, 4, 5]]),  # (2, 3)
        logits=jnp.ones((2, 3, 21)),  # (2, 3, 21)
    )
    result = fusion(stacked)
    assert result.sequences.shape == (2, 3)
    assert result.logits.shape == (2, 3, 21)


def test_decoding_fusion_best_of_k():
    """Test that best-of-K fusion selects single best sequence."""
    fusion = BestOfKDecodingFusion()
    # Create two sequences with different quality
    seq1_logits = jnp.ones((3, 21)) * 0.5  # Low quality
    seq2_logits = jnp.ones((3, 21)) * 2.0  # High quality
    stacked = DecodeOutput(
        sequences=jnp.array([[0, 1, 2], [3, 4, 5]]),  # (2, 3)
        logits=jnp.stack([seq1_logits, seq2_logits]),  # (2, 3, 21)
    )
    result = fusion(stacked)
    # Should select sequence at index 1 (higher logits)
    assert result.sequences.shape == (1, 3)
    assert result.logits.shape == (1, 3, 21)
    assert jnp.allclose(result.sequences, stacked.sequences[1:2])


def test_stageset_has_decoding_fusion_field():
    """Test that StageSet has decoding_fusion field."""
    from aminx.types.stages import StageSet

    stage_set = StageSet()
    assert hasattr(stage_set, "decoding_fusion")
    assert stage_set.decoding_fusion is None


def test_stageset_accepts_decoding_fusion():
    """Test that StageSet can be initialized with decoding_fusion."""
    import equinox as eqx

    from aminx.types.stages import StageSet

    fusion = IdentityDecodingFusion()
    stage_set = StageSet(decoding_fusion=fusion)
    assert stage_set.decoding_fusion is fusion

    # Test that eqx.tree_at can update it
    updated_stage_set = eqx.tree_at(
        lambda s: s.decoding_fusion,
        stage_set,
        BestOfKDecodingFusion(),
        is_leaf=lambda x: x is None,
    )
    assert isinstance(updated_stage_set.decoding_fusion, BestOfKDecodingFusion)


def test_runspecification_has_decoding_fusion():
    """Test that RunSpecification has decoding_fusion field."""
    from aminx.run.specs import RunSpecification

    spec = RunSpecification(inputs=["dummy.pdb"])
    assert hasattr(spec, "decoding_fusion")
    assert spec.decoding_fusion is None


def test_runspecification_can_set_decoding_fusion():
    """Test that RunSpecification accepts decoding_fusion."""
    from aminx.run.specs import RunSpecification

    fusion = IdentityDecodingFusion()
    spec = RunSpecification(inputs=["dummy.pdb"], decoding_fusion=fusion)
    assert spec.decoding_fusion is fusion
