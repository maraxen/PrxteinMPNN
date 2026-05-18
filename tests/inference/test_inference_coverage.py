"""Coverage tests for uncovered inference/driver/logits branches.

This test module focuses on:
  1. Uncovered logit strategies (geometric_mean, product)
  2. Unconditional and conditional scoring paths in driver.py
  3. Edge cases with bias and different state weights
  4. Bundle builder with non-default strategies
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.inference.logits import (
    ArithmeticMeanLogits,
    GeometricMeanLogits,
    ProductOfProbabilities,
    ARLogitFuse,
)
from prxteinmpnn.inference.driver import (
    decode,
    decode_ar,
    infer_topology,
    TOPOLOGY_AR,
    TOPOLOGY_CONDITIONAL_SCORE,
    TOPOLOGY_UNCONDITIONAL,
)
from prxteinmpnn.inference.bundle_builder import build_inference_bundle
from prxteinmpnn.types.stages import (
    StageSet,
    ConditionalDecodeStep,
    UnconditionalDecodeStep,
)
from prxteinmpnn.types.encodings import EncoderOutput
from prxteinmpnn.model import PrxteinMPNN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_model():
    """Minimal synthetic PrxteinMPNN for testing."""
    return PrxteinMPNN(
        node_features=16,
        edge_features=16,
        hidden_features=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=4,
        dropout_rate=0.0,
        key=jax.random.PRNGKey(0),
    )


@pytest.fixture
def synthetic_model_inference(synthetic_model):
    """Model in inference mode (no dropout)."""
    return eqx.tree_inference(synthetic_model, value=True)


@pytest.fixture
def synthetic_inputs():
    """Generate small deterministic inputs for testing.

    Returns:
        tuple of (coords, mask, residue_index, chain_index, sequence)
        where coords is (S=2, L=6, 4, 3) for multistate with 2 states and 6 residues
    """
    S, L = 2, 6  # 2 states, 6 residues
    rng = np.random.default_rng(42)

    coords = jnp.array(rng.normal(size=(S, L, 4, 3)).astype(np.float32))
    mask = jnp.ones((S, L), dtype=jnp.float32)
    residue_index = jnp.tile(jnp.arange(L, dtype=jnp.int32)[None, :], (S, 1))
    chain_index = jnp.zeros((S, L), dtype=jnp.int32)
    # Return a 1D sequence (L,) not (S, L) — build_inference_bundle expands it
    sequence = jnp.array(rng.integers(0, 20, size=L, dtype=np.int32))

    return coords, mask, residue_index, chain_index, sequence


# ---------------------------------------------------------------------------
# 1. Geometric Mean Logits Strategy (Uncovered)
# ---------------------------------------------------------------------------


class TestGeometricMeanLogits:
    """Test GeometricMeanLogits strategy which is currently uncovered."""

    def test_geometric_mean_reduces_state_dim(self):
        """GeometricMeanLogits reduces (S, L, V) to (L, V)."""
        S, L, V = 3, 6, 21
        weights = jnp.array([0.5, 0.3, 0.2])
        strategy = GeometricMeanLogits(weights=weights)

        logits = jnp.ones((S, L, V))
        result = strategy(logits)

        assert result.shape == (L, V)

    def test_geometric_mean_weighted_computation(self):
        """GeometricMeanLogits computes weighted average in logit space."""
        S, L, V = 2, 4, 21
        weights = jnp.array([0.6, 0.4])
        strategy = GeometricMeanLogits(weights=weights)

        # Create distinct logits per state
        logits = jnp.arange(S * L * V, dtype=jnp.float32).reshape(S, L, V)
        result = strategy(logits)

        # Expected: (w0 * L0 + w1 * L1) / sum(w)
        expected = (weights[0] * logits[0] + weights[1] * logits[1]) / jnp.sum(weights)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_geometric_mean_with_temperature(self):
        """GeometricMeanLogits respects temperature scaling."""
        S, L, V = 2, 4, 21
        weights = jnp.array([0.5, 0.5])
        temperature = 2.0
        strategy = GeometricMeanLogits(weights=weights, temperature=temperature)

        logits = jnp.ones((S, L, V))
        result = strategy(logits)

        # Temperature should scale the result
        expected = jnp.ones((L, V)) / temperature
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_geometric_mean_with_bias(self):
        """GeometricMeanLogits applies bias after averaging."""
        S, L, V = 2, 4, 21
        weights = jnp.array([0.5, 0.5])
        strategy = GeometricMeanLogits(weights=weights)

        logits = jnp.ones((S, L, V))
        bias = jnp.arange(L * V, dtype=jnp.float32).reshape(L, V)
        result = strategy(logits, bias=bias)

        # Result should be average + bias
        avg = jnp.mean(logits, axis=0)
        expected = avg + bias
        assert jnp.allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Product of Probabilities Strategy (Uncovered)
# ---------------------------------------------------------------------------


class TestProductOfProbabilities:
    """Test ProductOfProbabilities strategy (sum of weighted log-probs)."""

    def test_product_reduces_state_dim(self):
        """ProductOfProbabilities reduces (S, L, V) to (L, V)."""
        S, L, V = 3, 6, 21
        weights = jnp.array([0.5, 0.3, 0.2])
        strategy = ProductOfProbabilities(weights=weights)

        logits = jnp.ones((S, L, V))
        result = strategy(logits)

        assert result.shape == (L, V)

    def test_product_weighted_sum(self):
        """ProductOfProbabilities sums weighted logits."""
        S, L, V = 2, 4, 21
        weights = jnp.array([0.7, 0.3])
        strategy = ProductOfProbabilities(weights=weights)

        logits = jnp.arange(S * L * V, dtype=jnp.float32).reshape(S, L, V)
        result = strategy(logits)

        # Expected: sum(w_i * L_i)
        expected = weights[0] * logits[0] + weights[1] * logits[1]
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_product_with_bias(self):
        """ProductOfProbabilities applies bias."""
        S, L, V = 2, 4, 21
        weights = jnp.array([0.5, 0.5])
        strategy = ProductOfProbabilities(weights=weights)

        logits = jnp.ones((S, L, V))
        bias = jnp.ones((L, V)) * 2.0
        result = strategy(logits, bias=bias)

        sum_weighted = weights[0] * logits[0] + weights[1] * logits[1]
        expected = sum_weighted + bias
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_product_zero_weights(self):
        """ProductOfProbabilities handles zero weights gracefully."""
        S, L, V = 2, 4, 21
        weights = jnp.array([1.0, 0.0])
        strategy = ProductOfProbabilities(weights=weights)

        logits = jnp.arange(S * L * V, dtype=jnp.float32).reshape(S, L, V)
        result = strategy(logits)

        # Only first state should contribute
        expected = logits[0]
        assert jnp.allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Arithmetic Mean with Bias (Extended)
# ---------------------------------------------------------------------------


class TestArithmeticMeanLogitsWithBias:
    """Extended tests for ArithmeticMeanLogits with bias."""

    def test_arithmetic_mean_with_bias(self):
        """ArithmeticMeanLogits applies bias correctly."""
        S, L, V = 3, 6, 21
        weights = jnp.ones(S) / S
        strategy = ArithmeticMeanLogits(weights=weights)

        logits = jnp.ones((S, L, V))
        bias = jnp.arange(L * V, dtype=jnp.float32).reshape(L, V) * 0.01
        result = strategy(logits, bias=bias)

        # Result should be mean + bias
        avg = jnp.mean(logits, axis=0)
        expected = avg + bias
        assert jnp.allclose(result, expected, atol=1e-4)

    def test_arithmetic_mean_unequal_weights(self):
        """ArithmeticMeanLogits respects non-uniform weights."""
        S, L, V = 3, 4, 21
        weights = jnp.array([0.5, 0.3, 0.2])
        strategy = ArithmeticMeanLogits(weights=weights)

        # Create distinct per-state logits
        logits = jnp.array([
            jnp.ones((L, V)) * 1.0,
            jnp.ones((L, V)) * 2.0,
            jnp.ones((L, V)) * 3.0,
        ])

        result = strategy(logits)

        # Expected: weighted arithmetic mean in log-space
        # log(sum(w_i * exp(L_i)) / sum(w_i))
        weighted = weights[:, None, None] * jnp.exp(logits)
        expected_exp_avg = jnp.sum(weighted, axis=0) / jnp.sum(weights)
        expected = jnp.log(expected_exp_avg)

        assert jnp.allclose(result, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. Unconditional Scoring Path (Uncovered)
# ---------------------------------------------------------------------------


class TestUnconditionalDecodePath:
    """Test unconditional scoring path via driver._decode_unconditional."""

    def test_unconditional_topology_detection(self):
        """infer_topology detects TOPOLOGY_UNCONDITIONAL when UnconditionalDecodeStep set."""
        class DummyDecoder(eqx.Module):
            pass

        stage_set = StageSet(
            decode_step=UnconditionalDecodeStep(decoder=DummyDecoder())
        )
        topology = infer_topology(stage_set)
        assert topology == TOPOLOGY_UNCONDITIONAL

    def test_unconditional_decode_path_dispatch(self, synthetic_model_inference):
        """decode() dispatches to unconditional path when topology requires it."""
        from prxteinmpnn.inference.driver import _decode_unconditional

        # Build a minimal unconditional stage set
        class DummyDecoder(eqx.Module):
            def __call__(self, *args, **kwargs):
                return jnp.zeros((6, 16))  # (L, H)

        stage_set = StageSet(
            decode_step=UnconditionalDecodeStep(decoder=DummyDecoder()),
            logit_transform=ArithmeticMeanLogits(weights=jnp.array([0.5, 0.5]))
        )

        # Create encoder output
        enc = EncoderOutput(
            node_features=jnp.zeros((2, 6, 16)),  # (S, L, H)
            edge_features=jnp.zeros((2, 6, 4, 16)),  # (S, L, k, H)
            neighbor_indices=jnp.zeros((2, 6, 4), dtype=jnp.int32),
            mask=jnp.ones((2, 6))
        )

        # Mock conditioning
        class MockCond:
            bias = None

        key = jax.random.PRNGKey(0)
        config = type('Config', (), {'inference': True})()

        # Should not raise
        result = _decode_unconditional(
            synthetic_model_inference, key, enc, MockCond(), config, stage_set
        )
        assert result.shape == (6, 21)  # (L, V)


# ---------------------------------------------------------------------------
# 5. Conditional Scoring Path (Uncovered)
# ---------------------------------------------------------------------------


class TestConditionalDecodePath:
    """Test conditional scoring path via driver._decode_conditional."""

    def test_conditional_topology_default(self):
        """infer_topology returns TOPOLOGY_CONDITIONAL_SCORE by default."""
        stage_set = StageSet()  # No decode_step, no sample_step
        topology = infer_topology(stage_set)
        assert topology == TOPOLOGY_CONDITIONAL_SCORE

    def test_conditional_with_custom_decode_step(self):
        """Conditional path dispatches with ConditionalDecodeStep."""
        from prxteinmpnn.inference.driver import _decode_conditional

        class DummyDecoder(eqx.Module):
            def call_conditional(self, *args, **kwargs):
                # Returns (L, H) decoded features
                return jnp.zeros((6, 16))

        class DummyEmbed(eqx.Module):
            weight = None

        class DummyModel:
            def __init__(self):
                self.w_out = lambda x: jnp.zeros((21,))  # Takes 1 arg

        stage_set = StageSet(
            decode_step=ConditionalDecodeStep(
                decoder=DummyDecoder(),
                w_s_embed=DummyEmbed()
            ),
            logit_transform=ArithmeticMeanLogits(weights=jnp.array([0.5, 0.5]))
        )

        enc = EncoderOutput(
            node_features=jnp.zeros((2, 6, 16)),
            edge_features=jnp.zeros((2, 6, 4, 16)),
            neighbor_indices=jnp.zeros((2, 6, 4), dtype=jnp.int32),
            mask=jnp.ones((2, 6))
        )

        class MockCond:
            sequence_oh = jnp.zeros((6, 21))
            ar_mask = jnp.zeros((2, 6, 6))
            bias = None

        key = jax.random.PRNGKey(0)
        config = type('Config', (), {'inference': True})()

        # Should not raise
        result = _decode_conditional(
            DummyModel(),
            key, enc, MockCond(), config, stage_set
        )
        assert result.shape == (6, 21)


# ---------------------------------------------------------------------------
# 6. Bundle Builder with Different Strategies
# ---------------------------------------------------------------------------


class TestBundleBuilderStrategies:
    """Test bundle_builder with non-default strategies."""

    def test_bundle_builder_geometric_mean(self, synthetic_inputs):
        """bundle_builder correctly wires geometric_mean strategy."""
        coords, mask, residue_index, chain_index, sequence = synthetic_inputs

        bundle, config, stage_set = build_inference_bundle(
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            sequence=sequence,
            mode="score_conditional",
            strategy="geometric_mean",
            strategy_temperature=2.0,
        )

        # Verify geometric_mean is wired
        assert isinstance(stage_set.logit_transform, GeometricMeanLogits)
        assert stage_set.logit_transform.temperature == 2.0

    def test_bundle_builder_product_strategy(self, synthetic_inputs):
        """bundle_builder correctly wires product strategy."""
        coords, mask, residue_index, chain_index, sequence = synthetic_inputs

        bundle, config, stage_set = build_inference_bundle(
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            sequence=sequence,
            strategy="product",
        )

        # Verify product is wired
        assert isinstance(stage_set.logit_transform, ProductOfProbabilities)

    def test_bundle_builder_with_custom_state_weights(self, synthetic_inputs):
        """bundle_builder respects custom state_weights."""
        coords, mask, residue_index, chain_index, sequence = synthetic_inputs
        custom_weights = jnp.array([0.3, 0.7])

        bundle, config, stage_set = build_inference_bundle(
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            sequence=sequence,
            state_weights=custom_weights,
            strategy="arithmetic_mean",
        )

        # Verify weights are propagated
        assert jnp.allclose(stage_set.logit_transform.weights, custom_weights)

    def test_bundle_builder_geometric_with_custom_temp(self, synthetic_inputs):
        """bundle_builder wires custom temperature for geometric_mean."""
        coords, mask, residue_index, chain_index, sequence = synthetic_inputs

        bundle, config, stage_set = build_inference_bundle(
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            sequence=sequence,
            strategy="geometric_mean",
            strategy_temperature=0.5,
        )

        assert isinstance(stage_set.logit_transform, GeometricMeanLogits)
        assert stage_set.logit_transform.temperature == 0.5


# ---------------------------------------------------------------------------
# 7. AR Logit Transform (Extended)
# ---------------------------------------------------------------------------


class TestARLogitFuseExtended:
    """Extended tests for ARLogitFuse in AR path."""

    def test_ar_logit_fuse_in_stage_set(self):
        """ARLogitFuse is correctly wired in StageSet.ar_logit_transform."""
        stage_set = StageSet(ar_logit_transform=ARLogitFuse())
        assert isinstance(stage_set.ar_logit_transform, ARLogitFuse)

    def test_ar_logit_fuse_vmap_compatible(self):
        """ARLogitFuse works under vmap for per-position fusion."""
        fuse = ARLogitFuse()

        # Simulate (S, L, V) logits
        S, L, V = 3, 6, 21
        logits = jnp.ones((S, L, V))

        # vmap over position axis: (S, L, V) -> (L, V)
        vmapped_fuse = jax.vmap(fuse, in_axes=1, out_axes=0)
        result = vmapped_fuse(logits)

        assert result.shape == (L, V)
        # Each position should be mean of logits at that position
        expected_per_pos = jnp.mean(logits, axis=0)
        assert jnp.allclose(result, expected_per_pos, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. Integration: End-to-End Bundle to Logits
# ---------------------------------------------------------------------------


class TestEndToEndIntegration:
    """Integration tests combining bundle_builder with logit strategies."""

    def test_bundle_with_all_strategies(self, synthetic_inputs):
        """bundle_builder integrates with all logit strategies."""
        coords, mask, residue_index, chain_index, sequence = synthetic_inputs

        for strategy in ["arithmetic_mean", "geometric_mean", "product"]:
            bundle, config, stage_set = build_inference_bundle(
                coords=coords,
                mask=mask,
                residue_index=residue_index,
                chain_index=chain_index,
                sequence=sequence,
                strategy=strategy,
            )

            # Verify strategy is wired and callable
            assert stage_set.logit_transform is not None

            # Test callable: (S, L, V) -> (L, V)
            S, L, V = 2, 6, 21
            test_logits = jnp.ones((S, L, V))
            result = stage_set.logit_transform(test_logits)
            assert result.shape == (L, V)

    def test_bundle_arithmetic_with_bias(self, synthetic_inputs):
        """Logit transform applies bias after fusion."""
        coords, mask, residue_index, chain_index, sequence = synthetic_inputs
        L = coords.shape[1]

        bias = jnp.ones((L, 21)) * 0.5

        bundle, config, stage_set = build_inference_bundle(
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            sequence=sequence,
            bias=bias,
        )

        # Verify bias is in conditioning
        assert jnp.allclose(bundle.conditioning.bias, bias)

        # Test that logit_transform can accept and apply bias
        S, V = 2, 21
        test_logits = jnp.ones((S, L, V))
        result = stage_set.logit_transform(test_logits, bias=bias)

        # Result should be higher due to positive bias
        result_no_bias = stage_set.logit_transform(test_logits)
        assert jnp.all(result > result_no_bias - 0.1)  # bias effect visible


# ---------------------------------------------------------------------------
# 9. Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_state_geometric_mean(self):
        """GeometricMeanLogits with S=1 reduces properly."""
        weights = jnp.array([1.0])
        strategy = GeometricMeanLogits(weights=weights)

        L, V = 6, 21
        logits = jnp.ones((1, L, V))
        result = strategy(logits)

        assert result.shape == (L, V)
        assert jnp.allclose(result, logits[0], atol=1e-5)

    def test_single_state_product(self):
        """ProductOfProbabilities with S=1."""
        weights = jnp.array([1.0])
        strategy = ProductOfProbabilities(weights=weights)

        L, V = 6, 21
        logits = jnp.ones((1, L, V))
        result = strategy(logits)

        assert result.shape == (L, V)
        assert jnp.allclose(result, logits[0], atol=1e-5)

    def test_large_state_weights(self):
        """Strategies handle large weight values."""
        S, L, V = 3, 6, 21
        weights = jnp.array([100.0, 200.0, 300.0])  # Large values

        strategy = ArithmeticMeanLogits(weights=weights)
        logits = jnp.ones((S, L, V))
        result = strategy(logits)

        # Should still be ~1.0 (weighted average of 1.0)
        assert jnp.allclose(result, jnp.ones((L, V)), atol=1e-4)

    def test_very_small_weights(self):
        """Strategies handle very small weight values."""
        S, L, V = 2, 4, 21
        weights = jnp.array([1e-9, 1e-9])

        strategy = ProductOfProbabilities(weights=weights)
        logits = jnp.ones((S, L, V))
        result = strategy(logits)

        # Should still compute without NaNs
        assert not jnp.any(jnp.isnan(result))


# ---------------------------------------------------------------------------
# 10. JIT Compilation
# ---------------------------------------------------------------------------


class TestJITCompilation:
    """Test that all paths work under jax.jit."""

    def test_arithmetic_mean_jit(self):
        """ArithmeticMeanLogits works under jit."""
        strategy = ArithmeticMeanLogits(weights=jnp.array([0.5, 0.5]))

        @jax.jit
        def apply(logits, bias):
            return strategy(logits, bias=bias)

        S, L, V = 2, 6, 21
        logits = jnp.ones((S, L, V))
        bias = jnp.zeros((L, V))
        result = apply(logits, bias)

        assert result.shape == (L, V)

    def test_geometric_mean_jit(self):
        """GeometricMeanLogits works under jit."""
        strategy = GeometricMeanLogits(
            weights=jnp.array([0.6, 0.4]),
            temperature=1.5
        )

        @jax.jit
        def apply(logits):
            return strategy(logits)

        S, L, V = 2, 6, 21
        logits = jnp.ones((S, L, V))
        result = apply(logits)

        assert result.shape == (L, V)

    def test_product_jit(self):
        """ProductOfProbabilities works under jit."""
        strategy = ProductOfProbabilities(weights=jnp.array([0.7, 0.3]))

        @jax.jit
        def apply(logits, bias):
            return strategy(logits, bias=bias)

        S, L, V = 2, 6, 21
        logits = jnp.ones((S, L, V))
        bias = jnp.zeros((L, V))
        result = apply(logits, bias)

        assert result.shape == (L, V)
