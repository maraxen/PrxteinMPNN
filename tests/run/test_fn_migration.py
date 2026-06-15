"""Tests for AveragingMode enum and function-typed aggregation/decode fields."""

import jax.numpy as jnp
import pytest

from aminx.run.specs import (
    AveragingMode,
    ScoringSpecification,
    SamplingSpecification,
    JacobianSpecification,
)


def test_averaging_mode_enum():
    """Test AveragingMode enum is defined and has expected members."""
    assert hasattr(AveragingMode, "INPUTS")
    assert hasattr(AveragingMode, "NOISE_LEVELS")
    assert hasattr(AveragingMode, "INPUTS_AND_NOISE")


def test_averaging_mode_to_fn_inputs():
    """Test AveragingMode.INPUTS.to_fn() returns callable that averages encodings only."""
    fn = AveragingMode.INPUTS.to_fn()
    encodings = jnp.ones((4, 8))
    noised = jnp.ones((4, 8)) * 2
    result = fn(encodings, noised)

    # Should return mean of encodings (all 1s)
    assert result.shape == (8,)
    assert jnp.allclose(result, 1.0)


def test_averaging_mode_to_fn_noise_levels():
    """Test AveragingMode.NOISE_LEVELS.to_fn() returns callable that averages noised only."""
    fn = AveragingMode.NOISE_LEVELS.to_fn()
    encodings = jnp.ones((4, 8))
    noised = jnp.ones((4, 8)) * 2
    result = fn(encodings, noised)

    # Should return mean of noised (all 2s)
    assert result.shape == (8,)
    assert jnp.allclose(result, 2.0)


def test_averaging_mode_to_fn_inputs_and_noise():
    """Test AveragingMode.INPUTS_AND_NOISE.to_fn() returns callable that averages both."""
    fn = AveragingMode.INPUTS_AND_NOISE.to_fn()
    encodings = jnp.ones((4, 8))
    noised = jnp.ones((4, 8)) * 2
    result = fn(encodings, noised)

    # Should return mean of concatenated arrays: mean([1,1,1,1,2,2,2,2]) = 1.5
    assert result.shape == (8,)
    assert jnp.allclose(result, 1.5)


def test_scoring_spec_has_encoding_aggregation_fn():
    """Test ScoringSpecification has encoding_aggregation_fn callable field."""
    spec = ScoringSpecification(
        inputs=["dummy.pdb"], sequences_to_score=["ACDEFGHIKLMNPQRSTVWY"]
    )
    assert hasattr(spec, "encoding_aggregation_fn")
    assert callable(spec.encoding_aggregation_fn)

    # Should work as a function
    encodings = jnp.ones((4, 8))
    noised = jnp.ones((4, 8)) * 2
    result = spec.encoding_aggregation_fn(encodings, noised)
    assert result.shape == (8,)


def test_sampling_spec_has_encoding_aggregation_fn():
    """Test SamplingSpecification has encoding_aggregation_fn callable field."""
    spec = SamplingSpecification(inputs=["dummy.pdb"], num_samples=4)
    assert hasattr(spec, "encoding_aggregation_fn")
    assert callable(spec.encoding_aggregation_fn)


def test_jacobian_spec_has_encoding_aggregation_fn():
    """Test JacobianSpecification has encoding_aggregation_fn callable field."""
    spec = JacobianSpecification(inputs=["dummy.pdb"])
    assert hasattr(spec, "encoding_aggregation_fn")
    assert callable(spec.encoding_aggregation_fn)


def test_sampling_spec_decode_fn_typed():
    """Test SamplingSpecification.decode_fn field is typed as DecodeFn | None."""
    spec = SamplingSpecification(inputs=["dummy.pdb"], num_samples=4)
    assert hasattr(spec, "decode_fn")
    assert spec.decode_fn is None

    # Should accept a callable
    decode_fn = lambda output: output
    spec2 = SamplingSpecification(
        inputs=["dummy.pdb"], num_samples=4, decode_fn=decode_fn
    )
    assert spec2.decode_fn is decode_fn


def test_with_averaging_mode_classmethod():
    """Test with_averaging_mode classmethod exists and works."""
    spec = ScoringSpecification.with_averaging_mode(
        "inputs",
        inputs=["dummy.pdb"],
        sequences_to_score=["ACDEFGHIKLMNPQRSTVWY"],
    )
    assert callable(spec.encoding_aggregation_fn)

    # Should produce INPUTS mode function
    encodings = jnp.ones((4, 8))
    noised = jnp.ones((4, 8)) * 2
    result = spec.encoding_aggregation_fn(encodings, noised)
    assert jnp.allclose(result, 1.0)  # Only encodings averaged


def test_with_averaging_mode_enum_input():
    """Test with_averaging_mode accepts AveragingMode enum directly."""
    spec = ScoringSpecification.with_averaging_mode(
        AveragingMode.NOISE_LEVELS,
        inputs=["dummy.pdb"],
        sequences_to_score=["ACDEFGHIKLMNPQRSTVWY"],
    )
    assert callable(spec.encoding_aggregation_fn)

    encodings = jnp.ones((4, 8))
    noised = jnp.ones((4, 8)) * 2
    result = spec.encoding_aggregation_fn(encodings, noised)
    assert jnp.allclose(result, 2.0)  # Only noised averaged
