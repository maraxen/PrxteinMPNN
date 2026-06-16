"""Tests for backward-compat deprecation of average_encoding_mode kwarg."""

import pytest

from aminx.run.specs import (
    ScoringSpecification,
    SamplingSpecification,
    JacobianSpecification,
)


def test_scoring_spec_average_encoding_mode_deprecation():
    """Test ScoringSpecification emits DeprecationWarning when average_encoding_mode is passed."""
    with pytest.warns(DeprecationWarning, match="average_encoding_mode.*deprecated"):
        spec = ScoringSpecification(
            inputs=["dummy.pdb"],
            sequences_to_score=["ACDEFGHIKLMNPQRSTVWY"],
            average_encoding_mode="inputs",  # deprecated kwarg
        )
    # Spec should still be created successfully, just with warning
    assert spec is not None


def test_sampling_spec_average_encoding_mode_deprecation():
    """Test SamplingSpecification emits DeprecationWarning when average_encoding_mode is passed."""
    with pytest.warns(DeprecationWarning, match="average_encoding_mode.*deprecated"):
        spec = SamplingSpecification(
            inputs=["dummy.pdb"],
            num_samples=4,
            average_encoding_mode="inputs_and_noise",  # deprecated kwarg
        )
    assert spec is not None


def test_jacobian_spec_average_encoding_mode_deprecation():
    """Test JacobianSpecification emits DeprecationWarning when average_encoding_mode is passed."""
    with pytest.warns(DeprecationWarning, match="average_encoding_mode.*deprecated"):
        spec = JacobianSpecification(
            inputs=["dummy.pdb"],
            average_encoding_mode="noise_levels",  # deprecated kwarg
        )
    assert spec is not None


def test_average_encoding_mode_kwarg_is_stripped():
    """Test that average_encoding_mode kwarg is stripped and does not cause errors."""
    # This should not raise, just warn
    spec = ScoringSpecification(
        inputs=["dummy.pdb"],
        sequences_to_score=["ACDEFGHIKLMNPQRSTVWY"],
        average_encoding_mode="inputs",
    )
    # Verify spec was created successfully
    assert spec.inputs == ("dummy.pdb",) or spec.inputs == ["dummy.pdb"]
