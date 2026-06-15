"""Tests for _sync_run_spec guard flag to prevent double-fire on construction (#1906)."""

from unittest.mock import patch

from aminx.run.specs import (
    InspectionSpecification,
    JacobianSpecification,
    SamplingSpecification,
    ScoringSpecification,
)


def test_sync_run_spec_fires_once_sampling():
    """SamplingSpecification fires build_run_spec exactly once during construction.

    Verifies that the _run_spec_synced guard flag prevents double-firing:
    the parent RunSpecification.__post_init__ calls _sync_run_spec(),
    and the child SamplingSpecification.__post_init__ resets the flag
    and calls _sync_run_spec() again, but the guard ensures it fires only once.
    """
    with patch("aminx.run.specs.build_run_spec") as mock_build:
        mock_build.return_value = None
        SamplingSpecification(inputs=["dummy.pdb"], num_samples=4)
    assert mock_build.call_count == 1, f"Expected 1 call, got {mock_build.call_count}"


def test_sync_run_spec_fires_once_scoring():
    """ScoringSpecification fires build_run_spec exactly once during construction."""
    with patch("aminx.run.specs.build_run_spec") as mock_build:
        mock_build.return_value = None
        ScoringSpecification(inputs=["dummy.pdb"], sequences_to_score=["ACDEF"])
    assert mock_build.call_count == 1, f"Expected 1 call, got {mock_build.call_count}"


def test_sync_run_spec_fires_once_jacobian():
    """JacobianSpecification fires build_run_spec exactly once during construction."""
    with patch("aminx.run.specs.build_run_spec") as mock_build:
        mock_build.return_value = None
        JacobianSpecification(inputs=["dummy.pdb"])
    assert mock_build.call_count == 1, f"Expected 1 call, got {mock_build.call_count}"


def test_sync_run_spec_fires_once_inspection():
    """InspectionSpecification fires build_run_spec exactly once during construction."""
    with patch("aminx.run.specs.build_run_spec") as mock_build:
        mock_build.return_value = None
        InspectionSpecification(inputs=["dummy.pdb"])
    assert mock_build.call_count == 1, f"Expected 1 call, got {mock_build.call_count}"


def test_run_spec_populated_after_construction():
    """run_spec field is populated after construction for all spec types."""
    with patch("aminx.run.specs.build_run_spec") as mock_build:
        mock_spec = object()
        mock_build.return_value = mock_spec

        # Test each spec type
        for spec_cls in [
            SamplingSpecification,
            ScoringSpecification,
            JacobianSpecification,
            InspectionSpecification,
        ]:
            if spec_cls == ScoringSpecification:
                spec = spec_cls(inputs=["dummy.pdb"], sequences_to_score=["ACDEF"])
            else:
                spec = spec_cls(inputs=["dummy.pdb"])

            assert hasattr(spec, "run_spec"), f"{spec_cls.__name__} missing run_spec"
            assert (
                spec.run_spec is mock_spec
            ), f"{spec_cls.__name__} run_spec not set correctly"
