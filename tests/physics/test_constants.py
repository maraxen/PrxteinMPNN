"""Tests for physical constants."""

import pytest

from prxteinmpnn.physics.constants import (
    ANGSTROM_TO_NM,
    COULOMB_CONSTANT,
    COULOMB_CONSTANT_KCAL,
    DEFAULT_EPSILON,
    DEFAULT_SIGMA,
    KCAL_TO_KJ,
    KJ_TO_KCAL,
    MIN_DISTANCE,
    NM_TO_ANGSTROM,
)


def test_coulomb_constant_value():
    """Test that Coulomb constant has expected value."""
    # Standard value for protein simulations
    assert COULOMB_CONSTANT_KCAL == pytest.approx(332.0636, rel=1e-4)
    assert COULOMB_CONSTANT == COULOMB_CONSTANT_KCAL


def test_unit_conversions_are_inverses():
    """Test that unit conversion constants are inverses."""
    assert KCAL_TO_KJ * KJ_TO_KCAL == pytest.approx(1.0)
    assert NM_TO_ANGSTROM * ANGSTROM_TO_NM == pytest.approx(1.0)


def test_unit_conversion_values():
    """Test specific unit conversion values."""
    assert KCAL_TO_KJ == pytest.approx(4.184, rel=1e-4)
    assert NM_TO_ANGSTROM == pytest.approx(10.0)


def test_default_lj_parameters():
    """Test default LJ parameters are reasonable."""
    assert DEFAULT_SIGMA > 0
    assert DEFAULT_EPSILON > 0
    assert 2.0 < DEFAULT_SIGMA < 5.0  # Typical range for atoms
    assert 0.01 < DEFAULT_EPSILON < 1.0  # Typical range


def test_min_distance_is_small():
    """Test that minimum distance is sufficiently small for stability."""
    assert MIN_DISTANCE < 1e-5
    assert MIN_DISTANCE > 0