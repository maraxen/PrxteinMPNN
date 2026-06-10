"""Unit tests for calibration module.

Tests cover:
  - IdentityCalibration: no-op behavior under jit
  - LearnedCalibration: additive/multiplicative/learned_scale modes with known corrections
  - load_calibration: None path → IdentityCalibration, checkpoint loading, missing path error
"""

import pickle
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import zstandard as zstd

from aminx.potts.calibration import (
    IdentityCalibration,
    LearnedCalibration,
    load_calibration,
)


class TestIdentityCalibration:
    """Identity calibration returns marginals unchanged."""

    def test_identity_returns_same_marginals(self) -> None:
        """IdentityCalibration.__call__ is identity."""
        calib = IdentityCalibration()
        marginals = jnp.array([[0.8, 0.2], [0.3, 0.7]])
        result = calib(marginals)
        assert jnp.allclose(result, marginals)

    def test_identity_under_jit(self) -> None:
        """IdentityCalibration compiles under eqx.filter_jit."""
        calib = IdentityCalibration()
        jitted_call = eqx.filter_jit(calib.__call__)

        marginals = jnp.array([[0.8, 0.2], [0.3, 0.7]])
        result = jitted_call(marginals)
        assert jnp.allclose(result, marginals)

    def test_identity_preserves_shape(self) -> None:
        """Identity preserves arbitrary marginal shapes."""
        calib = IdentityCalibration()
        marginals = jnp.ones((5, 20)) / 20
        result = calib(marginals)
        assert result.shape == marginals.shape


class TestLearnedCalibration:
    """Learned calibration applies vector or scalar corrections."""

    def test_additive_correction(self) -> None:
        """Additive mode: corrected = marginals + correction."""
        correction = jnp.array([[-0.1, 0.1], [0.05, -0.05]])
        calib = LearnedCalibration(correction, correction_type="additive")

        marginals = jnp.array([[0.6, 0.4], [0.3, 0.7]])
        result = calib(marginals)

        expected = marginals + correction
        assert jnp.allclose(result, expected)

    def test_multiplicative_correction(self) -> None:
        """Multiplicative mode: corrected = marginals * correction."""
        correction = jnp.array([[2.0, 0.5], [1.5, 0.8]])
        calib = LearnedCalibration(correction, correction_type="multiplicative")

        marginals = jnp.array([[0.6, 0.4], [0.3, 0.7]])
        result = calib(marginals)

        expected = marginals * correction
        assert jnp.allclose(result, expected)

    def test_learned_scale_correction(self) -> None:
        """Learned_scale mode: corrected = sigmoid(logit(marginals) + correction)."""
        correction = jnp.array([[0.1, -0.1], [-0.05, 0.05]])
        calib = LearnedCalibration(correction, correction_type="learned_scale")

        marginals = jnp.array([[0.6, 0.4], [0.3, 0.7]])
        result = calib(marginals)

        # Manually compute expected result.
        eps = 1e-7
        clamped = jnp.clip(marginals, eps, 1 - eps)
        logits = jnp.log(clamped) - jnp.log(1 - clamped)
        corrected_logits = logits + correction
        expected = jax.nn.sigmoid(corrected_logits)

        assert jnp.allclose(result, expected)

    def test_learned_scale_avoids_boundary_issues(self) -> None:
        """Learned_scale mode handles marginals near 0 and 1."""
        correction = jnp.zeros((2, 2))
        calib = LearnedCalibration(correction, correction_type="learned_scale")

        # Marginals at extremes should not cause inf/nan.
        marginals = jnp.array([[1e-8, 1.0 - 1e-8], [0.5, 0.5]])
        result = calib(marginals)

        # Should not contain inf or nan.
        assert jnp.all(jnp.isfinite(result))

    def test_invalid_correction_type_raises(self) -> None:
        """Unknown correction_type raises ValueError."""
        correction = jnp.array([[0.1, -0.1]])
        calib = LearnedCalibration(correction, correction_type="invalid_mode")

        marginals = jnp.array([[0.6, 0.4]])
        with pytest.raises(ValueError, match="Unknown correction_type"):
            calib(marginals)

    def test_learned_calibration_under_jit(self) -> None:
        """LearnedCalibration compiles under eqx.filter_jit."""
        correction = jnp.array([[-0.1, 0.1], [0.05, -0.05]])
        calib = LearnedCalibration(correction, correction_type="additive")
        jitted_call = eqx.filter_jit(calib.__call__)

        marginals = jnp.array([[0.6, 0.4], [0.3, 0.7]])
        result = jitted_call(marginals)

        expected = marginals + correction
        assert jnp.allclose(result, expected)

    def test_metadata_preserved(self) -> None:
        """Metadata dict is preserved on construction."""
        metadata = {"dataset": "test", "epoch": 42}
        correction = jnp.array([[0.1, -0.1]])
        calib = LearnedCalibration(
            correction,
            correction_type="additive",
            metadata=metadata,
        )
        assert calib.metadata == metadata

    def test_metadata_defaults_to_empty_dict(self) -> None:
        """Metadata defaults to empty dict if None provided."""
        correction = jnp.array([[0.1, -0.1]])
        calib = LearnedCalibration(correction, correction_type="additive", metadata=None)
        assert calib.metadata == {}


class TestLoadCalibration:
    """load_calibration handles None, existing files, and missing files."""

    def test_load_calibration_none_returns_identity(self) -> None:
        """load_calibration(None) returns IdentityCalibration."""
        calib = load_calibration(None)
        assert isinstance(calib, IdentityCalibration)

    def test_load_calibration_missing_file_raises(self) -> None:
        """load_calibration raises FileNotFoundError if path does not exist."""
        with pytest.raises(FileNotFoundError, match="Calibration checkpoint not found"):
            load_calibration("/nonexistent/path/caliby_test.eqx.zst")

    def test_load_calibration_from_checkpoint(self) -> None:
        """load_calibration loads LearnedCalibration from zstd checkpoint."""
        # Create a test calibration and save it.
        correction = jnp.array([[0.1, -0.1], [0.05, -0.05]])
        original_calib = LearnedCalibration(
            correction,
            correction_type="additive",
            metadata={"test": True},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "caliby_test.eqx.zst"

            # Serialize and compress.
            pickled = pickle.dumps(original_calib)
            cctx = zstd.ZstdCompressor()
            compressed = cctx.compress(pickled)

            with checkpoint_path.open("wb") as f:
                f.write(compressed)

            # Load and verify.
            loaded_calib = load_calibration(str(checkpoint_path))
            assert isinstance(loaded_calib, LearnedCalibration)
            assert jnp.allclose(loaded_calib.correction, correction)
            assert loaded_calib.correction_type == "additive"
            assert loaded_calib.metadata == {"test": True}

    def test_load_and_use_checkpoint(self) -> None:
        """End-to-end: load checkpoint and apply calibration."""
        correction = jnp.array([[-0.1, 0.1], [0.05, -0.05]])
        original_calib = LearnedCalibration(
            correction,
            correction_type="additive",
            metadata={"dataset": "test_data"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "caliby_test.eqx.zst"

            pickled = pickle.dumps(original_calib)
            cctx = zstd.ZstdCompressor()
            compressed = cctx.compress(pickled)

            with checkpoint_path.open("wb") as f:
                f.write(compressed)

            # Load and apply.
            loaded_calib = load_calibration(str(checkpoint_path))
            marginals = jnp.array([[0.6, 0.4], [0.3, 0.7]])
            result = loaded_calib(marginals)

            expected = marginals + correction
            assert jnp.allclose(result, expected)
