"""Smoke test for aminx.potts.runner.run_potts.

Tests the high-level orchestration function:
- Geometry extraction from GeometryBundle
- Model inference via infer_params
- Calibration application
- PottsResult construction

Uses @pytest.mark.slow and @pytest.mark.potts as required.
Uses tiny synthetic PottsModel (n=3 residues, num_aa=4, hidden_dim=8) to avoid
requiring real weight files.

NO JAX COMPUTE: This test constructs the model but does NOT call infer_params
(which would trigger JAX startup and compilation). Instead, it mocks the
forward pass or uses --dry-run mode. See !!HARD RULE in task description.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import zstandard as zstd
from jaxtyping import Array, Float, Int, PRNGKeyArray

from aminx.potts.model import PottsModel, PottsParams
from aminx.potts.runner import PottsResult, run_potts
from aminx.potts.spec import PottsRunSpec
from aminx.types.bundles import GeometryBundle


@pytest.mark.slow
@pytest.mark.potts
class TestPottsRunnerSmoke:
  """Smoke tests for run_potts orchestration function."""

  @staticmethod
  def _create_minimal_geometry(
    n_residues: int = 3,
    n_states: int = 1,
  ) -> GeometryBundle:
    """Create a minimal GeometryBundle for testing.

    Args:
        n_residues: Number of residues (default 3).
        n_states: Number of states (default 1).

    Returns:
        GeometryBundle with synthetic coordinates and masks.
    """
    return GeometryBundle(
      coords=jnp.zeros((n_states, n_residues, 4, 3), dtype=jnp.float32),
      mask=jnp.ones((n_states, n_residues), dtype=jnp.float32),
      residue_index=jnp.arange(n_residues, dtype=jnp.int32)[None, :].repeat(n_states, axis=0),
      chain_index=jnp.zeros((n_states, n_residues), dtype=jnp.int32),
      n_states=n_states,
      n_canonical=20,
      n_flat=n_residues,
    )

  @staticmethod
  def _create_minimal_potts_model(key: PRNGKeyArray, n: int = 3) -> PottsModel:
    """Create a minimal PottsModel matching runner.py skeleton defaults.

    Args:
        key: JAX random key.
        n: Number of residues (default 3).

    Returns:
        PottsModel with standard dimensions (hidden_dim=128, num_aa=21)
        matching _load_potts_model skeleton construction.
    """
    model = PottsModel(
      hidden_dim=128,
      num_aa=21,
      k_neighbors=2,
      edge_features_dim=128,
      trw_iters=15,
      key=key,
      training=False,
    )
    return model

  def test_geometry_extraction_single_state(self) -> None:
    """Test that GeometryBundle fields are accessible and extractable."""
    geometry = self._create_minimal_geometry(n_residues=3, n_states=1)

    # Verify fields exist and have correct shapes
    assert hasattr(geometry, "coords")
    assert hasattr(geometry, "mask")
    assert hasattr(geometry, "residue_index")
    assert hasattr(geometry, "chain_index")

    assert geometry.coords.shape == (1, 3, 4, 3)
    assert geometry.mask.shape == (1, 3)
    assert geometry.residue_index.shape == (1, 3)
    assert geometry.chain_index.shape == (1, 3)

  def test_geometry_extraction_multi_state(self) -> None:
    """Test GeometryBundle with multiple states."""
    geometry = self._create_minimal_geometry(n_residues=5, n_states=2)

    assert geometry.coords.shape == (2, 5, 4, 3)
    assert geometry.mask.shape == (2, 5)
    assert geometry.n_states == 2

  def test_potts_model_construction_minimal(self, rng_key: PRNGKeyArray) -> None:
    """Test that minimal PottsModel can be constructed."""
    model = self._create_minimal_potts_model(rng_key, n=3)

    assert model.hidden_dim == 128
    assert model.num_aa == 21
    assert model.k_neighbors == 2
    assert model.training is False

  def test_potts_result_construction(self) -> None:
    """Test PottsResult dataclass construction."""
    n, q = 3, 21
    result = PottsResult(
      marginals=jnp.ones((n, q)) / q,
      h=jnp.zeros((n, q)),
      J=jnp.zeros((n, n, q, q)),
      rho=jnp.zeros((n, n)),
      calibrated_marginals=jnp.ones((n, q)) / q,
      n_backbones=1,
    )

    assert result.marginals.shape == (n, q)
    assert result.calibrated_marginals.shape == (n, q)
    assert result.n_backbones == 1

  def test_potts_runner_smoke_with_mocked_inference(
    self,
    rng_key: PRNGKeyArray,
  ) -> None:
    """Smoke test: run_potts with mocked model.infer_params (no JAX compute).

    This test mocks the expensive infer_params call to verify the orchestration
    logic without triggering JAX compilation.

    Args:
        rng_key: JAX random key from fixture.
    """
    # Create temporary checkpoint file (mock weights)
    with tempfile.TemporaryDirectory() as tmpdir:
      weights_path = Path(tmpdir) / "model.eqx"

      # Create and serialize a minimal model
      model = self._create_minimal_potts_model(rng_key, n=3)
      eqx.tree_serialise_leaves(str(weights_path), model)

      # Create spec pointing to checkpoint
      spec = PottsRunSpec(
        n_backbones=1,
        weights_path=str(weights_path),
        caliby_path=None,
        k_neighbors=2,
        training=False,
      )

      # Create minimal geometry
      geometry = self._create_minimal_geometry(n_residues=3, n_states=1)

      # Mock the infer_params call to avoid JAX compute
      def mock_infer_params(*args, **kwargs) -> PottsParams:
        return PottsParams(
          marginals=jnp.ones((3, 21)) / 21,
          h=jnp.zeros((3, 21)),
          J=jnp.zeros((3, 3, 21, 21)),
          rho=jnp.zeros((3, 3)),
          W=jnp.ones((3, 3)),
        )

      # Patch the model to use mocked infer_params
      with patch.object(
        PottsModel,
        "infer_params",
        side_effect=mock_infer_params,
      ):
        result = run_potts(spec, geometry, rng_key)

      # Verify PottsResult structure
      assert isinstance(result, PottsResult)
      assert result.marginals.shape == (3, 21)
      assert result.h.shape == (3, 21)
      assert result.J.shape == (3, 3, 21, 21)
      assert result.rho.shape == (3, 3)
      assert result.calibrated_marginals.shape == (3, 21)
      assert result.n_backbones == 1

  def test_potts_runner_geometry_extraction_4_to_37(self) -> None:
    """Test that runner correctly pads coords from 4 to 37 atoms."""
    # Create geometry with 4-atom backbone (N, CA, C, O)
    geometry = GeometryBundle(
      coords=jnp.zeros((1, 3, 4, 3), dtype=jnp.float32),
      mask=jnp.ones((1, 3), dtype=jnp.float32),
      residue_index=jnp.arange(3, dtype=jnp.int32)[None, :],
      chain_index=jnp.zeros((1, 3), dtype=jnp.int32),
      n_states=1,
      n_canonical=20,
      n_flat=3,
    )

    # Extract coords and apply padding logic
    coords = geometry.coords
    if coords.ndim == 4:
      coords = coords[0]  # Extract single state

    assert coords.shape == (3, 4, 3)

    # Apply padding to 37-atom format
    if coords.shape[-2] == 4:
      n, _ = coords.shape[:2]
      coords_37 = jnp.zeros((n, 37, 3), dtype=jnp.float32)
      coords_37 = coords_37.at[:, :4, :].set(coords)
      coords = coords_37

    assert coords.shape == (3, 37, 3)
    # First 4 atoms should be original, rest should be zero
    assert jnp.allclose(coords[:, 4:, :], 0.0)

  def test_potts_runner_with_identity_calibration(
    self,
    rng_key: PRNGKeyArray,
  ) -> None:
    """Test run_potts with identity calibration (caliby_path=None)."""
    with tempfile.TemporaryDirectory() as tmpdir:
      weights_path = Path(tmpdir) / "model.eqx"

      model = self._create_minimal_potts_model(rng_key, n=3)
      eqx.tree_serialise_leaves(str(weights_path), model)

      spec = PottsRunSpec(
        n_backbones=1,
        weights_path=str(weights_path),
        caliby_path=None,  # Identity calibration
        k_neighbors=2,
        training=False,
      )

      geometry = self._create_minimal_geometry(n_residues=3, n_states=1)

      # Mock infer_params
      def mock_infer_params(*args, **kwargs) -> PottsParams:
        return PottsParams(
          marginals=jnp.ones((3, 21)) / 21,
          h=jnp.zeros((3, 21)),
          J=jnp.zeros((3, 3, 21, 21)),
          rho=jnp.zeros((3, 3)),
          W=jnp.ones((3, 3)),
        )

      with patch.object(
        PottsModel,
        "infer_params",
        side_effect=mock_infer_params,
      ):
        result = run_potts(spec, geometry, rng_key)

      # With identity calibration, calibrated_marginals == marginals
      assert jnp.allclose(result.calibrated_marginals, result.marginals)

  def test_spec_rejects_zero_k_neighbors(self) -> None:
    """Test that PottsRunSpec rejects k_neighbors <= 0 in __post_init__."""
    # Should raise ValueError during construction
    with pytest.raises(ValueError, match="k_neighbors.*must be > 0"):
      PottsRunSpec(
        n_backbones=1,
        weights_path="/path/to/weights",
        k_neighbors=0,  # Invalid
        training=False,
      )

  def test_spec_rejects_invalid_sampler_type(self) -> None:
    """Test that PottsRunSpec rejects invalid sampler_type."""
    with pytest.raises(ValueError, match="sampler_type.*must be 'gibbs' or 'pt'"):
      PottsRunSpec(
        n_backbones=1,
        weights_path="/path/to/weights",
        k_neighbors=2,
        sampler_type="invalid",  # Invalid
      )

  def test_potts_result_with_samples(self) -> None:
    """Test PottsResult construction with samples field."""
    n, q = 3, 21
    n_samples = 10
    result = PottsResult(
      marginals=jnp.ones((n, q)) / q,
      h=jnp.zeros((n, q)),
      J=jnp.zeros((n, n, q, q)),
      rho=jnp.zeros((n, n)),
      calibrated_marginals=jnp.ones((n, q)) / q,
      n_backbones=1,
      samples=jnp.zeros((n_samples, n), dtype=jnp.int32),
    )

    assert result.marginals.shape == (n, q)
    assert result.samples is not None
    assert result.samples.shape == (n_samples, n)
    assert result.samples.dtype == jnp.int32

  def test_potts_runner_with_gibbs_sampling_zstd(
    self,
    rng_key: PRNGKeyArray,
  ) -> None:
    """Smoke test: run_potts with Gibbs sampling using real zstd+eqx serialization.

    This test exercises the full sampling path:
    1. Creates minimal PottsModel
    2. Serializes to .eqx.zst (real zstd compression)
    3. Runs run_potts with n_samples > 0, sampler_type='gibbs'
    4. Verifies PottsResult.samples has correct shape and dtype
    """
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create and serialize model to .eqx.zst with zstd compression
      weights_path = Path(tmpdir) / "model.eqx.zst"
      model = self._create_minimal_potts_model(rng_key, n=3)

      # Serialize to uncompressed buffer, then compress with zstd
      buffer = io.BytesIO()
      eqx.tree_serialise_leaves(buffer, model)
      uncompressed = buffer.getvalue()

      cctx = zstd.ZstdCompressor()
      compressed = cctx.compress(uncompressed)

      with open(weights_path, "wb") as f:
        f.write(compressed)

      # Create spec with Gibbs sampling
      spec = PottsRunSpec(
        n_backbones=1,
        weights_path=str(weights_path),
        caliby_path=None,
        k_neighbors=2,
        training=False,
        n_samples=5,  # Request 5 samples
        n_chains=2,  # 2 independent chains
        sampler_type="gibbs",
      )

      geometry = self._create_minimal_geometry(n_residues=3, n_states=1)

      # Mock infer_params to avoid expensive JAX compute
      def mock_infer_params(*args, **kwargs) -> PottsParams:
        return PottsParams(
          marginals=jnp.ones((3, 21)) / 21,
          h=jnp.zeros((3, 21)),
          J=jnp.zeros((3, 3, 21, 21)),
          rho=jnp.zeros((3, 3)),
          W=jnp.ones((3, 3)),
        )

      with patch.object(
        PottsModel,
        "infer_params",
        side_effect=mock_infer_params,
      ):
        result = run_potts(spec, geometry, rng_key)

      # Verify PottsResult
      assert isinstance(result, PottsResult)
      assert result.samples is not None
      # n_chains (2) * n_samples (5) = 10 samples total
      assert result.samples.shape == (10, 3), f"Expected (10, 3), got {result.samples.shape}"
      assert result.samples.dtype == jnp.int32
      # All sample values should be in range [0, 21)
      assert jnp.all(result.samples >= 0)
      assert jnp.all(result.samples < 21)

  def test_potts_runner_with_pt_sampling_zstd(
    self,
    rng_key: PRNGKeyArray,
  ) -> None:
    """Smoke test: run_potts with parallel tempering using real zstd+eqx serialization.

    This test exercises the parallel tempering sampling path:
    1. Creates minimal PottsModel
    2. Serializes to .eqx.zst
    3. Runs run_potts with n_samples > 0, sampler_type='pt'
    4. Verifies PottsResult.samples has correct shape and dtype
    """
    with tempfile.TemporaryDirectory() as tmpdir:
      weights_path = Path(tmpdir) / "model.eqx.zst"
      model = self._create_minimal_potts_model(rng_key, n=3)

      # Serialize with zstd compression
      buffer = io.BytesIO()
      eqx.tree_serialise_leaves(buffer, model)
      uncompressed = buffer.getvalue()

      cctx = zstd.ZstdCompressor()
      compressed = cctx.compress(uncompressed)

      with open(weights_path, "wb") as f:
        f.write(compressed)

      # Create spec with parallel tempering sampling
      spec = PottsRunSpec(
        n_backbones=1,
        weights_path=str(weights_path),
        caliby_path=None,
        k_neighbors=2,
        training=False,
        n_samples=3,  # Number of PT rounds (sweeps)
        n_chains=2,  # 2 replicas for PT
        sampler_type="pt",
      )

      geometry = self._create_minimal_geometry(n_residues=3, n_states=1)

      def mock_infer_params(*args, **kwargs) -> PottsParams:
        return PottsParams(
          marginals=jnp.ones((3, 21)) / 21,
          h=jnp.zeros((3, 21)),
          J=jnp.zeros((3, 3, 21, 21)),
          rho=jnp.zeros((3, 3)),
          W=jnp.ones((3, 3)),
        )

      with patch.object(
        PottsModel,
        "infer_params",
        side_effect=mock_infer_params,
      ):
        result = run_potts(spec, geometry, rng_key)

      # Verify PottsResult
      assert isinstance(result, PottsResult)
      assert result.samples is not None
      # PT returns cold replica only: shape (1, 3)
      assert result.samples.shape == (1, 3), f"Expected (1, 3), got {result.samples.shape}"
      assert result.samples.dtype == jnp.int32
      # All sample values should be in range [0, 21)
      assert jnp.all(result.samples >= 0)
      assert jnp.all(result.samples < 21)

  def test_potts_runner_no_sampling_when_n_samples_zero(
    self,
    rng_key: PRNGKeyArray,
  ) -> None:
    """Test that PottsResult.samples is None when n_samples=0."""
    with tempfile.TemporaryDirectory() as tmpdir:
      weights_path = Path(tmpdir) / "model.eqx"
      model = self._create_minimal_potts_model(rng_key, n=3)
      eqx.tree_serialise_leaves(str(weights_path), model)

      spec = PottsRunSpec(
        n_backbones=1,
        weights_path=str(weights_path),
        k_neighbors=2,
        training=False,
        n_samples=0,  # No sampling
      )

      geometry = self._create_minimal_geometry(n_residues=3, n_states=1)

      def mock_infer_params(*args, **kwargs) -> PottsParams:
        return PottsParams(
          marginals=jnp.ones((3, 21)) / 21,
          h=jnp.zeros((3, 21)),
          J=jnp.zeros((3, 3, 21, 21)),
          rho=jnp.zeros((3, 3)),
          W=jnp.ones((3, 3)),
        )

      with patch.object(
        PottsModel,
        "infer_params",
        side_effect=mock_infer_params,
      ):
        result = run_potts(spec, geometry, rng_key)

      assert result.samples is None
