"""Tests for PottsRunSpec: frozen config dataclass with validation and JSON serialization."""

import json
import sys
from unittest.mock import MagicMock

import pytest


from aminx.potts.spec import PottsRunSpec
from aminx.potts._trw_spec import PottsTRWRunSpec


class TestPottsRunSpecConstruction:
  """Test basic construction and field defaults."""

  def test_default_construction(self) -> None:
    """Test minimal construction with defaults."""
    spec = PottsRunSpec(
      weights_path="/path/to/weights",
      k_neighbors=10,
    )
    assert spec.weights_path == "/path/to/weights"
    assert spec.k_neighbors == 10
    assert spec.n_backbones == 1
    assert spec.caliby_path is None
    assert spec.training is False
    assert isinstance(spec.trw_spec, PottsTRWRunSpec)

  def test_explicit_n_backbones(self) -> None:
    """Test explicit n_backbones field."""
    spec = PottsRunSpec(
      n_backbones=3,
      weights_path="/path/to/weights",
      k_neighbors=10,
    )
    assert spec.n_backbones == 3

  def test_explicit_caliby_path(self) -> None:
    """Test explicit caliby_path field."""
    spec = PottsRunSpec(
      weights_path="/path/to/weights",
      caliby_path="/path/to/caliby",
      k_neighbors=10,
    )
    assert spec.caliby_path == "/path/to/caliby"

  def test_none_caliby_path_valid(self) -> None:
    """Test that caliby_path=None is valid (identity default)."""
    spec = PottsRunSpec(
      weights_path="/path/to/weights",
      caliby_path=None,
      k_neighbors=10,
    )
    assert spec.caliby_path is None

  def test_frozen_after_construction(self) -> None:
    """Test that spec is frozen (immutable after construction)."""
    spec = PottsRunSpec(
      weights_path="/path/to/weights",
      k_neighbors=10,
    )
    with pytest.raises(Exception):  # FrozenInstanceError
      spec.weights_path = "/different/path"


class TestPottsRunSpecValidation:
  """Test __post_init__ guards and validation."""

  def test_n_backbones_less_than_one_raises(self) -> None:
    """Test ValueError when n_backbones < 1."""
    with pytest.raises(ValueError, match="n_backbones must be >= 1"):
      PottsRunSpec(
        n_backbones=0,
        weights_path="/path/to/weights",
        k_neighbors=10,
      )

  def test_n_backbones_negative_raises(self) -> None:
    """Test ValueError when n_backbones is negative."""
    with pytest.raises(ValueError, match="n_backbones must be >= 1"):
      PottsRunSpec(
        n_backbones=-1,
        weights_path="/path/to/weights",
        k_neighbors=10,
      )

  def test_training_true_with_fori_loop_raises(self) -> None:
    """Test ValueError when training=True and trw_loop='fori'."""
    trw_spec = PottsTRWRunSpec(
      rho_backend="dense_pinv",
      message_backend="dense",
      trw_loop="fori",
    )
    with pytest.raises(ValueError, match="trw_loop=fori is unsafe for training"):
      PottsRunSpec(
        weights_path="/path/to/weights",
        trw_spec=trw_spec,
        k_neighbors=10,
        training=True,
      )

  def test_training_true_with_scan_loop_valid(self) -> None:
    """Test that training=True with trw_loop='scan' is valid."""
    trw_spec = PottsTRWRunSpec(
      rho_backend="dense_pinv",
      message_backend="dense",
      trw_loop="scan",
      checkpoint_trw_step=True,
    )
    spec = PottsRunSpec(
      weights_path="/path/to/weights",
      trw_spec=trw_spec,
      k_neighbors=10,
      training=True,
    )
    assert spec.training is True
    assert spec.trw_spec.trw_loop == "scan"

  def test_training_false_with_fori_loop_valid(self) -> None:
    """Test that training=False with trw_loop='fori' is valid (default behavior)."""
    trw_spec = PottsTRWRunSpec(
      rho_backend="dense_pinv",
      message_backend="dense",
      trw_loop="fori",
    )
    spec = PottsRunSpec(
      weights_path="/path/to/weights",
      trw_spec=trw_spec,
      k_neighbors=10,
      training=False,
    )
    assert spec.training is False
    assert spec.trw_spec.trw_loop == "fori"


class TestPottsRunSpecJSON:
  """Test JSON serialization round-trip."""

  def test_to_json_basic(self) -> None:
    """Test to_json produces valid JSON string."""
    spec = PottsRunSpec(
      weights_path="/path/to/weights",
      k_neighbors=10,
    )
    json_str = spec.to_json()
    assert isinstance(json_str, str)
    data = json.loads(json_str)
    assert data["weights_path"] == "/path/to/weights"
    assert data["k_neighbors"] == 10

  def test_json_round_trip_minimal(self) -> None:
    """Test minimal spec round-trips through JSON."""
    spec1 = PottsRunSpec(
      weights_path="/path/to/weights",
      k_neighbors=10,
    )
    json_str = spec1.to_json()
    spec2 = PottsRunSpec.from_json(json_str)

    assert spec2.weights_path == spec1.weights_path
    assert spec2.k_neighbors == spec1.k_neighbors
    assert spec2.n_backbones == spec1.n_backbones
    assert spec2.caliby_path == spec1.caliby_path
    assert spec2.training == spec1.training

  def test_json_round_trip_full(self) -> None:
    """Test full spec with all fields round-trips through JSON."""
    trw_spec = PottsTRWRunSpec(
      rho_backend="edge_lanczos",
      message_backend="tiled_scan",
      tile_size=16,
      lanczos_rank=64,
      checkpoint_trw_step=True,
      trw_loop="scan",
    )
    spec1 = PottsRunSpec(
      n_backbones=5,
      weights_path="/path/to/weights",
      caliby_path="/path/to/caliby",
      trw_spec=trw_spec,
      k_neighbors=20,
      training=True,
    )
    json_str = spec1.to_json()
    spec2 = PottsRunSpec.from_json(json_str)

    assert spec2.n_backbones == 5
    assert spec2.weights_path == spec1.weights_path
    assert spec2.caliby_path == spec1.caliby_path
    assert spec2.k_neighbors == 20
    assert spec2.training is True
    assert spec2.trw_spec.rho_backend == "edge_lanczos"
    assert spec2.trw_spec.message_backend == "tiled_scan"
    assert spec2.trw_spec.tile_size == 16
    assert spec2.trw_spec.lanczos_rank == 64
    assert spec2.trw_spec.checkpoint_trw_step is True
    assert spec2.trw_spec.trw_loop == "scan"

  def test_json_preserves_none_caliby_path(self) -> None:
    """Test that JSON round-trip preserves caliby_path=None."""
    spec1 = PottsRunSpec(
      weights_path="/path/to/weights",
      caliby_path=None,
      k_neighbors=10,
    )
    json_str = spec1.to_json()
    spec2 = PottsRunSpec.from_json(json_str)
    assert spec2.caliby_path is None


class TestPottsRunSpecClassmethods:
  """Test inference_default and training_default classmethods."""

  def test_inference_default(self) -> None:
    """Test inference_default classmethod."""
    spec = PottsRunSpec.inference_default(
      weights_path="/path/to/weights",
      k_neighbors=15,
    )

    assert spec.weights_path == "/path/to/weights"
    assert spec.k_neighbors == 15
    assert spec.n_backbones == 1
    assert spec.caliby_path is None
    assert spec.training is False
    assert spec.trw_spec.trw_loop == "fori"
    assert spec.trw_spec.checkpoint_trw_step is False

  def test_training_default_minimal(self) -> None:
    """Test training_default classmethod with minimal args."""
    spec = PottsRunSpec.training_default(
      weights_path="/path/to/weights",
      k_neighbors=15,
    )

    assert spec.weights_path == "/path/to/weights"
    assert spec.k_neighbors == 15
    assert spec.n_backbones == 1
    assert spec.caliby_path is None
    assert spec.training is True
    assert spec.trw_spec.trw_loop == "scan"
    assert spec.trw_spec.checkpoint_trw_step is True

  def test_training_default_with_caliby(self) -> None:
    """Test training_default with explicit caliby_path."""
    spec = PottsRunSpec.training_default(
      weights_path="/path/to/weights",
      k_neighbors=15,
      caliby_path="/path/to/caliby",
    )

    assert spec.caliby_path == "/path/to/caliby"
    assert spec.training is True

  def test_training_default_validates_on_construction(self) -> None:
    """Test that training_default produces spec that passes validation."""
    spec = PottsRunSpec.training_default(
      weights_path="/path/to/weights",
      k_neighbors=15,
    )
    # Should not raise; validates in __post_init__
    assert spec.training is True
    assert spec.trw_spec.trw_loop == "scan"

  def test_inference_default_round_trips(self) -> None:
    """Test that inference_default result round-trips through JSON."""
    spec1 = PottsRunSpec.inference_default(
      weights_path="/path/to/weights",
      k_neighbors=15,
    )
    json_str = spec1.to_json()
    spec2 = PottsRunSpec.from_json(json_str)

    assert spec2.weights_path == spec1.weights_path
    assert spec2.k_neighbors == spec1.k_neighbors
    assert spec2.training == spec1.training
    assert spec2.trw_spec.trw_loop == spec1.trw_spec.trw_loop

  def test_training_default_round_trips(self) -> None:
    """Test that training_default result round-trips through JSON."""
    spec1 = PottsRunSpec.training_default(
      weights_path="/path/to/weights",
      k_neighbors=15,
      caliby_path="/path/to/caliby",
    )
    json_str = spec1.to_json()
    spec2 = PottsRunSpec.from_json(json_str)

    assert spec2.weights_path == spec1.weights_path
    assert spec2.k_neighbors == spec1.k_neighbors
    assert spec2.training == spec1.training
    assert spec2.caliby_path == spec1.caliby_path
    assert spec2.trw_spec.trw_loop == spec1.trw_spec.trw_loop
    assert spec2.trw_spec.checkpoint_trw_step == spec1.trw_spec.checkpoint_trw_step
