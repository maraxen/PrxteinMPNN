"""Tests for TrainingSpecification."""
from pathlib import Path

import pytest

from prxteinmpnn.training.specs import TrainingSpecification


def test_training_spec_defaults():
    """Test that TrainingSpecification has sensible defaults."""
    spec = TrainingSpecification(
        inputs="data/train/",
        num_epochs=5,
    )
    assert spec.learning_rate == 1e-4
    assert spec.weight_decay == 0.01
    assert spec.batch_size == 32
    assert spec.precision == "bf16"
    assert spec.label_smoothing == 0.0
def test_training_spec_path_conversion(tmp_path: Path):
    """Test that string paths are converted to Path objects."""
    spec = TrainingSpecification(
        inputs="data/train/",
        validation_data="data/val/",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        num_epochs=5,
    )
    assert isinstance(spec.checkpoint_dir, Path)
    assert isinstance(spec.validation_data, Path)
def test_training_spec_checkpoint_dir_creation(tmp_path: Path):
    """Test that checkpoint directory is created if it doesn't exist."""
    checkpoint_dir = tmp_path / "new_checkpoints"
    assert not checkpoint_dir.exists()
    spec = TrainingSpecification(
        inputs="data/train/",
        checkpoint_dir=checkpoint_dir,
        num_epochs=5,
    )
    assert checkpoint_dir.exists()
def test_training_spec_requires_epochs_or_steps():
    """Test that either num_epochs or total_steps must be provided."""
    with pytest.raises(ValueError, match="Either total_steps or num_epochs must be provided"):
        TrainingSpecification(
            inputs="data/train/",
            num_epochs=None, # type: ignore
            total_steps=None,
        )
def test_training_spec_invalid_precision():
    """Test that invalid precision raises ValueError."""
    with pytest.raises(ValueError, match="precision must be one of"):
        TrainingSpecification(
            inputs="data/train/",
            num_epochs=5,
            precision="fp64",  # Invalid # type: ignore
        )
def test_training_spec_warmup_and_decay():
    """Test learning rate schedule parameters."""
    spec = TrainingSpecification(
        inputs="data/train/",
        num_epochs=10,
        warmup_steps=1000,
        total_steps=10000,
        learning_rate=1e-3,
    )
    assert spec.warmup_steps == 1000
    assert spec.total_steps == 10000
    assert spec.learning_rate == 1e-3
def test_training_spec_early_stopping():
    """Test early stopping configuration."""
    spec = TrainingSpecification(
        inputs="data/train/",
        validation_data="data/val/",
        num_epochs=10,
        early_stopping_patience=5,
        early_stopping_metric="val_accuracy",
    )
    assert spec.early_stopping_patience == 5
    assert spec.early_stopping_metric == "val_accuracy"
def test_training_spec_physics_features():
    """Test physics feature configuration."""
    spec = TrainingSpecification(
        inputs="data/train/",
        num_epochs=5,
        use_electrostatics=True,
        use_vdw=True,
        physics_feature_weight=2.0,
    )
    assert spec.use_electrostatics is True
    assert spec.use_vdw is True
    assert spec.physics_feature_weight == 2.0


def test_training_spec_save_at_epochs():
    """Test save_at_epochs configuration."""
    spec = TrainingSpecification(
        inputs="data/train/",
        num_epochs=10,
        save_at_epochs=[1, 5, 10],
    )
    assert spec.save_at_epochs == [1, 5, 10]
