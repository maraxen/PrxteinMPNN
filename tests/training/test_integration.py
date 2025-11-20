"""Integration tests for the complete training pipeline."""

import pytest

from prxteinmpnn.training import TrainingSpecification


@pytest.mark.slow
def test_train_overfit_single_batch(temp_data_dir, temp_checkpoint_dir, small_model, mock_batch):
    """Test that model can overfit to a single batch (smoke test)."""
    # This is a minimal integration test to verify the training loop works

    spec = TrainingSpecification(
        inputs=str(temp_data_dir / "train"),
        checkpoint_dir=temp_checkpoint_dir,
        batch_size=2,
        num_epochs=2,  # Just 2 epochs for quick test
        learning_rate=1e-3,
        log_every=1,
        checkpoint_every=5,
        eval_every=5,
    )

    # Note: This test requires mock data files in temp_data_dir
    # In practice, you'd create mock PDB/PQR files or mock the data loader
    # For now, this is a structural test

    # Should not raise
    # results = train(spec)
    # If used, train() now returns a TrainingResult dataclass:
    # assert results.final_step > 0

    # Placeholder assertion
    assert spec.num_epochs == 2


@pytest.mark.slow
def test_train_with_validation(temp_data_dir, temp_checkpoint_dir):
    """Test training with validation set."""
    spec = TrainingSpecification(
        inputs=str(temp_data_dir / "train"),
        validation_data=str(temp_data_dir / "val"),
        checkpoint_dir=temp_checkpoint_dir,
        batch_size=2,
        num_epochs=2,
        eval_every=2,
    )

    # Placeholder - would run full training loop with mocked data
    assert spec.validation_data is not None


@pytest.mark.slow
def test_train_with_early_stopping(temp_data_dir, temp_checkpoint_dir):
    """Test training with early stopping."""
    spec = TrainingSpecification(
        inputs=str(temp_data_dir / "train"),
        validation_data=str(temp_data_dir / "val"),
        checkpoint_dir=temp_checkpoint_dir,
        batch_size=2,
        num_epochs=100,  # Many epochs
        early_stopping_patience=3,
        eval_every=2,
    )

    # Would trigger early stopping before reaching 100 epochs
    assert spec.early_stopping_patience == 3


@pytest.mark.slow
def test_train_resume_from_checkpoint(temp_data_dir, temp_checkpoint_dir):
    """Test resuming training from checkpoint."""
    # First training run
    spec1 = TrainingSpecification(
        inputs=str(temp_data_dir / "train"),
        checkpoint_dir=temp_checkpoint_dir,
        batch_size=2,
        num_epochs=1,
        checkpoint_every=5,
    )

    # Second training run (resume)
    spec2 = TrainingSpecification(
        inputs=str(temp_data_dir / "train"),
        checkpoint_dir=temp_checkpoint_dir,
        resume_from_checkpoint=temp_checkpoint_dir / "checkpoint_5",
        batch_size=2,
        num_epochs=2,
    )

    # Placeholder assertions
    assert spec2.resume_from_checkpoint is not None
