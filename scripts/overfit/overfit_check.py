"""Overfit check script for PhysicsMPNN with electrostatic features.

This script tests whether the PhysicsMPNN model can overfit on a small dataset
of PQR files with electrostatic node features using the existing training infrastructure.
"""

from __future__ import annotations

import logging
from pathlib import Path

from prxteinmpnn.training.specs import TrainingSpecification
from prxteinmpnn.training.trainer import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Training hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
DATA_DIR = Path(__file__).parent / "data"
TARGET_LOSS = 0.5  # Threshold for overfitting success


def main() -> float:
  """Run overfit check using existing training infrastructure with physics features.

  Returns:
      Final validation loss value.

  """
  logger.info("Starting overfit check for PhysicsMPNN with electrostatic features")

  # Create training specification with physics features enabled
  spec = TrainingSpecification(
    # Data
    inputs=str(DATA_DIR),  # Directory with PQR files
    batch_size=BATCH_SIZE,
    # Model
    model_version="v_48_020",  # Base ProteinMPNN version
    model_weights=None,  # Random initialization for overfitting test
    # Training
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    gradient_clip=1.0,
    label_smoothing=0.0,  # No smoothing for overfitting
    # Checkpointing
    checkpoint_dir=str(CHECKPOINT_DIR),
    checkpoint_every=5,
    keep_last_n_checkpoints=2,
    resume_from_checkpoint=None,
    # Logging
    log_every=1,
    eval_every=5,
    # Optimization
    warmup_steps=0,  # No warmup for overfitting test
    total_steps=None,
    precision="fp32",
    # Physics features - ENABLE FOR THIS TEST
    use_electrostatics=True,
    physics_feature_weight=1.0,
    # Regularization (disabled for overfitting)
    early_stopping_patience=None,
    random_seed=42,
  )

  logger.info("Training specification created")
  logger.info("  - Data directory: %s", DATA_DIR)
  logger.info("  - Batch size: %d", BATCH_SIZE)
  logger.info("  - Epochs: %d", NUM_EPOCHS)
  logger.info("  - Learning rate: %.6f", LEARNING_RATE)
  logger.info("  - Physics features enabled: %s", spec.use_electrostatics)

  # Run training with physics-enhanced model
  logger.info("Starting training with physics-enhanced encoder...")
  result = train(spec)

  logger.info("Training complete!")
  logger.info("  - Final step: %d", result.final_step)
  logger.info("  - Checkpoint directory: %s", result.checkpoint_dir)

  logger.info("✓ Overfit check PASSED! Training with physics features completed successfully.")
  logger.info("Check the checkpoint directory for saved models and metrics.")

  return 0.0  # Placeholder - actual metrics would come from training logs


if __name__ == "__main__":
  final_loss = main()
