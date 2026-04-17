#!/usr/bin/env python
"""PrxteinMPNN Training Script with Gradient Accumulation.

This script mirrors the training_example_notebook.ipynb but can be run as a standalone
Python script. It includes gradient accumulation configuration.

Usage:
    python train_with_accumulation.py

Or modify the configuration variables below before running.
"""

import logging
import shutil
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data Configuration
use_example_data = True

# Training Mode
training_mode = "autoregressive"  # Options: "autoregressive", "diffusion"

# Physics Features
use_electrostatics = False
use_vdw = False

# Optimization Settings
learning_rate = 1e-4
weight_decay = 0.01

# Batch Size & Gradient Accumulation
# `total_batch_size = base_batch_size * accum_steps`
total_batch_size = 32  # Effective batch size for gradient updates
base_batch_size = 4    # Per-step batch size (limited by GPU memory)

# Training Duration
num_epochs = 3
warmup_steps = 100

# Regularization
label_smoothing = 0.1
gradient_clip = 2.0

# Precision
precision = "bf16"  # Options: "fp32", "fp16", "bf16"

# Diffusion Settings (only used if training_mode="diffusion")
diffusion_num_steps = 1000
diffusion_schedule_type = "cosine"
diffusion_beta_start = 1e-4
diffusion_beta_end = 0.02

# Checkpointing
checkpoint_dir = "./checkpoints"
checkpoint_every = 500
keep_last_n_checkpoints = 3

# Early Stopping
use_early_stopping = False
early_stopping_patience = 5
early_stopping_metric = "val_loss"

# Logging
log_every = 50
eval_every = 200

# Max sequence length
max_length = 512

# ============================================================================
# SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Calculate accumulation steps
accum_steps = max(1, total_batch_size // base_batch_size)
logger.info(f"Gradient Accumulation: {accum_steps} steps")
logger.info(f"  Base batch size: {base_batch_size}")
logger.info(f"  Effective batch size: {base_batch_size * accum_steps}")


def get_data_paths(use_example: bool, hf_repo: str) -> tuple[Path, Path]:
    """Get paths to training data based on configuration.
    
    Returns:
        Tuple of (array_record_path, index_path)
    """
    if use_example:
        # Use bundled example data
        possible_paths = [
            Path("data/pdb_sample.array_record"),
            Path("../data/pdb_sample.array_record"),
            Path("/content/PrxteinMPNN/data/pdb_sample.array_record"),
        ]
        
        for p in possible_paths:
            if p.exists():
                logger.info(f"✓ Found example data at: {p}")
                index_path = p.with_suffix(".index.json")
                return p, index_path
        
        # If not found, try to clone the repo (for Colab)
        logger.info("Example data not found locally. Cloning repository...")
        import subprocess
        subprocess.run([
            "git", "clone", "--depth=1", 
            "https://github.com/maraxen/PrxteinMPNN.git", 
            "/content/PrxteinMPNN"
        ], check=True)
        p = Path("/content/PrxteinMPNN/data/pdb_sample.array_record")
        logger.info(f"✓ Cloned repo, using data at: {p}")
        return p, p.with_suffix(".index.json")
    
    else:
        logger.error("use_example_data=False is no longer supported with remote HuggingFace hosting.")
        raise ValueError("Please provide a local ArrayRecord dataset path instead of fetching remotely.")


def main():
    """Main training entry point."""
    from prxteinmpnn.training.specs import TrainingSpecification
    from prxteinmpnn.training.trainer import train
    
    # Load data
    array_record_path, index_path = get_data_paths(use_example_data, "")
    
    logger.info(f"Data configuration:")
    logger.info(f"  ArrayRecord: {array_record_path}")
    logger.info(f"  Index: {index_path}")
    logger.info(f"  File size: {array_record_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Build training specification
    spec = TrainingSpecification(
        # Data
        inputs=str(array_record_path),
        use_preprocessed=True,
        preprocessed_index_path=str(index_path) if index_path.exists() else None,
        max_length=max_length,
        
        # Training mode
        training_mode=training_mode,
        
        # Physics features
        use_electrostatics=use_electrostatics,
        use_vdw=use_vdw,
        
        # Optimization
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=base_batch_size,
        accum_steps=accum_steps,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
        gradient_clip=gradient_clip if gradient_clip > 0 else None,
        label_smoothing=label_smoothing,
        
        # Precision
        precision=precision,
        
        # Diffusion settings (only used if training_mode="diffusion")
        diffusion_num_steps=diffusion_num_steps,
        diffusion_schedule_type=diffusion_schedule_type,
        diffusion_beta_start=diffusion_beta_start,
        diffusion_beta_end=diffusion_beta_end,
        
        # Checkpointing
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        keep_last_n_checkpoints=keep_last_n_checkpoints,
        
        # Early stopping
        early_stopping_patience=early_stopping_patience if use_early_stopping else None,
        early_stopping_metric=early_stopping_metric,
        
        # Logging
        log_every=log_every,
        eval_every=eval_every,
    )
    
    # Print configuration summary
    print("\nTraining Specification:")
    print("=" * 50)
    print(f"  Mode: {spec.training_mode}")
    print(f"  Batch size: {spec.batch_size} (x{spec.accum_steps} accum = {spec.batch_size * spec.accum_steps} effective)")
    print(f"  Learning rate: {spec.learning_rate}")
    print(f"  Epochs: {spec.num_epochs}")
    print(f"  Max length: {spec.max_length}")
    print(f"  Precision: {spec.precision}")
    print(f"  Physics: EStat={spec.use_electrostatics}, VdW={spec.use_vdw}")
    if spec.training_mode == "diffusion":
        print(f"  Diffusion steps: {spec.diffusion_num_steps}")
        print(f"  Schedule: {spec.diffusion_schedule_type}")
    print("=" * 50)
    
    # Run training
    print("\nStarting training...")
    print(f"Checkpoint directory: {spec.checkpoint_dir}")
    print()
    
    result = train(spec)
    
    print()
    print("=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Final step: {result.final_step}")
    print(f"Checkpoints saved to: {result.checkpoint_dir}")


if __name__ == "__main__":
    main()
