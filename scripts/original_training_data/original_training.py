"""Original Training Script for PrxteinMPNN using Preprocessed Data."""

import traceback
from pathlib import Path
import equinox as eqx
from prxteinmpnn.training.specs import TrainingSpecification
from prxteinmpnn.training.trainer import train
import jax

jax.config.update("jax_default_matmul_precision", "bfloat16")
# --- Define Paths to Preprocessed Data ---
# The inputs field is used for the main data file (.array_record)
TRAINING_ARRAY_RECORD_PATH = Path(
    "/workspace/src/prxteinmpnn/training/data/pdb_2021aug02.array_record"
)
# The preprocessed_index_path is used for the corresponding index file (.index.json)
TRAINING_INDEX_PATH = Path(
    "/workspace/src/prxteinmpnn/training/data/pdb_2021aug02.index.json"
)

# --- Define Checkpoint Directory ---
# This directory must exist for checkpointing (trainer.py handles creation in _init_checkpoint_and_model)
CHECKPOINT_DIR = Path("/workspace/prxteinmpnn_scratch_training_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# --- Create Training Specification ---
# PLATEAU BREAKTHROUGH STRATEGY:
# 1. Reduced learning rate (1e-3 → 3e-4) to escape local minima
# 2. Added weight decay (0.01) for better generalization
# 3. Gradient clipping (1.0) for training stability
# 4. Multi-level backbone noise for data augmentation
# 5. Label smoothing (0.1) to prevent overconfidence
# 6. Larger proteins (max_length=1024) for more complex patterns
# 7. Gradient accumulation (16 steps) to fit batch_size=512 in memory

spec = TrainingSpecification(
    # Set this flag to True to use the array_record data loader
    use_preprocessed=True,  #

    # Path to the main preprocessed data file (.array_record)
    inputs=str(TRAINING_ARRAY_RECORD_PATH),

    # Path to the index file (.index.json) which maps to the data
    preprocessed_index_path=TRAINING_INDEX_PATH,  #
    
    validation_data=str(TRAINING_ARRAY_RECORD_PATH),
    validation_preprocessed_index_path=TRAINING_INDEX_PATH,

    # --- Training from Scratch - Optimized Hyperparameters ---
    batch_size=512,  # Effective batch size
    num_epochs=300,
    
    # LEARNING RATE: Standard for transformer training from scratch
    learning_rate=5e-4,  # Sweet spot between 1e-3 (too high) and 3e-4 (too conservative)
    
    # OPTIMIZER: Regularization for better generalization
    weight_decay=0.01,  # L2 regularization - standard for AdamW
    
    # WARMUP: Critical for stable training from scratch
    warmup_steps=10000,  # ~0.6% of total steps (standard: 1-5% for large models)
    
    # GRADIENT CLIPPING: Prevents training instability
    gradient_clip=1.0,  # Standard clipping threshold
    
    checkpoint_dir=CHECKPOINT_DIR,
    total_steps=375_001,  # 187500 * 2 + 1
    
    # CHECKPOINT NOTE: The saved final_model.eqx has a different architecture
    # Starting fresh with the new configuration (lower LR, augmentation, etc.)
    resume_from_checkpoint=False,
    # To resume from final_model.eqx (if architectures match):
    # resume_from_checkpoint=CHECKPOINT_DIR / "final_model.eqx",
    # Or to resume from the latest Orbax checkpoint:
    # resume_from_checkpoint=CHECKPOINT_DIR,
    
    # Use a default model version (e.g., from the README) and weights
    model_weights=None,
    model_version="v_48_020",
    # A random seed is recommended for reproducibility
    random_seed=42,
    
    eval_every=50000,
    log_every=1000,
    checkpoint_every=20000,
    
    # GRADIENT ACCUMULATION: Splits batch into micro-batches for memory efficiency
    # Effective batch = 512, split into 16 micro-batches of 32 each
    accum_steps=16,
    
    # SEQUENCE LENGTH: Longer sequences = more complex structural patterns
    max_length=512,  # Balanced between 384 (too short) and 1024 (memory intensive)
    truncation_strategy="random_crop",
    
    # DATA AUGMENTATION: Backbone coordinate noise (Å)
    backbone_noise=(0.2,),  # Single noise level; trainer currently uses the first value
    
    # LABEL SMOOTHING: Prevents overconfidence, improves generalization
    label_smoothing=0.1,  # Standard 10% smoothing
    
    use_vdw=False,
    use_electrostatics=False,
    precision="bf16",

)

# --- Run the Training Loop ---
print(f"Starting training with inputs: {spec.inputs}")
print(f"Using index file: {spec.preprocessed_index_path}")
print(f"Checkpoint directory: {spec.checkpoint_dir}")

try:
    results = train(spec)
    print(f"\nTraining completed successfully! Final step: {results.final_step}")
    final_path = CHECKPOINT_DIR / "final_model.eqx"
    eqx.tree_serialise_leaves(final_path, results.final_model)
    print(f"Final model weights saved to {final_path}")

except Exception as e:
    print(f"\nAn error occurred during training: {e}")
    traceback.print_exc()
finally:
    print("Training process has ended.")


# --- Cleanup (Optional) ---
# import shutil
# if CHECKPOINT_DIR.exists():
#     shutil.rmtree(CHECKPOINT_DIR)
#     print(f"Cleaned up checkpoint directory: {CHECKPOINT_DIR}")