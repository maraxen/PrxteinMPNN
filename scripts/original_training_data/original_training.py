"""Original Training Script for PrxteinMPNN using Preprocessed Data."""

import traceback
import jax.random
from pathlib import Path
import equinox as eqx
from prxteinmpnn.training.specs import TrainingSpecification
from prxteinmpnn.training.trainer import train

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
spec = TrainingSpecification(
    # Set this flag to True to use the array_record data loader
    use_preprocessed=True,  #

    # Path to the main preprocessed data file (.array_record)
    inputs=str(TRAINING_ARRAY_RECORD_PATH),

    # Path to the index file (.index.json) which maps to the data
    preprocessed_index_path=TRAINING_INDEX_PATH,  #
    
    validation_data=str(TRAINING_ARRAY_RECORD_PATH),
    validation_preprocessed_index_path=TRAINING_INDEX_PATH,

    # --- Other required/recommended parameters ---
    batch_size=16,  # 128 first 50k steps, then 64 with 384 len until 100k steps, then 32 with 512 len until 150k, then 16 with 1024 len
    num_epochs=500,
    learning_rate=5e-4, # decreased from 1e-3 to 5e-4 at 200k steps
    checkpoint_dir=CHECKPOINT_DIR,
    total_steps=1e9,
    resume_from_checkpoint=True,
    
    
    # Use a default model version (e.g., from the README) and weights
    model_weights=None,
    model_version="v_48_020",
    # A random seed is recommended for reproducibility
    random_seed=42,
    
    eval_every=50000,
    log_every=1000,
    checkpoint_every=200000, # changed to 200k at 200k steps
    
    max_length=1024, # changed to 384 at 50k steps, then to 512 at 100k steps
    truncation_strategy="random_crop",
    use_vdw=False,
    use_electrostatics=False,
    backbone_noise=(0.2,),

)

# --- Run the Training Loop ---
print(f"Starting training with inputs: {spec.inputs}")
print(f"Using index file: {spec.preprocessed_index_path}")
print(f"Checkpoint directory: {spec.checkpoint_dir}")

try:
    results = train(spec)
    print(f"\nTraining completed successfully! Final step: {results.final_step}")
    

except Exception as e:
    print(f"\nAn error occurred during training: {e}")
    traceback.print_exc()
finally:
    print("Training process has ended.")
    final_path = CHECKPOINT_DIR / "final_model.eqx"
    eqx.tree_serialise_leaves(final_path, results.final_model)
    print(f"Final model weights saved to {final_path}")

# --- Cleanup (Optional) ---
# import shutil
# if CHECKPOINT_DIR.exists():
#     shutil.rmtree(CHECKPOINT_DIR)
#     print(f"Cleaned up checkpoint directory: {CHECKPOINT_DIR}")