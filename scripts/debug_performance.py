import time
import sys
from pathlib import Path
import grain.python as grain
import jax
from prxteinmpnn.io.loaders import create_protein_dataset
import logging
from absl import logging as absl_logging

# 1. Force ABSL (used by Grain) to show INFO logs (where the table lives)
absl_logging.set_verbosity(absl_logging.INFO)
absl_logging.get_absl_handler().python_handler.stream = sys.stdout

# 2. Configure standard python logging as well
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# 1. Enable Grain Debug Mode to visualize bottlenecks
grain.config.update("py_debug_mode", True)

# --- CONFIGURATION ---
# --- Define Paths to Preprocessed Data ---
# The inputs field is used for the main data file (.array_record)
TRAINING_ARRAY_RECORD_PATH = Path(
    "/workspace/src/prxteinmpnn/training/data/pdb_2021aug02.array_record"
)
# The preprocessed_index_path is used for the corresponding index file (.index.json)
TRAINING_INDEX_PATH = Path(
    "/workspace/src/prxteinmpnn/training/data/pdb_2021aug02.index.json"
)

USE_PREPROCESSED = True
BATCH_SIZE = 128
# ---------------------

def run_diagnosis():
    print(f"Initializing dataset from: {TRAINING_ARRAY_RECORD_PATH}")
    
    # Initialize the dataset exactly as the trainer does
    ds = create_protein_dataset(
        inputs=TRAINING_ARRAY_RECORD_PATH,
        batch_size=BATCH_SIZE,
        use_preprocessed=USE_PREPROCESSED,
        preprocessed_index_path=TRAINING_INDEX_PATH if USE_PREPROCESSED else None,
        # Physics features disabled as requested
        use_electrostatics=False, 
        use_vdw=False,
	truncation_strategy="random_crop",
	max_length=256,
    )

    print("Starting iteration loop (running for ~1 minute or 50 batches)...")
    it = iter(ds)
    
    start_time = time.time()
    try:
        # Run enough steps to let Grain gather stable statistics
        for i in range(200):
            _ = next(it)
            if i % 10 == 0:
                print(f"Processed batch {i}...")
    except StopIteration:
        print("Reached end of dataset.")
    except Exception as e:
        print(f"Error during iteration: {e}")

    total_time = time.time() - start_time
    print(f"\nFinished. Total time: {total_time:.2f}s")
    print("Check the 'Grain Dataset Execution Summary' table printed above.")

if __name__ == "__main__":
    # Prevent JAX from preallocating all GPU memory during this CPU test
    jax.config.update("jax_platform_name", "cpu")
    run_diagnosis()
