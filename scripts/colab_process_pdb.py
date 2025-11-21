# @title Setup & Install
# @markdown Run this cell to install dependencies and setup the environment.

import logging
import os
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(command):
  try:
    subprocess.run(command, check=True, shell=True)
  except subprocess.CalledProcessError as e:
    logger.error(f"Command failed: {command}\n{e}")
    sys.exit(1)


# Install dependencies if not already installed (checking for a key package like jax)
try:
  import jax
except ImportError:
  run_command(
    "pip install -q array_record huggingface_hub dm-haiku optax jax jaxlib chex equinox msgpack msgpack-numpy",
  )

# Clone the repository to get the latest code
if not os.path.exists("PrxteinMPNN"):
  run_command("git clone https://github.com/maraxen/PrxteinMPNN.git")

# Add src to path
sys.path.append(os.path.abspath("PrxteinMPNN/src"))

# Now imports should work if dependencies are installed
import msgpack_numpy as m
from huggingface_hub import HfApi, login

# Import from the cloned repo
try:
  from prxteinmpnn.io.parsing.mappings import string_to_protein_sequence
  from prxteinmpnn.physics.force_fields import load_force_field_from_hub
  from prxteinmpnn.training.data.download_and_process import (
    convert_to_array_record,
    download_data,
    extract_data,
    parse_pt_file,
  )
  from prxteinmpnn.utils.data_structures import ProteinTuple
  from prxteinmpnn.utils.residue_constants import (
    atom_types,
    restype_1to3,
    restypes,
    van_der_waals_radius,
  )
except ImportError:
  # If running locally where src is already in path, this might be redundant or fail if not cloned.
  # Assuming the script might be run in an environment where prxteinmpnn is installed or in path.
  from prxteinmpnn.training.data.download_and_process import (
    convert_to_array_record,
    extract_data,
  )

m.patch()

# @title Configuration
OUTPUT_DIR = Path("/content/data")
HF_REPO_ID = "maraxen/pdb_2021aug02_array_record"  # @param {type:"string"}

# @title Authenticate
# @markdown Ensure you have set 'HF_TOKEN' in Colab secrets.
try:
  from google.colab import userdata

  try:
    hf_token = userdata.get("HF_TOKEN")
    login(token=hf_token)
    print("Logged in to HuggingFace.")
  except Exception as e:
    print(f"Could not login: {e}")
    print("Please ensure 'HF_TOKEN' is set in Colab secrets.")
except ImportError:
  print("Not running in Google Colab. Skipping Colab-specific authentication.")
  # Assuming user might be logged in via CLI or environment variable

# @title Process Data
# @markdown This cell downloads, extracts, processes, and uploads the data.


def main():
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  # 1. Download
  # Note: The full dataset URL might be different from the sample.
  # Using the URL provided in the prompt:
  FULL_PDB_URL = "https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz"

  tar_path = OUTPUT_DIR / "pdb_2021aug02.tar.gz"
  if not tar_path.exists():
    logger.info(f"Downloading {FULL_PDB_URL}...")
    subprocess.run(["wget", FULL_PDB_URL, "-P", str(OUTPUT_DIR)], check=True)

  # 2. Extract
  extract_dir = OUTPUT_DIR / "pdb_2021aug02"
  extract_dir.mkdir(exist_ok=True)
  # Only extract if looks empty
  if not list(extract_dir.iterdir()):
    extract_data(tar_path, extract_dir)

  # 3. Find .pt files
  pt_files = list(extract_dir.rglob("*.pt"))
  logger.info(f"Found {len(pt_files)} .pt files.")

  if not pt_files:
    logger.warning("No .pt files found.")
    return

  # 4. Convert
  output_record = OUTPUT_DIR / "pdb_2021aug02.array_record"
  index_record = OUTPUT_DIR / "pdb_2021aug02.index.json"

  # Use the imported function which should now have the fix if we pulled the latest code.
  # If the repo on GitHub doesn't have the fix yet, we might need to patch it here dynamically
  # or define the function inline.
  # For now, assuming the user will push the fix before running this.
  convert_to_array_record(pt_files, output_record, index_record)

  # 5. Upload
  logger.info(f"Uploading to {HF_REPO_ID}...")
  api = HfApi()
  api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

  api.upload_file(
    path_or_fileobj=output_record,
    path_in_repo="data.array_record",
    repo_id=HF_REPO_ID,
    repo_type="dataset",
  )

  api.upload_file(
    path_or_fileobj=index_record, path_in_repo="index.json", repo_id=HF_REPO_ID, repo_type="dataset",
  )
  logger.info("Upload complete!")


if __name__ == "__main__":
  main()
