"""Download all PrxteinMPNN weights from Hugging Face Hub to local src/weights directory.

This script will download all .eqx weight files for all model versions and types
and place them in the src/weights directory for local use.
"""

import logging
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from prxteinmpnn.io.weights import (
  ALL_MODEL_VERSIONS,
  ALL_MODEL_WEIGHTS,
  load_weights,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
  """Download all PrxteinMPNN weights from Hugging Face Hub to local weights directory."""
  for model_weights in ALL_MODEL_WEIGHTS:
    for model_version in ALL_MODEL_VERSIONS:
      filename = f"{model_weights}_{model_version}.eqx"
      local_path = WEIGHTS_DIR / filename
      logger.info("Downloading %s ...", filename)
      # Download and save weights to local_path
      load_weights(
        model_version=model_version,
        model_weights=model_weights,
        local_path=None,  # force download from HF
        skeleton=None,
        use_eqx_format=True,
      )
      # Move downloaded file to local_path if not already there
      # (hf_hub_download caches to ~/.cache by default)
      # So we copy it to our weights dir for local use
      # But load_weights returns the loaded object, not the file path
      # So we must call hf_hub_download directly to get the file
      path = hf_hub_download(
        repo_id="maraxen/prxteinmpnn",
        filename=f"eqx/{filename}",
        repo_type="model",
      )
      if not local_path.exists():
        shutil.copy2(path, local_path)
      logger.info("Saved to %s", local_path)


if __name__ == "__main__":
  main()
