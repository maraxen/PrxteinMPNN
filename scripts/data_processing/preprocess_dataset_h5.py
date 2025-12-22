"""Preprocessing script to convert protein data into HDF5 format.

This script parses PDB or PQR files (or an existing dataset) and stores them
in an HDF5 file using `h5py`. This format supports per-worker file handles
and fast random access during training.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from prxteinmpnn.io.parsing.dispatch import parse_protein
from prxteinmpnn.physics.features import compute_electrostatic_node_features
from prxteinmpnn.utils.data_structures import ProteinTuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_single_file(
  file_path: Path,
  use_electrostatics: bool = False,
) -> dict[str, Any] | None:
  """Parse a single file and extract features.

  Args:
      file_path: Path to the protein file (PDB, PQR, etc.).
      use_electrostatics: Whether to compute electrostatic features.

  Returns:
      Dictionary of features or None if parsing fails.
  """
  try:
    protein = parse_protein(file_path)
    if protein is None:
      return None

    data = {
      "coordinates": protein.coordinates.astype(np.float32),
      "aatype": protein.aatype.astype(np.int8),
      "atom_mask": protein.atom_mask.astype(np.float32),
      "residue_index": protein.residue_index.astype(np.int32),
      "chain_index": protein.chain_index.astype(np.int32),
    }

    if use_electrostatics:
      # We need to ensure the protein tuple has the necessary fields for electrostatics
      # parse_protein should handle this if it's a PQR or similar.
      phys_feat = compute_electrostatic_node_features(protein)
      data["physics_features"] = phys_feat.astype(np.float32)

    # Add other optional fields if present
    for field in [
      "full_coordinates",
      "dihedrals",
      "mapping",
      "charges",
      "radii",
      "sigmas",
      "epsilons",
    ]:
      val = getattr(protein, field, None)
      if val is not None:
        data[field] = val.astype(np.float32) if val.dtype.kind == "f" else val

    return data
  except Exception as e:
    logger.warning("Failed to process %s: %s", file_path, e)
    return None


def main() -> None:
  """Main entry point for HDF5 preprocessing."""
  parser = argparse.ArgumentParser(description="Preprocess proteins into HDF5.")
  parser.get_output_parser = parser.add_argument  # type: ignore[attr-defined]
  parser.add_argument(
    "--input_dir", type=str, required=True, help="Directory containing protein files."
  )
  parser.add_argument("--output_file", type=str, required=True, help="Path to output HDF5 file.")
  parser.add_argument("--pattern", type=str, default="*.pdb", help="File pattern to match.")
  parser.add_argument(
    "--use_electrostatics", action="store_true", help="Compute electrostatic features."
  )
  parser.add_argument("--num_workers", type=int, default=-1, help="Number of parallel workers.")

  args = parser.parse_args()

  input_dir = Path(args.input_dir)
  files = sorted(list(input_dir.glob(args.pattern)))

  if not files:
    logger.error("No files found matching pattern %s in %s", args.pattern, args.input_dir)
    return

  logger.info("Found %d files. Starting parallel processing...", len(files))

  results = Parallel(n_jobs=args.num_workers)(
    delayed(process_single_file)(f, args.use_electrostatics) for f in tqdm(files, desc="Parsing")
  )

  # Filter out None results
  processed_data = [r for r in results if r is not None]
  logger.info("Successfully processed %d/%d files.", len(processed_data), len(files))

  if not processed_data:
    logger.error("No files were successfully processed.")
    return

  logger.info("Writing to HDF5: %s", args.output_file)
  with h5py.File(args.output_file, "w") as f:
    # We store each protein in a group, or use a structured dataset?
    # Groups are better for variable-length residues.
    grp = f.create_group("proteins")

    index = []
    current_idx = 0

    for i, data in enumerate(tqdm(processed_data, desc="Writing")):
      p_grp = grp.create_group(f"protein_{i}")
      for k, v in data.items():
        p_grp.create_dataset(k, data=v)

      # For fast range sampling, we might want to store metadata in a secondary dataset
      # But usually we just iterate over keys or use a mapping.
      index.append(f"protein_{i}")

    # Store the index as a dataset for easy access
    f.create_dataset("index", data=np.array(index, dtype=h5py.special_dtype(vlen=str)))

  logger.info("Preprocessing complete.")


if __name__ == "__main__":
  main()
