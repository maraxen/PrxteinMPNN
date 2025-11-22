"""Preprocessing pipeline for PQR files to array_record with physics features.

This module implements efficient parallel preprocessing with:
- Scatter-gather parallelization (workers write temporary shards)
- Resumable processing (JSONL metadata tracking)
- Progress monitoring
- Feature validation
- Index building

Based on the md_cath dataset preprocessing patterns.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
  from collections.abc import Sequence

import jax.numpy as jnp
import msgpack
import msgpack_numpy as m
import numpy as np
import tqdm
from array_record.python.array_record_module import (
  ArrayRecordReader,
  ArrayRecordWriter,
)

from prxteinmpnn.io.parsing.biotite import processed_structure_to_protein_tuples
from prxteinmpnn.io.parsing.pqr import parse_pqr_to_processed_structure
from prxteinmpnn.physics.features import compute_electrostatic_node_features
from prxteinmpnn.physics.force_fields import load_force_field_from_hub

# Patch msgpack to handle numpy arrays
m.patch()

logger = logging.getLogger(__name__)
FORCE_FIELD_REPO: str = "maraxen/eqx-ff"


@dataclass(frozen=True)
class PreprocessingSpecification:
  """Specifies parameters for the preprocessing pipeline."""

  input_dir: Path
  output_file: Path
  num_workers: int = 8
  force_field: str = "amber14-all"
  resume_from_checkpoint: bool = True
  compute_lj: bool = False  # Not implemented yet
  compression: str = "zstd:9"
  validate_features: bool = True
  group_size: int = 1

  estat_noise: Sequence[float] | float | None = None
  estat_noise_mode: Literal["direct", "temperature"] = "direct"
  vdw_noise: Sequence[float] | float | None = None
  vdw_noise_mode: Literal["direct", "temperature"] = "direct"

  # Derived paths
  index_file: Path = field(init=False)
  metadata_file: Path = field(init=False)
  temp_dir: Path = field(init=False)

  def __post_init__(self) -> None:
    """Initialize derived fields."""
    object.__setattr__(
      self,
      "index_file",
      self.output_file.with_suffix(".index.json"),
    )
    object.__setattr__(
      self,
      "metadata_file",
      self.output_file.with_suffix(".metadata.jsonl"),
    )
    object.__setattr__(
      self,
      "temp_dir",
      self.output_file.with_suffix(f"._temp_{uuid.uuid4().hex}"),
    )


def _worker_process_protein(args: tuple) -> tuple[str, Path | None]:
  """Worker function: Process one PQR file and write to temporary shard.

  Args:
      args: Tuple of (pqr_path, spec, temp_dir_path, force_field)

  Returns:
      Tuple of (protein_id, Path_to_temp_shard | None)
      Returns None for path if processing failed.

  """
  pqr_path, spec, temp_dir_path, _force_field_data = args
  protein_id = pqr_path.stem

  try:
    # Use new pipeline: parse PQR directly to ProcessedStructure
    processed_structure = parse_pqr_to_processed_structure(pqr_path, chain_id=None)

    # Convert to ProteinTuple
    protein_generator = processed_structure_to_protein_tuples(
      processed_structure,
      source_name=str(pqr_path),
      extract_dihedrals=False,
      populate_physics=False,  # PQR already has physics params
    )
    protein_tuple = next(protein_generator)
    assert protein_tuple.full_coordinates is not None  # noqa: S101

    # Compute physics features
    estat_val = spec.estat_noise
    if isinstance(estat_val, (list, tuple)):
      estat_val = estat_val[0] if estat_val else 0.0
    elif estat_val is None:
      estat_val = 0.0

    physics_features = compute_electrostatic_node_features(
      protein_tuple,
      noise_scale=estat_val,
      noise_mode=spec.estat_noise_mode,
    )

    record_data = {
      # Basic structure info
      "protein_id": protein_id,
      "source_file": str(pqr_path),
      # Coordinates and sequence
      "coordinates": np.array(protein_tuple.coordinates),  # (N, 37, 3) backbone
      "aatype": np.array(protein_tuple.aatype),  # (N,)
      "atom_mask": np.array(protein_tuple.atom_mask),  # (N, 37)
      # Indexing
      "residue_index": np.array(protein_tuple.residue_index),  # (N,)
      "chain_index": np.array(protein_tuple.chain_index),  # (N,)
      "mask": np.array(protein_tuple.atom_mask[:, 1]),  # (N,) CA mask
      # Physics features (the main output)
      "physics_features": np.array(physics_features),  # (N, 5)
      # Full atomic data (for verification/debugging)
      "full_coordinates": np.array(protein_tuple.full_coordinates),  # (n_atoms, 3)
      "charges": np.array(protein_tuple.charges),  # (n_atoms,)
      "radii": np.array(protein_tuple.radii),  # (n_atoms,)
      # Estat metadata
      "estat_backbone_mask": (
        np.array(protein_tuple.estat_backbone_mask)
        if protein_tuple.estat_backbone_mask is not None
        else np.zeros(len(protein_tuple.full_coordinates), dtype=bool)
      ),
      "estat_resid": (
        np.array(protein_tuple.estat_resid)
        if protein_tuple.estat_resid is not None
        else np.zeros(len(protein_tuple.full_coordinates), dtype=np.int32)
      ),
      "estat_chain_index": (
        np.array(protein_tuple.estat_chain_index)
        if protein_tuple.estat_chain_index is not None
        else np.zeros(len(protein_tuple.full_coordinates), dtype=np.int32)
      ),
    }

    # Validate features if requested
    if spec.validate_features:
      for key, value in record_data.items():
        if isinstance(value, np.ndarray | jnp.ndarray) and not np.all(
          np.isfinite(value),
        ):
          logger.warning("Non-finite values in %s for %s", key, pqr_path)
          return (protein_id, None)

      # Check that physics features are reasonable
      physics_mag = physics_features[:, -1]  # Force magnitude
      max_reasonable_force = 1e6  # Use a named constant for clarity
      if np.any(physics_mag > max_reasonable_force):
        logger.warning(
          "Unusually large forces (max=%.2e) in %s",
          np.max(physics_mag),
          pqr_path,
        )

    shard_path = temp_dir_path / f"shard-{uuid.uuid4().hex}.array_record"
    writer = ArrayRecordWriter(
      str(shard_path),
      f"{spec.compression},group_size:{spec.group_size}",
    )

    try:
      writer.write(msgpack.packb(record_data, use_bin_type=True))
    finally:
      writer.close()

  except StopIteration:
    logger.exception("No frames found in %s", pqr_path)
    return (protein_id, None)
  except Exception:
    logger.exception("Failed to process %s", pqr_path)
    return (protein_id, None)
  else:
    return (protein_id, shard_path)


def _load_checkpoint_metadata(metadata_file: Path) -> dict[str, Any]:
  """Load processing checkpoint from JSONL metadata file."""
  if not metadata_file.exists():
    return {
      "processed_files": set(),
      "total_records": 0,
      "failed_files": set(),
    }

  processed_files = set()
  failed_files = set()
  total_records = 0

  try:
    with metadata_file.open("r") as f:
      for line in f:
        try:
          entry = json.loads(line.strip())
          if entry.get("status") == "success":
            processed_files.add(entry["protein_id"])
            total_records += 1
          elif entry.get("status") == "failed":
            failed_files.add(entry["protein_id"])
        except json.JSONDecodeError:
          logger.warning("Skipping malformed line in metadata file: %s", line)
  except OSError:
    logger.exception("Could not read metadata file %s", metadata_file)
    return {
      "processed_files": set(),
      "total_records": 0,
      "failed_files": set(),
    }

  return {
    "processed_files": processed_files,
    "total_records": total_records,
    "failed_files": failed_files,
  }


def _append_metadata(
  metadata_file: Path,
  protein_id: str,
  status: str,
  error_message: str | None = None,
) -> None:
  """Append entry to JSONL metadata file."""
  entry = {
    "protein_id": protein_id,
    "status": status,
    "timestamp": str(Path(__file__).stat().st_mtime),  # Placeholder
  }
  if error_message:
    entry["error"] = error_message

  try:
    with metadata_file.open("a") as f:
      f.write(json.dumps(entry) + "\n")
  except OSError:
    logger.exception("Could not append to metadata file %s", metadata_file)


def _merge_shards_to_final(
  shard_paths: list[Path],
  output_file: Path,
  metadata_file: Path,
  index_file: Path,
  compression: str,
  group_size: int,
) -> dict[str, int]:
  """Merge temporary shards into final array_record file."""
  logger.info("Merging %d shards into %s", len(shard_paths), output_file)

  final_writer = ArrayRecordWriter(
    str(output_file),
    f"{compression},group_size:{group_size}",
  )

  protein_index = {}
  global_record_index = 0

  try:
    for shard_path in tqdm.tqdm(sorted(shard_paths), desc="Merging shards"):
      if not shard_path.exists():
        logger.warning("Shard file not found, skipping: %s", shard_path)
        continue

      shard_reader = ArrayRecordReader(str(shard_path))

      try:
        num_shard_records = shard_reader.num_records()
        if num_shard_records > 0:
          records_from_shard = shard_reader.read(0, num_shard_records)

          for record_bytes in records_from_shard:
            # Write to final file
            final_writer.write(record_bytes)

            # Update index
            try:
              unpacked = msgpack.unpackb(record_bytes, raw=False)
              protein_id = unpacked["protein_id"]
              protein_index[protein_id] = global_record_index

              # Append to metadata
              _append_metadata(metadata_file, protein_id, "success")

              global_record_index += 1
            except (msgpack.exceptions.UnpackException, KeyError):
              logger.exception("Failed to read record from shard %s", shard_path)

      except Exception:
        logger.exception("Failed to read shard %s", shard_path)
      finally:
        shard_reader.close()

  finally:
    final_writer.close()

  # Write index to JSON
  logger.info("Writing index with %d entries to %s", len(protein_index), index_file)
  try:
    with index_file.open("w") as f:
      json.dump(protein_index, f, indent=2)
  except OSError:
    logger.exception("Could not write index file %s", index_file)

  return protein_index


def _log_spec(spec: PreprocessingSpecification) -> None:
  """Log the preprocessing specification."""
  logger.info("Starting preprocessing with spec:")
  logger.info("  Input: %s", spec.input_dir)
  logger.info("  Output: %s", spec.output_file)
  logger.info("  Index: %s", spec.index_file)
  logger.info("  Metadata: %s", spec.metadata_file)
  logger.info("  Temp dir: %s", spec.temp_dir)
  logger.info("  Workers: %d", spec.num_workers)


def _setup_multiprocessing() -> None:
  """Set the multiprocessing start method."""
  try:
    mp.set_start_method("spawn", force=True)
  except (RuntimeError, ValueError):
    logger.warning("Could not set multiprocessing start method to 'spawn'.")


def _load_and_serialize_force_field(
  force_field_name: str,
  repo_id: str,
) -> dict[str, Any] | None:
  """Load force field and serialize it for workers."""
  logger.info("Loading force field: %s", force_field_name)
  try:
    force_field = load_force_field_from_hub(
      force_field_name,
      repo_id=repo_id,
    )
    return {
      "charges": np.array(force_field.charges_by_id),
      "sigmas": np.array(force_field.sigmas_by_id),
      "epsilons": np.array(force_field.epsilons_by_id),
      "atom_key_to_id": force_field.atom_key_to_id,
    }
  except Exception:
    logger.exception("Failed to load force field. Aborting.")
    return None


def _discover_pqr_files(input_dir: Path) -> list[Path]:
  """Find all .pqr files in the input directory."""
  pqr_files = sorted(input_dir.glob("*.pqr"))
  logger.info("Found %d PQR files", len(pqr_files))
  return pqr_files


def _filter_files_by_checkpoint(
  pqr_files: list[Path],
  metadata_file: Path,
  *,
  resume: bool,
) -> tuple[list[Path], set[str], set[str]]:
  """Filter file list based on checkpoint metadata."""
  checkpoint_data = _load_checkpoint_metadata(metadata_file)
  processed_files = checkpoint_data["processed_files"]
  failed_files = checkpoint_data["failed_files"]
  files_to_process = pqr_files

  if resume and (processed_files or failed_files):
    logger.info(
      "Resuming from checkpoint: %d processed, %d failed",
      len(processed_files),
      len(failed_files),
    )
    files_to_process = [
      f for f in pqr_files if f.stem not in processed_files and f.stem not in failed_files
    ]
    logger.info("Remaining: %d files to process", len(files_to_process))

  return files_to_process, processed_files, failed_files


def _run_scatter_phase(
  files_to_process: list[Path],
  spec: PreprocessingSpecification,
  force_field_data: dict[str, Any],
) -> list[Path]:
  """Run the parallel processing 'scatter' phase.

  Returns:
      A list of paths to the generated shards.
      Logs failures directly to the metadata file.

  """
  logger.info("Starting scatter phase: parallel processing to temporary shards")
  spec.temp_dir.mkdir(parents=True, exist_ok=True)
  worker_args = [(pqr_file, spec, spec.temp_dir, force_field_data) for pqr_file in files_to_process]

  shard_paths = []

  if spec.num_workers == 0:
    logger.info("Running in single-process mode (num_workers=0)")
    for args in tqdm.tqdm(worker_args, desc="Processing proteins"):
      protein_id, shard_path = _worker_process_protein(args)
      if shard_path:
        shard_paths.append(shard_path)
      else:
        _append_metadata(spec.metadata_file, protein_id, "failed")
  else:
    with mp.Pool(processes=spec.num_workers) as pool:
      for protein_id, shard_path in tqdm.tqdm(
        pool.imap_unordered(_worker_process_protein, worker_args),
        total=len(worker_args),
        desc="Processing proteins",
      ):
        if shard_path:
          shard_paths.append(shard_path)
        else:
          _append_metadata(spec.metadata_file, protein_id, "failed")

  logger.info("Scatter phase complete: %d shards created", len(shard_paths))
  return shard_paths


def _cleanup_temp_files(shard_paths: list[Path], temp_dir: Path) -> None:
  """Remove temporary shard files and directory."""
  logger.info("Cleaning up temporary shards")
  try:
    for shard_path in shard_paths:
      shard_path.unlink(missing_ok=True)
    temp_dir.rmdir()
  except OSError as e:
    logger.warning("Could not clean up temp directory %s: %s", temp_dir, e)


def _log_summary_and_get_results(
  spec: PreprocessingSpecification,
  initial_processed: set[str],
  initial_failed: set[str],
) -> dict[str, Any]:
  """Reload metadata, log final summary, and return results dict."""
  final_stats = _load_checkpoint_metadata(spec.metadata_file)
  total_success = final_stats["total_records"]
  total_failed = len(final_stats["failed_files"])

  num_success_this_run = total_success - len(initial_processed)
  num_failed_this_run = total_failed - len(initial_failed)

  logger.info("=" * 60)
  logger.info("Preprocessing complete!")
  logger.info("  Successfully processed (this run): %d", num_success_this_run)
  logger.info("  Failed (this run): %d", num_failed_this_run)
  logger.info("  Total processed (all runs): %d", total_success)
  logger.info("  Total failed (all runs): %d", total_failed)
  logger.info("  Output file: %s", spec.output_file)
  logger.info("  Index file: %s", spec.index_file)
  logger.info("  Metadata file: %s", spec.metadata_file)
  logger.info("=" * 60)

  return {
    "output_file": spec.output_file,
    "index_file": spec.index_file,
    "metadata_file": spec.metadata_file,
    "num_proteins": total_success,
    "num_failed": total_failed,
  }


def preprocess_dataset(spec: PreprocessingSpecification) -> dict[str, Any]:
  """Run the main preprocessing pipeline.

  Implements efficient parallel preprocessing with:
  1. Load force field (once, shared across workers)
  2. Discover PQR files
  3. Check checkpoint (skip already-processed files)
  4. Scatter: Parallel processing to temporary shards
  5. Gather: Merge shards into final array_record
  6. Build index for fast lookup

  Args:
      spec: Preprocessing specification

  Returns:
      Dictionary with:
      - output_file: Path to generated array_record
      - index_file: Path to generated index
      - metadata_file: Path to metadata JSONL
      - num_proteins: Number of proteins processed
      - num_failed: Number of failed proteins

  """
  _log_spec(spec)
  _setup_multiprocessing()

  force_field_data = _load_and_serialize_force_field(
    spec.force_field,
    FORCE_FIELD_REPO,
  )
  if force_field_data is None:
    return {
      "output_file": spec.output_file,
      "index_file": spec.index_file,
      "metadata_file": spec.metadata_file,
      "num_proteins": 0,
      "num_failed": 0,
    }

  pqr_files = _discover_pqr_files(spec.input_dir)
  if not pqr_files:
    logger.warning("No PQR files found in %s. Exiting.", spec.input_dir)
    return {
      "output_file": spec.output_file,
      "index_file": spec.index_file,
      "metadata_file": spec.metadata_file,
      "num_proteins": 0,
      "num_failed": 0,
    }

  files_to_process, processed_files, failed_files = _filter_files_by_checkpoint(
    pqr_files,
    spec.metadata_file,
    resume=spec.resume_from_checkpoint,
  )

  if not files_to_process:
    logger.info("All files already processed!")
    return _log_summary_and_get_results(spec, processed_files, failed_files)

  shard_paths = _run_scatter_phase(files_to_process, spec, force_field_data)

  if not shard_paths:
    logger.error(
      "No shards were created! All %d submitted proteins failed processing.",
      len(files_to_process),
    )
    return _log_summary_and_get_results(spec, processed_files, failed_files)

  logger.info("Starting gather phase: merging shards into final array_record")
  _ = _merge_shards_to_final(
    shard_paths,
    spec.output_file,
    spec.metadata_file,
    spec.index_file,
    spec.compression,
    spec.group_size,
  )

  _cleanup_temp_files(shard_paths, spec.temp_dir)
  return _log_summary_and_get_results(spec, processed_files, failed_files)
