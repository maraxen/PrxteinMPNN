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
from array_record.python.array_record_module import (  # type: ignore[unresolved-import]
  ArrayRecordReader,
  ArrayRecordWriter,
)

from prxteinmpnn.io.parsing.biotite import processed_structure_to_protein_tuples
from prxteinmpnn.io.parsing.pqr import parse_pqr_to_processed_structure
from proxide.physics.features import (
  compute_electrostatic_node_features,
  compute_vdw_node_features,
)
from proxide.physics.force_fields import load_force_field

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
  force_field: str = "ff14SB"
  resume_from_checkpoint: bool = True
  compute_lj: bool = True
  compute_estat: bool = True
  compression: str = "zstd:9"
  validate_features: bool = True
  group_size: int = 1
  metadata_file: Path = field(init=False)
  index_file: Path = field(init=False)
  temp_dir: Path = field(init=False)
  estat_noise: float | Sequence[float] | None = None
  estat_noise_mode: Literal["direct", "thermal"] = "direct"
  vdw_noise: float | Sequence[float] | None = None
  vdw_noise_mode: Literal["direct", "thermal"] = "direct"

  def __post_init__(self) -> None:
    """Set dependent paths."""
    object.__setattr__(self, "metadata_file", self.output_file.with_suffix(".jsonl"))
    object.__setattr__(self, "index_file", self.output_file.with_suffix(".index"))
    object.__setattr__(self, "temp_dir", self.output_file.parent / f"tmp_{self.output_file.stem}")


def _worker_process_protein(
  args: tuple[Path, PreprocessingSpecification, Path, dict[str, Any]],
) -> tuple[str, Path | None]:
  """Worker function to process a single protein structure.

  Args:
      args: Tuple containing (pqr_path, spec, temp_dir_path, force_field_data).

  Returns:
      A tuple of (protein_id, shard_path). shard_path is None if processing failed.

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
    feats = []

    if spec.compute_lj:
      v_noise = spec.vdw_noise
      if isinstance(v_noise, (list, tuple)):
        v_noise = v_noise[0] if v_noise else 0.0
      elif v_noise is None:
        v_noise = 0.0

      v_feat = compute_vdw_node_features(
        protein_tuple,
        noise_scale=v_noise,
        noise_mode=spec.vdw_noise_mode,
      )
      feats.append(v_feat)

    if spec.compute_estat:
      e_noise = spec.estat_noise
      if isinstance(e_noise, (list, tuple)):
        e_noise = e_noise[0] if e_noise else 0.0
      elif e_noise is None:
        e_noise = 0.0

      e_feat = compute_electrostatic_node_features(
        protein_tuple,
        noise_scale=e_noise,
        noise_mode=spec.estat_noise_mode,
      )
      feats.append(e_feat)

    if not feats:
      msg = "No physics features requested in preprocessing"
      raise ValueError(msg)

    physics_features = jnp.concatenate(feats, axis=-1)

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
      "physics_features": np.array(physics_features),  # (N, 5) or (N, 10)
      # Full atomic data (for verification/debugging)
      "full_coordinates": np.array(protein_tuple.full_coordinates),  # (n_atoms, 3)
      "charges": np.array(protein_tuple.charges),  # (n_atoms,)
      "radii": np.array(protein_tuple.radii),  # (n_atoms,)
      # Estat metadata
      "estat_backbone_mask": (
        np.array(protein_tuple.estat_backbone_mask)
        if protein_tuple.estat_backbone_mask is not None
        else np.zeros(
          len(protein_tuple.full_coordinates) if protein_tuple.full_coordinates is not None else 0,
          dtype=bool,
        )
      ),
      "estat_resid": (
        np.array(protein_tuple.estat_resid)
        if protein_tuple.estat_resid is not None
        else np.zeros(
          len(protein_tuple.full_coordinates) if protein_tuple.full_coordinates is not None else 0,
          dtype=np.int32,
        )
      ),
      "estat_chain_index": (
        np.array(protein_tuple.estat_chain_index)
        if protein_tuple.estat_chain_index is not None
        else np.zeros(
          len(protein_tuple.full_coordinates) if protein_tuple.full_coordinates is not None else 0,
          dtype=np.int32,
        )
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
      physics_mag = physics_features[:, -1]  # Force magnitude (last col of last feat block)
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
    writer.write(msgpack.packb(record_data))
    writer.close()

    return (protein_id, shard_path)

  except Exception:
    logger.exception("Failed to process %s", pqr_path)
    return (protein_id, None)


def _load_checkpoint_metadata(metadata_file: Path) -> dict[str, Any]:
  """Load existing progress from the metadata JSONL file."""
  if not metadata_file.exists():
    return {"processed_files": set(), "failed_files": set(), "total_records": 0}

  processed = set()
  failed = set()
  count = 0

  with metadata_file.open("r") as f:
    for line in f:
      if not line.strip():
        continue
      try:
        entry = json.loads(line)
        p_id = entry["id"]
        if entry["status"] == "success":
          processed.add(p_id)
          count += 1
        else:
          failed.add(p_id)
      except (json.JSONDecodeError, KeyError):
        logger.warning("Skipping invalid metadata line: %s", line.strip())
        continue

  return {"processed_files": processed, "failed_files": failed, "total_records": count}


def _append_metadata(metadata_file: Path, protein_id: str, status: str) -> None:
  """Append a single result entry to the metadata file."""
  with metadata_file.open("a") as f:
    f.write(json.dumps({"id": protein_id, "status": status}) + "\n")


def _filter_files(
  pqr_files: list[Path],
  metadata_file: Path,
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
    "success_count": total_success,
    "failure_count": total_failed,
  }


def run_preprocessing_pipeline(spec: PreprocessingSpecification) -> dict[str, Any]:
  """Execute the full preprocessing pipeline.

  Args:
      spec: PreprocessingSpecification object.

  Returns:
      A dictionary with execution statistics and paths.

  """
  logger.info("Starting PrxteinMPNN preprocessing pipeline")
  logger.info("Input dir: %s", spec.input_dir)
  logger.info("Output file: %s", spec.output_file)

  # 1. Discover files
  pqr_files = sorted(spec.input_dir.glob("*.pqr"))
  if not pqr_files:
    msg = f"No PQR files found in {spec.input_dir}"
    raise FileNotFoundError(msg)

  # 2. Filter based on checkpoint
  files_to_process, initial_processed, initial_failed = _filter_files(
    pqr_files,
    spec.metadata_file,
    spec.resume_from_checkpoint,
  )

  if not files_to_process:
    logger.info("All files already processed according to checkpoint. Exiting.")
    return _log_summary_and_get_results(spec, initial_processed, initial_failed)

  # 3. Load force field (once to avoid redundant downloads/parsing in workers)
  logger.info("Loading force field: %s", spec.force_field)
  ff_obj = load_force_field(spec.force_field)
  # Convert to a dict structure that's easily picklable for workers
  # Realistically, workers will use pqr info for charges/radii, but we might
  # need some ff data in the future.
  ff_data = {"name": spec.force_field, "temp_marker": True}

  # 4. Run scatter phase (parallel processing)
  shard_paths = _run_scatter_phase(files_to_process, spec, ff_data)

  if not shard_paths:
    logger.error("No shards were successfully created. Exiting.")
    return _log_summary_and_get_results(spec, initial_processed, initial_failed)

  # 5. Run gather phase (merge shards into final array_record)
  logger.info("Starting gather phase: merging %d shards", len(shard_paths))
  writer = ArrayRecordWriter(str(spec.output_file), spec.compression)

  for shard_path in tqdm.tqdm(shard_paths, desc="Merging shards"):
    reader = ArrayRecordReader(str(shard_path))
    for record in reader:
      writer.write(record)
      # Log success to metadata
      record_data = msgpack.unpackb(record)
      _append_metadata(spec.metadata_file, record_data["protein_id"], "success")

  writer.close()

  # 6. Cleanup
  _cleanup_temp_files(shard_paths, spec.temp_dir)

  # 7. Final summary
  return _log_summary_and_get_results(spec, initial_processed, initial_failed)
