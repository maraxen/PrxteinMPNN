"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from io import StringIO

  from grain.python import IterDataset

  from prxteinmpnn.run.specs import Specs
  from prxteinmpnn.utils.types import Model

import equinox as eqx
from proxide.ops.dataset import create_protein_dataset

from prxteinmpnn.io.weights import LIGAND_DEFAULT_CHECKPOINT, load_ligand_model, load_model


def _loader_inputs(inputs: Sequence[str | StringIO] | str | StringIO) -> Sequence[str | StringIO]:
  return (inputs,) if not isinstance(inputs, Sequence) else inputs  # type: ignore[invalid-return-type]


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    while True:
      chunk = handle.read(1 << 20)
      if not chunk:
        break
      digest.update(chunk)
  return digest.hexdigest()


def _resolve_registry_model_path(spec: Specs) -> str | None:
  if spec.model_local_path is not None:
    return str(spec.model_local_path)
  if spec.checkpoint_registry_path is None:
    return None

  registry_path = Path(spec.checkpoint_registry_path)
  if not registry_path.exists():
    msg = f"Checkpoint registry file not found: {registry_path}"
    raise FileNotFoundError(msg)

  payload = json.loads(registry_path.read_text(encoding="utf-8"))
  entries = payload.get("entries")
  if not isinstance(entries, list):
    msg = "Checkpoint registry must contain an 'entries' list."
    raise ValueError(msg)

  requested_checkpoint = spec.checkpoint_id
  if requested_checkpoint is None and spec.model_family == "ligandmpnn":
    requested_checkpoint = LIGAND_DEFAULT_CHECKPOINT
  if requested_checkpoint is None:
    requested_checkpoint = f"{spec.model_weights}_{spec.model_version}"

  match: dict | None = None
  for entry in entries:
    if not isinstance(entry, dict):
      continue
    if entry.get("model_family") != spec.model_family:
      continue
    if entry.get("checkpoint_id") != requested_checkpoint:
      continue
    match = entry
    break

  if match is None:
    msg = (
      "No checkpoint registry entry found for "
      f"model_family={spec.model_family!r}, checkpoint_id={requested_checkpoint!r}."
    )
    raise ValueError(msg)

  artifact_path_raw = match.get("artifact_path")
  if not isinstance(artifact_path_raw, str) or not artifact_path_raw:
    msg = "Checkpoint registry entry must include non-empty 'artifact_path'."
    raise ValueError(msg)

  artifact_path = Path(artifact_path_raw)
  if not artifact_path.is_absolute():
    artifact_path = registry_path.parent / artifact_path
  artifact_path = artifact_path.resolve()
  if not artifact_path.exists():
    msg = f"Checkpoint artifact not found: {artifact_path}"
    raise FileNotFoundError(msg)

  expected_sha = match.get("sha256")
  if expected_sha is not None:
    if not isinstance(expected_sha, str):
      msg = "Checkpoint registry entry field 'sha256' must be a string when provided."
      raise ValueError(msg)
    observed_sha = _sha256_file(artifact_path)
    if observed_sha.lower() != expected_sha.lower():
      msg = (
        f"Checkpoint checksum mismatch for {artifact_path}: "
        f"expected {expected_sha}, observed {observed_sha}."
      )
      raise ValueError(msg)

  return str(artifact_path)


def prep_protein_stream_and_model(
  spec: Specs,
) -> tuple[IterDataset, Model]:
  """Prepare the protein data stream and model parameters.

  Args:
      spec: A RunSpecification object containing configuration options.

  Returns:
      A tuple containing the protein data iterator and model in inference mode.

  """
  parse_kwargs = {
    "chain_id": spec.chain_id,
    "model": spec.model,
    "altloc": spec.altloc,
    "topology": spec.topology,
  }
  protein_iterator = create_protein_dataset(
    _loader_inputs(spec.inputs),
    batch_size=spec.batch_size,
    foldcomp_database=spec.foldcomp_database,
    parse_kwargs=parse_kwargs,
    pass_mode=spec.pass_mode,
    use_preprocessed=spec.use_preprocessed,
    preprocessed_index_path=spec.preprocessed_index_path,
    split=spec.split,
    use_electrostatics=spec.use_electrostatics,
    estat_noise=spec.estat_noise,
    estat_noise_mode=spec.estat_noise_mode,
    use_vdw=spec.use_vdw,
    vdw_noise=spec.vdw_noise,
    vdw_noise_mode=spec.vdw_noise_mode,
    max_length=spec.max_length,
    truncation_strategy=spec.truncation_strategy,
  )
  local_model_path = _resolve_registry_model_path(spec)
  if spec.model_family == "ligandmpnn":
    model = load_ligand_model(
      checkpoint_id=spec.checkpoint_id or LIGAND_DEFAULT_CHECKPOINT,
      local_path=local_model_path,
      ligand_mpnn_use_side_chain_context=spec.ligand_mpnn_use_side_chain_context,
    )
  else:
    model = load_model(
      model_version=spec.model_version,
      model_weights=spec.model_weights,
      local_path=local_model_path,
    )

  # Set model to inference mode (disables dropout, etc.)
  model = eqx.nn.inference_mode(model, value=True)

  return protein_iterator, model
