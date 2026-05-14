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
  from prxteinmpnn.types.arrays import Model

import equinox as eqx
from proxide.ops.dataset import create_protein_dataset

from prxteinmpnn.io.weights import load_model
from prxteinmpnn.run.resources import proxide_dataset_resource_kwargs


def _loader_inputs(inputs: Sequence[str | StringIO] | str | StringIO) -> Sequence[str | StringIO]:
  return (inputs,) if not isinstance(inputs, Sequence) else inputs  # type: ignore[invalid-return-type]


def _artifact_path_for_registry_entry(registry_path: Path, artifact_path: str) -> Path:
  path = Path(artifact_path)
  return path if path.is_absolute() else (registry_path.parent / path).resolve()


def _resolve_local_checkpoint_from_registry(spec: Specs) -> str | None:
  if spec.checkpoint_registry_path is None:
    return None
  if spec.checkpoint_id is None:
    msg = "checkpoint_id is required when checkpoint_registry_path is set"
    raise ValueError(msg)
  reg_path = spec.checkpoint_registry_path
  payload = json.loads(reg_path.read_text(encoding="utf-8"))
  entries = payload.get("entries", [])
  found = None
  for entry in entries:
    if entry.get("model_family") == spec.model_family and entry.get("checkpoint_id") == spec.checkpoint_id:
      found = entry
      break
  if found is None:
    msg = (
      f"No checkpoint registry entry found for model_family={spec.model_family!r} "
      f"checkpoint_id={spec.checkpoint_id!r}"
    )
    raise ValueError(msg)
  artifact = _artifact_path_for_registry_entry(reg_path, str(found["artifact_path"]))
  if "sha256" in found:
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != found["sha256"]:
      msg = f"Registry checksum mismatch for {artifact}: expected {found['sha256']}, got {digest}"
      raise ValueError(msg)
  return str(artifact)


def prep_protein_stream_and_model(
  spec: Specs,
) -> tuple[IterDataset, Model]:
  """Prepare the protein data stream and model parameters.

  Args:
      spec: A RunSpecification object containing configuration options.

  Returns:
      A tuple containing the protein data iterator and model in inference mode.

  Checkpoint resolution for :func:`~prxteinmpnn.io.weights.load_model`:

  - If ``spec.model_local_path`` is set, its path is passed as ``local_path`` and any
    ``checkpoint_registry_path`` entry is **not** read (explicit local artifact wins).
  - Else if ``spec.checkpoint_registry_path`` is set, the registry JSON must contain
    top-level ``"entries"``: a list of objects with ``model_family``,
    ``checkpoint_id``, ``artifact_path`` (relative paths resolve next to the registry
    file), and optional ``sha256``. The entry matching ``spec.model_family`` and
    ``spec.checkpoint_id`` is chosen; if ``sha256`` is present it must match the
    artifact bytes or :class:`ValueError` is raised. Missing entry raises
    :class:`ValueError`. ``checkpoint_id`` must be set on the spec when using the
    registry.
  - Otherwise weights load from packaged resources via ``checkpoint_id`` /
    ``model_weights`` + ``model_version`` as in :func:`load_model`.

  """
  parse_kwargs = {
    "chain_id": spec.chain_id,
    "model": spec.model,
    "altloc": spec.altloc,
    "topology": spec.topology,
  }
  ram_mb, workers, buf = proxide_dataset_resource_kwargs(spec, context="inference")
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
    ram_budget_mb=ram_mb,
    max_workers=workers,
    max_buffer_size=buf,
  )
  local_ckpt = None
  if spec.model_local_path is not None:
    local_ckpt = str(spec.model_local_path)
  elif reg_path := _resolve_local_checkpoint_from_registry(spec):
    local_ckpt = reg_path

  if spec.checkpoint_id is not None:
    model = load_model(
      checkpoint_id=spec.checkpoint_id,
      model_weights=spec.model_weights,
      model_version=spec.model_version,
      local_path=local_ckpt,
    )
  else:
    model = load_model(
      model_version=spec.model_version,
      model_weights=spec.model_weights,
      local_path=local_ckpt,
    )

  # Set model to inference mode (disables dropout, etc.)
  model = eqx.nn.inference_mode(model, value=True)

  return protein_iterator, model
