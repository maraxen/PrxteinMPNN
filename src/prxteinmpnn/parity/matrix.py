"""Parity path matrix helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ParityPath:
  """Describe a single parity path and its acceptance policy."""

  id: str
  tier: str
  code_paths: tuple[str, ...]
  reference: str
  method: str
  metrics: tuple[str, ...]
  acceptance: dict[str, object]


def project_root() -> Path:
  """Return repository root."""
  return Path(__file__).resolve().parents[3]


def manifest_path() -> Path:
  """Return the parity path manifest path."""
  return project_root() / "tests/parity/parity_matrix.json"


def load_parity_matrix(path: Path | None = None) -> tuple[ParityPath, ...]:
  """Load the parity matrix manifest."""
  payload = json.loads((path or manifest_path()).read_text(encoding="utf-8"))
  paths = payload.get("paths")
  if not isinstance(paths, list):
    msg = "Parity matrix manifest must contain a 'paths' list."
    raise TypeError(msg)

  parsed: list[ParityPath] = []
  for item in paths:
    if not isinstance(item, dict):
      msg = "Parity matrix entries must be objects."
      raise TypeError(msg)

    code_paths = item.get("code_paths")
    metrics = item.get("metrics")
    acceptance = item.get("acceptance")
    if not isinstance(code_paths, list) or not all(isinstance(v, str) for v in code_paths):
      msg = f"Parity path {item.get('id', '<unknown>')} has invalid code_paths."
      raise TypeError(msg)
    if not isinstance(metrics, list) or not all(isinstance(v, str) for v in metrics):
      msg = f"Parity path {item.get('id', '<unknown>')} has invalid metrics."
      raise TypeError(msg)
    if not isinstance(acceptance, dict):
      msg = f"Parity path {item.get('id', '<unknown>')} has invalid acceptance."
      raise TypeError(msg)

    parsed.append(
      ParityPath(
        id=str(item.get("id")),
        tier=str(item.get("tier")),
        code_paths=tuple(code_paths),
        reference=str(item.get("reference", "")),
        method=str(item.get("method", "")),
        metrics=tuple(metrics),
        acceptance=acceptance,
      ),
    )

  return tuple(parsed)
