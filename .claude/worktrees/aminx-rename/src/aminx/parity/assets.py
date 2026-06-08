"""Parity asset manifest helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

AssetBase = Literal["project", "reference"]


@dataclass(frozen=True, slots=True)
class ParityAsset:
  """Describe a parity asset and where it is expected to live."""

  id: str
  base: AssetBase
  path: str
  required_for: tuple[str, ...]
  sha256: str | None
  source: str


def project_root() -> Path:
  """Return the repository root."""
  return Path(__file__).resolve().parents[3]


def manifest_path() -> Path:
  """Return the parity asset manifest path."""
  return project_root() / "tests/parity/parity_assets.json"


def _load_manifest_text(path: Path) -> dict[str, object]:
  return json.loads(path.read_text(encoding="utf-8"))


def load_parity_assets(path: Path | None = None) -> tuple[ParityAsset, ...]:
  """Load the parity asset manifest from disk."""
  manifest = _load_manifest_text(path or manifest_path())
  assets = manifest.get("assets")
  if not isinstance(assets, list):
    msg = "Parity asset manifest must contain an 'assets' list."
    raise TypeError(msg)

  parsed: list[ParityAsset] = []
  for item in assets:
    if not isinstance(item, dict):
      msg = "Parity asset entries must be objects."
      raise TypeError(msg)

    required_for = item.get("required_for")
    if not isinstance(required_for, list) or not all(isinstance(v, str) for v in required_for):
      msg = f"Parity asset {item.get('id', '<unknown>')} has invalid required_for."
      raise TypeError(msg)

    base = cast("AssetBase", item.get("base"))
    if base not in ("project", "reference"):
      msg = f"Parity asset {item.get('id', '<unknown>')} has invalid base."
      raise TypeError(msg)

    sha_value = item.get("sha256")
    if sha_value is not None and not isinstance(sha_value, str):
      msg = f"Parity asset {item.get('id', '<unknown>')} has invalid sha256."
      raise TypeError(msg)

    parsed.append(
      ParityAsset(
        id=str(item.get("id")),
        base=base,
        path=str(item.get("path")),
        required_for=tuple(required_for),
        sha256=sha_value,
        source=str(item.get("source", "")),
      ),
    )

  return tuple(parsed)


def _resolve_reference_root(reference_root: Path | None) -> Path | None:
  if reference_root is not None:
    return reference_root

  env_value = os.environ.get("REFERENCE_PATH")
  if env_value:
    candidate = Path(env_value)
    if candidate.is_dir():
      return candidate

  for candidate in (
    project_root() / "reference_ligandmpnn_clone",
    project_root().parent / "reference_ligandmpnn_clone",
  ):
    if candidate.is_dir():
      return candidate

  return None


def _resolve_asset_path(
  asset: ParityAsset,
  *,
  project_root_path: Path,
  reference_root_path: Path | None,
) -> Path | None:
  if asset.base == "project":
    return project_root_path / asset.path
  if reference_root_path is None:
    return None
  return reference_root_path / asset.path


def _file_sha256(path: Path) -> str:
  digest = sha256()
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def verify_parity_assets(
  *,
  tiers: set[str] | None = None,
  project_root_path: Path | None = None,
  reference_root_path: Path | None = None,
  manifest: tuple[ParityAsset, ...] | None = None,
) -> list[str]:
  """Return human-readable parity asset validation errors."""
  project_root_value = project_root_path or project_root()
  reference_root_value = _resolve_reference_root(reference_root_path)
  required_tiers = tiers or set()
  assets = manifest or load_parity_assets()
  failures: list[str] = []

  for asset in assets:
    if required_tiers and not (required_tiers & set(asset.required_for)):
      continue

    resolved = _resolve_asset_path(
      asset,
      project_root_path=project_root_value,
      reference_root_path=reference_root_value,
    )
    if resolved is None:
      root_hint = (
        "REFERENCE_PATH or reference_ligandmpnn_clone"
        if asset.base == "reference"
        else "project root"
      )
      failures.append(f"{asset.id}: unresolved {asset.base} asset; set {root_hint}")
      continue

    if not resolved.exists():
      failures.append(f"{asset.id}: missing {resolved}")
      continue

    if asset.sha256 is not None and _file_sha256(resolved) != asset.sha256:
      failures.append(f"{asset.id}: checksum mismatch for {resolved}")

  return failures
