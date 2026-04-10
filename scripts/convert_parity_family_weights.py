"""Convert checkpoint-family parity assets with scripts/convert_weights.py."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ConversionJob:
  """A single converted checkpoint target tied to a reference source."""

  asset_id: str
  family: str
  source_path: Path
  target_path: Path


def _load_manifest_assets(path: Path) -> list[dict[str, object]]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  assets = payload.get("assets")
  if not isinstance(assets, list):
    msg = "Parity asset manifest must contain an 'assets' list."
    raise TypeError(msg)
  if not all(isinstance(asset, dict) for asset in assets):
    msg = "Parity asset entries must be objects."
    raise TypeError(msg)
  return list(assets)


def _resolve_reference_root(
  *,
  project_root: Path,
  configured: str | None,
) -> Path | None:
  candidates: list[Path] = []
  if configured:
    candidates.append(Path(configured))

  env_reference = os.environ.get("REFERENCE_PATH")
  if env_reference:
    candidates.append(Path(env_reference))

  candidates.extend(
    [
      project_root / "reference_ligandmpnn_clone",
      project_root.parent / "reference_ligandmpnn_clone",
    ],
  )

  for candidate in candidates:
    resolved = candidate.resolve()
    if resolved.is_dir():
      return resolved

  return None


def _resolve_asset_path(
  *,
  base: object,
  relative_path: object,
  project_root: Path,
  reference_root: Path | None,
) -> Path | None:
  if not isinstance(relative_path, str):
    return None

  if base == "project":
    return project_root / relative_path
  if base == "reference" and reference_root is not None:
    return reference_root / relative_path
  return None


def _collect_jobs(
  *,
  assets: list[dict[str, object]],
  tier: str,
  families: set[str],
  project_root: Path,
  reference_root: Path | None,
) -> tuple[list[ConversionJob], list[str]]:
  asset_by_id = {
    str(asset.get("id")): asset
    for asset in assets
    if isinstance(asset.get("id"), str)
  }
  jobs: list[ConversionJob] = []
  failures: list[str] = []

  for asset in assets:
    required_for = asset.get("required_for")
    if not isinstance(required_for, list) or tier not in required_for:
      continue

    if asset.get("asset_kind") != "converted_checkpoint":
      continue

    asset_id = asset.get("id")
    family = asset.get("family")
    if not isinstance(asset_id, str) or not isinstance(family, str):
      failures.append("Encountered converted checkpoint asset with invalid id/family metadata.")
      continue

    if families and family not in families:
      continue

    reference_id = asset.get("reference_asset_id")
    if not isinstance(reference_id, str):
      failures.append(f"{asset_id}: missing reference_asset_id.")
      continue

    reference_asset = asset_by_id.get(reference_id)
    if reference_asset is None:
      failures.append(f"{asset_id}: reference asset {reference_id!r} not found in manifest.")
      continue

    source_path = _resolve_asset_path(
      base=reference_asset.get("base"),
      relative_path=reference_asset.get("path"),
      project_root=project_root,
      reference_root=reference_root,
    )
    target_path = _resolve_asset_path(
      base=asset.get("base"),
      relative_path=asset.get("path"),
      project_root=project_root,
      reference_root=reference_root,
    )

    if source_path is None:
      failures.append(f"{asset_id}: unable to resolve reference source path.")
      continue
    if target_path is None:
      failures.append(f"{asset_id}: unable to resolve target output path.")
      continue

    jobs.append(
      ConversionJob(
        asset_id=asset_id,
        family=family,
        source_path=source_path,
        target_path=target_path,
      ),
    )

  return jobs, failures


def main() -> int:  # noqa: PLR0911, PLR0915
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--project-root",
    default=".",
    help="Project root path containing scripts/convert_weights.py and parity manifest.",
  )
  parser.add_argument(
    "--reference-path",
    default=None,
    help="Path to LigandMPNN reference checkout (overrides REFERENCE_PATH).",
  )
  parser.add_argument(
    "--tier",
    default="parity_audit",
    help="Manifest tier to convert (default: parity_audit).",
  )
  parser.add_argument(
    "--family",
    action="append",
    dest="families",
    default=[],
    help="Family filter (protein, soluble, membrane, ligand, sc). May be repeated.",
  )
  parser.add_argument(
    "--skip-existing",
    action="store_true",
    help="Skip conversion when the target .eqx file already exists.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print conversion actions without running scripts/convert_weights.py.",
  )
  args = parser.parse_args()

  project_root = Path(args.project_root).resolve()
  manifest_path = project_root / "tests/parity/parity_assets.json"
  convert_script = project_root / "scripts/convert_weights.py"

  if not manifest_path.exists():
    print(f"Missing parity asset manifest: {manifest_path}")
    return 1
  if not convert_script.exists():
    print(f"Missing conversion script: {convert_script}")
    return 1

  assets = _load_manifest_assets(manifest_path)
  reference_root = _resolve_reference_root(project_root=project_root, configured=args.reference_path)

  jobs, failures = _collect_jobs(
    assets=assets,
    tier=args.tier,
    families=set(args.families),
    project_root=project_root,
    reference_root=reference_root,
  )

  if failures:
    print("Failed to prepare conversion jobs:")
    for failure in failures:
      print(f"- {failure}")
    return 1

  if not jobs:
    requested = ", ".join(sorted(set(args.families))) or "all families"
    print(f"No conversion jobs found for tier={args.tier!r} and families={requested}.")
    return 1

  if reference_root is None:
    print("Reference root not found. Set --reference-path or REFERENCE_PATH.")
    return 1

  converted = 0
  skipped = 0
  conversion_failures: list[str] = []

  for job in jobs:
    if not job.source_path.exists():
      conversion_failures.append(f"{job.asset_id}: missing source checkpoint {job.source_path}")
      continue

    if args.skip_existing and job.target_path.exists():
      skipped += 1
      print(f"Skipping existing {job.target_path}")
      continue

    print(
      f"[{job.family}] {job.asset_id}: {job.source_path.name} -> {job.target_path.name}",
    )

    if args.dry_run:
      continue

    job.target_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(  # noqa: S603
      [
        sys.executable,
        str(convert_script),
        "--input",
        str(job.source_path),
        "--output",
        str(job.target_path),
      ],
      cwd=project_root,
      check=False,
    )
    if result.returncode != 0:
      conversion_failures.append(f"{job.asset_id}: conversion failed with exit code {result.returncode}")
      continue

    if not job.target_path.exists():
      conversion_failures.append(f"{job.asset_id}: conversion completed but output missing")
      continue

    converted += 1

  if conversion_failures:
    print("Conversion failures:")
    for failure in conversion_failures:
      print(f"- {failure}")
    return 1

  if args.dry_run:
    print(f"Dry run complete. Planned {len(jobs)} conversion job(s).")
    return 0

  print(f"Conversion complete. Converted {converted} checkpoint(s); skipped {skipped} existing checkpoint(s).")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
