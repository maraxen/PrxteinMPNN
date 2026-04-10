"""Preflight checks for parity-heavy and checkpoint-family audit execution."""

from __future__ import annotations

import argparse
import os
from importlib.util import find_spec
from pathlib import Path

from prxteinmpnn.parity.assets import verify_parity_assets


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--reference-path",
    default=os.environ.get("REFERENCE_PATH", "reference_ligandmpnn_clone"),
    help="Path to checked-out LigandMPNN reference repository.",
  )
  parser.add_argument(
    "--project-root",
    default=".",
    help="Project root path containing converted model_params assets.",
  )
  parser.add_argument(
    "--tier",
    action="append",
    dest="tiers",
    default=[],
    help="Parity tier to validate (default: parity_heavy). May be repeated.",
  )
  parser.add_argument(
    "--require-parser-backend",
    action="store_true",
    help="Require proxide rust parsing backend import to succeed.",
  )
  args = parser.parse_args()

  tiers = set(args.tiers) if args.tiers else {"parity_heavy"}

  reference_root = Path(args.reference_path).resolve()
  project_root = Path(args.project_root).resolve()
  missing: list[str] = []

  if "parity_heavy" in tiers and find_spec("torch") is None:
    missing.append("python module: torch")

  asset_failures = verify_parity_assets(
    tiers=tiers,
    project_root_path=project_root,
    reference_root_path=reference_root,
  )
  missing.extend(asset_failures)

  if args.require_parser_backend and find_spec("proxide.io.parsing.rust") is None:
    missing.append("python module: proxide.io.parsing.rust")

  if missing:
    print("Parity preflight failed:")
    for item in missing:
      print(f"- {item}")
    return 1

  tier_list = ", ".join(sorted(tiers))
  print(f"Parity preflight passed for tiers: {tier_list}.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
