"""Validate parity asset cache entries and checksums."""

from __future__ import annotations

import argparse

from prxteinmpnn.parity.assets import verify_parity_assets


def main() -> int:
  """Validate parity assets and exit non-zero on failure."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--tier",
    action="append",
    dest="tiers",
    default=[],
    help="Parity tier to validate. May be repeated.",
  )
  args = parser.parse_args()

  failures = verify_parity_assets(tiers=set(args.tiers) or None)
  if failures:
    print("Parity asset validation failed:")
    for failure in failures:
      print(f"- {failure}")
    return 1

  print("Parity asset validation passed.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
