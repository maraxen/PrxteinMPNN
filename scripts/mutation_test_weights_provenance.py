#!/usr/bin/env python3
"""Mutation-test the weight-provenance suite: would it catch a broken resolver?

A green suite says the tests pass, not that they would fail on wrong code. This applies a
known defect to ``src/aminx/io/weights.py``, runs the suite, and reports whether the suite
noticed -- then restores the file.

It exists because it found something. On the first run (2026-08-24, PR #145 review) two
mutations ESCAPED a 7-test suite that looked complete:

  * hardcoding ``source="hub"`` in the packaged branch  -> 7 passed
  * making the packaged branch ignore ``filename``      -> 7 passed

Both slipped through because the only test reaching that branch asserted
``source in {...}`` -- a membership check -- and re-derived its expected file through the same
resolver it was testing. After ``tests/io/test_weight_provenance_packaged.py`` was added, all
mutations are caught (8/8).

SCOPE, so this is not cited as stronger evidence than it is: ``pyproject.toml``'s
``pythonpath = ["src", ...]`` means pytest always imports from the SOURCE tree, where the
packaged checkpoints exist. In the built wheel they do not, so the packaged branch is dead
code there and the two mutations that only that branch reaches are unobservable in the shipped
artifact. This measures the source checkout.

Run:  uv run --no-sync python scripts/mutation_test_weights_provenance.py
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "src/aminx/io/weights.py"
SUITES = [
  "tests/io/test_weight_provenance.py",
  "tests/io/test_weight_provenance_packaged.py",
]

log = logging.getLogger("mutation_test")

#: (label, original source fragment, mutated fragment)
MUTATIONS: list[tuple[str, str, str]] = [
  (
    "drop revision= from the hub call",
    'return "hub", hf_hub_download(repo_id=HF_REPO_ID, filename=filename, revision=revision)',
    'return "hub", hf_hub_download(repo_id=HF_REPO_ID, filename=filename)',
  ),
  (
    "fail-closed becomes warn-and-fall-through",
    "      raise FileNotFoundError(msg)",
    "      log.warning(msg)",
  ),
  (
    "explicit_dir ignored entirely",
    "  explicit_dir = _env_or_none(WEIGHTS_DIR_ENV)",
    "  explicit_dir = None",
  ),
  (
    "source hardcoded to 'hub' in the packaged branch",
    '        return "packaged", str(packaged)',
    '        return "hub", str(packaged)',
  ),
  (
    "_hub_revision_from_path searches left-to-right again",
    '  index = len(parts) - 1 - parts[::-1].index("snapshots") + 1',
    '  index = parts.index("snapshots") + 1',
  ),
  (
    "_hub_revision_from_path stops validating the commit sha",
    "  if not _COMMIT_SHA.fullmatch(candidate):",
    "  if False:",
  ),
  (
    "checkpoint filename normalisation dropped",
    "  filename = normalise_checkpoint_filename(filename)\n",
    "",
  ),
  (
    "packaged branch ignores filename",
    '    resource_path = files("aminx.model_params").joinpath(filename)',
    '    resource_path = files("aminx.model_params").joinpath("proteinmpnn_v_48_002.eqx.zst")',
  ),
]


def run_suite() -> tuple[bool, str]:
  """Return ``(passed, summary_line)`` for the provenance suites."""
  proc = subprocess.run(
    ["uv", "run", "--no-sync", "pytest", *SUITES, "-q", "--no-header", "-p", "no:cacheprovider"],
    cwd=REPO,
    capture_output=True,
    text=True,
    timeout=900,
    check=False,
  )
  lines = (proc.stdout + proc.stderr).splitlines()
  summary = next(
    (ln.strip() for ln in reversed(lines) if "passed" in ln or "failed" in ln),
    "no summary line",
  )
  return proc.returncode == 0, summary


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--verbose", action="store_true")
  args = parser.parse_args()
  logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s",
  )

  pristine = TARGET.read_text()
  passed, summary = run_suite()
  log.info("baseline: %s -- %s", "PASS" if passed else "FAIL", summary)
  if not passed:
    log.error("baseline is not green; mutation results would be meaningless")
    return 2

  escaped: list[str] = []
  try:
    for label, old, new in MUTATIONS:
      if old not in pristine:
        log.warning("SKIP    %s (anchor not found -- has the source moved?)", label)
        escaped.append(f"{label} [anchor missing]")
        continue
      TARGET.write_text(pristine.replace(old, new, 1))
      passed, summary = run_suite()
      # Restore by rewriting the original text. NOT `git checkout` -- that restores from the
      # index and silently discards uncommitted work, which it did the first time this ran.
      TARGET.write_text(pristine)
      if passed:
        escaped.append(label)
      log.info("%s %s -- %s", "ESCAPED" if passed else "CAUGHT ", label, summary)
  finally:
    TARGET.write_text(pristine)

  if TARGET.read_text() != pristine:
    log.error("target was NOT restored; inspect %s before continuing", TARGET)
    return 2

  log.info("\ncaught %d/%d", len(MUTATIONS) - len(escaped), len(MUTATIONS))
  for label in escaped:
    log.error("ESCAPED: %s", label)
  return 1 if escaped else 0


if __name__ == "__main__":
  sys.exit(main())
