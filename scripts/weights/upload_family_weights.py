#!/usr/bin/env python3
"""Propose the family weight artifacts to the Hugging Face repo as a pull request.

Dry run by default: it verifies every file against ``src/aminx/io/family_weights.json``, builds the
model-card section and prints what would be uploaded. ``--execute`` is the only path that touches the
network, and it refuses unless the licence decision record has ``status: accepted`` (the reviewer's
sign-off), every artifact hashes to its pinned value, and ``HF_TOKEN`` is set. The upload is a PR on the
HF repo (``create_pr=True``), so nothing reaches ``main`` until someone merges it there.

After the PR is merged on the Hub, bump ``HF_REVISION`` in ``src/aminx/io/weights.py`` to the merge
commit and rerun ``tests/io/test_family_weights.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger("upload_family_weights")

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "src/aminx/io/family_weights.json"
RECORD_GLOB = "*redistribution-of-pottsmpnn-lasermpnn-and-protonpottsmpnn*.md"
HF_REPO_ID = "maraxen/aminx"

CARD_SECTION = """
## Family checkpoints (aminx 0.2.1)

Converted copies of third-party weights, redistributed under their upstream MIT licences. Each upstream
copyright and permission notice is in `NOTICE-weights.md` in the aminx source tree and reproduced
below. These are the upstream parameters re-serialised as Equinox `.eqx`; they are not retrained. Parity
of each against its upstream is measured per path and is not complete (see the aminx parity ledger).

| checkpoint id | upstream | upstream file |
|---|---|---|
{rows}

Load by id with aminx (`--checkpoint-id <id>`); every file is pinned by sha256 in
`aminx/io/family_weights.json`. Each `.eqx` has a `.json` manifest next to it that must be kept with it
(it records static fields the `.eqx` format does not store, for example LASErMPNN's `build_hydrogens`).

Upstreams: PottsMPNN (<https://github.com/KeatingLab/PottsMPNN>, MIT, (c) 2022 Justas Dauparas,
(c) 2026 Foster Birnbaum, Amy E. Keating), LASErMPNN (<https://github.com/polizzilab/LASErMPNN>, MIT,
(c) 2025 Benjamin Fry, Nicholas Polizzi), Proton-PottsMPNN (MIT, (c) 2026 Christian P. Jacobsen).
Please cite the upstream publications if you use these weights.
"""

UPSTREAM = {
  "pottsmpnn": ("PottsMPNN", "KeatingLab/PottsMPNN @ 0cb0a588"),
  "lasermpnn": ("LASErMPNN", "polizzilab/LASErMPNN @ e70f2c6d"),
  "protonpottsmpnn": ("Proton-PottsMPNN", "epoch-0125 checkpoint, pin 09682abf"),
}


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def record_status() -> str:
  """``status`` from the licence decision record's frontmatter."""
  record = next((REPO / ".praxia/docs/decisions").glob(RECORD_GLOB))
  match = re.search(r"^status:\s*(\S+)", record.read_text(encoding="utf-8"), flags=re.MULTILINE)
  return match.group(1) if match else "unknown"


def verify(artifacts: Path) -> list[tuple[Path, str]]:
  """Every pinned artifact and manifest must exist and hash to its pinned value."""
  table = json.loads(TABLE.read_text(encoding="utf-8"))["checkpoints"]
  if not table:
    msg = f"{TABLE} is empty; run scripts/weights/build_family_weights.py first"
    raise SystemExit(msg)
  files: list[tuple[Path, str]] = []
  for row in table:
    for name, expected in ((row["filename"], row["sha256"]), (row["manifest_filename"], row["manifest_sha256"])):
      path = artifacts / name
      if not path.is_file():
        msg = f"missing {path}"
        raise SystemExit(msg)
      digest = _sha256(path)
      if digest != expected:
        msg = f"{path} has sha256 {digest}, pinned {expected}"
        raise SystemExit(msg)
      files.append((path, name))
  return files


def card_section() -> str:
  table = json.loads(TABLE.read_text(encoding="utf-8"))["checkpoints"]
  rows = "\n".join(
    f"| `{row['checkpoint_id']}` | {UPSTREAM[row['family']][0]}, {UPSTREAM[row['family']][1]} | "
    f"`{row['filename']}` |"
    for row in table
  )
  return CARD_SECTION.format(rows=rows)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--artifacts-dir", type=Path, required=True)
  parser.add_argument("--repo", default=HF_REPO_ID)
  parser.add_argument("--notice", type=Path, default=REPO / "NOTICE-weights.md")
  parser.add_argument("--execute", action="store_true", help="actually open the PR on the Hub")
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  files = verify(args.artifacts_dir)
  total = sum(path.stat().st_size for path, _ in files)
  for path, name in files:
    logger.info("would upload %-45s %12d bytes", name, path.stat().st_size)
  logger.info("%d files, %.1f MB, plus NOTICE-weights.md and a README.md section", len(files), total / 1e6)

  status = record_status()
  logger.info("licence decision record status: %s", status)
  if not args.execute:
    logger.info("dry run; nothing uploaded. Re-run with --execute after the record is accepted.")
    return 0
  if status != "accepted":
    logger.error("refusing to upload: the licence decision record is %r, not 'accepted'", status)
    return 2
  if not os.environ.get("HF_TOKEN"):
    logger.error("refusing to upload: HF_TOKEN is not set")
    return 2

  from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download  # noqa: PLC0415

  api = HfApi()
  current = Path(hf_hub_download(repo_id=args.repo, filename="README.md", revision="main")).read_text(
    encoding="utf-8",
  )
  readme = current.rstrip("\n") + "\n" + card_section()
  operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for path, name in files]
  operations.append(CommitOperationAdd(path_in_repo="NOTICE-weights.md", path_or_fileobj=str(args.notice)))
  operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")))
  info = api.create_commit(
    repo_id=args.repo,
    operations=operations,
    commit_message="Add PottsMPNN, LASErMPNN and ProtonPottsMPNN checkpoints (aminx 0.2.1)",
    create_pr=True,
  )
  logger.info("opened %s", info.pr_url)
  return 0


if __name__ == "__main__":
  sys.exit(main())
