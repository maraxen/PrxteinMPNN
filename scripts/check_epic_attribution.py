#!/usr/bin/env python3
"""Flag decode-path-adjacent commits missing an mpnn_ext epic-ID reference.

Per mpnn_ext/.praxia/docs/roadmaps/consolidated-cross-project/260702_00-mandate.md
§4.1 item 2: any aminx commit implementing an mpnn_ext-epic requirement must
reference the driving epic ID in its commit message (pattern like ``#1234`` or
``epic #1234``), and should be accompanied by an aminx-side decision doc
cross-linking to the mpnn_ext epic doc. This script enforces the commit-message
half mechanically; the decision-doc half is a manual review item.

Only *source code* paths under ``src/`` are checked against the keyword list —
prose in ``.praxia/`` logs, docs, or fixtures mentioning "wave"/"schedule" in
passing must not trip the lint.

Grandfathered: the five original wave-color commits (54d6d84, 0be59ef,
4060e9d, 0670197, 1cec556) landed before this lint existed and are already
retroactively attributed by hand in
``.praxia/docs/decisions/260702_wave-color-commits-retroactive-attribution.md``.
They're exempted here by SHA so this lint doesn't perpetually fail on its own
introducing PR; this is a one-time grandfather list, not a pattern to append
to for future violations — new violations must fix the commit message (or, if
already merged, get their own decision doc AND a matching exemption PR that
links it).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

DECODE_PATH_KEYWORDS = re.compile(
    r"\b(wave|chromatic|schedul|decod|coloring|colouring)\b", re.IGNORECASE
)
EPIC_ID_PATTERN = re.compile(r"(?:epic\s*)?#\d{3,5}\b", re.IGNORECASE)
SOURCE_PATH_PREFIX = "src/"

GRANDFATHERED_SHA_PREFIXES = frozenset(
    {"54d6d84", "0be59ef", "4060e9d", "0670197", "1cec556"}
)


def _run(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def commit_shas(rev_range: str) -> list[str]:
    return [line for line in _run("rev-list", rev_range).splitlines() if line]


def commit_message(sha: str) -> str:
    return _run("log", "-1", "--format=%B", sha)


def changed_source_paths(sha: str) -> list[str]:
    names = _run("show", "--name-only", "--format=", sha).splitlines()
    return [n for n in names if n.startswith(SOURCE_PATH_PREFIX)]


def commit_touches_decode_path(sha: str) -> bool:
    return any(DECODE_PATH_KEYWORDS.search(path) for path in changed_source_paths(sha))


def check_commit(sha: str) -> str | None:
    """Return a violation message, or None if the commit is compliant."""
    if sha[:7] in GRANDFATHERED_SHA_PREFIXES:
        return None
    message = commit_message(sha)
    if not commit_touches_decode_path(sha):
        return None
    if EPIC_ID_PATTERN.search(message):
        return None
    subject = message.strip().splitlines()[0] if message.strip() else "(empty)"
    return (
        f"{sha[:8]} touches decode/schedule/wave/coloring source code but its "
        f'commit message has no epic-ID reference (e.g. "#2871" or '
        f'"epic #2871"): {subject!r}'
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rev_range",
        nargs="?",
        default="origin/main..HEAD",
        help="git rev-list range to check (default: origin/main..HEAD)",
    )
    args = parser.parse_args()

    try:
        shas = commit_shas(args.rev_range)
    except subprocess.CalledProcessError as exc:
        print(f"Could not resolve rev range {args.rev_range!r}: {exc.stderr}", file=sys.stderr)
        return 1

    violations = [msg for sha in shas for msg in [check_commit(sha)] if msg]

    if violations:
        print("Boundary-enforcement lint FAILED (aminx backlog #2954):\n")
        for v in violations:
            print(f"  - {v}")
        print(
            "\nDecode-path-adjacent commits driven by an mpnn_ext research epic "
            "must reference that epic's ID in the commit message, and should be "
            "accompanied by an aminx-side decision doc cross-linking to the "
            "mpnn_ext epic doc (see .praxia/docs/decisions/260702_wave-color-"
            "commits-retroactive-attribution.md for the pattern)."
        )
        return 1

    print(f"Boundary-enforcement lint passed ({len(shas)} commit(s) checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
