#!/usr/bin/env python3
"""Flag decode-path-adjacent commits missing an mpnn_ext epic-ID reference.

Per mpnn_ext/.praxia/docs/roadmaps/consolidated-cross-project/260702_00-mandate.md
§4.1 item 2: any aminx commit implementing an mpnn_ext-epic requirement must
reference the driving epic ID in its commit message (pattern like ``#1234`` or
``epic #1234``), and should be accompanied by an aminx-side decision doc
cross-linking to the mpnn_ext epic doc. This script enforces the commit-message
half mechanically; the decision-doc half is a manual review item.

Motivated by backlog aminx#2954: five wave-color commits (54d6d84, 0be59ef,
4060e9d, 0670197, 1cec556) landed in aminx with no epic-ID reference despite an
explicit prior instruction to file that work under mpnn_ext instead.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

DECODE_PATH_KEYWORDS = re.compile(
    r"\b(wave|chromatic|schedul|decode|coloring|colouring)\b", re.IGNORECASE
)
EPIC_ID_PATTERN = re.compile(r"(?:epic\s*)?#\d{3,5}\b", re.IGNORECASE)


def _run(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def commit_shas(rev_range: str) -> list[str]:
    return [line for line in _run("rev-list", rev_range).splitlines() if line]


def commit_message(sha: str) -> str:
    return _run("log", "-1", "--format=%B", sha)


def commit_diff_touches_decode_path(sha: str) -> bool:
    diff = _run("show", "--unified=0", "--format=", sha)
    return bool(DECODE_PATH_KEYWORDS.search(diff))


def check_commit(sha: str) -> str | None:
    """Return a violation message, or None if the commit is compliant."""
    message = commit_message(sha)
    if not commit_diff_touches_decode_path(sha):
        return None
    if EPIC_ID_PATTERN.search(message):
        return None
    subject = message.strip().splitlines()[0] if message.strip() else "(empty)"
    return (
        f"{sha[:8]} touches decode/schedule/wave/coloring code but its commit "
        f'message has no epic-ID reference (e.g. "#2871" or "epic #2871"): '
        f"{subject!r}"
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
