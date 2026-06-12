"""Tests for xtrax boundary discipline enforcement.

Verifies that:
(a) aminx modules do not import from xtrax internals (private modules)
(b) xtrax modules do not reference aminx protein-specific symbols

Both rules enforce the boundary preservation outlined in ADR 260605 and the
xtrax-foundations spec (260611).
"""

import subprocess
from pathlib import Path

import pytest


def run_ast_grep_rule(rule_path: str, scan_path: str | None = None) -> tuple[bool, str]:
    """Run an ast-grep rule and return (success, output).

    Args:
        rule_path: Path to the rule YAML, relative to repo root.
        scan_path: Optional explicit path to scan; overrides files: in the rule.

    Returns:
        (True, "") if rule passes (0 violations)
        (False, output) if rule fails (violations found)

    Raises:
        pytest.skip: if ast-grep (sg) is not installed
    """
    cmd = ["sg", "scan", "--rule", rule_path]
    if scan_path is not None:
        cmd.append(scan_path)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        return result.returncode == 0, result.stderr
    except FileNotFoundError:
        pytest.skip(
            "ast-grep (sg) not found in PATH. Install via: npm install -g @ast-grep/cli"
        )


def test_aminx_xtrax_internals_boundary():
    """Test that aminx modules do not import xtrax internals.

    Rule: aminx-xtrax-internals-boundary.yml
    Forbidden imports:
      - from xtrax._internal import ...
      - from xtrax._utils import ...
      - from xtrax.impl import ...
      - from xtrax.core import ...
    """
    rule_path = ".ast-grep/rules/aminx-xtrax-internals-boundary.yml"
    success, output = run_ast_grep_rule(rule_path)

    assert success, (
        f"aminx→xtrax-internals boundary violated:\n{output}\n\n"
        "Fix: Remove imports from xtrax._internal, xtrax._utils, xtrax.impl, xtrax.core.\n"
        "Use public xtrax API only (xtrax.training, xtrax.engine, xtrax.tiling, etc.)"
    )


def test_xtrax_protein_symbols_guard():
    """Test that xtrax modules do not reference protein-specific symbols.

    Rule: xtrax-protein-symbols-guard.yml
    Protected symbols:
      - atom_37 (37-atom PDB standard)
      - residue_index (PDB residue numbering)
      - tie_group_map (amino acid tying for symmetric design)

    These symbols are protein-domain-specific and violate xtrax's
    domain-agnostic interface.

    Scans the sibling ../xtrax/src/xtrax directory when present (editable-pin dev
    setup). Skips if the directory does not exist.
    """
    rule_path = ".ast-grep/rules/xtrax-protein-symbols.yml"
    repo_root = Path(__file__).parent.parent

    # Resolve the main repo root via git so this works in both the main checkout
    # and any worktrees (which live at deeply-nested paths under .claude/worktrees/).
    git_common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    if git_common.returncode != 0:
        pytest.skip("Could not determine git root; skipping xtrax boundary check")
    common_dir = Path(git_common.stdout.strip())
    # --git-common-dir is relative when run from the main checkout, absolute from a worktree.
    if not common_dir.is_absolute():
        common_dir = (repo_root / common_dir).resolve()
    # common_dir = <aminx-root>/.git  =>  projects_dir = <aminx-root>/..
    projects_dir = common_dir.parent.parent
    xtrax_src = projects_dir / "xtrax" / "src" / "xtrax"

    if not xtrax_src.exists():
        pytest.skip(
            f"sibling xtrax repo not found at {xtrax_src}; "
            "boundary guard only enforces against co-located source"
        )

    success, output = run_ast_grep_rule(rule_path, str(xtrax_src))

    assert success, (
        f"xtrax protein-symbols guard violated:\n{output}\n\n"
        "Fix: Remove references to atom_37, residue_index, tie_group_map from xtrax.\n"
        "These are protein-specific and belong in aminx only."
    )
