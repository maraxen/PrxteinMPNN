"""Tests for xtrax boundary discipline enforcement.

Verifies that:
(a) aminx modules do not import from xtrax internals (private modules)
(b) xtrax modules do not reference aminx protein-specific symbols

Both rules enforce the boundary preservation outlined in ADR 260605 and the
xtrax-foundations spec (260611).
"""

import subprocess
from pathlib import Path


def run_ast_grep_rule(rule_path: str) -> tuple[bool, str]:
    """Run an ast-grep rule and return (success, output).

    Returns:
        (True, "") if rule passes (0 violations)
        (False, output) if rule fails (violations found)
    """
    try:
        result = subprocess.run(
            ["sg", "scan", "--rule", rule_path],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        return result.returncode == 0, result.stderr
    except FileNotFoundError:
        raise RuntimeError("ast-grep (sg) not found in PATH. Install via: cargo install ast-grep")


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
    """
    rule_path = ".ast-grep/rules/xtrax-protein-symbols.yml"
    success, output = run_ast_grep_rule(rule_path)

    assert success, (
        f"xtrax protein-symbols guard violated:\n{output}\n\n"
        "Fix: Remove references to atom_37, residue_index, tie_group_map from xtrax.\n"
        "These are protein-specific and belong in aminx only."
    )
