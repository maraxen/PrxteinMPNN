"""Test potts import boundary gate via ast-grep.

Verifies that the potts-import-boundary YAML rule:
  (a) FIRES on a planted violation placed in src/aminx/potts/ (matches files: glob).
  (b) Is CLEAN on the real src/aminx/potts/ tree (0 violations) when no probe
      files are present.
  (c) EXEMPTS designer.py even when it contains a banned import (validates
      ignores: glob functionality).

The rule enforces the isolation between aminx.potts (parallel model) and
aminx (MPNN stageset): potts.model, potts.poe, and potts.sampling must NOT
import from aminx.inference.decode, aminx.host.plan, aminx.types.stages, or
aminx.inference.logits. Only potts.designer is exempt from this restriction.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest


_RULE_PATH = ".ast-grep/rules/potts-import-boundary.yml"
_REPO_ROOT = Path(__file__).parent.parent.parent


def run_ast_grep_rule(rule_path: str, scan_path: str | None = None) -> tuple[bool, str]:
    """Run an ast-grep rule and return (success, output).

    Mirrors the helper in tests/lint/test_rs6b_flat_field_gate.py.

    Args:
        rule_path: Path to the rule YAML, relative to repo root.
        scan_path: Optional explicit path to scan; overrides files: in the rule.

    Returns:
        (True, "") if rule passes (0 violations)
        (False, output) if rule reports violations (non-zero exit)

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
            cwd=str(_REPO_ROOT),
        )
        return result.returncode == 0, result.stderr
    except FileNotFoundError:
        pytest.skip(
            "ast-grep (sg) not found in PATH. Install via: npm install -g @ast-grep/cli"
        )


def test_potts_boundary_fires_on_planted_violation() -> None:
    """Assert the rule FIRES when src/aminx/potts/ contains a banned import.

    Plants a temporary file directly inside src/aminx/potts/ so the rule's
    'files: src/aminx/potts/**/*.py' glob matches it, then asserts the rule
    reports at least one violation. The file is removed in the finally block
    regardless of test outcome.
    """
    violation_code = """\
from aminx.inference.decode import decode_one

def potts_decode():
    return decode_one()
"""
    potts_dir = _REPO_ROOT / "src" / "aminx" / "potts"
    probe_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(potts_dir),
            suffix="_potts_boundary_probe.py",
            mode="w",
            delete=False,
        ) as f:
            f.write(violation_code)
            probe_path = f.name

        success, output = run_ast_grep_rule(_RULE_PATH)

    finally:
        if probe_path and os.path.exists(probe_path):
            os.unlink(probe_path)

    assert not success, (
        f"Rule should have fired on banned import from aminx.inference.decode in planted violation, but passed.\n"
        f"Rule output:\n{output}"
    )


def test_potts_boundary_clean_on_real_tree() -> None:
    """Assert the rule produces 0 violations on the real src/aminx/potts/ tree.

    Uses the rule's own files:/ignores: configuration (no explicit scan_path),
    which covers potts/**/*.py and excludes designer.py per the rule YAML.
    """
    success, output = run_ast_grep_rule(_RULE_PATH)

    assert success, (
        f"potts-import-boundary gate found violations in src/aminx/potts/:\n{output}\n\n"
        "Fix: potts.model, potts.poe, and potts.sampling must NOT import from:\n"
        "  aminx.inference.decode, aminx.host.plan, aminx.types.stages,\n"
        "  aminx.inference.logits.\n"
        "Exemption: designer.py is intentionally excluded."
    )


def test_potts_boundary_exempts_designer() -> None:
    """Assert designer.py is EXEMPT from the rule even with a banned import.

    Temporarily appends a banned import line to the real src/aminx/potts/designer.py,
    runs the rule, and asserts success (0 violations because designer.py is in
    ignores:). Then restores the original content. This proves the ignores: glob
    works correctly.
    """
    designer_path = _REPO_ROOT / "src" / "aminx" / "potts" / "designer.py"
    original_content: str | None = None
    try:
        # Save original content
        original_content = designer_path.read_text()

        # Append a banned import (that would normally be flagged)
        modified_content = original_content + "\nfrom aminx.inference.logits import compute_logits\n"
        designer_path.write_text(modified_content)

        # Run the rule; should pass because designer.py is exempt
        success, output = run_ast_grep_rule(_RULE_PATH)

    finally:
        # Restore original content
        if original_content is not None:
            designer_path.write_text(original_content)

    assert success, (
        f"Rule should NOT fire on designer.py even with banned imports, but failed.\n"
        f"designer.py is in ignores: configuration.\n"
        f"Rule output:\n{output}"
    )
