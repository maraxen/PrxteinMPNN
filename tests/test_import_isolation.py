"""Regression gate: verify no runtime imports of ensemble or psa leak into prxteinmpnn core.

These tests pass immediately (prxteinmpnn.ensemble is not runtime-imported even before
extraction, because specs.py has `from __future__ import annotations` which makes all
field annotations lazy). They serve as a non-regression harness throughout the sprint.
"""
from __future__ import annotations

import subprocess
import sys


def _run_isolation_check(module_path: str) -> None:
    code = f"""
import sys
import prxteinmpnn
import prxteinmpnn.run.specs
assert "{module_path}" not in sys.modules, (
    "{{!r}} was imported at runtime — must be TYPE_CHECKING-only".format("{module_path}")
)
print("OK: {module_path} not in sys.modules")
"""
    import os
    env = {**os.environ, "PYTHONPATH": "/home/marielle/projects/tev_design/prxteinmpnn/src"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"Isolation check failed for {{module_path!r}}:\n"
        f"stdout: {{result.stdout}}\nstderr: {{result.stderr}}"
    )


def test_ensemble_not_imported_at_runtime() -> None:
    """prxteinmpnn.ensemble must not appear in sys.modules after core import."""
    _run_isolation_check("prxteinmpnn.ensemble")


def test_psa_not_imported_at_runtime() -> None:
    """prxteinmpnn.psa must not appear in sys.modules after core import."""
    _run_isolation_check("prxteinmpnn.psa")


def test_ensemble_dbscan_not_imported_at_runtime() -> None:
    """The ConformationalStates source module must not load at runtime."""
    _run_isolation_check("prxteinmpnn.ensemble.dbscan")
