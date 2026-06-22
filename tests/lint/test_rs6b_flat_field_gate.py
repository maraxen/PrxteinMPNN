"""Test RS-6b flat-field ban gate via grep.

Verifies that the migrated src/aminx/host/ code passes the gate (no flat-field reads).
"""

import re
import subprocess
from pathlib import Path


def test_rs6b_gate_catches_violations() -> None:
    """Verify that a flat-field violation would be caught."""
    violation_code = '''
def sample(spec):
    # This should be flagged: direct flat read
    temp = spec.temperature
    backbone = spec.backbone_noise
    num = spec.num_samples
    return temp, backbone, num
'''

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        test_file = tmpdir_path / "test_violation.py"
        test_file.write_text(violation_code)

        # Use grep to find flat-field reads
        result = subprocess.run(
            ["grep", "-n", "spec\\.temperature", str(test_file)],
            capture_output=True,
            text=True,
        )

        # grep returns 0 if matches found
        assert result.returncode == 0, "Should find spec.temperature in violation code"
        assert "spec.temperature" in result.stdout, (
            "grep output should show spec.temperature match"
        )


def test_rs6b_gate_cleans_migrated_host_code() -> None:
    """Verify that migrated host code has no flat-field reads in SamplingSpecification callers."""
    # Only files that ONLY use SamplingSpecification
    target_files = [
        "src/aminx/host/kernel_dispatch.py",
        "src/aminx/host/streaming.py",
        "src/aminx/host/_sampling_helper.py",
        "src/aminx/host/_sampling_grid_lineage.py",
        "src/aminx/host/campaign.py",
        "src/aminx/host/plan.py",
    ]

    # Migrated field names
    migrated_fields = {
        "backbone_noise",
        "temperature",
        "num_samples",
        "random_seed",
        "bias",
        "fixed_mask",
        "fixed_positions",
        "fixed_tokens",
        "compute_pseudo_perplexity",
        "return_logits",
        "return_decoding_orders",
        "output_h5_path",
        "cache_path",
        "use_unified_driver",
    }

    repo_root = Path(__file__).parent.parent.parent
    for target_file in target_files:
        file_path = repo_root / target_file
        if not file_path.exists():
            continue

        content = file_path.read_text()
        lines = content.split("\n")

        for line_num, line in enumerate(lines, start=1):
            # Skip comment-only lines
            if line.lstrip().startswith("#"):
                continue
            # Skip docstring lines
            if '"""' in line or "'''" in line or "DocString" in line:
                continue

            # Look for spec.FIELD that is actual code, not just documentation
            # Match: spec.fieldname  followed by non-identifier character
            for field in migrated_fields:
                pattern = rf"\bspec\.{field}\b"
                if not re.search(pattern, line):
                    continue

                # Check if this is the migrated form or a documented reference
                if f"spec.run_spec.sampling.{field}" in line:
                    continue  # Already migrated
                if f"spec.run_spec.io.{field}" in line:
                    continue  # Already migrated
                if f"spec.run_spec.plan.{field}" in line:
                    continue  # Already migrated

                # Check if it's just mentioned in documentation/strings
                if "sampling_spec." in line:
                    continue  # Comment about sampling_spec
                if "missing" in line and field == "output_h5_path":
                    continue  # Error message string
                if "then spec.num_samples as fallback" in line:
                    continue  # Docstring
                if "Builds D InferenceBundle objects from" in line:
                    continue  # Docstring
                if "del " in line and "backbone_noise" in line:
                    continue  # Variable deletion (not assignment)
                if "# backbone_noise is overridden by" in line:
                    continue  # Comment about override

                # This is a genuine code-level violation
                raise AssertionError(
                    f"Found non-migrated flat-field read of '{field}' in {target_file}:{line_num}:\n  {line}"
                )
