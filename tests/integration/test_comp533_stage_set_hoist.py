"""Integration tests for COMP-533: stage_set hoist verification.

This module verifies that make_stage_set is hoisted correctly from
inference.logits into runner as a module-level call, not called repeatedly
per batch or per structure. Tests are added by T7 and T8.
"""

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_make_stage_set_called_once_per_sample():
    """Verify make_stage_set is called exactly once per sample() invocation.

    Patches the USE-SITE binding in runner (not the definition in inference.logits)
    so the mock actually intercepts the call. This proves the hoist is working:
    make_stage_set should be called once at the runner level when sample() is invoked,
    not once per structure, batch, or decode operation.
    """
    from prxteinmpnn.inference.logits import make_stage_set as real_make_stage_set
    from prxteinmpnn.host.runner import sample

    call_log = []

    def tracking_make_stage_set(*args, **kwargs):
        call_log.append((args, kwargs))
        return real_make_stage_set(*args, **kwargs)

    # Minimal spec for sample() call
    spec_kwargs = {
        "inputs": "test_dummy_input.pdb",  # Will fail to load, but we intercept early
        "output_h5_path": None,  # In-memory mode to avoid I/O
        "average_node_features": False,
    }

    # Patch the USE-SITE binding: prxteinmpnn.host.runner.make_stage_set
    # This is where runner.py binds the name after `from prxteinmpnn.inference.logits import make_stage_set`
    with patch("prxteinmpnn.host.runner.make_stage_set", side_effect=tracking_make_stage_set):
        try:
            # sample() will fail early due to missing input file, but that's OK—
            # we just need to verify make_stage_set is called before the failure
            from prxteinmpnn.run.specs import SamplingSpecification

            spec = SamplingSpecification(**spec_kwargs)
            sample(spec)
        except (FileNotFoundError, Exception):
            # Expected: sample() fails because input file doesn't exist
            # But make_stage_set should have been called once before failure
            pass

    assert len(call_log) == 1, (
        f"Expected make_stage_set called once at runner level, "
        f"got {len(call_log)} calls. Calls: {call_log}"
    )


def test_kernel_dispatch_has_no_make_stage_set_import():
    """Structural check: kernel_dispatch.py does not import make_stage_set.

    T3 removed the make_stage_set import from kernel_dispatch.py as part of
    the hoist refactoring. Verify via AST that no ImportFrom node imports
    make_stage_set from inference.logits in kernel_dispatch.
    """
    kd_path = Path(__file__).parent.parent.parent / "src/prxteinmpnn/host/kernel_dispatch.py"
    assert kd_path.exists(), f"kernel_dispatch.py not found at {kd_path}"

    tree = ast.parse(kd_path.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "inference.logits" in node.module:
                for alias in node.names:
                    assert alias.name != "make_stage_set", (
                        f"kernel_dispatch.py still imports make_stage_set from "
                        f"inference.logits at line {node.lineno} — T3 should have removed this"
                    )
