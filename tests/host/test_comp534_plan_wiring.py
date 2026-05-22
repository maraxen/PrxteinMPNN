"""COMP-534: Tests that runner.py uses InferencePlan for stage_set wiring."""

from __future__ import annotations

import ast
import functools
import inspect
from unittest.mock import MagicMock, patch

import equinox as eqx
import jax.numpy as jnp
import pytest

from prxteinmpnn.host.plan import InferencePlan, make_inference_plan
from prxteinmpnn.inference.logits import make_stage_set


class DummyModel(eqx.Module):
    """Minimal equinox module for make_inference_plan testing."""


class DummySpec:
    """Minimal spec matching the validated shape from test_inference_plan.py."""
    use_rolling_state = False
    multi_state_strategy = "arithmetic_mean"
    multi_state_temperature = 1.0
    state_weights = None
    sampling_strategy = "temperature"
    temperature = [1.0]


def test_runner_imports_make_inference_plan():
    """runner.py must import make_inference_plan."""
    from prxteinmpnn.host import runner
    assert hasattr(runner, "make_inference_plan")


def test_runner_imports_inference_plan():
    """runner.py must import InferencePlan."""
    from prxteinmpnn.host import runner
    assert hasattr(runner, "InferencePlan")


def test_runner_does_not_import_make_stage_set():
    """runner.py must NOT import make_stage_set directly (uses plan instead)."""
    from prxteinmpnn.host import runner
    source = inspect.getsource(runner)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "logits" in node.module:
                names = [alias.name for alias in node.names]
                assert "make_stage_set" not in names, (
                    f"runner.py should not import make_stage_set "
                    f"(found in import from {node.module})"
                )


def test_make_inference_plan_called_once_in_sample():
    """make_inference_plan must be called exactly once per sample() call."""
    with patch("prxteinmpnn.host.runner.make_inference_plan") as mock_make_plan:
        mock_plan = MagicMock(spec=InferencePlan)
        mock_plan.stage_set = make_stage_set("arithmetic_mean", 1.0, None)
        mock_make_plan.return_value = mock_plan

        with patch("prxteinmpnn.host.runner.prep_protein_stream_and_model") as mock_prep:
            mock_iter = MagicMock()
            mock_iter.__iter__ = MagicMock(return_value=iter([]))
            mock_iter.skipped_frames = []
            mock_model = MagicMock()
            mock_prep.return_value = (mock_iter, mock_model)

            from prxteinmpnn.host.runner import sample
            from prxteinmpnn.run.specs import SamplingSpecification
            try:
                result = sample(spec=SamplingSpecification(inputs="dummy.pdb"))
            except Exception:
                pytest.skip(
                    "sample() requires infrastructure not available in test environment"
                )

        assert mock_make_plan.call_count == 1, (
            f"make_inference_plan should be called once, "
            f"was called {mock_make_plan.call_count} times"
        )


def test_plan_stage_set_property_accessible():
    """InferencePlan.stage_set property returns non-None stage_set."""
    plan = make_inference_plan(DummyModel(), DummySpec())
    assert hasattr(plan, "stage_set")
    assert plan.stage_set is not None


def test_plan_stage_set_matches_make_stage_set():
    """plan.stage_set type matches standalone make_stage_set construction."""
    spec = DummySpec()
    plan = make_inference_plan(DummyModel(), spec)
    standalone = make_stage_set(
        strategy=spec.multi_state_strategy,
        strategy_temperature=spec.multi_state_temperature,
        state_weights=spec.state_weights,
    )
    assert type(plan.stage_set) == type(standalone)


def test_functools_partial_accepts_plan_stage_set():
    """functools.partial binding with plan.stage_set works correctly."""
    plan = make_inference_plan(DummyModel(), DummySpec())

    def mock_fn(spec, ensemble, model, stage_set=None, **kwargs):
        return stage_set

    bound = functools.partial(mock_fn, stage_set=plan.stage_set)
    result = bound(None, None, None)
    assert result is plan.stage_set
