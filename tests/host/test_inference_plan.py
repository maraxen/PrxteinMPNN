"""COMP-8: Tests for InferencePlan in host/plan.py.

RED phase — tests fail until:
  - InferencePlan class exists in host/plan.py
  - InferencePlan exposes .sample() and .score() as distinct entrypoints
  - InferenceComponents named tuple (encode_fn, driver, stage_set) exists
  - make_inference_plan(model, spec) factory exists
"""

from __future__ import annotations

import inspect
import pytest


# ---------------------------------------------------------------------------
# 1. Import contract
# ---------------------------------------------------------------------------

def test_inference_plan_importable():
    """InferencePlan is importable from host.plan."""
    from aminx.host.plan import InferencePlan  # noqa: F401


def test_inference_components_importable():
    """InferenceComponents is importable from host.plan."""
    from aminx.host.plan import InferenceComponents  # noqa: F401


def test_make_inference_plan_importable():
    """make_inference_plan factory is importable from host.plan."""
    from aminx.host.plan import make_inference_plan  # noqa: F401


# ---------------------------------------------------------------------------
# 2. InferencePlan interface
# ---------------------------------------------------------------------------

def test_inference_plan_has_sample_method():
    """InferencePlan has a .sample() method."""
    from aminx.host.plan import InferencePlan
    assert hasattr(InferencePlan, "sample"), "InferencePlan must have .sample()"
    assert callable(InferencePlan.sample)


def test_inference_plan_has_score_method():
    """InferencePlan has a .score() method."""
    from aminx.host.plan import InferencePlan
    assert hasattr(InferencePlan, "score"), "InferencePlan must have .score()"
    assert callable(InferencePlan.score)


def test_inference_plan_sample_score_are_distinct():
    """sample() and score() are separate methods with different implementations."""
    from aminx.host.plan import InferencePlan
    # They must not be the same function object
    assert InferencePlan.sample is not InferencePlan.score


def test_inference_plan_has_stage_set_attribute():
    """InferencePlan exposes a stage_set attribute (resolved from spec)."""
    import equinox as eqx
    from aminx.host.plan import InferencePlan, make_inference_plan

    # Create a minimal plan instance via make_inference_plan
    class DummyModel(eqx.Module):
        pass

    class DummySpec:
        sampling_strategy = "temperature"
        use_rolling_state = False
        multi_state_strategy = "arithmetic_mean"
        multi_state_temperature = 1.0
        state_weights = None
        temperature = [1.0]

    plan = make_inference_plan(DummyModel(), DummySpec())
    assert hasattr(plan, "stage_set"), "InferencePlan instance must have a stage_set attribute"


# ---------------------------------------------------------------------------
# 3. InferenceComponents named tuple contract
# ---------------------------------------------------------------------------

def test_inference_components_is_named_tuple():
    """InferenceComponents is a NamedTuple with encode_fn, stage_set."""
    from aminx.host.plan import InferenceComponents
    # NamedTuple has _fields
    assert hasattr(InferenceComponents, "_fields"), "InferenceComponents must be a NamedTuple"
    fields = InferenceComponents._fields
    assert "encode_fn" in fields, f"Missing encode_fn in {fields}"
    assert "stage_set" in fields, f"Missing stage_set in {fields}"


# ---------------------------------------------------------------------------
# 4. make_inference_plan factory
# ---------------------------------------------------------------------------

def test_make_inference_plan_accepts_model_and_spec():
    """make_inference_plan(model, spec) has model and spec parameters."""
    from aminx.host.plan import make_inference_plan
    sig = inspect.signature(make_inference_plan)
    assert "model" in sig.parameters
    assert "spec" in sig.parameters


def test_make_inference_plan_returns_inference_plan():
    """make_inference_plan returns an InferencePlan instance."""
    import equinox as eqx
    from aminx.host.plan import InferencePlan, make_inference_plan

    # Minimal duck-type model
    class DummyModel(eqx.Module):
        pass

    # Minimal duck-type spec
    class DummySpec:
        sampling_strategy = "temperature"
        use_rolling_state = False
        multi_state_strategy = "arithmetic_mean"
        multi_state_temperature = 1.0
        state_weights = None
        temperature = [1.0]

    plan = make_inference_plan(DummyModel(), DummySpec())
    assert isinstance(plan, InferencePlan), (
        f"make_inference_plan must return InferencePlan, got {type(plan)}"
    )


def test_make_inference_plan_with_geometric_mean():
    """make_inference_plan with multi_state_strategy='geometric_mean' wires GeometricMeanLogits."""
    from aminx.host.plan import make_inference_plan
    from aminx.inference.logits import GeometricMeanLogits
    import equinox as eqx

    class DummyModel(eqx.Module):
        pass

    class GeoSpec:
        sampling_strategy = "temperature"
        use_rolling_state = False
        multi_state_strategy = "geometric_mean"
        multi_state_temperature = 1.0
        state_weights = None
        temperature = [1.0]

    plan = make_inference_plan(DummyModel(), GeoSpec())
    assert isinstance(plan.stage_set.logit_transform, GeometricMeanLogits)


# ---------------------------------------------------------------------------
# 5. STE (Straight-Through Estimator) decode mode routing
# ---------------------------------------------------------------------------

def test_make_inference_plan_straight_through_uses_ste_decode():
    """make_inference_plan with sampling_strategy='straight_through' uses STEDecode mode."""
    from aminx.host.plan import make_inference_plan
    from aminx.inference.decode.ste import STEDecode
    import equinox as eqx

    # Create minimal model with decoder stubs
    class DummyModel(eqx.Module):
        decoder: object = None
        w_s_embed: object = None

    class STESpec:
        sampling_strategy = "straight_through"
        use_rolling_state = False
        multi_state_strategy = "arithmetic_mean"
        multi_state_temperature = 1.0
        state_weights = None
        temperature = [1.0]
        average_node_features = False

    model = DummyModel(decoder=object(), w_s_embed=type('obj', (object,), {'weight': None})())
    plan = make_inference_plan(model, STESpec())
    assert isinstance(plan.decode_fn, STEDecode), (
        f"straight_through sampling_strategy should use STEDecode, "
        f"got {type(plan.decode_fn)}"
    )


def test_make_inference_plan_temperature_uses_conditional_mode():
    """make_inference_plan with sampling_strategy='temperature' uses ConditionalDecode."""
    from aminx.host.plan import make_inference_plan
    from aminx.inference.decode.conditional import ConditionalDecode
    import equinox as eqx

    class DummyModel(eqx.Module):
        pass

    class TempSpec:
        sampling_strategy = "temperature"
        use_rolling_state = False
        multi_state_strategy = "arithmetic_mean"
        multi_state_temperature = 1.0
        state_weights = None
        temperature = [1.0]
        average_node_features = False

    plan = make_inference_plan(DummyModel(), TempSpec())
    assert isinstance(plan.decode_fn, ConditionalDecode), (
        f"temperature sampling_strategy should use ConditionalDecode, "
        f"got {type(plan.decode_fn)}"
    )
