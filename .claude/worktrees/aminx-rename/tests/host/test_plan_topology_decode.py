# tests/host/test_plan_topology_decode.py
"""Test planner validator rules for decode-axis composability (Sprint 6).

Tasks 11–12: Validates mode-specific rejection (Rule 1) and STE↔DecodeStep pairing (Rule 2).
"""

from __future__ import annotations

import pytest
import jax.numpy as jnp
from unittest.mock import MagicMock

from aminx.host.plan import PlanTopologyError, _validate_plan_topology
from aminx.inference.decode.mode import (
    ConditionalMode,
    UnconditionalMode,
    AutoregressiveMode,
    STEMode,
)
from aminx.tiling.dispatch import DispatchRejected
from aminx.tiling.strategy import Scan
from aminx.types.stages import StageSet, UnconditionalDecodeStep


# --- Rule 1: factory-level rejection with mode-name context ---


def test_factory_reject_conditional_mode_scan_includes_mode_name_and_heterogeneous():
    """ConditionalMode + Scan on state axis → DispatchRejected with mode name + heterogeneous."""
    from aminx.inference.decode.factory import make_decode_fn

    # Create a minimal stub model
    stub_model = MagicMock()

    with pytest.raises(DispatchRejected) as excinfo:
        make_decode_fn(
            model=stub_model,
            mode=ConditionalMode(),
            strategy=Scan(init=jnp.zeros(()), transition=lambda c, x: (c, x)),
        )

    msg = str(excinfo.value).lower()
    assert "conditionalmode" in msg
    assert "heterogeneous" in msg


def test_factory_reject_unconditional_mode_scan_includes_mode_name_and_heterogeneous():
    """UnconditionalMode + Scan on state axis → DispatchRejected with mode name + heterogeneous."""
    from aminx.inference.decode.factory import make_decode_fn

    stub_model = MagicMock()

    with pytest.raises(DispatchRejected) as excinfo:
        make_decode_fn(
            model=stub_model,
            mode=UnconditionalMode(),
            strategy=Scan(init=jnp.zeros(()), transition=lambda c, x: (c, x)),
        )

    msg = str(excinfo.value).lower()
    assert "unconditionalmode" in msg
    assert "heterogeneous" in msg


def test_factory_reject_autoregressive_mode_scan_includes_mode_name_and_heterogeneous():
    """AutoregressiveMode + Scan on state axis → DispatchRejected with mode name + heterogeneous."""
    from aminx.inference.decode.factory import make_decode_fn

    stub_model = MagicMock()

    with pytest.raises(DispatchRejected) as excinfo:
        make_decode_fn(
            model=stub_model,
            mode=AutoregressiveMode(),
            strategy=Scan(init=jnp.zeros(()), transition=lambda c, x: (c, x)),
        )

    msg = str(excinfo.value).lower()
    assert "autoregressivemode" in msg
    assert "heterogeneous" in msg


def test_factory_reject_ste_mode_scan_includes_mode_name_and_heterogeneous():
    """STEMode + Scan on state axis → DispatchRejected with mode name + heterogeneous."""
    from aminx.inference.decode.factory import make_decode_fn

    stub_model = MagicMock()

    with pytest.raises(DispatchRejected) as excinfo:
        make_decode_fn(
            model=stub_model,
            mode=STEMode(),
            strategy=Scan(init=jnp.zeros(()), transition=lambda c, x: (c, x)),
        )

    msg = str(excinfo.value).lower()
    assert "stemode" in msg
    assert "heterogeneous" in msg


# --- Rule 2: validator-level STE + UnconditionalDecodeStep rejection ---


def test_validator_rejects_ste_decode_with_unconditional_decode_step():
    """Validator rejects plan with STEDecode + UnconditionalDecodeStep on stage_set."""
    from aminx.inference.decode.ste import STEDecode

    # Build a minimal plan-like mock with decode_fn field
    plan = MagicMock()

    # Create minimal instances for the bad pair
    unc_step = UnconditionalDecodeStep(decoder=MagicMock())
    ste_decode = MagicMock(spec=STEDecode)

    plan.stage_set = StageSet(decode_step=unc_step)
    plan.decode_fn = ste_decode

    with pytest.raises(PlanTopologyError, match="STE.*Unconditional|Unconditional.*STE"):
        _validate_plan_topology(plan, plan.stage_set)


def test_validator_allows_ste_decode_with_conditional_decode_step():
    """Validator allows plan with STEDecode + ConditionalDecodeStep (the correct pairing)."""
    from aminx.inference.decode.ste import STEDecode
    from aminx.types.stages import ConditionalDecodeStep

    plan = MagicMock()

    # Create minimal instances for the correct pair
    cond_step = ConditionalDecodeStep(
        decoder=MagicMock(),
        w_s_embed=MagicMock(),
    )
    ste_decode = MagicMock(spec=STEDecode)

    plan.stage_set = StageSet(decode_step=cond_step)
    plan.decode_fn = ste_decode

    # Must not raise (check ConditionalDecodeStep signature if needed)
    _validate_plan_topology(plan, plan.stage_set)
