"""Regression: aminx RunSpec carries its PrecisionConfig as ``precision_config``.

xtrax >= 0.4.1b1 declares ``RunSpec.precision`` as ``str | None`` and rejects any other
type in ``__check_init__``, so aminx must not reuse that name for a PrecisionConfig.
"""

from __future__ import annotations

from aminx.run.run_spec_portable_json import (
    run_spec_portable_from_dict,
    run_spec_portable_to_dict,
)
from aminx.run.spec import PrecisionConfig, RunSpec


def test_run_spec_accepts_precision_config_under_installed_xtrax() -> None:
    spec = RunSpec(
        seed=0,
        axes=[],
        carry_specs={},
        boundaries=None,
        precision_config=PrecisionConfig(compute="bf16"),
    )
    assert spec.precision_config.compute == "bf16"
    # xtrax >= 0.4.1b1 defines ``precision: str | None`` (default None); 0.4.0a11 has no such
    # field at all. Either way aminx must not have stored a PrecisionConfig under that name.
    assert getattr(spec, "precision", None) is None


def test_precision_config_round_trips_through_portable_json() -> None:
    wire = {
        "version": 2,
        "io": {"sink_kind": "none", "output_dir": None, "manifest_path": None},
        "multistate": {"mode": "single", "n_states": 1, "combine_strategy": "arithmetic_mean"},
        "resource": {
            "n_devices": 1,
            "sample_batch_size": 32,
            "structure_batch_size": 32,
            "max_buffer_size": None,
        },
        "precision": {"compute": "bf16"},
    }
    spec = run_spec_portable_from_dict(wire)
    assert spec.precision_config.compute == "bf16"
    out = run_spec_portable_to_dict(spec)
    # Wire key is unchanged; only the Python attribute was renamed.
    assert out["precision"] == {"compute": "bf16"}
    assert run_spec_portable_from_dict(out).precision_config.compute == "bf16"
