"""Measure the laser_encoder float32 SCALE-RELATIVE parity floor (option 3, chosen 261013).

Pre-registered in ``laser_encoder_scale_relative_floor.bth.toml``, committed before this file
existed. Read that sidecar for the hypothesis and outcome criteria.

THE INSTRUMENT IS NOT RE-EXPRESSED. Every function that derives, admits, controls or grades comes
from ``laser_score_scale_relative_floor`` (graded run 3af3be02), whose measurement in turn comes
from ``laser_f32_tolerance_floor``. Both modules are imported unchanged, so their bytes and hashes
do not move. The only differences are the wave this module points them at and the script hash that
keys its own resume units, so laser_score's units can never be reused for laser_encoder.

Invocation:
  AMINX_LASER_ORACLE_DIR=<b3_laser_owned> JAX_PLATFORMS=cpu \\
  bth run --project-slug aminx -- uv run --no-sync python3 \\
    scripts/analysis/laser_encoder_scale_relative_floor.py --work-dir <dir> --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

import laser_score_scale_relative_floor as base  # noqa: E402
from laser_f32_tolerance_floor import OracleAbsent, _add_tests_to_path, _repo_root  # noqa: E402, PLC2701

_LOG = logging.getLogger("laser_encoder_scale_relative_floor")

_WAVE = "laser_encoder"

# Point the shared instrument at this wave. _unit_key and _collect read base._WAVE at call time.
base._WAVE = _WAVE  # noqa: SLF001


def _self_test() -> dict[str, Any]:
  """The inherited self-test, plus the checks that the retargeting itself took effect."""
  inherited = base._self_test()  # noqa: SLF001
  failed = list(inherited["failed"])
  checks = int(inherited["n_checks"])

  checks += 1
  if base._WAVE != _WAVE:  # noqa: SLF001
    failed.append("wave_not_retargeted")

  # A unit key must depend on the wave, so laser_score units can never satisfy a laser_encoder
  # lookup even under an identical script hash.
  checks += 1
  key_enc = base._unit_key("sha", "p", "f32")  # noqa: SLF001
  base._WAVE = "laser_score"  # noqa: SLF001
  key_score = base._unit_key("sha", "p", "f32")  # noqa: SLF001
  base._WAVE = _WAVE  # noqa: SLF001
  if key_enc == key_score:
    failed.append("unit_key_ignores_wave")

  # Amendment 1: an identically-zero field (max|ref| = 0 and max|err| = 0, as lig_vectors is) is
  # excluded from the floor rather than turning it into inf, and is still reported as degenerate.
  checks += 1
  population = [
    base._synth(f"p{i}", "f", 1.0, 1000.0)  # noqa: SLF001
    for i in range(5)
  ] + [base._synth(f"p{i}", "zero", 0.0, 0.0) for i in range(5)]  # noqa: SLF001
  degenerate, scaled = _split_degenerate(population)
  try:
    floor = base._floor(scaled)[0]  # noqa: SLF001
    base._leave_one_out(scaled)  # noqa: SLF001
  except ValueError:
    floor = math.inf
  if len(degenerate) != 5 or not math.isfinite(floor):
    failed.append("zero_scale_field_breaks_floor")

  return {"passed": not failed, "n_checks": checks, "failed": failed}


def _split_degenerate(stats: list[Any]) -> tuple[list[str], list[Any]]:
  """``(degenerate pair::field ids, the non-degenerate entries)`` (amendment 1)."""
  degenerate = base._degenerate(stats)  # noqa: SLF001
  excluded = set(degenerate)
  return degenerate, [s for s in stats if f"{s.pair}::{s.field}" not in excluded]


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Measure the laser_encoder float32 scale-relative parity floor.",
  )
  parser.add_argument("--out", type=Path, required=True, help="result JSON path")
  parser.add_argument("--work-dir", type=Path, required=True, help="resume directory")
  parser.add_argument(
    "--self-test-only",
    action="store_true",
    help="run the instrument self-test and exit without touching an oracle",
  )
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

  script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  shared_sha = hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()
  self_test = _self_test()
  if args.self_test_only:
    print(json.dumps(self_test, indent=2))  # noqa: T201
    return 0 if self_test["passed"] else 1

  payload: dict[str, Any] = {
    "self_test_passed": bool(self_test["passed"]),
    "self_test_n_checks": int(self_test["n_checks"]),
    "self_test_failed": list(self_test["failed"]),
    "oracle_absent": [],
    "wave": _WAVE,
    "headroom": base._HEADROOM,  # noqa: SLF001
    "float32_eps": float(np.finfo(np.float32).eps),
    "degenerate_scale_threshold": base._DEGENERATE_REL,  # noqa: SLF001
    "control_graded_relative": base._CONTROL_GRADED,  # noqa: SLF001
    "control_scale_margin": base._CONTROL_SCALE_MARGIN,  # noqa: SLF001
    "script_sha256": script_sha,
    "shared_instrument_sha256": shared_sha,
    "work_dir": str(args.work_dir),
    "n_reused": 0,
    "n_computed": 0,
  }

  if not self_test["passed"]:
    payload["verdict"] = "instrument_unverified"
    base._emit(args.out, payload)  # noqa: SLF001
    return 1

  _add_tests_to_path(_repo_root())

  try:
    f64, f32 = base._collect(args.work_dir, script_sha, payload)  # noqa: SLF001
  except OracleAbsent as exc:
    payload["oracle_absent"] = [str(exc)]
    payload["verdict"] = "oracle_absent"
    base._emit(args.out, payload)  # noqa: SLF001
    return 0

  payload["n_pairs"] = len({s.pair for s in f32})
  payload["n_fields"] = len({s.field for s in f32})
  payload["f64_passes"] = all(s.n_violating == 0 for s in f64)
  payload["f32_violated_elementwise"] = any(s.n_violating for s in f32)
  degenerate, scaled = _split_degenerate(f32)
  excluded = set(degenerate)
  payload["degenerate_fields"] = degenerate
  payload["degenerate_field_names"] = sorted({d.split("::", 1)[1].rsplit("__", 1)[-1] for d in degenerate})
  payload["degenerate_max_abs"] = max(
    (s.max_abs for s in f32 if f"{s.pair}::{s.field}" in excluded),
    default=0.0,
  )
  if not scaled:
    msg = "every f32 entry is degenerate; nothing to derive a floor from"
    raise RuntimeError(msg)

  # Amendment 1: floor, leave-one-out and control run over the non-degenerate entries only.
  floor, floor_pair, floor_field, second = base._floor(scaled)  # noqa: SLF001
  payload["floor_scale_relative"] = floor
  payload["floor_pair"] = floor_pair
  payload["floor_field"] = floor_field
  payload["second_worst_scale_relative"] = second
  payload["floor_scale_relative_per_field"] = base._per_field_floor(scaled)  # noqa: SLF001

  if math.isfinite(floor) and floor > 0.0:
    tol = base._tol_from(floor)  # noqa: SLF001
  else:
    tol = math.inf if math.isinf(floor) else 0.0
  payload["proposed_tol"] = tol

  not_adm, loo_pair, loo_ratio = base._leave_one_out(scaled)  # noqa: SLF001
  payload["n_loo_not_admitted"] = not_adm
  payload["loo_all_admitted"] = not_adm == 0
  payload["loo_worst_pair"] = loo_pair
  payload["loo_worst_ratio"] = loo_ratio

  n_control_admitted, control_ratio = base._control_scale_relative(scaled, tol)  # noqa: SLF001
  payload["control_rejected"] = n_control_admitted == 0
  payload["control_scale_ratio"] = control_ratio
  payload["below_control_scale"] = control_ratio <= 1.0 / base._CONTROL_SCALE_MARGIN  # noqa: SLF001

  payload["verdict"] = base._grade(payload)  # noqa: SLF001
  base._emit(args.out, payload)  # noqa: SLF001
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
