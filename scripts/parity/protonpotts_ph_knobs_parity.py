"""The ``protonpotts_ph_knobs`` wave: aminx block descent with ``sweep_order`` knn / energy against the upstream engine (P4k).

Pre-registered in ``protonpotts_ph_knobs_parity.bth.toml`` (committed before this script). Per (cell, precision, block-descent
call) the aminx side is given what upstream was given (sealed table and E_idx, native sequence, binder mask, the production
configuration with the call's ``temperature`` and ``sweep_order`` from the dump meta, and the recorded uniforms) and runs the
SHIPPED visit-order path: ``ph_design._visit_order`` on the plan, then ``block_table`` and ``block_descent`` on that order, as
``ph_design._run_optimiser`` composes them. It must reproduce the recorded visit order exactly, then match upstream's plan,
pooled z-scales, sequence, draw count and scores by the same comparison as the ``protonpotts_ph_block`` wave.

Reused by import from ``protonpotts_ph_block_parity``: ``Cell``, ``CELLS``, ``UPSTREAM_CDF_ORDER``, ``_to_aminx``,
``derive_plan``, ``compare_call``. Copied: ``run_call`` (the block wave's version hard-wires ``plan.designable`` as the sweep
order; this one feeds ``_visit_order``'s output to both ``block_table`` and ``block_descent``).

AMENDMENT (261011, ``protonpotts_ph_knobs_parity.bth.toml``): f32 calls' final-energy and selectivity-gap bands are
``10 x max(upstream f32-f64 spread, aminx f32-f64 spread of the same sequence)``, never below 1e-6 relative; see
``amended_f32_scores``. f64 calls and the block wave are unchanged.

Run where aminx is importable::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/parity/protonpotts_ph_knobs_parity.py \\
        --encoder-dir <P4e out dir> --ph-dir <P4k out dir> --features-dir ~/projects/aminx-oracles-protonpotts
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig  # noqa: E402
from aminx.families.protonpotts_mpnn.ph_descent import block_descent, rep_class_mask  # noqa: E402
from aminx.families.protonpotts_mpnn.ph_design import _visit_order  # noqa: E402
from aminx.families.protonpotts_mpnn.ph_plan import block_table, valid_token_mask  # noqa: E402

# Sibling wave: the block-descent parity script in this directory (on sys.path because this file runs as a script).
from protonpotts_ph_block_parity import (  # noqa: E402
  CELLS,
  F32_FACTOR,
  F32_MIN_REL,
  UPSTREAM_CDF_ORDER,
  Cell,
  _close,
  _to_aminx,
  compare_call,
  derive_plan,
  scores,
)

logger = logging.getLogger("protonpotts_ph_knobs_parity")

SENTINEL = 10**6
N_CALLS_EXPECTED = 48
# Knob values the dump records per call that the production configuration must reproduce (samples_per_site is not
# compared: the dump records 1 on the T=0 calls, and the block wave's production configuration uses the default 2).
KNOBS_MUST_MATCH = (
  "block_size", "combined_lambda", "neighbour_k", "max_mutations", "zscale_mode",
  "self_weight", "global_weight", "adjacent_repeat_weight",
)  # fmt: skip
# The dump sidecar's [outcomes.pass] flags, exactly as registered.
DUMP_CLAIMS = (
  "engine_matches_sealed", "complete", "shim_consistent", "f64_is_float64", "precisions_differ", "reproducible",
  "knn_order_live", "energy_order_live",
)  # fmt: skip
CONTROLS = ("position_knn", "position_energy", "swapped", "shifted")


def check_knobs(call: dict, base: PHDesignConfig) -> None:
  """Fail loudly if the dump's recorded knobs disagree with the configuration this wave runs under."""
  knobs = call["knobs"]
  if knobs["sweep_order"] != call["sweep_order"] or float(knobs["temperature"]) != float(call["temperature"]):
    raise ValueError(f"call {call['index']} ({call['label']}): knobs disagree with the call's sweep_order/temperature")
  for key in KNOBS_MUST_MATCH:
    if getattr(base, key) != knobs[key]:
      raise ValueError(f"call {call['index']} ({call['label']}): knob {key}={knobs[key]!r} != base {getattr(base, key)!r}")


def config_for(base: PHDesignConfig, call: dict, sweep_order: str | None = None) -> PHDesignConfig:
  """The production configuration at the call's temperature, with the call's sweep order (or an override)."""
  return dataclasses.replace(
    base, temperature=float(call["temperature"]), sweep_order=sweep_order or call["sweep_order"]
  )  # fmt: skip


def run_call(cell: Cell, call: dict, config: PHDesignConfig, *, shift_uniforms: bool = False):  # noqa: ANN201
  """Shipped composition: ``_visit_order`` -> ``block_table`` -> ``block_descent`` on the visit order.

  Mirrors ``protonpotts_ph_block_parity.run_call`` except the designable array and the block table are built from the
  visit order returned by ``ph_design._visit_order``. Returns ``(plan, blocks, result, visit)``.
  """
  plan = derive_plan(cell, config)
  e_idx = np.asarray(cell.e_idx)
  visit = _visit_order(config, plan, cell.table, e_idx, cell.native)
  blocks, block_valid = block_table(e_idx, visit, config.block_size)
  pins = plan.pins
  depth = max(len(p.dep_idxs) for p in pins)
  pin_dep = np.zeros((len(pins), depth), dtype=np.int32)
  pin_dep_valid = np.zeros((len(pins), depth), dtype=bool)
  for i, pin in enumerate(pins):
    pin_dep[i, : len(pin.dep_idxs)] = pin.dep_idxs
    pin_dep_valid[i, : len(pin.dep_idxs)] = True
  recorded = cell.arrays[f"c{call['index']}_uniforms"]
  need = max(1, config.block_max_rounds * len(plan.designable))
  uniforms = np.zeros(max(need, recorded.shape[0]), dtype=cell.dtype)
  uniforms[: recorded.shape[0]] = recorded
  if shift_uniforms and recorded.shape[0] > 1:
    uniforms[: recorded.shape[0]] = np.roll(recorded, -1)
  residue_number = np.where(cell.binder, cell.res_id, -SENTINEL - 100 * np.arange(cell.length))
  result = block_descent(
    cell.table, cell.e_idx, jnp.asarray(_to_aminx(cell.arrays[f"c{call['index']}_S_in"])),
    jnp.asarray(visit, dtype=jnp.int32), jnp.asarray(blocks), jnp.asarray(block_valid),
    jnp.asarray([p.position for p in pins], dtype=jnp.int32), jnp.asarray([p.prot_idx for p in pins], dtype=jnp.int32),
    jnp.asarray(pin_dep), jnp.asarray(pin_dep_valid),
    jnp.asarray(valid_token_mask(config.forbidden_tokens)),
    jnp.asarray(rep_class_mask(config.repetitive_window_parents)), jnp.asarray(residue_number, dtype=jnp.int32),
    config=config, uniforms=jnp.asarray(uniforms), cdf_order=jnp.asarray(UPSTREAM_CDF_ORDER),
  )  # fmt: skip
  return plan, blocks, result, visit


def amended_f32_scores(cell: Cell, other: Cell, call: dict, plan, seq: np.ndarray) -> dict:
  """Amendment 261011: the f32 score band is 10 x max(upstream f32-f64 spread, aminx f32-f64 spread of the same sequence).

  ``cell`` is the f32 cell, ``other`` the f64 cell. The aminx f64 score is ``scores(other, plan, seq)`` (same plan, same
  sequence, f64 table). The floor is never below ``F32_MIN_REL`` relative to the target scale.
  """
  index = call["index"]
  design = cell.designs[index]
  upstream = other.designs[index]
  want_h = design["final_potts_energy"]
  want_s = design["selective_energy"]
  up_floor_h = abs(want_h - upstream["final_potts_energy"])
  up_floor_s = abs((want_s or 0.0) - (upstream["selective_energy"] or 0.0))
  energy, gap = scores(cell, plan, seq)
  energy64, gap64 = scores(other, plan, seq)
  aminx_floor_h = abs(energy - energy64)
  aminx_floor_s = abs(gap - gap64)
  scale_h = max(1.0, abs(want_h))
  scale_s = max(1.0, abs(want_s or 0.0))
  energy_band = max(F32_FACTOR * max(up_floor_h, aminx_floor_h), F32_MIN_REL * scale_h)
  gap_band = max(F32_FACTOR * max(up_floor_s, aminx_floor_s), F32_MIN_REL * scale_s)
  scores_ok = _close(energy, want_h, energy_band) and _close(gap, want_s, gap_band)
  return {
    "scores_ok": bool(scores_ok),
    "energy_band": float(energy_band),
    "gap_band": float(gap_band),
    "gap_abs_diff": float(abs(gap - want_s)),
    "aminx_gap_floor": float(aminx_floor_s),
  }


def check_call(cell: Cell, other: Cell, call: dict, config: PHDesignConfig, *, f32: bool, shift: bool = False) -> dict:
  """One call through the shipped path, compared with upstream: the visit order first, then ``compare_call``.

  f32 calls take the amended score band (``amended_f32_scores``) in place of ``compare_call``'s score verdict; f64 calls
  keep ``compare_call``'s verdict. Controls go through here too, so they are graded identically.
  """
  plan, blocks, result, visit = run_call(cell, call, config, shift_uniforms=shift)
  verdict = compare_call(cell, other, call, (plan, blocks, result), f32=f32)
  if f32:
    verdict.update(amended_f32_scores(cell, other, call, plan, np.asarray(result.seq)))
  order_ok = [int(v) for v in visit] == [int(v) for v in call["visit_order"]]
  verdict["order_ok"] = order_ok
  verdict["ok"] = bool(
    verdict["plan_ok"] and verdict["tokens_ok"] and verdict["draws_ok"] and verdict["zscales_ok"]
    and verdict["scores_ok"] and order_ok
  )  # fmt: skip
  return verdict


def control_config(base: PHDesignConfig, call: dict, name: str) -> tuple[PHDesignConfig, bool] | None:
  """The (configuration, shift) a control runs on this call, or None if the control does not apply to it."""
  sweep = call["sweep_order"]
  if name == "position_knn":
    return (config_for(base, call, "position"), False) if sweep == "knn" else None
  if name == "position_energy":
    return (config_for(base, call, "position"), False) if sweep == "energy" else None
  if name == "swapped":
    return config_for(base, call, {"knn": "energy", "energy": "knn"}[sweep]), False
  if name == "shifted":
    return (config_for(base, call), True) if float(call["temperature"]) > 0 else None
  raise ValueError(f"unknown control {name!r}")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--encoder-dir", type=Path, required=True)
  parser.add_argument("--ph-dir", type=Path, required=True)
  parser.add_argument(
    "--features-dir", type=Path, required=True,
    help="accepted for the registered invocation; the dump already verified its native S against the sealed features",
  )  # fmt: skip
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

  manifest = json.loads((args.ph_dir / "ph_knobs_manifest.json").read_text(encoding="utf-8"))
  flags = manifest["flags"]
  dump_flags_pass = flags.get("n_cells") == len(CELLS) and all(bool(flags.get(k)) for k in DUMP_CLAIMS)
  base = PHDesignConfig()

  per_call: dict[str, dict] = {}
  counts = {"ok": 0, "order": 0, "plan": 0, "tokens": 0, "draws": 0, "zscales": 0, "scores": 0}
  worst_z = worst_e = 0.0
  rejected = dict.fromkeys(CONTROLS, False)
  applied = dict.fromkeys(CONTROLS, 0)
  for cell_name in CELLS:
    cells = {p: Cell(args.encoder_dir, args.ph_dir, cell_name, p) for p in ("f64", "f32")}
    for precision, cell in cells.items():
      other = cells["f32" if precision == "f64" else "f64"]
      for call in cell.block_calls():
        check_knobs(call, base)
        verdict = check_call(cell, other, call, config_for(base, call), f32=precision == "f32")
        per_call[f"{cell_name}/{precision}/{call['label']}/{call['index']}"] = verdict
        counts["ok"] += verdict["ok"]
        counts["order"] += verdict["order_ok"]
        counts["plan"] += verdict["plan_ok"]
        counts["tokens"] += verdict["tokens_ok"]
        counts["draws"] += verdict["draws_ok"]
        counts["zscales"] += verdict["zscales_ok"]
        counts["scores"] += verdict["scores_ok"]
        if precision == "f64":
          worst_z = max(worst_z, verdict["zscale_rel_diff"])
          worst_e = max(worst_e, verdict["energy_rel_diff"])
        logger.info("%s/%s/%s %s", cell_name, precision, call["label"], "ok" if verdict["ok"] else verdict)

    # Controls: float64 comparison on every cell (the control cell is no longer special), each applicable call.
    f64_cell = cells["f64"]
    for call in f64_cell.block_calls():
      for name in CONTROLS:
        spec = control_config(base, call, name)
        if spec is None:
          continue
        config, shift = spec
        applied[name] += 1
        try:
          verdict = check_call(f64_cell, cells["f32"], call, config, f32=False, shift=shift)
        except (ValueError, IndexError):
          rejected[name] = True
          continue
        rejected[name] |= not verdict["ok"]
    logger.info("controls on %s done", cell_name)
  n_calls = len(per_call)

  mutants = {name: ("failed" if rejected[name] else "passed") for name in CONTROLS}
  for name, verdict in mutants.items():
    logger.info("control %-16s applied to %d calls: %s", name, applied[name], "REJECTED" if verdict == "failed" else "NOT REJECTED")

  clean = (
    "pass"
    if counts["ok"] == n_calls == N_CALLS_EXPECTED and dump_flags_pass
    else "fail"
  )  # fmt: skip
  results = {
    "clean": clean, "n_cells": len(CELLS), "n_calls": n_calls, "n_calls_ok": int(counts["ok"]),
    "n_orders_exact": int(counts["order"]), "n_plans_ok": int(counts["plan"]), "n_zscales_ok": int(counts["zscales"]),
    "n_tokens_exact": int(counts["tokens"]), "n_draws_ok": int(counts["draws"]), "n_scores_ok": int(counts["scores"]),
    "dump_flags_pass": bool(dump_flags_pass), "worst_zscale_rel_f64": worst_z, "worst_energy_rel_f64": worst_e,
    "n_listed": len(mutants), "n_failed": sum(v == "failed" for v in mutants.values()), "mutants": mutants,
    "undetected": sorted(k for k, v in mutants.items() if v == "passed"), "per_call": per_call,
  }  # fmt: skip
  path = os.environ.get("BTH_RESULTS_PATH")
  if not path:
    msg = "protonpotts_ph_knobs_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
  logger.info("clean=%s", clean)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
