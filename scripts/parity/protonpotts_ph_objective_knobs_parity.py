"""The ``protonpotts_ph_objective_knobs`` wave: aminx block descent with ``global_weight`` / ``adjacent_repeat_weight`` / ``self_weight`` and (P4m4, amendment 2) ``zscale_mode="single_mutation"`` against upstream.

AMENDMENT 2 (261011, task 261011_zscale-sm; sidecar amended in the same commit, before any run): the oracle is the P4m4 dump
(``dump_protonpotts_ph_objective_knobs_v4.py``: P4m3's eight configurations plus six ``block_zsm_*`` ones), 20 block calls per cell and
precision, 160 in all (``N_CALLS_EXPECTED``, derived below from the configuration list). A call's ``zscale_mode`` is taken from the label
(``ZSM_LABELS``), NOT from the record, and ``KNOBS_MUST_MATCH`` (which still contains ``zscale_mode``) checks it against the record. A sixth
control, ``block_zscales``, runs the single_mutation calls with ``zscale_mode="block"``. No threshold changed.

Pre-registered in ``protonpotts_ph_objective_knobs_parity.bth.toml`` (committed before the port and before this script). Per
(cell, precision, block-descent call) the aminx side is given what upstream was given (sealed table and E_idx, native sequence, binder
mask, the production configuration with the call's ``temperature`` and the three objective knob values recorded in the dump meta, and
the recorded uniforms) and runs the SHIPPED path, ``ph_design._visit_order`` -> ``block_table`` -> ``ph_design._run_optimiser`` ->
``block_descent``. It must match upstream's plan, pooled z-scales (all three, ``sdGlob`` included), final tokens, draw count and scores
(final energy, selectivity gap, and the global term ``g`` of the final sequence).

Reused by import from ``protonpotts_ph_block_parity``: ``Cell``, ``CELLS``, ``UPSTREAM_CDF_ORDER``, ``_to_aminx``, ``derive_plan``,
``compare_call``, ``scores``; from ``protonpotts_ph_knobs_parity``: ``amended_f32_scores`` (the f32 band adopted by amendment there and
a priori here: ``10 x max(upstream f32-f64 spread, aminx f32-f64 spread of the same sequence)``, never below 1e-6 relative). The same
band is applied to ``g``. The g oracle is upstream's recorded ``final_global_dH``; the aminx side is
``ph_descent.global_protonation_dH`` on the final sequence with the masks ``global_masks(S_in, binder)``.

Controls (float64, on every call of the configurations each applies to; rejected when the comparison fails on at least one call):
``ignore_global`` (global_weight 0), ``flip_global`` (weight negated), ``ignore_adjrep`` (adjacent_repeat_weight 0), ``ignore_self``
(self_weight 1.0), ``shifted`` (uniforms shifted by one draw on a T > 0 call), ``block_zscales`` (zscale_mode "block" on the
single_mutation calls).

Run where aminx is importable::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/parity/protonpotts_ph_objective_knobs_parity.py \\
        --encoder-dir <P4e out dir> --ph-dir <P4m4 out dir> --features-dir ~/projects/aminx-oracles-protonpotts

``--smoke`` runs the first cell, three calls, no controls, and does not grade (``clean`` is ``"smoke"``).
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
from aminx.families.protonpotts_mpnn.ph_descent import global_masks, global_protonation_dH  # noqa: E402
from aminx.families.protonpotts_mpnn.ph_design import _run_optimiser, _visit_order  # noqa: E402
from aminx.families.protonpotts_mpnn.ph_plan import block_table  # noqa: E402

# Sibling waves in this directory (on sys.path because this file runs as a script).
from protonpotts_ph_block_parity import (  # noqa: E402
  CELLS,
  F32_FACTOR,
  F32_MIN_REL,
  F64_REL,
  UPSTREAM_CDF_ORDER,
  Cell,
  _to_aminx,
  compare_call,
  derive_plan,
)
from protonpotts_ph_knobs_parity import amended_f32_scores  # noqa: E402

logger = logging.getLogger("protonpotts_ph_objective_knobs_parity")

SENTINEL = 10**6
# Block calls the P4m4 dump records per (cell, precision), derived from its configuration list (v4 ``CONFIGS``): T0 configurations draw
# one sample, T05 configurations two. v3 (P4m3) had the first eight labels, 11 calls; the six block_zsm_* labels add 9.
LABEL_CALLS = {
  "block_glob_p1_T0": 1, "block_glob_p1_T05": 2, "block_glob_m1_T0": 1, "block_adjrep_T0": 1, "block_adjrep_T05": 2,
  "block_self0_T0": 1, "block_self0_T05": 2, "block_self4_T0": 1,
  "block_zsm_T0": 1, "block_zsm_T05": 2, "block_zsm_glob1_T0": 1, "block_zsm_glob1_T05": 2, "block_zsm_self4_T0": 1,
  "block_zsm_self4_T05": 2,
}  # fmt: skip
ZSM_LABELS = frozenset(label for label in LABEL_CALLS if label.startswith("block_zsm_"))
N_CALLS_EXPECTED = len(CELLS) * 2 * sum(LABEL_CALLS.values())  # 4 cells x 2 precisions x 20 = 160 (was 88 for the 8-label P4m3 dump)
# Knob values the dump records per call that the production configuration must reproduce; the three objective knobs are the
# ones under test and are taken FROM the record, so they are not listed. P4m3 runs everything at sweep_order "position".
KNOBS_MUST_MATCH = ("block_size", "combined_lambda", "neighbour_k", "max_mutations", "zscale_mode")
OBJECTIVE_KNOBS = ("global_weight", "adjacent_repeat_weight", "self_weight")
CONTROLS = ("ignore_global", "flip_global", "ignore_adjrep", "ignore_self", "shifted", "block_zscales")


def base_for(call: dict) -> PHDesignConfig:
  """The production configuration whose ``zscale_mode`` is the one this call's LABEL asks for (not read from the record)."""
  return PHDesignConfig(zscale_mode="single_mutation" if call["label"] in ZSM_LABELS else "block")


def config_for(base: PHDesignConfig, call: dict, **override: object) -> PHDesignConfig:
  """The production configuration at the call's temperature with the call's recorded objective knobs (and any override)."""
  knobs = call["knobs"]
  if knobs["sweep_order"] != "position" or float(knobs["temperature"]) != float(call["temperature"]):
    raise ValueError(f"call {call['index']} ({call['label']}): sweep_order/temperature in the knobs disagree with the call")
  for key in KNOBS_MUST_MATCH:
    if getattr(base, key) != knobs[key]:
      raise ValueError(f"call {call['index']} ({call['label']}): knob {key}={knobs[key]!r} != base {getattr(base, key)!r}")
  recorded = {key: float(knobs[key]) for key in OBJECTIVE_KNOBS}
  return dataclasses.replace(base, temperature=float(call["temperature"]), sweep_order="position", **{**recorded, **override})


def run_call(cell: Cell, call: dict, config: PHDesignConfig, *, shift_uniforms: bool = False):  # noqa: ANN201
  """Shipped composition: ``_visit_order`` -> ``block_table`` -> ``_run_optimiser`` (-> ``block_descent``).

  The visit order and the block table are also built here, as ``_run_optimiser`` builds them, only so the plan comparison can read the
  blocks. Returns ``(plan, blocks, result, visit, s_in)``.
  """
  index = call["index"]
  plan = derive_plan(cell, config)
  e_idx = np.asarray(cell.e_idx)
  s_in = _to_aminx(cell.arrays[f"c{index}_S_in"])
  visit = _visit_order(config, plan, cell.table, e_idx, s_in)
  blocks, _block_valid = block_table(e_idx, visit, config.block_size)
  recorded = cell.arrays[f"c{index}_uniforms"]
  need = max(1, config.block_max_rounds * len(plan.designable))
  uniforms = np.zeros(max(need, recorded.shape[0]), dtype=cell.dtype)
  uniforms[: recorded.shape[0]] = recorded
  if shift_uniforms and recorded.shape[0] > 1:
    uniforms[: recorded.shape[0]] = np.roll(recorded, -1)
  residue = np.where(cell.binder, cell.res_id, -SENTINEL - 100 * np.arange(cell.length))
  result = _run_optimiser(
    plan, cell.table, cell.e_idx, e_idx, s_in, residue, config, jnp.asarray(uniforms), jnp.asarray(UPSTREAM_CDF_ORDER),
    cell.binder,
  )  # fmt: skip
  return plan, blocks, result, visit, s_in


def global_g(cell: Cell, seq: np.ndarray, s_in: np.ndarray) -> float:
  """The global term ``g`` of ``seq`` on this cell's table, with the residue classes taken from the pre-pin native ``s_in``."""
  masks = global_masks(jnp.asarray(s_in, dtype=jnp.int32), jnp.asarray(cell.binder))
  return float(global_protonation_dH(cell.table, cell.e_idx, jnp.asarray(seq, dtype=jnp.int32), masks))


def check_global(cell: Cell, other: Cell, call: dict, seq: np.ndarray, s_in: np.ndarray, *, f32: bool) -> tuple[bool, float]:
  """``g`` of the final sequence against upstream's recorded ``final_global_dH``: f64 1e-9 relative, f32 the amended band."""
  want = call["final_global_dH"]
  if want is None:
    return False, float("inf")
  got = global_g(cell, seq, s_in)
  scale = max(1.0, abs(want))
  if f32:
    up_floor = abs(want - other.meta["calls"][call["index"]]["final_global_dH"])
    aminx_floor = abs(got - global_g(other, seq, s_in))
    band = max(F32_FACTOR * max(up_floor, aminx_floor), F32_MIN_REL * scale)
  else:
    band = F64_REL * scale
  return bool(abs(got - want) <= band), float(abs(got - want) / scale)


def check_call(cell: Cell, other: Cell, call: dict, config: PHDesignConfig, *, f32: bool, shift: bool = False) -> dict:
  """One call through the shipped path, compared with upstream on plan, z-scales (3), tokens, draws and scores (energy, gap, g)."""
  plan, blocks, result, visit, s_in = run_call(cell, call, config, shift_uniforms=shift)
  seq = np.asarray(result.seq)
  verdict = compare_call(cell, other, call, (plan, blocks, result), f32=f32)
  if f32:
    verdict.update(amended_f32_scores(cell, other, call, plan, seq))
  g_ok, g_rel = check_global(cell, other, call, seq, s_in, f32=f32)
  verdict["g_ok"] = g_ok
  verdict["g_rel_diff"] = g_rel
  verdict["scores_ok"] = bool(verdict["scores_ok"] and g_ok)  # energy, selectivity gap and g
  # Informational, not graded: the aminx masks come from S_in (the pre-pin seed), upstream's from the structure's residue names.
  verdict["seed_is_native"] = bool(np.array_equal(s_in, cell.native))
  # sweep_order "position" runs no ranker, so the dump records visit_order as null: nothing to compare (the plan check covers it).
  recorded_order = call.get("visit_order")
  verdict["order_ok"] = [int(v) for v in visit] == [int(v) for v in recorded_order] if recorded_order is not None else True
  verdict["ok"] = bool(
    verdict["plan_ok"] and verdict["tokens_ok"] and verdict["draws_ok"] and verdict["zscales_ok"]
    and verdict["scores_ok"] and verdict["order_ok"]
  )  # fmt: skip
  return verdict


def f32_upstream_drift(cell: Cell, other: Cell, call: dict, verdict: dict) -> bool:
  """Amendment 1: an f32 call whose ONLY failing check is tokens_ok, where aminx f32 tokens equal upstream f64 tokens exactly and
  upstream's own f32 tokens differ from its f64 tokens -- upstream f32 drift, reported separately and never counted as a pass."""
  checks = ("plan_ok", "draws_ok", "zscales_ok", "scores_ok", "order_ok")
  if verdict["ok"] or verdict["tokens_ok"] or not all(verdict[k] for k in checks):
    return False
  index = call["index"]
  plan, _blocks, result, _visit, _s_in = run_call(cell, call, config_for(base_for(call), call))
  aminx_f32 = np.asarray(result.seq)
  up_f32 = _to_aminx(cell.arrays[f"c{index}_S_out"])
  up_f64 = _to_aminx(other.arrays[f"c{index}_S_out"])
  return bool(np.array_equal(aminx_f32, up_f64) and not np.array_equal(up_f32, up_f64))


def control_config(base: PHDesignConfig, call: dict, name: str) -> tuple[PHDesignConfig, bool] | None:
  """The (configuration, shift) a control runs on this call, or None if the control does not apply to it."""
  config = config_for(base, call)
  if name == "ignore_global":
    return (config_for(base, call, global_weight=0.0), False) if config.global_weight != 0.0 else None
  if name == "flip_global":
    return (config_for(base, call, global_weight=-config.global_weight), False) if config.global_weight != 0.0 else None
  if name == "ignore_adjrep":
    return (config_for(base, call, adjacent_repeat_weight=0.0), False) if config.adjacent_repeat_weight != 0.0 else None
  if name == "ignore_self":
    return (config_for(base, call, self_weight=1.0), False) if config.self_weight != 1.0 else None
  if name == "shifted":
    return (config, True) if config.temperature > 0 else None
  if name == "block_zscales":  # a port that feeds block-mode z-scales where single_mutation is expected
    return (config_for(base, call, zscale_mode="block"), False) if config.zscale_mode == "single_mutation" else None
  raise ValueError(f"unknown control {name!r}")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--encoder-dir", type=Path, required=True)
  parser.add_argument("--ph-dir", type=Path, required=True)
  parser.add_argument(
    "--features-dir", type=Path, required=True,
    help="accepted for the registered invocation; the dump already verified its native S against the sealed features",
  )  # fmt: skip
  parser.add_argument("--smoke", action="store_true", help="first cell, three calls, no controls; does not grade")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

  manifest = json.loads((args.ph_dir / "ph_objective_knobs_manifest.json").read_text(encoding="utf-8"))
  flags = manifest["flags"]
  dump_flags_pass = flags.get("n_cells") == len(CELLS) and all(v is True for k, v in flags.items() if k != "n_cells")
  cell_names = CELLS[:1] if args.smoke else CELLS

  per_call: dict[str, dict] = {}
  counts = {"ok": 0, "plan": 0, "tokens": 0, "draws": 0, "zscales": 0, "scores": 0, "drift": 0}
  worst_z = worst_e = worst_g = 0.0
  rejected = dict.fromkeys(CONTROLS, False)
  applied = dict.fromkeys(CONTROLS, 0)
  for cell_name in cell_names:
    cells = {p: Cell(args.encoder_dir, args.ph_dir, cell_name, p) for p in ("f64", "f32")}
    for precision, cell in cells.items():
      other = cells["f32" if precision == "f64" else "f64"]
      calls = cell.block_calls()[:3] if args.smoke else cell.block_calls()
      for call in calls:
        verdict = check_call(cell, other, call, config_for(base_for(call), call), f32=precision == "f32")
        verdict["f32_upstream_drift"] = precision == "f32" and f32_upstream_drift(cell, other, call, verdict)
        counts["drift"] += verdict["f32_upstream_drift"]
        per_call[f"{cell_name}/{precision}/{call['label']}/{call['index']}"] = verdict
        counts["ok"] += verdict["ok"]
        counts["plan"] += verdict["plan_ok"]
        counts["tokens"] += verdict["tokens_ok"]
        counts["draws"] += verdict["draws_ok"]
        counts["zscales"] += verdict["zscales_ok"]
        counts["scores"] += verdict["scores_ok"]
        if precision == "f64":
          worst_z = max(worst_z, verdict["zscale_rel_diff"])
          worst_e = max(worst_e, verdict["energy_rel_diff"])
          worst_g = max(worst_g, verdict["g_rel_diff"])
        logger.info("%s/%s/%s %s", cell_name, precision, call["label"], "ok" if verdict["ok"] else verdict)
    if args.smoke:
      continue

    # Controls: float64 comparison on every cell, each applicable call.
    f64_cell = cells["f64"]
    for call in f64_cell.block_calls():
      for name in CONTROLS:
        spec = control_config(base_for(call), call, name)
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
    logger.info("control %-14s applied to %d calls: %s", name, applied[name], "REJECTED" if verdict == "failed" else "NOT REJECTED")

  if args.smoke:
    clean = "smoke"
  else:
    # Amendment 1: upstream f32 drift calls complete the count but are never passes.
    clean = "pass" if counts["ok"] + counts["drift"] == n_calls == N_CALLS_EXPECTED and dump_flags_pass else "fail"
  results = {
    "clean": clean, "n_cells": len(cell_names), "n_calls": n_calls, "n_calls_ok": int(counts["ok"]),
    "n_plans_ok": int(counts["plan"]), "n_zscales_ok": int(counts["zscales"]), "n_tokens_exact": int(counts["tokens"]),
    "n_draws_ok": int(counts["draws"]), "n_scores_ok": int(counts["scores"]), "n_f32_upstream_drift": int(counts["drift"]),
    "dump_flags_pass": bool(dump_flags_pass),
    "worst_zscale_rel_f64": worst_z, "worst_energy_rel_f64": worst_e, "worst_global_rel_f64": worst_g,
    "n_listed": len(mutants), "n_failed": sum(v == "failed" for v in mutants.values()), "mutants": mutants,
    "undetected": sorted(k for k, v in mutants.items() if v == "passed"), "per_call": per_call,
  }  # fmt: skip
  path = os.environ.get("BTH_RESULTS_PATH")
  if not path:
    if args.smoke:
      logger.info("smoke: %s", {k: v for k, v in results.items() if k != "per_call"})
      return 0
    msg = "protonpotts_ph_objective_knobs_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
  logger.info("clean=%s", clean)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
