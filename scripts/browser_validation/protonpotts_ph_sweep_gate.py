"""X7d gate: JS block_descent with sweep_order knn / energy (Node, and headless Chromium at 1 and 4 ORT threads) vs aminx design_structure.

task_id 261011_x7d-sweep-order, debt #2621. Sidecar: protonpotts_ph_sweep_gate.bth.toml (pre-registered before the JS sweepOrder port and before
this script existed); that file is the contract. Reference: ``design_structure`` with ``PHDesignConfig.sweep_order`` knn / energy on X7b's cells
and X7b's reference arm (``protonpotts_ph_loop_gate._design``) and comparator (``same_designs``, factored out of X7b's main unchanged), plus an
EXACT visit-order check and NO fragility exemptions. The Python visit order is obtained from aminx without editing src/: ``ph_design._plan_for``
then ``ph_design._visit_order`` called with the very inputs ``design_structure`` passes (merged table, int32 e_idx, native tokens, binder mask,
res_id, config).

Units are (arm, thread setting, cell, case): node@1, chromium@1, chromium@4. Each runs in its own process with its own timeout and a completion
stamp under --work-dir keyed by the sha256 of its inputs (PDB, job, manifest shas, JS / page digests, arm, thread setting), so a preemption loses
at most one unit; a rerun reuses matching stamps and lists them in ``units_reused``. ORDER control runs are further units of the same kind (the same
JS with sweepOrder "position"). Controls are evaluated per setting; a control counts as detected only if it fired at EVERY setting.

The Chromium arm drives the NEW harness page ph_sweep_index.html + ph_sweep_page.mjs (the graded X7c page ph_page.mjs does not serialise visitOrder
and is untouched), through browser/layer_c/run_p07.mjs unchanged. The Node arm uses the NEW runner run_ph_sweep_node.mjs for the same reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

from scripts.browser_validation.layer_a_common import emit  # noqa: E402
from scripts.browser_validation.layer_c_common import LAYER_C_DIR, find_node, node_env_report  # noqa: E402
from scripts.browser_validation.potts_browser_parity import git_state, run_harness  # noqa: E402
from scripts.browser_validation.protonpotts_export_gate import STRUCTURES, _bucket_for, _load_model, _sha256  # noqa: E402
from scripts.browser_validation.protonpotts_ph_loop_gate import (  # noqa: E402
  BLOCK_ROUNDS,
  ENERGY_BAR,
  MAX_MUTATIONS,
  SAMPLES,
  _design,
  _rel,
  same_designs,
)
from scripts.browser_validation.protonpotts_ph_methods_browser_parity import (  # noqa: E402
  _read_stamp,
  count_units,
  isolated_at_4,
  threads_effective,
)
from scripts.browser_validation.protonpotts_ph_methods_node_gate import (  # noqa: E402
  _energies_under,
  _perturbed_model,
  inputs_sha256,
)
from scripts.browser_validation.protonpotts_score_gate import _verify_artifacts  # noqa: E402

logger = logging.getLogger("protonpotts_ph_sweep_gate")

SEED = 4031
CENTER_TYPES = ("HIS-P", "ASP-P", "GLU-P")
CASES = (  # (name, sweep_order, temperature)
  ("knn_t0", "knn", 0.0),
  ("energy_t0", "energy", 0.0),
  ("knn_t005", "knn", 0.05),
  ("energy_t005", "energy", 0.05),
)
ARMS = ("node@1", "chromium@1", "chromium@4")
CONTROL_NAMES = ("order", "tokens", "energy")
SCORER_DIR = REPO / "browser" / "protonpotts-scorer"
NODE_RUNNER = "run_ph_sweep_node.mjs"
NODE_FILES = ("ph_design.mjs", "ph_plan.mjs", "protonpotts_inputs.mjs", "protonpotts_scorer.mjs", NODE_RUNNER)
PAGE_FILES = ("ph_sweep_page.mjs", "ph_design.mjs", "ph_plan.mjs", "protonpotts_inputs.mjs", "protonpotts_scorer.mjs")
PAGE_HTML = "ph_sweep_index.html"
UNIT_TIMEOUT_MS = 7_200_000


# ---- pure: comparison and aggregation (no browser, no jax) ----------------------------------------------------------------------------------


def first_diff_index(a: list[Any], b: list[Any]) -> int | None:
  """First index at which two sequences differ (a length difference counts at the shorter length); None when equal."""
  for i, (x, y) in enumerate(zip(a, b, strict=False)):
    if x != y:
      return i
  return None if len(a) == len(b) else min(len(a), len(b))


def locate_difference(js_designs: list[dict[str, Any]], ref_designs: list[Any], ref_visit: list[int]) -> dict[str, Any] | None:
  """The first disagreement as {sample, field, index} (None when none): visit order, sequence, designable, pins, label, draws, energies."""
  if len(js_designs) != len(ref_designs):
    return {"sample": None, "field": "n_designs", "index": min(len(js_designs), len(ref_designs))}
  for js, ref in zip(js_designs, ref_designs, strict=True):
    sample = int(js.get("sample", -1))
    got_visit = js.get("visit_order")
    if got_visit is None:
      return {"sample": sample, "field": "visit_order", "index": None}
    i = first_diff_index([int(x) for x in got_visit], [int(x) for x in ref_visit])
    if i is not None:
      return {"sample": sample, "field": "visit_order", "index": i}
    i = first_diff_index([int(x) for x in js["sequence"]], [int(x) for x in np.asarray(ref.sequence)])
    if i is not None:
      return {"sample": sample, "field": "sequence", "index": i}
    i = first_diff_index([int(x) for x in js["designable"]], [int(x) for x in ref.designable])
    if i is not None:
      return {"sample": sample, "field": "designable", "index": i}
    js_pins = [(p["position"], p["protIdx"], p["resId"]) for p in js["pins"]]
    ref_pins = [(p.position, p.prot_idx, p.res_id) for p in ref.pins]
    i = first_diff_index(js_pins, ref_pins)
    if i is not None:
      return {"sample": sample, "field": "pins", "index": i}
    if js["label"] != ref.label:
      return {"sample": sample, "field": "label", "index": None}
    if int(js["n_draws"]) != int(ref.n_draws):
      return {"sample": sample, "field": "n_draws", "index": None}
    if _rel(ref.final_potts_energy, js["final_potts_energy"]) > ENERGY_BAR:
      return {"sample": sample, "field": "final_potts_energy", "index": None}
    if _rel(ref.selective_energy, js["selective_energy"]) > ENERGY_BAR:
      return {"sample": sample, "field": "selective_energy", "index": None}
  return None


def grade_unit(js_designs: list[dict[str, Any]], ref_designs: list[Any], ref_visit: list[int]) -> dict[str, Any]:
  """X7b's comparator (``same_designs``, unchanged) AND the exact visit order of every design. No exemptions."""
  row = same_designs(js_designs, ref_designs)
  visit_ok = bool(js_designs) and all(
    d.get("visit_order") is not None and [int(x) for x in d["visit_order"]] == [int(x) for x in ref_visit] for d in js_designs)
  row["visit_order_equal"] = bool(visit_ok)
  row["passed"] = bool(row["passed"] and visit_ok)
  row["first_difference"] = None if row["passed"] else locate_difference(js_designs, ref_designs, ref_visit)
  return row


def new_controls() -> dict[str, dict[str, int]]:
  return {k: {"applies": 0, "flagged": 0} for k in CONTROL_NAMES}


def record_control(controls: dict[str, dict[str, int]], name: str, *, applies: bool, flagged: bool) -> None:
  if applies:
    controls[name]["applies"] += 1
    controls[name]["flagged"] += int(bool(flagged))


def aggregate_controls(by_setting: dict[str, dict[str, dict[str, int]]], expected: tuple[str, ...]) -> dict[str, bool]:
  """A control is detected only if, at EVERY expected setting, it applied at least once and flagged on every case it applied to."""
  def fired(setting: str, name: str) -> bool:
    c = by_setting.get(setting, {}).get(name)
    return bool(c) and c["applies"] > 0 and c["flagged"] == c["applies"]

  return {f"control_{n}_detected": bool(expected) and all(fired(s, n) for s in expected) for n in CONTROL_NAMES}


def uniforms_for(cell_index: int, temperature: float) -> list[np.ndarray] | None:
  """SEED 4031 draws, one array per sample; a function of (cell, temperature) only, so knn and energy cases share them and every unit is reproducible alone."""
  if temperature <= 0:
    return None
  rng = np.random.default_rng([SEED, cell_index, 1])
  return [rng.uniform(size=BLOCK_ROUNDS * MAX_MUTATIONS).astype(np.float32) for _ in range(SAMPLES)]


def js_config(binder_chain: str, sweep_order: str, temperature: float) -> dict[str, Any]:
  return {"binderChain": binder_chain, "centerTypes": list(CENTER_TYPES), "temperature": temperature, "samplesPerSite": SAMPLES,
          "maxMutations": MAX_MUTATIONS, "blockMaxRounds": BLOCK_ROUNDS, "sweepOrder": sweep_order}


# ---- reference ---------------------------------------------------------------------------------------------------------------------------------


def reference_visit_order(table: Any, e_idx: Any, native: np.ndarray, binder: np.ndarray, res_id: np.ndarray, config: Any) -> list[int]:  # noqa: ANN401
  """The visit order design_structure's optimiser uses, without editing src/: ``_plan_for`` then ``_visit_order`` on design_structure's own inputs."""
  import jax.numpy as jnp  # noqa: PLC0415

  from aminx.families.protonpotts_mpnn import ph_design  # noqa: PLC0415

  table_j = jnp.asarray(table)
  e_idx_np = np.asarray(e_idx, dtype=np.int32)
  native_np = np.asarray(native, dtype=np.int32)
  plan = ph_design._plan_for(table_j, e_idx_np, native_np, np.asarray(binder, dtype=bool), np.asarray(res_id), config)  # noqa: SLF001
  if plan is None or not plan.designable:
    return []
  return [int(p) for p in ph_design._visit_order(config, plan, table_j, e_idx_np, native_np)]  # noqa: SLF001


# ---- units -------------------------------------------------------------------------------------------------------------------------------------


def _digests(names: tuple[str, ...]) -> dict[str, str]:
  return {n: _sha256(SCORER_DIR / n) for n in sorted(names)}


def assemble_base_site(site: Path, scoring: Path, ph: Path, ort_dir: Path) -> None:
  """Everything but cells.json and the PDB: the sweep page, the JS modules, both artifact sets, the ORT dist."""
  if site.exists():
    shutil.rmtree(site)
  for sub in ("models/scoring", "models/ph", "ort", "data"):
    (site / sub).mkdir(parents=True)
  shutil.copy(SCORER_DIR / PAGE_HTML, site / "index.html")
  for name in PAGE_FILES:
    shutil.copy(SCORER_DIR / name, site / name)
  for src, dst in ((scoring, "scoring"), (ph, "ph")):
    manifest = json.loads((src / "MANIFEST.json").read_text())
    shutil.copy(src / "MANIFEST.json", site / "models" / dst / "MANIFEST.json")
    for graph_file in sorted({g["file"] for b in manifest["buckets"] for g in b["graphs"].values()}):
      shutil.copy(src / graph_file, site / "models" / dst / graph_file)
  dist = ort_dir / "node_modules" / "onnxruntime-web" / "dist"
  for f in dist.iterdir():
    if f.is_file():
      shutil.copy(f, site / "ort" / f.name)


def _finish(unit_dir: Path, want_sha: str, payload: dict[str, Any]) -> dict[str, Any]:
  out_path = unit_dir / "payload.json"
  out_path.write_text(json.dumps(payload))
  digest = _sha256(out_path)
  (unit_dir / "stamp.json").write_text(json.dumps({"inputs_sha256": want_sha, "output_sha256": digest, "complete": True}))
  return {"payload": payload, "error": "", "output_sha256": digest}


def run_chromium_unit(node: str, site: Path, unit_dir: Path, job: dict[str, Any], pdb: Path, threads: int, timeout_ms: int, want_sha: str) -> dict[str, Any]:
  unit_dir.mkdir(parents=True, exist_ok=True)
  shutil.copy(pdb, site / "data" / pdb.name)
  (site / "cells.json").write_text(json.dumps({"cells": [job | {"pdb": f"data/{pdb.name}"}]}))
  harness = run_harness(node, site, unit_dir / "harness", threads, timeout_ms)
  meta = {"chromiumVersion": str(harness.get("chromiumVersion") or "")}
  if not harness.get("harnessOk"):
    return {"payload": None, "error": f"harness: {str(harness.get('harnessError'))[:800]}", "output_sha256": None}
  page = harness["result"]
  rec = next((c for c in page.get("cells", []) if c.get("id") == job["id"]), None)
  if rec is None or not rec.get("ok"):
    return {"payload": None, "error": str((rec or {}).get("error") or "cell missing from page result")[:2000], "output_sha256": None}
  env = {"cross_origin_isolated": bool(page.get("cross_origin_isolated")), "num_threads_effective": int(page.get("num_threads_effective", -1))}
  return _finish(unit_dir, want_sha, {"meta": meta, "env": env, "designs": rec["designs"]})


def run_node_unit(node: str, ort_dir: Path, scoring: Path, ph: Path, unit_dir: Path, job: dict[str, Any], pdb: Path, timeout_s: int, want_sha: str) -> dict[str, Any]:
  unit_dir.mkdir(parents=True, exist_ok=True)
  cases_path, out_path = unit_dir / "node_cases.json", unit_dir / "node_out.json"
  out_path.unlink(missing_ok=True)
  cases_path.write_text(json.dumps({"cases": [{"id": job["id"], "bucket": job["bucket"], "pdb_file": str(pdb), "labels": job["labels"],
                                               "config": job["config"], "uniforms": job["uniforms"]}]}))
  proc = subprocess.run(  # noqa: S603 -- fixed argv we built
    [node, str(SCORER_DIR / NODE_RUNNER), "--ort-dir", str(ort_dir), "--scoring-models", str(scoring), "--ph-models", str(ph),
     "--cases", str(cases_path), "--out", str(out_path)],
    capture_output=True, text=True, timeout=timeout_s, check=False,
  )
  if proc.returncode != 0 or not out_path.exists():
    return {"payload": None, "error": f"node runner rc={proc.returncode}: {proc.stderr[-1500:]}", "output_sha256": None}
  node_out = json.loads(out_path.read_text())
  rec = next((r for r in node_out["results"] if r["id"] == job["id"]), None)
  if rec is None or not rec.get("ok"):
    return {"payload": None, "error": str((rec or {}).get("error") or "case missing from node result")[:2000], "output_sha256": None}
  env = {"cross_origin_isolated": None, "num_threads_effective": int(node_out.get("num_threads") or 0)}
  return _finish(unit_dir, want_sha, {"meta": {}, "env": env, "designs": rec["designs"]})


# ---- driver ------------------------------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--scoring-models", type=Path, required=True, help="X6a artifact dir")
  parser.add_argument("--scoring-manifest-sha256", required=True)
  parser.add_argument("--ph-models", type=Path, required=True, help="X7a artifact dir")
  parser.add_argument("--ph-manifest-sha256", required=True)
  parser.add_argument("--potts-root", type=Path,
                      default=Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser())
  parser.add_argument("--state-npz", type=Path,
                      default=Path(os.environ.get("AMINX_PROTONPOTTS_STATE", "~/scratch/v6_state.npz")).expanduser())
  parser.add_argument("--node-bin", default=os.environ.get("NODE_BIN"))
  parser.add_argument("--ort-dir", type=Path, default=Path(os.environ.get("ORT_WEB_DIR", str(LAYER_C_DIR))))
  parser.add_argument("--unit-timeout-ms", type=int, default=UNIT_TIMEOUT_MS, help="milliseconds per unit process")
  parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated arm@threads settings (smoke subsetting)")
  parser.add_argument("--cells", default="", help="comma-separated cell names (smoke subsetting; default all)")
  parser.add_argument("--cases", default="", help="comma-separated case names (smoke subsetting; default all)")
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  settings = tuple(a for a in args.arms.split(",") if a)
  for s in settings:
    if s not in ARMS:
      msg = f"unknown arm {s!r}; choose from {ARMS}"
      raise SystemExit(msg)
  want_cells = {c for c in args.cells.split(",") if c}
  want_cases = {c for c in args.cases.split(",") if c}

  # EXACTLY the sidecar's [result_schema] keys.
  result: dict[str, Any] = {
    "scoring_manifest_sha256": args.scoring_manifest_sha256, "ph_manifest_sha256": args.ph_manifest_sha256, "artifacts_verified": False,
    "jax_arm_ok": False, "node_ok": False, "browser_complete": False, "isolated_at_4": False, "threads_effective": {},
    "jax_version": "", "onnxruntime_web_version": "", "node_version": "", "chromium_version": "", "git_hash": "", "git_clean": False,
    "n_cases": 0, "n_pass": 0, "cases": [], "units_reused": [], "control_order_detected": False, "control_tokens_detected": False,
    "control_energy_detected": False, "error": "",
  }  # fmt: skip
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  controls_by_setting: dict[str, dict[str, dict[str, int]]] = {s: new_controls() for s in settings}
  unit_envs: list[dict[str, Any]] = []
  ran_by_arm: dict[str, list[bool]] = {"node": [], "chromium": []}
  try:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    if jax.config.jax_enable_x64:
      msg = "jax_enable_x64 is set; refusing"
      raise RuntimeError(msg)  # noqa: TRY301
    from aminx.families.protonpotts_mpnn.energy import protonpotts_table  # noqa: PLC0415
    from aminx.families.protonpotts_mpnn.features import featurize_pdb, kept_residues  # noqa: PLC0415
    from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig  # noqa: PLC0415

    _verify_artifacts(args.scoring_models, args.scoring_manifest_sha256)
    _verify_artifacts(args.ph_models, args.ph_manifest_sha256)
    result["artifacts_verified"] = True
    result["jax_version"] = jax.__version__
    result["git_hash"], result["git_clean"] = git_state(REPO)
    node = find_node(args.node_bin)
    if node is None:
      msg = "no node binary found (--node-bin, $NODE_BIN, $PATH, nvm)"
      raise RuntimeError(msg)  # noqa: TRY301
    env = node_env_report(node)
    result["node_version"] = str(env.get("node") or "")
    result["onnxruntime_web_version"] = str(env.get("onnxruntime-web") or "")
    model = _load_model(args.state_npz)
    bad_model = _perturbed_model(model)
    digests = {"node": _digests(NODE_FILES), "chromium": _digests((*PAGE_FILES, PAGE_HTML))}
    site = args.work_dir / "site"
    if any(s.startswith("chromium") for s in settings):
      assemble_base_site(site, args.scoring_models, args.ph_models, args.ort_dir)
    timeout_ms = args.unit_timeout_ms

    def execute(setting: str, kind: str, uid: str, job: dict[str, Any], pdb: Path, pdb_sha: str) -> dict[str, Any]:
      """Run (or reuse the stamp of) one unit. Returns {"payload"|None, "error", "reused"}."""
      arm, threads = setting.split("@")
      want = inputs_sha256({"kind": kind, "pdb_sha256": pdb_sha, "job": job, "arm": arm, "threads": int(threads),
                            "scoring_manifest_sha256": args.scoring_manifest_sha256, "ph_manifest_sha256": args.ph_manifest_sha256,
                            "js_and_page_sha256": digests[arm], "onnxruntime_web_version": result["onnxruntime_web_version"],
                            "node_version": result["node_version"]})  # fmt: skip
      unit_dir = args.work_dir / "units" / setting / uid
      reused = _read_stamp(unit_dir, want)
      if reused is not None:
        result["units_reused"].append({"unit": f"{setting}__{uid}", "dir": str(unit_dir), "inputs_sha256": want,
                                       "output_sha256": reused["output_sha256"]})  # fmt: skip
        return {"payload": reused["payload"], "error": "", "reused": True}
      try:
        if arm == "node":
          got = run_node_unit(node, args.ort_dir, args.scoring_models, args.ph_models, unit_dir, job, pdb, timeout_ms // 1000, want)
        else:
          got = run_chromium_unit(node, site, unit_dir, job, pdb, int(threads), timeout_ms, want)
      except Exception as exc:  # noqa: BLE001 -- one unit's failure must not lose the others
        logger.exception("%s %s failed", setting, uid)
        got = {"payload": None, "error": f"{type(exc).__name__}: {exc}"[:2000]}
      return {"payload": got["payload"], "error": got["error"], "reused": False}

    for cell_index, (name, (rel, _sha)) in enumerate(STRUCTURES.items()):
      if want_cells and name not in want_cells:
        continue
      pdb = args.potts_root / "energy_benchmark_datasets" / rel
      pdb_sha = hashlib.sha256(pdb.read_bytes()).hexdigest()
      feats = featurize_pdb(pdb, None)
      kept = kept_residues(pdb)
      native = np.asarray(feats["S"][0], dtype=np.int32)
      length = native.shape[0]
      chains = [k[0] for k in kept]
      binder_chain = chains[-1]
      binder = np.asarray([c == binder_chain for c in chains])
      res_id = np.asarray([k[1] for k in kept], dtype=np.int32)
      args_t = (jnp.asarray(feats["X"][0, :, :4, :], dtype=jnp.float32), jnp.ones(length, jnp.float32),
                jnp.asarray(feats["R_idx"][0]), jnp.asarray(feats["chain_labels"][0]), jnp.ones(length, bool))  # fmt: skip
      table, e_idx = jax.jit(lambda *a, _m=model: protonpotts_table(_m, *a))(*args_t)
      bucket = _bucket_for(length)

      def cfg(order: str, temperature: float, _chain: str = binder_chain) -> Any:  # noqa: ANN401
        return PHDesignConfig(binder_chain=_chain, center_types=CENTER_TYPES, temperature=temperature, samples_per_site=SAMPLES,
                              max_mutations=MAX_MUTATIONS, block_max_rounds=BLOCK_ROUNDS, sweep_order=order)

      def job_for(uid: str, order: str, temperature: float, uniforms: Any, _bucket: int = bucket, _chain: str = binder_chain) -> dict[str, Any]:  # noqa: ANN401
        return {"id": uid, "bucket": _bucket, "labels": None, "config": js_config(_chain, order, temperature),
                "uniforms": None if uniforms is None else [u.tolist() for u in uniforms]}

      position_visit = reference_visit_order(table, e_idx, native, binder, res_id, cfg("position", 0.05))
      selected = [(c, o, t) for c, o, t in CASES if not want_cases or c in want_cases]
      refs: dict[str, dict[str, Any]] = {}
      for case, order, temperature in selected:
        uniforms = uniforms_for(cell_index, temperature)
        config = cfg(order, temperature)
        ref = _design(table, e_idx, native, binder, res_id, config, uniforms)
        visit = reference_visit_order(table, e_idx, native, binder, res_id, config)
        rolled = _design(table, e_idx, native, binder, res_id, config, [np.roll(u, 1) for u in uniforms]) if temperature > 0 else None
        refs[case] = {"ref": ref, "visit": visit, "rolled": rolled, "uniforms": uniforms, "order": order, "temperature": temperature,
                      "live": visit != position_visit}  # fmt: skip
        logger.info("reference %s__%s: %d designs, visit order differs from position: %s", name, case, len(ref), visit != position_visit)

      for setting in settings:
        arm, threads_s = setting.split("@")
        threads = int(threads_s)
        controls = controls_by_setting[setting]
        # ORDER control runs: the JS with sweepOrder "position", once per temperature present, graded against every selected knn / energy reference.
        order_runs: dict[float, list[dict[str, Any]] | None] = {}
        for temperature in sorted({r["temperature"] for r in refs.values()}):
          uniforms = uniforms_for(cell_index, temperature)
          uid = f"{name}__order_ctrl_t{temperature:g}"
          got = execute(setting, "order_ctrl", uid, job_for(uid, "position", temperature, uniforms), pdb, pdb_sha)
          ran_by_arm[arm].append(got["payload"] is not None)
          order_runs[temperature] = None if got["payload"] is None else got["payload"]["designs"]
          if got["payload"] is None:
            logger.error("%s %s: %s", setting, uid, got["error"])
        for case, order, temperature in selected:
          r = refs[case]
          uid = f"{name}__{case}"
          row: dict[str, Any] = {"id": f"{setting}__{uid}", "arm": arm, "threads": threads, "cell": name, "case": case,
                                 "sweep_order": order, "temperature": temperature, "ran": False, "passed": False, "reused": False,
                                 "error": "", "ref_visit_order_differs_from_position": bool(r["live"])}  # fmt: skip
          result["cases"].append(row)
          got = execute(setting, "main", uid, job_for(uid, order, temperature, r["uniforms"]), pdb, pdb_sha)
          ran_by_arm[arm].append(got["payload"] is not None)
          row["reused"] = got["reused"]
          if got["payload"] is None:
            row["error"] = got["error"] or "unit failed"
            logger.error("%s %s: %s", setting, uid, row["error"])
            continue
          payload = got["payload"]
          chromium = payload["meta"].get("chromiumVersion", "")
          if chromium and not result["chromium_version"]:
            result["chromium_version"] = chromium
          unit_env = {"threads": threads, "arm": arm, **payload["env"]}
          row.update(cross_origin_isolated=payload["env"]["cross_origin_isolated"], num_threads_effective=payload["env"]["num_threads_effective"])
          if arm == "chromium":
            unit_envs.append(unit_env)
          js = payload["designs"]
          row["ran"] = True
          row.update(grade_unit(js, r["ref"], r["visit"]))
          # ORDER control: position-order JS vs this reference through the SAME comparator; applies only where the knob is live here.
          pos = order_runs.get(temperature)
          if pos is not None:
            order_graded = grade_unit(pos, r["ref"], r["visit"])
            record_control(controls, "order", applies=bool(r["live"]), flagged=not order_graded["passed"])
            row["order_control"] = {"applies": bool(r["live"]), "flagged": not order_graded["passed"],
                                    "first_difference": order_graded["first_difference"]}  # fmt: skip
          # TOKENS control (T > 0): reference with every uniform array rolled by one.
          if r["rolled"] is not None:
            tok_graded = grade_unit(js, r["rolled"], r["visit"])
            record_control(controls, "tokens", applies=True, flagged=not tok_graded["passed"])
            row["tokens_control"] = {"applies": True, "flagged": not tok_graded["passed"]}
          # ENERGY control: once per setting, on its first ran unit (X7b's): JS sequences under a perturbed head must disagree with the JS energies.
          if controls["energy"]["applies"] == 0:
            seqs = [[int(x) for x in d["sequence"]] for d in js]
            p_energy = _energies_under(bad_model, args_t, seqs)
            flagged = all(_rel(p, d["final_potts_energy"]) > ENERGY_BAR for p, d in zip(p_energy, js, strict=True))
            record_control(controls, "energy", applies=True, flagged=flagged)
            row["energy_control"] = {"applies": True, "flagged": bool(flagged)}
          logger.info("%s %s: passed=%s visit_equal=%s reused=%s first_difference=%s", setting, uid, row["passed"],
                      row.get("visit_order_equal"), row["reused"], row.get("first_difference"))
    result["jax_arm_ok"] = True
    result["node_ok"] = bool(ran_by_arm["node"]) and all(ran_by_arm["node"])
    result["browser_complete"] = bool(ran_by_arm["chromium"]) and all(ran_by_arm["chromium"])
    result["n_cases"], result["n_pass"] = count_units(result["cases"])
    result["isolated_at_4"] = isolated_at_4(unit_envs)
    result["threads_effective"] = threads_effective(
      [{"threads": f"{u['arm']}@{u['threads']}", "num_threads_effective": u["num_threads_effective"]} for u in unit_envs]
      + [{"threads": "node@1", "num_threads_effective": c["num_threads_effective"]} for c in result["cases"]
         if c.get("arm") == "node" and c.get("ran")])  # fmt: skip
    result |= aggregate_controls(controls_by_setting, settings)
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("ph sweep gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  logger.info("elapsed %.1fs", time.perf_counter() - t0)
  emit(result, args.out)
  logger.info("done: node_ok=%s browser_complete=%s isolated@4=%s cases pass %s/%s controls order=%s tokens=%s energy=%s",
              result["node_ok"], result["browser_complete"], result["isolated_at_4"], result["n_pass"], result["n_cases"],
              result["control_order_detected"], result["control_tokens_detected"], result["control_energy_detected"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
