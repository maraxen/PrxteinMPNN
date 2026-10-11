"""X11c gate: the JS pH host-loop methods in REAL headless Chromium (1 and 4 ORT threads) vs aminx design_methods on replayed draws.

task_id 261011_x11c-browser-methods. Sidecar: protonpotts_ph_methods_browser_parity.bth.toml (pre-registered before this script and before the
harness page existed). Everything that grades is X11b's (protonpotts_ph_methods_node_gate), imported, not restated: cells, cases, draw recording
and replay, the reference arm and its consistency check, the comparator (PROB_BOUND, TIE_MARGIN, exemptions, energy bar) and the four planted
controls. The browser arm is X7c's: a site served over HTTP with COOP/COEP, driven through browser/layer_c/run_p07.mjs UNCHANGED, with the NEW
harness page browser/protonpotts-scorer/ph_methods_page.mjs (ph_design.mjs is loaded as-is; the graded X7c page and site/ are untouched).

Units are (thread setting, cell, case). Each runs in its own Chromium process with its own timeout and a completion stamp under --work-dir, keyed
by the sha256 of its inputs (PDB, job, manifest shas, JS and page digests, thread setting), so a preemption loses at most one unit. Controls are
evaluated per thread setting; a control is detected only when it fired at EVERY thread setting that ran. ``isolated_at_4`` is true only when the
4-thread units ran cross-origin isolated with more than one effective thread.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
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
from scripts.browser_validation.protonpotts_ph_methods_node_gate import (  # noqa: E402
  CASES,
  CENTER_TYPES,
  SAMPLES,
  _cdf_order_array,
  _evaluate_controls,
  _perturbed_model,
  _ref_dict,
  case_seed,
  compare_designs,
  inputs_sha256,
  js_config,
  normalize_js,
  record_case,
  reference_designs,
)
from scripts.browser_validation.protonpotts_score_gate import _verify_artifacts  # noqa: E402

logger = logging.getLogger("protonpotts_ph_methods_browser_parity")

SCORER_DIR = REPO / "browser" / "protonpotts-scorer"
PAGE_FILES = ("ph_methods_page.mjs", "ph_design.mjs", "ph_plan.mjs", "protonpotts_inputs.mjs", "protonpotts_scorer.mjs")
PAGE_HTML = "ph_methods_index.html"
THREADS = (1, 4)
SEED_OFFSET = 300000
UNIT_TIMEOUT_MS = 7_200_000
CONTROL_NAMES = ("tokens", "randint", "perm", "energy")


# ---- pure: aggregation (no browser, no jax) ---------------------------------------------------------------------------------------------------


def new_controls() -> dict[str, dict[str, bool]]:
  return {k: {"applies": False, "flagged": False} for k in CONTROL_NAMES}


def aggregate_controls(by_threads: dict[int, dict[str, dict[str, bool]]], expected: tuple[int, ...] = THREADS) -> dict[str, bool]:
  """A control is detected only if it fired at EVERY expected thread setting (a missing thread setting means not detected)."""
  return {
    f"control_{name}_detected": bool(by_threads) and all(t in by_threads and bool(by_threads[t][name]["flagged"]) for t in expected)
    for name in CONTROL_NAMES
  }


def isolated_at_4(units: list[dict[str, Any]]) -> bool:
  """True only when there is at least one 4-thread unit and every one reported crossOriginIsolated with more than one effective thread."""
  at4 = [u for u in units if u.get("threads") == 4]
  return bool(at4) and all(bool(u.get("cross_origin_isolated")) and int(u.get("num_threads_effective", -1)) > 1 for u in at4)


def threads_effective(units: list[dict[str, Any]]) -> dict[str, int]:
  """{"<threads requested>": min effective thread count reported over that setting's units} (min: one fallback unit must show)."""
  out: dict[str, int] = {}
  for u in units:
    if u.get("num_threads_effective") is None:
      continue
    key, n = str(u["threads"]), int(u["num_threads_effective"])
    out[key] = min(out.get(key, n), n)
  return out


def count_units(rows: list[dict[str, Any]]) -> tuple[int, int]:
  """(n_cases, n_pass): units across every thread setting."""
  return len(rows), sum(bool(r.get("passed")) for r in rows)


# ---- browser arm --------------------------------------------------------------------------------------------------------------------------------


def _digests() -> dict[str, str]:
  names = (*PAGE_FILES, PAGE_HTML)
  return {n: _sha256(SCORER_DIR / n) for n in sorted(names)}


def assemble_base_site(site: Path, scoring: Path, ph: Path, ort_dir: Path) -> None:
  """Everything but cells.json and the PDB: page, JS modules, both artifact sets, the ORT dist."""
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


def _read_stamp(unit_dir: Path, want_sha: str) -> dict[str, Any] | None:
  """Reuse only when the stamp is complete, its inputs hash matches and the payload file still hashes to the stamp."""
  stamp_path, out_path = unit_dir / "stamp.json", unit_dir / "payload.json"
  if not (stamp_path.exists() and out_path.exists()):
    return None
  try:
    stamp = json.loads(stamp_path.read_text())
    payload = json.loads(out_path.read_text())
  except (OSError, ValueError):
    return None
  if not (stamp.get("complete") is True and stamp.get("inputs_sha256") == want_sha):
    return None
  if _sha256(out_path) != stamp.get("output_sha256"):
    return None
  return {"payload": payload, "output_sha256": stamp["output_sha256"]}


def _run_unit(node: str, site: Path, unit_dir: Path, cell: dict[str, Any], pdb: Path, threads: int, timeout_ms: int, want_sha: str) -> dict[str, Any]:
  """One (thread setting, cell, case) in its own Chromium process. Returns {"payload": {...}|None, "error": str, "output_sha256": str|None}."""
  unit_dir.mkdir(parents=True, exist_ok=True)
  shutil.copy(pdb, site / "data" / pdb.name)
  (site / "cells.json").write_text(json.dumps({"cells": [cell | {"pdb": f"data/{pdb.name}"}]}))
  harness = run_harness(node, site, unit_dir / "harness", threads, timeout_ms)
  meta = {"chromiumVersion": str(harness.get("chromiumVersion") or "")}
  if not harness.get("harnessOk"):
    return {"payload": None, "error": f"harness: {str(harness.get('harnessError'))[:800]}", "output_sha256": None, "meta": meta}
  payload = harness["result"]
  rec = next((c for c in payload.get("cells", []) if c.get("id") == cell["id"]), None)
  if rec is None or not rec.get("ok"):
    return {"payload": None, "error": str((rec or {}).get("error") or "cell missing from page result")[:2000], "output_sha256": None, "meta": meta}
  out_path = unit_dir / "payload.json"
  out_path.write_text(json.dumps({"meta": meta, "result": payload}))
  digest = _sha256(out_path)
  (unit_dir / "stamp.json").write_text(json.dumps({"inputs_sha256": want_sha, "output_sha256": digest, "complete": True}))
  return {"payload": {"meta": meta, "result": payload}, "error": "", "output_sha256": digest, "meta": meta}


# ---- driver -----------------------------------------------------------------------------------------------------------------------------------


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
  parser.add_argument("--unit-timeout-ms", type=int, default=UNIT_TIMEOUT_MS, help="milliseconds per (thread setting, cell, case) Chromium run")
  parser.add_argument("--seed-offset", type=int, default=SEED_OFFSET, help="added to every case seed (300000 = X11c held-out draws)")
  parser.add_argument("--threads", default=",".join(str(t) for t in THREADS), help="comma-separated ORT thread settings")
  parser.add_argument("--cells", default="", help="comma-separated cell names to run (smoke subsetting; default all)")
  parser.add_argument("--cases", default="", help="comma-separated case names to run (smoke subsetting; default all)")
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  threads_list = tuple(int(t) for t in args.threads.split(",") if t)
  want_cells = {c for c in args.cells.split(",") if c}
  want_cases = {c for c in args.cases.split(",") if c}

  result: dict[str, Any] = {
    "scoring_manifest_sha256": args.scoring_manifest_sha256, "ph_manifest_sha256": args.ph_manifest_sha256,
    "artifacts_verified": False, "jax_arm_ok": False, "browser_complete": False, "reference_consistent": False,
    "isolated_at_4": False, "threads_effective": {}, "jax_version": "", "onnxruntime_web_version": "", "chromium_version": "",
    "git_hash": "", "git_clean": False, "seed_offset": int(args.seed_offset), "n_cases": 0, "n_pass": 0, "n_fragile_designs": 0,
    "max_prob_discrepancy": 0.0, "cases": [], "units_reused": [], "control_tokens_detected": False, "control_randint_detected": False,
    "control_perm_detected": False, "control_energy_detected": False, "error": "",
  }  # fmt: skip
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  controls_by_t: dict[int, dict[str, dict[str, bool]]] = {t: new_controls() for t in threads_list}
  consistent_flags: list[bool] = []
  unit_envs: list[dict[str, Any]] = []
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
    env = node_env_report(find_node(args.node_bin) or "node")
    result["onnxruntime_web_version"] = str(env.get("onnxruntime-web") or "")
    node = find_node(args.node_bin)
    if node is None:
      msg = "no node binary found (--node-bin, $NODE_BIN, $PATH, nvm)"
      raise RuntimeError(msg)  # noqa: TRY301
    model = _load_model(args.state_npz)
    bad_model = _perturbed_model(model)
    digests = _digests()
    cdf_order = _cdf_order_array()
    site = args.work_dir / "site"
    assemble_base_site(site, args.scoring_models, args.ph_models, args.ort_dir)
    result["jax_arm_ok"] = True

    all_ran = True
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
      binder_mask = np.asarray([c == binder_chain for c in chains])
      res_id = np.asarray([k[1] for k in kept], dtype=np.int32)
      coords = jnp.asarray(feats["X"][0, :, :4, :], dtype=jnp.float32)
      args_t = (coords, jnp.ones(length, jnp.float32), jnp.asarray(feats["R_idx"][0]),
                jnp.asarray(feats["chain_labels"][0]), jnp.ones(length, bool))  # fmt: skip
      table, e_idx = jax.jit(lambda *a, _m=model: protonpotts_table(_m, *a))(*args_t)
      for case_index, (case, method, temperature) in enumerate(CASES):
        if want_cases and case not in want_cases:
          continue
        cid = f"{name}__{case}"
        config = PHDesignConfig(binder_chain=binder_chain, method=method, center_types=CENTER_TYPES,
                                temperature=temperature, samples_per_site=SAMPLES)  # fmt: skip
        seed = case_seed(case_index, cell_index, args.seed_offset)
        rec = record_case(table, e_idx, native, binder_mask, res_id, config, seed)
        ref_raw = reference_designs(table, e_idx, native, binder_mask, res_id, config, rec["replay"])
        ref = [_ref_dict(d, rec["events"][int(d.sample)], cdf_order, rec["argmins"][int(d.sample)]) for d in ref_raw]
        ref_by_sample = {r["sample"]: r for r in ref}
        consistent = len(ref) == SAMPLES and all(
          s in ref_by_sample and ref_by_sample[s]["sequence"] == rec["sequences"][s].tolist() for s in range(SAMPLES))
        consistent_flags.append(consistent)
        cell = {"id": cid, "bucket": _bucket_for(length), "labels": None, "config": js_config(config), "replay": rec["replay"]}
        ctx = {"replay": rec["replay"], "config": config, "method": method, "temperature": temperature, "table": table,
               "e_idx": e_idx, "native": native, "binder_mask": binder_mask, "res_id": res_id, "cdf_order": cdf_order,
               "ref": ref, "bad_model": bad_model, "args_t": args_t}  # fmt: skip
        for threads in threads_list:
          uid = f"t{threads}__{cid}"
          row: dict[str, Any] = {"id": uid, "threads": threads, "cell": name, "case": case, "method": method,
                                 "temperature": temperature, "ran": False, "passed": False, "reused": False, "error": "",
                                 "reference_consistent": bool(consistent)}  # fmt: skip
          result["cases"].append(row)
          want = inputs_sha256({"pdb_sha256": pdb_sha, "job": {k: v for k, v in cell.items() if k != "id"} | {"id": cid},
                                "threads": threads, "scoring_manifest_sha256": args.scoring_manifest_sha256,
                                "ph_manifest_sha256": args.ph_manifest_sha256, "js_and_page_sha256": digests})  # fmt: skip
          unit_dir = args.work_dir / "units" / uid
          reused = _read_stamp(unit_dir, want)
          if reused is not None:
            got = {"payload": reused["payload"], "error": "", "output_sha256": reused["output_sha256"]}
            result["units_reused"].append({"unit": uid, "dir": str(unit_dir), "inputs_sha256": want,
                                           "output_sha256": reused["output_sha256"]})  # fmt: skip
            row["reused"] = True
          else:
            try:
              got = _run_unit(node, site, unit_dir, cell, pdb, threads, args.unit_timeout_ms, want)
            except Exception as exc:  # noqa: BLE001 -- one unit's failure must not lose the others
              got = {"payload": None, "error": f"{type(exc).__name__}: {exc}"[:2000], "output_sha256": None}
              logger.exception("%s failed", uid)
          if got["payload"] is None:
            all_ran = False
            row["error"] = got["error"] or "unit failed"
            continue
          page = got["payload"]["result"]
          chromium = got["payload"]["meta"].get("chromiumVersion", "")
          if chromium and not result["chromium_version"]:
            result["chromium_version"] = chromium
          unit_env = {"threads": threads, "cross_origin_isolated": bool(page.get("cross_origin_isolated")),
                      "num_threads_effective": int(page.get("num_threads_effective", -1))}  # fmt: skip
          row.update(unit_env, ort_version=str(page.get("ort_version") or ""))
          unit_envs.append(unit_env)
          designs = next(c for c in page["cells"] if c["id"] == cid)["designs"]
          js = [normalize_js(d) for d in designs]
          row["ran"] = True
          graded = compare_designs(js, ref, exempt_fragile=True, cdf_order=cdf_order)
          row.update(n_designs=len(js), n_fragile_designs=graded["fragile"], passed=bool(graded["pass"]),
                     n_replay_exhausted=sum(bool(d["replay_exhausted"]) for d in js),
                     max_discrepancy=graded["max_discrepancy"], mismatches=graded["mismatches"][:20],
                     first_fragile=[r["first_fragile"] for r in ref])  # fmt: skip
          result["n_fragile_designs"] += int(graded["fragile"])
          result["max_prob_discrepancy"] = max(result["max_prob_discrepancy"], float(graded["max_discrepancy"]))
          _evaluate_controls(row, controls_by_t[threads], js, ctx)
          logger.info("%s: passed=%s fragile=%s reused=%s", uid, row["passed"], graded["fragile"], row["reused"])
    expected_units = len(result["cases"])
    result["browser_complete"] = expected_units > 0 and all_ran and sum(bool(r["ran"]) for r in result["cases"]) == expected_units
    result["reference_consistent"] = bool(consistent_flags) and all(consistent_flags)
    result["n_cases"], result["n_pass"] = count_units(result["cases"])
    result["isolated_at_4"] = isolated_at_4(unit_envs)
    result["threads_effective"] = threads_effective(unit_envs)
    result |= aggregate_controls(controls_by_t, threads_list)
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("ph methods browser parity gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  logger.info("elapsed %.1fs", time.perf_counter() - t0)
  emit(result, args.out)
  logger.info("done: complete=%s isolated@4=%s cases pass %s/%s", result["browser_complete"], result["isolated_at_4"],
              result["n_pass"], result["n_cases"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
