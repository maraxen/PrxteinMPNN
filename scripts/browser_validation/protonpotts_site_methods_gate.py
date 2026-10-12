"""X12 deployed-site gate: the built ProtonPotts site reproduces the graded X11c (pH host-loop methods) and X7d (sweep_order) harness pages bitwise.

task_id 261011_x12-site-methods, debt #2625. Sidecar: protonpotts_site_methods_gate.bth.toml (pre-registered before this script existed); that
file is the contract. Built from X10's machinery (protonpotts_site_gate): the dist build served from a SUBPATH, PAGES (no COOP/COEP, 1 thread) and
COI (COOP/COEP, 4 threads) conditions, the window.__PP_RUN__ driver inside a subpath iframe, the integrity tamper copy, the environment checks and
the unit/stamp discipline are imported from it, not restated.

Two paths, each compared with its graded harness page at the SAME thread count (PAGES site@1 vs harness@1; COI site@4 vs harness@4):
  METHODS: jobs of X11c (cells x CASES, draw replays recorded by X11b's record_case at --seed-offset, default 400000); harness page
           ph_methods_page.mjs via run_harness; site via the hook with params.config = the FULL js_config and params.replay = the same replay.
  SWEEP:   jobs of X7d (cells x CASES, centre types HIS-P ASP-P GLU-P, uniforms from --sweep-seed, default 5031); harness page ph_sweep_page.mjs;
           site via the hook with the full config (incl. sweepOrder) and params.uniforms.

KNOWN LIMIT, not hidden: the deployed worker (site/protonpotts/worker.mjs, untouched here) serialises only the design fields in SITE_COMPARED_FIELDS
(+ a derived `symbols`). The harness pages additionally return events / argmins / replay_exhausted (methods) and visit_order (sweep); the site cannot
return them, so they are listed in ``excluded_fields`` as NOT_RETURNED_BY_SITE and are NOT compared. A harness field that is neither compared, excluded
as run-description, nor listed as not returned makes the comparison fail (``unaccounted_harness_field``), so nothing is dropped silently.

Units are (path, arm site|harness, thread setting, cell, case[, control variant]); each runs in its own process with its own timeout and a completion
stamp under --work-dir keyed by the sha256 of its inputs (job incl. replay / uniforms, PDB, manifests, page / module digests or the site tree digest).
The harness page grades nothing; every comparison lives here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

from scripts.browser_validation import protonpotts_ph_methods_browser_parity as methods_page  # noqa: E402
from scripts.browser_validation import protonpotts_ph_methods_node_gate as methods_gate  # noqa: E402
from scripts.browser_validation import protonpotts_ph_sweep_gate as sweep_gate  # noqa: E402
from scripts.browser_validation.layer_a_common import emit  # noqa: E402
from scripts.browser_validation.layer_c_common import LAYER_C_DIR, find_node, node_env_report  # noqa: E402
from scripts.browser_validation.potts_browser_parity import git_state  # noqa: E402
from scripts.browser_validation.protonpotts_export_gate import STRUCTURES, _bucket_for, _load_model, _sha256  # noqa: E402
from scripts.browser_validation.protonpotts_score_gate import _verify_artifacts  # noqa: E402
from scripts.browser_validation.protonpotts_site_gate import (  # noqa: E402
  DRIVER_HTML,
  FLIP_OFFSET,
  KEEP_OUT_OF_DIGEST,
  ORT_VERSION,
  _env_ok,
  _f32_equal,
  _link_tree,
  _reusable,
  _run_p07,
  _same_bytes,
  _site_problems,
  flip_byte,
  inputs_sha256,
  tree_digest,
)

logger = logging.getLogger("protonpotts_site_methods_gate")

SCORER_REL = "browser/protonpotts-scorer"
PATHS = ("methods", "sweep")
# (label, threads, isolated): PAGES = GitHub Pages / localfold condition, COI = COOP/COEP.
SETTINGS = (("pages", 1, False), ("coi", 4, True))
METHODS_SEED_OFFSET = 400000
SWEEP_SEED = 5031
UNIT_TIMEOUT_MS = 7_200_000
PLANTED = "__planted"
ORDER_VARIANT = "__order_position"

# Per-design fields the deployed worker returns AND the harness pages return: compared exactly.
SITE_COMPARED_FIELDS = ("sequence", "sample", "method", "label", "designable", "pins", "final_potts_energy", "selective_energy",
                        "selective_energies", "n_draws")
FLOAT_FIELDS = frozenset({"final_potts_energy", "selective_energy", "selective_energies"})
# Harness fields the deployed worker does not serialise (worker.mjs untouched): not compared, said so in the record.
NOT_RETURNED_BY_SITE = ("events", "argmins", "replay_exhausted", "visit_order")
# Run-description fields (harness page-level or per-cell, and the site's derived / site-only keys): excluded from the comparison.
RUN_DESCRIPTION_FIELDS = ("wall_ms", "wall_ms_all", "warmup_ms_all", "rep_sig_all", "session_create_ms", "ctrl_measured_ms", "user_agent",
                          "chromiumVersion", "num_threads_requested", "num_threads_after_init", "n_cells", "symbols (site-only derived)",
                          "bucket", "l_total")
EXCLUDED_FIELDS = [*(f"NOT_RETURNED_BY_SITE:{f}" for f in NOT_RETURNED_BY_SITE), *(f"RUN_DESCRIPTION:{f}" for f in RUN_DESCRIPTION_FIELDS)]
ACCOUNTED_HARNESS_FIELDS = frozenset(SITE_COMPARED_FIELDS) | frozenset(NOT_RETURNED_BY_SITE)


# ---------------------------------------------------------------- pure helpers (unit-tested)


def _exact_equal(a: Any, b: Any) -> bool:  # noqa: ANN401
  """Strict structural equality: bool is not int, int is not float, dict keys and list lengths must agree."""
  if isinstance(a, bool) or isinstance(b, bool):
    return isinstance(a, bool) and isinstance(b, bool) and a == b
  if isinstance(a, dict):
    return isinstance(b, dict) and a.keys() == b.keys() and all(_exact_equal(a[k], b[k]) for k in a)
  if isinstance(a, (list, tuple)):
    return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_exact_equal(x, y) for x, y in zip(a, b, strict=True))
  return type(a) is type(b) and a == b


def _float_field_equal(a: Any, b: Any) -> bool:  # noqa: ANN401
  """Float32 bit equality of a number or a list of numbers; None equals only None."""
  if a is None or b is None:
    return a is None and b is None
  try:
    return _f32_equal(a, b)
  except (TypeError, ValueError):
    return False


def compare_designs(ref_designs: Any, got_designs: Any) -> dict[str, Any]:  # noqa: ANN401
  """Harness (ref) vs deployed (got) designs, every SITE_COMPARED_FIELDS entry exact (float32 bits for FLOAT_FIELDS).

  Run-description keys and the site-only `symbols` are ignored; harness keys that are neither compared nor listed as not returned by the site are
  reported as unaccounted (a failure), so a new harness field cannot be dropped silently.
  """
  if not isinstance(ref_designs, list) or not isinstance(got_designs, list) or len(ref_designs) != len(got_designs) or not ref_designs:
    return {"exact": False, "mismatches": ["designs"]}
  mismatches: list[str] = []
  for i, (r, g) in enumerate(zip(ref_designs, got_designs, strict=True)):
    for field in SITE_COMPARED_FIELDS:
      if field not in r or field not in g:
        mismatches.append(f"designs[{i}].{field}:missing")
        continue
      same = _float_field_equal(r[field], g[field]) if field in FLOAT_FIELDS else _exact_equal(r[field], g[field])
      if not same:
        mismatches.append(f"designs[{i}].{field}")
    mismatches += [f"designs[{i}].{f}:unaccounted_harness_field" for f in sorted(set(r) - ACCOUNTED_HARNESS_FIELDS)]
  return {"exact": not mismatches, "mismatches": mismatches}


def control_flags(rows: list[dict[str, Any]], expected_keys: list[Any]) -> bool:
  """A control is detected only when every expected key has at least one row and EVERY row for it is a mismatch (exact false)."""
  if not expected_keys:
    return False
  for key in expected_keys:
    mine = [r for r in rows if r["key"] == key]
    if not mine or any(bool(r["exact"]) for r in mine):
      return False
  return True


def _decide(result: dict[str, Any]) -> str:
  """The pre-registered outcome ladder (first match wins: error, instrument_unverified, fail, pass)."""
  if (not result.get("artifacts_verified") or not result.get("harness_pages_match_graded") or not result.get("site_built")
      or not result.get("all_units_complete") or result.get("error")):
    return "error"
  if not (result.get("planted_detected") and result.get("order_detected") and result.get("integrity_refused")
          and result.get("environment_ok")):
    return "instrument_unverified"
  return "pass" if result.get("all_exact") else "fail"


def blob_problems(commit: str, expected: dict[str, str | None], actual: dict[str, str]) -> list[str]:
  """Files whose working-tree git blob hash differs from the blob at `commit` (None = absent at the commit)."""
  out = []
  for rel, want in sorted(expected.items()):
    if want is None:
      out.append(f"{rel}: not present at {commit}")
    elif actual.get(rel) != want:
      out.append(f"{rel}: blob {actual.get(rel)} != {want} at {commit}")
  return out


def roll_sweep_uniforms(uniforms: list[list[float]]) -> list[list[float]]:
  """Every sample's uniform stream rolled by one position in float32 (the sweep PLANTED control)."""
  return [np.roll(np.asarray(u, dtype=np.float32), 1).tolist() for u in uniforms]


def planted_job(job: dict[str, Any]) -> dict[str, Any] | None:
  """The job with a planted one-position roll (METHODS: every replay uniform list; SWEEP: every sample's uniforms), or None when the roll changes nothing."""
  if job["path"] == "methods":
    if not job.get("replay"):
      return None
    rolled = methods_gate.roll_uniforms(job["replay"])
    changed = any(not np.array_equal(np.asarray(a["uniforms"], dtype=np.float64), np.asarray(b["uniforms"], dtype=np.float64))
                  for a, b in zip(job["replay"], rolled, strict=True))
    return {**job, "replay": rolled, "id": f"{job['id']}{PLANTED}", "planted_of": job["id"]} if changed else None
  if not job.get("uniforms"):
    return None
  rolled_u = roll_sweep_uniforms(job["uniforms"])
  changed = any(not np.array_equal(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)) for a, b in zip(job["uniforms"], rolled_u, strict=True))
  return {**job, "uniforms": rolled_u, "id": f"{job['id']}{PLANTED}", "planted_of": job["id"]} if changed else None


def pick_planted(jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
  """The first job (declaration order) with T > 0 whose roll changes something, perturbed; None when there is none."""
  for job in jobs:
    if float(job["temperature"]) > 0:
      planted = planted_job(job)
      if planted is not None:
        return planted
  return None


def order_job(job: dict[str, Any]) -> dict[str, Any]:
  """The sweep job with sweepOrder forced to "position" (same uniforms): the deployed worker must not return the knn / energy output for it."""
  return {**job, "config": {**job["config"], "sweepOrder": "position"}, "id": f"{job['id']}{ORDER_VARIANT}", "order_of": job["id"]}


def pick_order_job(jobs: list[dict[str, Any]], cell: str = "") -> dict[str, Any] | None:
  """The sweep job the ORDER control perturbs: the first (in `cell` when given) with T > 0, else the first."""
  pool = [j for j in jobs if not cell or j["cell"] == cell]
  return next((j for j in pool if float(j["temperature"]) > 0), pool[0] if pool else None)


def translate(job: dict[str, Any], pdb_rel: str) -> dict[str, Any]:
  """One job as the deployed hook's message: kind design, the FULL config, and replay (methods) or uniforms (sweep) forwarded through params."""
  params: dict[str, Any] = {"labels": job.get("labels"), "config": job["config"], "bucket": job["bucket"]}
  if job.get("replay") is not None:
    params["replay"] = job["replay"]
  if job.get("uniforms") is not None:
    params["uniforms"] = job["uniforms"]
  return {"id": job["id"], "kind": "design", "pdb": pdb_rel, "params": params}


def harness_cell(job: dict[str, Any]) -> dict[str, Any]:
  """One job as the harness page's cells.json entry ({id, bucket, pdb, labels, config, replay | uniforms}); pdb is filled in by the caller."""
  cell = {"id": job["id"], "bucket": job["bucket"], "labels": job.get("labels"), "config": job["config"]}
  if job["path"] == "methods":
    cell["replay"] = job["replay"]
  else:
    cell["uniforms"] = job.get("uniforms")
  return cell


def sweep_uniforms(seed: int, cell_index: int, temperature: float) -> list[list[float]] | None:
  """X7d's draws with a different SEED: one float32 array per sample, a function of (seed, cell, temperature) only (None at T <= 0)."""
  if temperature <= 0:
    return None
  rng = np.random.default_rng([seed, cell_index, 1])
  return [rng.uniform(size=sweep_gate.BLOCK_ROUNDS * sweep_gate.MAX_MUTATIONS).astype(np.float32).tolist() for _ in range(sweep_gate.SAMPLES)]


# ---------------------------------------------------------------- git, jobs, staging


def _git(*args: str) -> str:
  return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True, timeout=120).stdout.strip()  # noqa: S603, S607


def _lib_files() -> list[str]:
  """PROTONPOTTS_LIB_FILES from tools/build_site.py (the modules the site bundles), loaded by path."""
  spec = importlib.util.spec_from_file_location("_x12_build_site", REPO / "tools" / "build_site.py")
  if spec is None or spec.loader is None:
    msg = "cannot load tools/build_site.py"
    raise RuntimeError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return [str(f) for f in module.PROTONPOTTS_LIB_FILES]


def _graded_report(path_name: str, commit: str, run_id: str, lib_files: list[str]) -> tuple[dict[str, Any], list[str]]:
  """Blob hashes of this path's harness page + html and every site lib module: working tree vs `commit`."""
  names = ("ph_methods_page.mjs", "ph_methods_index.html") if path_name == "methods" else ("ph_sweep_page.mjs", "ph_sweep_index.html")
  rels = [f"{SCORER_REL}/{n}" for n in (*names, *lib_files)]
  expected: dict[str, str | None] = {}
  actual: dict[str, str] = {}
  for rel in rels:
    try:
      expected[rel] = _git("rev-parse", f"{commit}:{rel}")
    except subprocess.CalledProcessError:
      expected[rel] = None
    try:
      actual[rel] = _git("hash-object", "--", rel)
    except subprocess.CalledProcessError:
      actual[rel] = ""
  return {"commit": commit, "run": run_id, "blobs": actual}, blob_problems(commit, expected, actual)


def _selected(args: argparse.Namespace) -> tuple[set[str], set[str]]:
  return {c for c in args.cells.split(",") if c}, {c for c in args.cases.split(",") if c}


def _job_cache_key(parts: dict[str, Any]) -> str:
  return inputs_sha256(parts)


def build_methods_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:  # noqa: C901
  """X11c's jobs (cells x CASES) with draw replays recorded at --seed-offset; each job cached under --work-dir/jobs keyed by its inputs."""
  want_cells, want_cases = _selected(args)
  jobs: list[dict[str, Any]] = []
  code_id = _git("rev-parse", "HEAD:src/aminx/families/protonpotts_mpnn")
  state_sha = _sha256(args.state_npz) if args.state_npz.is_file() else ""
  cache_dir = args.work_dir / "jobs" / "methods"
  cache_dir.mkdir(parents=True, exist_ok=True)
  for cell_index, (name, (rel, _sha)) in enumerate(STRUCTURES.items()):
    if want_cells and name not in want_cells:
      continue
    pdb = args.potts_root / "energy_benchmark_datasets" / rel
    pdb_sha = hashlib.sha256(pdb.read_bytes()).hexdigest()
    ctx: dict[str, Any] = {}

    def context(_pdb: Path = pdb, _ctx: dict[str, Any] = ctx) -> dict[str, Any]:
      if _ctx:
        return _ctx
      import jax  # noqa: PLC0415
      import jax.numpy as jnp  # noqa: PLC0415

      if jax.config.jax_enable_x64:
        msg = "jax_enable_x64 is set; refusing"
        raise RuntimeError(msg)
      from aminx.families.protonpotts_mpnn.energy import protonpotts_table  # noqa: PLC0415
      from aminx.families.protonpotts_mpnn.features import featurize_pdb, kept_residues  # noqa: PLC0415

      model = _load_model(args.state_npz)
      feats = featurize_pdb(_pdb, None)
      kept = kept_residues(_pdb)
      native = np.asarray(feats["S"][0], dtype=np.int32)
      length = native.shape[0]
      chains = [k[0] for k in kept]
      binder_chain = chains[-1]
      args_t = (jnp.asarray(feats["X"][0, :, :4, :], dtype=jnp.float32), jnp.ones(length, jnp.float32), jnp.asarray(feats["R_idx"][0]),
                jnp.asarray(feats["chain_labels"][0]), jnp.ones(length, bool))  # fmt: skip
      table, e_idx = jax.jit(lambda *a, _m=model: protonpotts_table(_m, *a))(*args_t)
      _ctx.update(native=native, length=length, binder_chain=binder_chain, binder_mask=np.asarray([c == binder_chain for c in chains]),
                  res_id=np.asarray([k[1] for k in kept], dtype=np.int32), table=table, e_idx=e_idx)
      return _ctx

    for case_index, (case, method, temperature) in enumerate(methods_gate.CASES):
      if want_cases and case not in want_cases:
        continue
      cid = f"{name}__{case}"
      seed = methods_gate.case_seed(case_index, cell_index, args.seed_offset)
      key = _job_cache_key({"pdb_sha256": pdb_sha, "seed": seed, "case": case, "method": method, "temperature": temperature,
                            "samples": methods_gate.SAMPLES, "centre": list(methods_gate.CENTER_TYPES), "state_sha256": state_sha,
                            "code": code_id})  # fmt: skip
      cache = cache_dir / f"{cid}.json"
      job = None
      if cache.is_file():
        try:
          stored = json.loads(cache.read_text())
          job = stored["job"] if stored.get("key") == key else None
        except (OSError, ValueError, KeyError):
          job = None
      if job is None:
        from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig  # noqa: PLC0415

        c = context()
        config = PHDesignConfig(binder_chain=c["binder_chain"], method=method, center_types=methods_gate.CENTER_TYPES, temperature=temperature,
                                samples_per_site=methods_gate.SAMPLES)  # fmt: skip
        rec = methods_gate.record_case(c["table"], c["e_idx"], c["native"], c["binder_mask"], c["res_id"], config, seed)
        job = {"path": "methods", "cell": name, "case": case, "id": cid, "pdb_file": str(pdb), "pdb_sha256": pdb_sha,
               "bucket": _bucket_for(c["length"]), "labels": None, "config": methods_gate.js_config(config), "replay": rec["replay"],
               "temperature": float(temperature), "method": method, "seed": seed}  # fmt: skip
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": key, "job": job}, allow_nan=False))
        tmp.replace(cache)
      jobs.append(job)
  return jobs


def build_sweep_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
  """X7d's jobs (cells x CASES, centre types HIS-P ASP-P GLU-P) with uniforms from --sweep-seed."""
  from aminx.families.protonpotts_mpnn.features import featurize_pdb, kept_residues  # noqa: PLC0415

  want_cells, want_cases = _selected(args)
  jobs: list[dict[str, Any]] = []
  for cell_index, (name, (rel, _sha)) in enumerate(STRUCTURES.items()):
    if want_cells and name not in want_cells:
      continue
    pdb = args.potts_root / "energy_benchmark_datasets" / rel
    pdb_sha = hashlib.sha256(pdb.read_bytes()).hexdigest()
    length = int(np.asarray(featurize_pdb(pdb, None)["S"][0]).shape[0])
    binder_chain = [k[0] for k in kept_residues(pdb)][-1]
    for case, order, temperature in sweep_gate.CASES:
      if want_cases and case not in want_cases:
        continue
      jobs.append({"path": "sweep", "cell": name, "case": case, "id": f"{name}__{case}", "pdb_file": str(pdb), "pdb_sha256": pdb_sha,
                   "bucket": _bucket_for(length), "labels": None, "config": sweep_gate.js_config(binder_chain, order, temperature),
                   "uniforms": sweep_uniforms(args.sweep_seed, cell_index, temperature), "temperature": float(temperature),
                   "order": order, "seed": args.sweep_seed})  # fmt: skip
  return jobs


def _pdb_rel(job: dict[str, Any]) -> str:
  return f"pdbs/{job['cell']}.pdb"


def _unit_problems(spec: dict[str, Any], payload: dict[str, Any] | None) -> list[str]:
  if payload is None:
    return ["no payload"]
  if not payload.get("harnessOk"):
    return [f"harnessOk false: {str(payload.get('harnessError'))[:300]}"]
  res = payload.get("result") or {}
  mid = spec["msg_id"]
  if spec["kind"] == "harness":
    cell = next((c for c in res.get("cells", []) if c.get("id") == mid), None)
    if not cell or not cell.get("ok"):
      return [f"{mid}: missing or not ok ({str((cell or {}).get('error'))[:200]})"]
    return [] if cell.get("designs") else [f"{mid}: no designs"]
  reply = next((r.get("reply") for r in res.get("replies", []) if r.get("id") == mid), None)
  if reply is None:
    return [f"{mid}: no reply"]
  if spec["kind"] == "integrity":
    return [] if reply.get("ok") is False else [f"{mid}: tampered graph was not refused"]
  if reply.get("ok") is not True or not isinstance(reply.get("result"), dict) or not reply["result"].get("designs"):
    return [f"{mid}: not ok or empty ({str(reply.get('error'))[:300]})"]
  return []


def _site_reply(payload: dict[str, Any] | None, mid: str) -> dict[str, Any] | None:
  if payload is None:
    return None
  return next((r.get("reply") for r in (payload.get("result") or {}).get("replies", []) if r.get("id") == mid), None)


def _harness_designs(payload: dict[str, Any] | None, mid: str) -> list[dict[str, Any]] | None:
  if payload is None:
    return None
  cell = next((c for c in (payload.get("result") or {}).get("cells", []) if c.get("id") == mid), None)
  return None if cell is None else cell.get("designs")


# ---------------------------------------------------------------- the gate


def _run_gate(args: argparse.Namespace, result: dict[str, Any]) -> None:  # noqa: C901, PLR0912, PLR0915
  dist = args.dist / "protonpotts"
  paths = tuple(p for p in args.paths.split(",") if p)
  settings = tuple(s for s in SETTINGS if s[0] in args.settings.split(","))
  for p in paths:
    if p not in PATHS:
      msg = f"unknown path {p!r}; choose from {PATHS}"
      raise ValueError(msg)
  problems: list[str] = []

  # 1. artifacts: export manifests by sha256; dist manifests byte-identical to the exports'.
  for models, sha in ((args.scoring_models, args.scoring_manifest_sha256), (args.ph_models, args.ph_manifest_sha256)):
    _verify_artifacts(models, sha)
  for kind, models in (("scoring", args.scoring_models), ("ph", args.ph_models)):
    if not _same_bytes(dist / "models" / kind / "MANIFEST.json", models / "MANIFEST.json"):
      problems.append(f"dist models/{kind}/MANIFEST.json differs from the {kind} export")
  result["artifacts_verified"] = not problems
  if problems:
    result["error"] = "; ".join(problems)[:4000]
    return

  # 2. the graded pages and every bundled lib module are byte-identical (git blob) to those at the graded commits.
  lib_files = _lib_files()
  page_problems: list[str] = []
  commits = {"methods": (args.graded_methods_commit, args.graded_methods_run), "sweep": (args.graded_sweep_commit, args.graded_sweep_run)}
  for p in paths:
    report, probs = _graded_report(p, commits[p][0], commits[p][1], lib_files)
    result["graded_runs"][p] = report
    page_problems += [f"{p}: {x}" for x in probs]
  result["harness_pages_match_graded"] = not page_problems
  if page_problems:
    result["error"] = "; ".join(page_problems)[:4000]
    return

  # 3. the built site: files present, lib byte-identical to the repo, worker imports rewritten, lib list == build_site's.
  site_problems = _site_problems(dist)
  present_lib = sorted(p.name for p in (dist / "lib").glob("*.mjs")) if (dist / "lib").is_dir() else []
  if present_lib != sorted(lib_files):
    site_problems.append(f"dist lib {present_lib} != PROTONPOTTS_LIB_FILES {sorted(lib_files)}")
  result["site_built"] = not site_problems
  result["site_tree_sha256"] = tree_digest(dist) if dist.is_dir() else ""
  if site_problems:
    result["error"] = "; ".join(site_problems)[:4000]
    return

  node = find_node(args.node)
  if node is None:
    msg = "no node binary found (--node, $NODE_BIN, $PATH, nvm)"
    raise RuntimeError(msg)
  env = node_env_report(node)
  result["node_version"] = env.get("node") or ""
  result["onnxruntime_web_version"] = env.get("onnxruntime-web") or ""

  # 4. jobs.
  jobs: dict[str, list[dict[str, Any]]] = {}
  if "methods" in paths:
    jobs["methods"] = build_methods_jobs(args)
  if "sweep" in paths:
    jobs["sweep"] = build_sweep_jobs(args)
  if not any(jobs.values()):
    msg = "no jobs selected"
    raise RuntimeError(msg)

  # 5. staging: harness sites (graded pages), deployed root (subpath + driver + PDBs), integrity root (one flipped ph graph).
  work = args.work_dir
  work.mkdir(parents=True, exist_ok=True)
  harness_site = {"methods": work / "harness_site_methods", "sweep": work / "harness_site_sweep"}
  assemblers = {"methods": methods_page.assemble_base_site, "sweep": sweep_gate.assemble_base_site}
  page_digests: dict[str, dict[str, str]] = {}
  for p in paths:
    assemblers[p](harness_site[p], args.scoring_models, args.ph_models, args.ort_dir)
    names = (*(methods_page.PAGE_FILES if p == "methods" else sweep_gate.PAGE_FILES),
             methods_page.PAGE_HTML if p == "methods" else sweep_gate.PAGE_HTML, *lib_files)
    page_digests[p] = {n: _sha256(REPO / SCORER_REL / n) for n in sorted(set(names))}
  deployed = work / "deployed_root"
  _link_tree(dist, deployed / "aminx" / "protonpotts")
  (deployed / "index.html").write_text(DRIVER_HTML)
  pdb_dir = deployed / "pdbs"
  pdb_dir.mkdir(parents=True, exist_ok=True)
  for p in paths:
    for job in jobs[p]:
      shutil.copy2(job["pdb_file"], pdb_dir / Path(_pdb_rel(job)).name)
  manifests = {"scoring": args.scoring_manifest_sha256, "ph": args.ph_manifest_sha256}
  site_tree = result["site_tree_sha256"]
  driver_sha = hashlib.sha256(DRIVER_HTML.encode()).hexdigest()

  # control jobs
  planted: dict[str, dict[str, Any] | None] = {p: pick_planted(jobs[p]) for p in paths}
  order_base = pick_order_job(jobs.get("sweep", []), args.order_cell) if "sweep" in paths else None
  order_variant = order_job(order_base) if order_base is not None else None
  probe = next(j for p in paths for j in jobs[p])
  ph_manifest = json.loads((args.ph_models / "MANIFEST.json").read_text())
  flip_entry = next((b for b in ph_manifest["buckets"] if int(b["bucket"]) == int(probe["bucket"])), None)
  if flip_entry is None:
    msg = f"ph manifest has no bucket {probe['bucket']}"
    raise RuntimeError(msg)
  flipped_file = str(flip_entry["graphs"][sorted(flip_entry["graphs"])[0]]["file"])
  integrity = work / "integrity_root"
  _link_tree(dist, integrity / "aminx" / "protonpotts")
  target = integrity / "aminx" / "protonpotts" / "models" / "ph" / flipped_file
  original = target.read_bytes()
  target.unlink()
  target.write_bytes(flip_byte(original, FLIP_OFFSET))
  (integrity / "index.html").write_text(DRIVER_HTML)
  _link_tree(pdb_dir, integrity / "pdbs")

  # 6. unit specs.
  specs: list[dict[str, Any]] = []

  def add_harness(p: str, threads: int, job: dict[str, Any]) -> None:
    name = f"{p}.harness.t{threads}.{job['id']}"
    inputs = {"unit": name, "kind": "harness", "job": job, "threads": threads, "pdb_sha256": job["pdb_sha256"], "manifests": manifests,
              "page_and_module_sha256": page_digests[p], "onnxruntime_web_version": result["onnxruntime_web_version"],
              "node_version": result["node_version"]}  # fmt: skip
    specs.append({"name": name, "kind": "harness", "role": "harness", "path": p, "arm": "harness", "threads": threads, "setting": f"t{threads}",
                  "isolate": True, "job": job, "msg_id": job["id"], "inputs": inputs})

  def add_site(p: str, label: str, threads: int, isolate: bool, job: dict[str, Any], role: str, base: dict[str, Any]) -> None:  # noqa: FBT001
    name = f"{p}.site.{label}.{job['id']}"
    msg = translate(job, _pdb_rel(base))
    inputs = {"unit": name, "kind": "site", "job": job, "threads": threads, "isolate": isolate, "pdb_sha256": base["pdb_sha256"],
              "manifests": manifests, "site_tree_sha256": site_tree, "driver_sha256": driver_sha}  # fmt: skip
    specs.append({"name": name, "kind": "site", "role": role, "path": p, "arm": "site", "threads": threads, "setting": label,
                  "isolate": isolate, "job": job, "base_id": base["id"], "msg": msg, "msg_id": job["id"], "root": deployed, "inputs": inputs})

  for p in paths:
    for label, threads, isolate in settings:
      for job in jobs[p]:
        add_harness(p, threads, job)
        add_site(p, label, threads, isolate, job, "main", job)
      if planted[p] is not None:
        base = next(j for j in jobs[p] if j["id"] == planted[p]["planted_of"])
        add_site(p, label, threads, isolate, planted[p], "planted", base)
      if p == "sweep" and order_variant is not None:
        add_site(p, label, threads, isolate, order_variant, "order", order_base)
  imsg = translate(probe, _pdb_rel(probe))
  specs.append({"name": "integrity.site.pages", "kind": "integrity", "role": "integrity", "path": probe["path"], "arm": "site", "threads": 1,
                "setting": "pages", "isolate": False, "job": probe, "msg": imsg, "msg_id": probe["id"], "root": integrity,
                "inputs": {"unit": "integrity.site.pages", "kind": "integrity", "job": probe, "site_tree_sha256": site_tree,
                           "flipped_file": flipped_file, "flip_offset": FLIP_OFFSET, "manifests": manifests, "driver_sha256": driver_sha}})  # fmt: skip

  # 7. run every unit (own process and timeout each), persisting completed units with a stamp.
  payloads: dict[str, dict[str, Any] | None] = {}
  units: list[dict[str, Any]] = []
  reused_names: list[dict[str, str]] = []
  for spec in specs:
    name = spec["name"]
    key = inputs_sha256(spec["inputs"])
    out = work / "units" / name
    payload = _reusable(out, key)
    reused = payload is not None
    error = ""
    try:
      if not reused:
        if out.is_dir():
          (out / "stamp.json").unlink(missing_ok=True)
        if spec["kind"] == "harness":
          job = spec["job"]
          site = harness_site[spec["path"]]
          shutil.copy(job["pdb_file"], site / "data" / Path(job["pdb_file"]).name)
          (site / "cells.json").write_text(json.dumps({"cells": [harness_cell(job) | {"pdb": f"data/{Path(job['pdb_file']).name}"}]}))
          payload = _run_p07(node, site, out, spec["threads"], args.timeout_ms, True)  # noqa: FBT003
        else:
          (spec["root"] / "x10_msgs.json").write_text(json.dumps([spec["msg"]]))
          payload = _run_p07(node, spec["root"], out, spec["threads"], args.timeout_ms, spec["isolate"])
    except Exception as exc:  # noqa: BLE001 -- a unit failure is recorded, the remaining units still run
      logger.exception("unit %s failed", name)
      error = f"{type(exc).__name__}: {exc}"[:500]
      payload = None
    unit_problems = [error] if error else _unit_problems(spec, payload)
    complete = not unit_problems
    if complete and not reused:
      out.mkdir(parents=True, exist_ok=True)
      (out / "stamp.json").write_text(json.dumps({"unit": name, "inputs_sha256": key, "complete": True,
                                                  "artifact_sha256": tree_digest(out, KEEP_OUT_OF_DIGEST)}, indent=2))
    if reused:
      reused_names.append({"unit": name, "dir": str(out), "inputs_sha256": key})
    payloads[name] = payload if complete else None
    units.append({"name": name, "path": spec["path"], "arm": spec["arm"], "role": spec["role"], "setting": spec["setting"],
                  "threads": spec["threads"], "isolate": spec["isolate"], "cell": spec["job"]["cell"], "case": spec["job"]["case"],
                  "inputs_sha256": key, "reused": reused, "complete": complete, "problems": unit_problems[:10]})
    logger.info("unit %s: complete=%s reused=%s problems=%d", name, complete, reused, len(unit_problems))
  rows = {u["name"]: u for u in units}
  result["units_reused"] = reused_names
  result["all_units_complete"] = all(u["complete"] for u in units)

  # 8. grading: each site unit against the harness unit of the same path, thread count and job.
  env_ok = True
  versions: set[str] = set()
  planted_rows: list[dict[str, Any]] = []
  order_rows: list[dict[str, Any]] = []
  main_rows: list[dict[str, Any]] = []
  for spec in specs:
    pay = payloads.get(spec["name"])
    if spec["kind"] == "harness":
      if pay is not None:
        res = pay.get("result") or {}
        harness_env = {"cross_origin_isolated": bool(res.get("cross_origin_isolated")), "num_threads_effective": res.get("num_threads_effective")}
        rows[spec["name"]]["env"] = harness_env
        env_ok &= harness_env["cross_origin_isolated"] is True and harness_env["num_threads_effective"] == spec["threads"]
      else:
        env_ok = False
      continue
    if spec["kind"] == "integrity":
      continue
    reply = _site_reply(pay, spec["msg_id"])
    if reply is None or reply.get("ok") is not True:
      env_ok = False
      continue
    site_env = reply.get("env") or {}
    rows[spec["name"]]["env"] = site_env
    versions.add(str(site_env.get("ort_version")))
    env_ok &= _env_ok(site_env, isolated=spec["isolate"], threads=spec["threads"])
    got = reply["result"]["designs"]
    refs = [spec["job"]["id"]] if spec["role"] == "main" else [spec["base_id"]]
    if spec["role"] == "order":  # compared with every knn / energy harness output of this cell at this temperature
      refs = [j["id"] for j in jobs["sweep"] if j["cell"] == spec["job"]["cell"] and j["temperature"] == spec["job"]["temperature"]]
    comps = []
    for ref_id in refs:
      h_name = f"{spec['path']}.harness.t{spec['threads']}.{ref_id}"
      h_designs = _harness_designs(payloads.get(h_name), ref_id)
      comp = compare_designs(h_designs, got) if h_designs is not None else {"exact": False, "mismatches": ["no_harness_output"]}
      comps.append({"vs_harness_unit": h_name, **comp})
      if spec["role"] == "main":
        main_rows.append({"key": (spec["path"], spec["setting"]), "exact": comp["exact"], "unit": spec["name"], "harness_present": h_designs is not None})
      elif spec["role"] == "planted":
        # A missing harness output cannot show a control firing: count it as "not flagged" (exact) so the control is not detected.
        planted_rows.append({"key": (spec["path"], spec["setting"]), "exact": bool(comp["exact"] or h_designs is None)})
      else:
        order_rows.append({"key": spec["setting"], "exact": bool(comp["exact"] or h_designs is None)})
    rows[spec["name"]]["comparison"] = comps if spec["role"] != "main" else comps[0]
  integ = payloads.get("integrity.site.pages")
  rep = _site_reply(integ, probe["id"])
  result["integrity_refused"] = bool(rep and rep.get("ok") is False and flipped_file in str(rep.get("error")))
  rows["integrity.site.pages"]["flipped_file"] = flipped_file
  result["ort_version_deployed"] = ",".join(sorted(versions))
  result["environment_ok"] = bool(env_ok and versions == {ORT_VERSION})
  expected_planted = [(p, s[0]) for p in paths if planted[p] is not None for s in settings]
  result["planted_detected"] = len(expected_planted) == len(paths) * len(settings) and control_flags(planted_rows, expected_planted)
  result["order_detected"] = (("sweep" not in paths) or order_variant is not None) and control_flags(
    order_rows, [s[0] for s in settings] if "sweep" in paths else [])
  expected_main = sum(len(jobs[p]) for p in paths) * len(settings)
  result["all_exact"] = len(main_rows) == expected_main and all(r["exact"] and r["harness_present"] for r in main_rows)
  result["units"] = units


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--dist", type=Path, required=True, help="built dist root containing protonpotts/ (tools/build_site.py --protonpotts-*-dir)")
  parser.add_argument("--scoring-models", type=Path, required=True)
  parser.add_argument("--scoring-manifest-sha256", required=True)
  parser.add_argument("--ph-models", type=Path, required=True)
  parser.add_argument("--ph-manifest-sha256", required=True)
  parser.add_argument("--graded-methods-commit", required=True, help="commit of the graded X11c run (harness page + lib blobs must equal it)")
  parser.add_argument("--graded-sweep-commit", required=True, help="commit of the graded X7d run")
  parser.add_argument("--graded-methods-run", default="", help="bathos run id of the graded X11c run (recorded only)")
  parser.add_argument("--graded-sweep-run", default="", help="bathos run id of the graded X7d run (recorded only)")
  parser.add_argument("--potts-root", type=Path, default=Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser())
  parser.add_argument("--state-npz", type=Path, default=Path(os.environ.get("AMINX_PROTONPOTTS_STATE", "~/scratch/v6_state.npz")).expanduser())
  parser.add_argument("--node", default=os.environ.get("NODE_BIN"))
  parser.add_argument("--ort-dir", type=Path, default=Path(os.environ.get("ORT_WEB_DIR", str(LAYER_C_DIR))))
  parser.add_argument("--timeout-ms", type=int, default=UNIT_TIMEOUT_MS, help="per unit (one process each)")
  parser.add_argument("--seed-offset", type=int, default=METHODS_SEED_OFFSET, help="METHODS draw-seed offset (400000)")
  parser.add_argument("--sweep-seed", type=int, default=SWEEP_SEED, help="SWEEP uniforms SEED (5031)")
  parser.add_argument("--paths", default=",".join(PATHS), help="comma-separated: methods,sweep (smoke subsetting)")
  parser.add_argument("--settings", default=",".join(s[0] for s in SETTINGS), help="comma-separated: pages,coi (smoke subsetting)")
  parser.add_argument("--cells", default="", help="comma-separated cell names (smoke subsetting; default all)")
  parser.add_argument("--cases", default="", help="comma-separated case names, both paths (smoke subsetting; default all)")
  parser.add_argument("--order-cell", default="", help="cell whose first T>0 sweep job the ORDER control perturbs (default: first selected)")
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  git_hash, git_clean = git_state(REPO)
  result: dict[str, Any] = {
    "scoring_manifest_sha256": args.scoring_manifest_sha256, "ph_manifest_sha256": args.ph_manifest_sha256,
    "artifacts_verified": False, "harness_pages_match_graded": False, "graded_runs": {}, "site_built": False, "site_tree_sha256": "",
    "ort_version_deployed": "", "onnxruntime_web_version": "", "node_version": "", "git_hash": git_hash, "git_clean": git_clean,
    "units": [], "units_reused": [], "excluded_fields": list(EXCLUDED_FIELDS), "all_units_complete": False, "planted_detected": False,
    "order_detected": False, "integrity_refused": False, "environment_ok": False, "all_exact": False, "error": "",
    "smoke_subset": bool(args.cells or args.cases or args.paths != ",".join(PATHS) or args.settings != ",".join(s[0] for s in SETTINGS)),
  }  # fmt: skip
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    _run_gate(args, result)
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("site methods gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["outcome"] = _decide(result)
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("outcome=%s", result["outcome"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
