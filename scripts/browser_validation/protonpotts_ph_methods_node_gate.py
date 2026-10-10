"""X11b gate: the browser JS pH host-loop methods (converged_mcmc, converged_mcmc_combined, two_phase, gibbs) vs aminx design_methods, under Node.

task_id 261010_ph-methods-js. Sidecar: protonpotts_ph_methods_node_gate.bth.toml (pre-registered before this script and before the JS
methods existed). The X6a scoring and X7a pH artifacts are verified by sha256 against their manifests, not re-exported.

Per (cell, case), in order: RECORD the draws the Python method functions consume (seeded Draws, mirroring design_methods' dispatch); REFERENCE
design_methods(replay=record); JS: run_ph_methods_node.mjs on the same PDB text with the same record as replay. Every case is one process with
its own timeout and a stamp under --work-dir, so a preemption loses at most one (cell, case).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
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
from scripts.browser_validation.potts_browser_parity import git_state  # noqa: E402
from scripts.browser_validation.protonpotts_export_gate import (  # noqa: E402
  STRUCTURES,
  _bucket_for,
  _load_model,
  _sha256,
)
from scripts.browser_validation.protonpotts_ph_loop_gate import ENERGY_BAR, _rel  # noqa: E402
from scripts.browser_validation.protonpotts_score_gate import _verify_artifacts  # noqa: E402

logger = logging.getLogger("protonpotts_ph_methods_node_gate")

FRAGILE_MARGIN = 1.0e-5  # of the total mass, the pre-registered margin
SAMPLES = 2
CENTER_TYPES = ("HIS-P",)
SEED_BASE = 1000
CASES: tuple[tuple[str, str, float], ...] = (
  ("mcmc_t005", "converged_mcmc", 0.05),
  ("mcmc_t0", "converged_mcmc", 0.0),
  ("combined_t005", "converged_mcmc_combined", 0.05),
  ("two_phase_t005", "two_phase", 0.05),
  ("gibbs_t005", "gibbs", 0.05),
)
MCMC_METHODS = ("converged_mcmc", "converged_mcmc_combined", "two_phase")
NODE_RUNNER = REPO / "browser" / "protonpotts-scorer" / "run_ph_methods_node.mjs"
JS_DIGEST_GLOBS = ("ph_design.mjs", "ph_plan.mjs", "protonpotts_*.mjs", "run_ph_methods_node.mjs")
UNIT_TIMEOUT_S = 7200
CONTROL_ERRORS = (IndexError, ValueError)  # replay / validation errors raised by Draws (and the design checks that follow it)


# ---- pure: outcome ladder, comparator, fragility, controls ------------------------------------------------------------------------------------


def _decide(result: dict[str, Any]) -> str:
  """The pre-registered outcome ladder, in declaration order (the first matching branch wins)."""
  if not (
    result.get("jax_arm_ok") is True and result.get("artifacts_verified") is True
    and result.get("node_ok") is True and result.get("reference_consistent") is True
  ):  # fmt: skip
    return "error"
  controls = ("control_tokens_detected", "control_randint_detected", "control_perm_detected", "control_energy_detected")
  if not all(result.get(k) is True for k in controls):
    return "instrument_unverified"
  n_cases = int(result.get("n_cases") or 0)
  if n_cases > 0 and int(result.get("n_pass") or 0) == n_cases:
    return "pass"
  return "fail"


def is_fragile_pick(probs: np.ndarray, uniform: float, cdf_order: np.ndarray | None, margin: float = FRAGILE_MARGIN) -> bool:
  """A pick is fragile when its uniform lies within ``margin`` of the total mass of a cumulative boundary (cumsum in ``cdf_order``, float64)."""
  if not np.isfinite(uniform):
    return False
  order = np.arange(len(probs)) if cdf_order is None else np.asarray(cdf_order)
  cumulative = np.cumsum(np.asarray(probs)[order].astype(np.float64))
  total = float(cumulative[-1])
  return bool(np.min(np.abs(cumulative - float(uniform) * total)) <= margin * total)


def first_fragile_index(events: list[Any], cdf_order: np.ndarray | None) -> int | None:
  """Index of the first fragile pick among ``events`` (objects with ``probs``, ``uniform``, ``choice``), or None."""
  for k, event in enumerate(events):
    if is_fragile_pick(event.probs, event.uniform, cdf_order):
      return k
  return None


def first_pick_difference(js_choices: list[int], ref_choices: list[int]) -> int | None:
  """First index where the two pick streams differ; a stream that ends early differs at its end; None when identical."""
  n = min(len(js_choices), len(ref_choices))
  for k in range(n):
    if int(js_choices[k]) != int(ref_choices[k]):
      return k
  return None if len(js_choices) == len(ref_choices) else n


def compare_designs(js: list[dict[str, Any]], ref: list[dict[str, Any]], *, exempt_fragile: bool = True) -> dict[str, Any]:
  """Grade JS designs against reference designs, paired in order (both sorted by final energy).

  Exact: designable, pins (position, prot token, residue number), label. A sequence mismatch is fragile-exempt only when ``exempt_fragile`` and the
  reference's first fragile pick is at or before the first differing pick; an exempt design skips its sequence, n_draws and energies (they follow
  from the sequence) but never its plan fields. Energies: ``_rel`` (max|d|/max(1,|ref|)) <= ENERGY_BAR.
  """
  if len(js) != len(ref) or not ref:
    return {"pass": False, "fragile": 0, "mismatches": [{"index": None, "field": "count", "js": len(js), "ref": len(ref)}]}
  mismatches: list[dict[str, Any]] = []
  fragile = 0
  for k, (j, r) in enumerate(zip(js, ref, strict=True)):
    plan_bad = [name for name in ("designable", "pins", "label") if _plan_field(j, name) != _plan_field(r, name)]
    if plan_bad:
      mismatches.append({"index": k, "field": ",".join(plan_bad)})
    if j["sequence"] != r["sequence"]:
      diff = first_pick_difference(j.get("choices", []), r.get("choices", []))
      first = r.get("first_fragile")
      if exempt_fragile and first is not None and diff is not None and diff >= first:
        fragile += 1
        continue
      mismatches.append({"index": k, "field": "sequence", "first_pick_difference": diff, "first_fragile": first})
      continue
    if int(j["n_draws"]) != int(r["n_draws"]):
      mismatches.append({"index": k, "field": "n_draws", "js": int(j["n_draws"]), "ref": int(r["n_draws"])})
    for name in ("final_potts_energy", "selective_energy"):
      if _rel(r[name], j[name]) > ENERGY_BAR:
        mismatches.append({"index": k, "field": name, "js": float(j[name]), "ref": float(r[name])})
  return {"pass": not mismatches, "fragile": fragile, "mismatches": mismatches}


def _plan_field(design: dict[str, Any], name: str) -> Any:  # noqa: ANN401
  if name == "designable":
    return [int(d) for d in design["designable"]]
  if name == "pins":
    return [[int(p[0]), int(p[1]), int(p[2])] for p in design["pins"]]
  return design["label"]


def roll_uniforms(replay: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Every sample's uniform list rolled by one (a new list; the input is untouched)."""
  return [dict(e, uniforms=np.roll(np.asarray(e["uniforms"], dtype=np.float64), 1).tolist()) if e["uniforms"] else dict(e) for e in replay]


def roll_randints(replay: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Every sample's randint list rolled by one (a new list; the input is untouched)."""
  return [
    dict(e, randints=np.roll(np.asarray(e["randints"], dtype=np.int64), 1).tolist()) if len(e["randints"]) >= 2 else dict(e)
    for e in replay
  ]  # fmt: skip


def reverse_perms(replay: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Every gibbs permutation reversed (a new list; the input is untouched)."""
  return [dict(e, perms=[list(reversed(p)) for p in e["perms"]]) for e in replay]


def control_applies(control: str, method: str, temperature: float, replay: list[dict[str, Any]]) -> bool:
  """Which cases a planted control applies to (pre-registered)."""
  if control == "tokens":
    return temperature > 0 and any(len(e["uniforms"]) > 0 for e in replay)
  if control == "randint":
    return method in MCMC_METHODS and len(replay[0]["randints"]) >= 2
  if control == "perm":
    return method == "gibbs" and len(replay[0]["perms"]) > 0
  return True  # energy: every case


def energy_control_reference(ref: list[dict[str, Any]], perturbed: np.ndarray) -> list[dict[str, Any]]:
  """The reference with each final energy replaced by the energy of the same sequence under the perturbed head (a new list)."""
  return [dict(r, final_potts_energy=float(e)) for r, e in zip(ref, perturbed, strict=True)]


def js_config(config: Any) -> dict[str, Any]:  # noqa: ANN401
  """A PHDesignConfig as the camelCase PhDesigner config (every field the host-loop methods read)."""
  return {
    "method": config.method,
    "binderChain": config.binder_chain,
    "backend": config.backend,
    "selective": bool(config.selective),
    "selectiveSource": config.selective_source,
    "placementBy": config.placement_by,
    "centerTypes": list(config.center_types),
    "depMap": {k: list(v) for k, v in config.dep_map},
    "forbiddenTokens": list(config.forbidden_tokens),
    "temperature": float(config.temperature),
    "samplesPerSite": int(config.samples_per_site),
    "maxMutations": int(config.max_mutations),
    "infillScope": config.infill_scope,
    "neighbourK": int(config.neighbour_k),
    "cvPatience": int(config.cv_patience),
    "cvMax": int(config.cv_max),
    "twoPhaseFrac": float(config.two_phase_frac),
    "combinedLambda": float(config.combined_lambda),
  }


def case_seed(case_index: int, cell_index: int) -> int:
  return SEED_BASE + 10 * case_index + cell_index


def inputs_sha256(payload: dict[str, Any]) -> str:
  return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def normalize_js(design: dict[str, Any]) -> dict[str, Any]:
  return {
    "sample": int(design["sample"]),
    "sequence": [int(x) for x in design["sequence"]],
    "designable": [int(x) for x in design["designable"]],
    "pins": [[int(p["position"]), int(p["protIdx"]), int(p["resId"])] for p in design["pins"]],
    "label": design["label"],
    "n_draws": int(design["n_draws"]),
    "final_potts_energy": float(design["final_potts_energy"]),
    "selective_energy": float(design["selective_energy"]),
    "choices": [int(e["choice"]) for e in (design.get("events") or [])],
  }


# ---- JAX arm: record, reference, controls (imports stay inside functions: the module imports without jax) ------------------------------------


def _method_context(table: Any, e_idx: Any, config: Any) -> Any:  # noqa: ANN401
  from aminx.families.protonpotts_mpnn.ph_methods import MethodContext  # noqa: PLC0415
  from aminx.families.protonpotts_mpnn.ph_plan import valid_token_mask  # noqa: PLC0415
  from aminx.families.protonpotts_mpnn.ph_sample import _cdf_order  # noqa: PLC0415

  return MethodContext(table, e_idx, valid_token_mask(config.forbidden_tokens), np.asarray(_cdf_order()), None)


def _replay_entry(draws: Any) -> dict[str, Any]:  # noqa: ANN401
  return {
    "uniforms": [float(e.uniform) for e in draws.events],
    "randints": [int(r) for r in draws.randint_log],
    "perms": [np.asarray(p).tolist() for p in draws.perm_log],
  }  # fmt: skip


def record_case(table: Any, e_idx: Any, native: np.ndarray, binder_mask: np.ndarray, res_id: np.ndarray, config: Any, seed: int) -> dict[str, Any]:  # noqa: ANN401
  """RECORD: run the Python method functions with seeded Draws, mirroring design_methods' dispatch. Returns the replay and the picks per sample.

  gibbs: ONE Draws(seed) shared by the rows, each row run in turn (the same stream design_methods consumes). Its events are attributed per row
  by slicing the shared event list; the one record is replicated to ``samples_per_site`` entries (design_methods reads replay[0]).
  """
  from aminx.families.protonpotts_mpnn.ph_methods import (  # noqa: PLC0415
    Draws,
    converge,
    gibbs,
    plan_for_methods,
    two_phase,
  )

  ctx = _method_context(table, e_idx, config)
  n = int(config.samples_per_site)
  length = int(native.shape[0])
  if config.method == "gibbs":
    draws = Draws(seed=seed, cdf_order=ctx.cdf_order)
    seqs: list[np.ndarray] = []
    events: list[list[Any]] = []
    for _ in range(n):
      start = len(draws.events)
      seqs.append(np.asarray(gibbs(ctx, config, native.reshape(1, length), binder_mask, draws)[0], dtype=np.int32))
      events.append(list(draws.events[start:]))
    record = _replay_entry(draws)
    return {"replay": [record] * n, "events": events, "sequences": seqs}
  plan = plan_for_methods(ctx, native, binder_mask, res_id, np.asarray(e_idx, dtype=np.int32), config)
  if plan is None or not plan.designable:
    msg = "no designable placement for the record"
    raise RuntimeError(msg)
  replay, events, seqs = [], [], []
  for sample in range(n):
    draws = Draws(seed=seed + sample, cdf_order=ctx.cdf_order)
    if config.method == "two_phase":
      result = two_phase(ctx, config, plan.pins, plan.designable, native, draws)
    else:
      result = converge(ctx, config, plan.pins, plan.designable, native, draws)
    replay.append(_replay_entry(draws))
    events.append(list(draws.events))
    seqs.append(np.asarray(result.sequence, dtype=np.int32))
  return {"replay": replay, "events": events, "sequences": seqs}


def reference_designs(table: Any, e_idx: Any, native: np.ndarray, binder_mask: np.ndarray, res_id: np.ndarray, config: Any, replay: list[dict[str, Any]]) -> list[Any]:  # noqa: ANN401
  from aminx.families.protonpotts_mpnn.ph_methods import design_methods  # noqa: PLC0415

  return design_methods(None, None, table, e_idx, native, binder_mask, res_id, config, replay=replay)


def _ref_dict(design: Any, events: list[Any], cdf_order: np.ndarray) -> dict[str, Any]:  # noqa: ANN401
  return {
    "sample": int(design.sample),
    "sequence": [int(x) for x in design.sequence],
    "designable": [int(d) for d in design.designable],
    "pins": [[int(p.position), int(p.prot_idx), int(p.res_id)] for p in design.pins],
    "label": str(design.label),
    "n_draws": int(design.n_draws),
    "final_potts_energy": float(design.final_potts_energy),
    "selective_energy": float(design.selective_energy),
    "choices": [int(e.choice) for e in events],
    "first_fragile": first_fragile_index(events, cdf_order),
  }


def _energies_under(bad_model: Any, args_t: tuple[Any, ...], seqs: list[list[int]]) -> np.ndarray:  # noqa: ANN401
  import jax.numpy as jnp  # noqa: PLC0415

  from aminx.families.protonpotts_mpnn.energy import protonpotts_energies  # noqa: PLC0415

  batch = jnp.asarray(np.asarray(seqs, dtype=np.int32))
  return np.asarray(protonpotts_energies(bad_model, *args_t, batch))


def _perturbed_model(model: Any) -> Any:  # noqa: ANN401
  import equinox as eqx  # noqa: PLC0415
  import jax  # noqa: PLC0415
  import jax.numpy as jnp  # noqa: PLC0415

  noise_key = jax.random.PRNGKey(1)
  perturbed = jax.tree_util.tree_map(
    lambda leaf: leaf + 1e-2 * jax.random.normal(noise_key, leaf.shape, leaf.dtype)
    if (eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating)) else leaf, model.potts_head,
  )  # fmt: skip
  return eqx.tree_at(lambda m: m.potts_head, model, perturbed)


def _run_control_reference(table: Any, e_idx: Any, native: np.ndarray, binder_mask: np.ndarray, res_id: np.ndarray, config: Any, replay: list[dict[str, Any]], cdf_order: np.ndarray) -> dict[str, Any]:  # noqa: ANN401
  """Re-run the reference on a planted replay. A replay or validation error from Draws is itself a detection (recorded, not raised)."""
  try:
    designs = reference_designs(table, e_idx, native, binder_mask, res_id, config, replay)
  except CONTROL_ERRORS as exc:
    return {"raised": f"{type(exc).__name__}: {exc}"[:500], "designs": None}
  return {"raised": None, "designs": [_ref_dict(d, [], cdf_order) for d in designs]}


# ---- node arm -------------------------------------------------------------------------------------------------------------------------------


def _js_digests() -> dict[str, str]:
  paths: list[Path] = []
  base = REPO / "browser" / "protonpotts-scorer"
  for pattern in JS_DIGEST_GLOBS:
    paths.extend(Path(p) for p in sorted(glob.glob(str(base / pattern))))
  return {p.name: _sha256(p) for p in sorted(set(paths))}


def _node_version(node_bin: str) -> str:
  return subprocess.run([node_bin, "--version"], capture_output=True, text=True, check=True, timeout=60).stdout.strip()  # noqa: S603


def _ort_version(ort_web_dir: str) -> str:
  pkg = Path(ort_web_dir) / "node_modules" / "onnxruntime-web" / "package.json"
  return str(json.loads(pkg.read_text())["version"])


def _read_stamp(unit_dir: Path, want_sha: str) -> dict[str, Any] | None:
  """A stamped unit is reused only when its stamp is complete, its inputs hash matches and its output file still hashes to the stamp."""
  stamp_path, out_path = unit_dir / "stamp.json", unit_dir / "node_out.json"
  if not (stamp_path.exists() and out_path.exists()):
    return None
  try:
    stamp = json.loads(stamp_path.read_text())
    node = json.loads(out_path.read_text())
  except (OSError, ValueError):
    return None
  if not (stamp.get("complete") is True and stamp.get("inputs_sha256") == want_sha):
    return None
  if _sha256(out_path) != stamp.get("output_sha256"):
    return None
  return {"node": node, "output_sha256": stamp["output_sha256"]}


def _run_unit(job: dict[str, Any], unit_dir: Path, args: argparse.Namespace, want_sha: str) -> dict[str, Any]:
  unit_dir.mkdir(parents=True, exist_ok=True)
  job_path, out_path = unit_dir / "job.json", unit_dir / "node_out.json"
  job_path.write_text(json.dumps(job))
  proc = subprocess.run(  # noqa: S603 -- fixed argv we built
    [args.node_bin, str(NODE_RUNNER), "--ort-dir", args.ort_web_dir, "--scoring-models", str(args.scoring_models),
     "--ph-models", str(args.ph_models), "--job", str(job_path), "--out", str(out_path)],
    capture_output=True, text=True, timeout=args.unit_timeout, check=False,
  )  # fmt: skip
  if proc.returncode != 0 or not out_path.exists():
    return {"node": {"ok": False, "error": f"node rc={proc.returncode}: {proc.stderr[-2000:]}", "designs": None}, "output_sha256": None}
  node = json.loads(out_path.read_text())
  if node.get("ok") is True:
    stamp = {"inputs_sha256": want_sha, "output_sha256": _sha256(out_path), "complete": True}
    (unit_dir / "stamp.json").write_text(json.dumps(stamp))
  return {"node": node, "output_sha256": _sha256(out_path) if out_path.exists() else None}


# ---- driver -----------------------------------------------------------------------------------------------------------------------------------


def _evaluate_controls(row: dict[str, Any], controls: dict[str, dict[str, Any]], js_designs: list[dict[str, Any]], ctx: dict[str, Any]) -> None:  # noqa: ANN401
  """Run every control that applies to this case, grade it with the same comparator (exemption off) and record it in ``controls``."""
  replay, cfg, method, temperature = ctx["replay"], ctx["config"], ctx["method"], ctx["temperature"]
  planted = {
    "tokens": (roll_uniforms(replay) if control_applies("tokens", method, temperature, replay) else None),
    "randint": (roll_randints(replay) if control_applies("randint", method, temperature, replay) else None),
    "perm": (reverse_perms(replay) if control_applies("perm", method, temperature, replay) else None),
  }
  for name, planted_replay in planted.items():
    if planted_replay is None:
      continue
    run = _run_control_reference(ctx["table"], ctx["e_idx"], ctx["native"], ctx["binder_mask"], ctx["res_id"], cfg, planted_replay, ctx["cdf_order"])
    if run["raised"] is not None:
      flagged = True
    else:
      flagged = not compare_designs(js_designs, run["designs"], exempt_fragile=False)["pass"]
    row.setdefault("controls", {})[name] = {"applies": True, "raised": run["raised"], "flagged": bool(flagged)}
    controls[name]["flagged"] = controls[name]["flagged"] or bool(flagged)
    controls[name]["applies"] = True
  energy_ref = ctx["ref"]
  perturbed = _energies_under(ctx["bad_model"], ctx["args_t"], [r["sequence"] for r in energy_ref])
  flagged = not compare_designs(js_designs, energy_control_reference(energy_ref, perturbed), exempt_fragile=False)["pass"]
  row.setdefault("controls", {})["energy"] = {"applies": True, "raised": None, "flagged": bool(flagged)}
  controls["energy"]["flagged"] = controls["energy"]["flagged"] or bool(flagged)
  controls["energy"]["applies"] = True


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
  parser.add_argument("--ort-web-dir", default=os.environ.get("ORT_WEB_DIR"))
  parser.add_argument("--unit-timeout", type=int, default=UNIT_TIMEOUT_S, help="seconds per (cell, case) Node process")
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  result: dict[str, Any] = {
    "scoring_manifest_sha256": args.scoring_manifest_sha256, "ph_manifest_sha256": args.ph_manifest_sha256,
    "artifacts_verified": False, "jax_arm_ok": False, "node_ok": False, "reference_consistent": False,
    "jax_version": "", "onnxruntime_web_version": "", "node_version": "", "git_hash": "", "git_clean": False,
    "n_cases": 0, "n_pass": 0, "n_fragile_designs": 0, "cases": [], "units_reused": [],
    "control_tokens_detected": False, "control_randint_detected": False, "control_perm_detected": False,
    "control_energy_detected": False, "error": "",
  }  # fmt: skip
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  controls: dict[str, dict[str, Any]] = {k: {"applies": False, "flagged": False} for k in ("tokens", "randint", "perm", "energy")}
  consistent_flags: list[bool] = []
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
    result["git_hash"], result["git_clean"] = _git_state_safe()
    result["onnxruntime_web_version"] = _ort_version(args.ort_web_dir)
    result["node_version"] = _node_version(args.node_bin)
    model = _load_model(args.state_npz)
    bad_model = _perturbed_model(model)
    digests = _js_digests()
    result["jax_arm_ok"] = True

    all_ran = True
    for cell_index, (name, (rel, _sha)) in enumerate(STRUCTURES.items()):
      pdb = args.potts_root / "energy_benchmark_datasets" / rel
      pdb_bytes = pdb.read_bytes()
      pdb_sha = hashlib.sha256(pdb_bytes).hexdigest()
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
        cid = f"{name}__{case}"
        config = PHDesignConfig(binder_chain=binder_chain, method=method, center_types=CENTER_TYPES,
                                temperature=temperature, samples_per_site=SAMPLES)  # fmt: skip
        row: dict[str, Any] = {"id": cid, "cell": name, "case": case, "method": method, "temperature": temperature,
                               "ran": False, "passed": False, "reused": False, "error": ""}  # fmt: skip
        result["cases"].append(row)
        seed = case_seed(case_index, cell_index)
        rec = record_case(table, e_idx, native, binder_mask, res_id, config, seed)
        cdf_order = _cdf_order_array()
        ref_raw = reference_designs(table, e_idx, native, binder_mask, res_id, config, rec["replay"])
        ref = [_ref_dict(d, rec["events"][int(d.sample)], cdf_order) for d in ref_raw]
        ref_by_sample = {r["sample"]: r for r in ref}
        consistent = len(ref) == SAMPLES and all(
          s in ref_by_sample and ref_by_sample[s]["sequence"] == rec["sequences"][s].tolist() for s in range(SAMPLES))
        consistent_flags.append(consistent)
        row["reference_consistent"] = bool(consistent)
        job = {"pdb": pdb_bytes.decode(), "labels": None, "bucket": _bucket_for(length),
               "config": js_config(config), "replay": rec["replay"]}  # fmt: skip
        want = inputs_sha256({"pdb_sha256": pdb_sha, "job": {k: v for k, v in job.items() if k != "pdb"},
                              "scoring_manifest_sha256": args.scoring_manifest_sha256,
                              "ph_manifest_sha256": args.ph_manifest_sha256, "js_sha256": digests})  # fmt: skip
        unit_dir = args.work_dir / "units" / cid
        reused = _read_stamp(unit_dir, want)
        if reused is not None:
          node_part = {"node": reused["node"], "output_sha256": reused["output_sha256"]}
          result["units_reused"].append({"unit": cid, "dir": str(unit_dir), "inputs_sha256": want,
                                         "output_sha256": reused["output_sha256"]})  # fmt: skip
          row["reused"] = True
        else:
          node_part = _run_unit(job, unit_dir, args, want)
        node = node_part["node"]
        if node.get("ok") is not True:
          all_ran = False
          row["error"] = str(node.get("error") or "node failed")[:2000]
          continue
        js = [normalize_js(d) for d in node["designs"]]
        row["ran"] = True
        graded = compare_designs(js, ref, exempt_fragile=True)
        row.update(n_designs=len(js), n_fragile_designs=graded["fragile"], passed=bool(graded["pass"]),
                   mismatches=graded["mismatches"][:20],
                   first_fragile=[r["first_fragile"] for r in ref])  # fmt: skip
        result["n_fragile_designs"] += int(graded["fragile"])
        ctx = {"replay": rec["replay"], "config": config, "method": method, "temperature": temperature, "table": table,
               "e_idx": e_idx, "native": native, "binder_mask": binder_mask, "res_id": res_id, "cdf_order": cdf_order,
               "ref": ref, "bad_model": bad_model, "args_t": args_t}  # fmt: skip
        _evaluate_controls(row, controls, js, ctx)
        logger.info("%s: passed=%s fragile=%s reused=%s", cid, row["passed"], graded["fragile"], row["reused"])
    result["node_ok"] = all_ran
    result["reference_consistent"] = bool(consistent_flags) and all(consistent_flags)
    result["n_cases"] = len(result["cases"])
    result["n_pass"] = sum(bool(c.get("passed")) for c in result["cases"])
    result["control_tokens_detected"] = bool(controls["tokens"]["flagged"])
    result["control_randint_detected"] = bool(controls["randint"]["flagged"])
    result["control_perm_detected"] = bool(controls["perm"]["flagged"])
    result["control_energy_detected"] = bool(controls["energy"]["flagged"])
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("ph methods node gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("outcome=%s cases pass %s/%s", _decide(result), result["n_pass"], result["n_cases"])
  return 0


def _cdf_order_array() -> np.ndarray:
  from aminx.families.protonpotts_mpnn.ph_sample import _cdf_order  # noqa: PLC0415

  return np.asarray(_cdf_order())


def _git_state_safe() -> tuple[str, bool]:
  try:
    return git_state(REPO)
  except (OSError, subprocess.SubprocessError):
    return "", False


if __name__ == "__main__":
  raise SystemExit(main())
