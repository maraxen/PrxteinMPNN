"""X10 deployed-site gate: the built dist/protonpotts/ site reproduces the graded ProtonPotts harness pages bitwise.

task_id 261010_protonpotts-site. Sidecar: protonpotts_site_gate.bth.toml (pre-registered before this script existed).

The deployed site (dist/protonpotts, assembled by tools/build_site.py) is served from a SUBPATH (/aminx/protonpotts/, as GitHub Pages
serves a project site) and driven through its own hook window.__PP_RUN__ by a driver page. Its replies are compared EXACTLY with the
outputs of the graded harness pages (X6c/X7c/X8c sites, cells.json as graded) at the same thread count:
  pages : deployed, no COOP/COEP, 1 thread      vs harness_t1
  coi   : deployed, COOP/COEP,      4 threads   vs harness_t4
Controls: a planted one-token / rolled-uniform difference per arm must mismatch; a byte-flipped L256 energy graph must be refused.
The page grades nothing; every comparison lives here.
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
from scripts.browser_validation.potts_browser_parity import git_state  # noqa: E402
from scripts.browser_validation.protonpotts_export_gate import _sha256  # noqa: E402
from scripts.browser_validation.protonpotts_score_gate import _verify_artifacts  # noqa: E402

logger = logging.getLogger("protonpotts_site_gate")

ARMS = ("scoring", "ph", "sampler")
THREADS = (1, 4)
RUNNER = LAYER_C_DIR / "run_p07.mjs"
SCORER_DIR = REPO / "browser" / "protonpotts-scorer"
GRADED_DIRS = {
  "scoring": "protonpotts_browser_parity",
  "ph": "protonpotts_ph_browser_parity",
  "sampler": "protonpotts_sampler_browser_parity",
}
INDEX_FOR = {"scoring": "protonpotts_index.html", "ph": "ph_index.html", "sampler": "sampler_index.html"}
PAGE_FILES = {
  "scoring": ("protonpotts_page.mjs", "protonpotts_inputs.mjs", "protonpotts_scorer.mjs"),
  "ph": ("ph_page.mjs", "ph_design.mjs", "ph_plan.mjs", "protonpotts_inputs.mjs", "protonpotts_scorer.mjs"),
  "sampler": ("sampler_page.mjs", "decoder_sampler.mjs", "protonpotts_inputs.mjs", "protonpotts_scorer.mjs"),
}
# Graded-site manifest path -> export kind whose MANIFEST.json it must equal byte for byte.
MANIFESTS = {
  "scoring": (("models/MANIFEST.json", "scoring"),),
  "ph": (("models/scoring/MANIFEST.json", "scoring"), ("models/ph/MANIFEST.json", "ph")),
  "sampler": (("models/MANIFEST.json", "decoder"),),
}
REPO_PAGE_PATHS = tuple(sorted({*(name for names in PAGE_FILES.values() for name in names), *INDEX_FOR.values(),
                                "ph_plan.mjs", "ph_design.mjs", "decoder_sampler.mjs"}))
LIB_FILES = ("protonpotts_inputs.mjs", "protonpotts_scorer.mjs", "ph_plan.mjs", "ph_design.mjs", "decoder_sampler.mjs")
SITE_FILES = ("index.html", "app.mjs", "worker.mjs", "config.js")
MODEL_KINDS = ("scoring", "ph", "decoder")
ORT_VERSION = "1.30.0"
PLANTED = "__planted"
FLIP_OFFSET = 100
INTEGRITY_BUCKET = 256
FLOAT_FIELDS = frozenset({"final_potts_energy", "selective_energy", "selective_energies"})
KEEP_OUT_OF_DIGEST = frozenset({"stamp.json", "run_p07.log"})

DRIVER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>X10 deployed-site driver</title>
</head>
<body>
<iframe id="pp" src="./aminx/protonpotts/index.html" title="ProtonPotts"></iframe>
<script type="module">
// X10 driver (tracked by scripts/browser_validation/protonpotts_site_gate.py). Sends each message in ./x10_msgs.json through the
// deployed site's own hook window.__PP_RUN__ and records the replies. It grades nothing.
const qs = new URLSearchParams(window.location.search);
const numThreads = Number(qs.get("numThreads") || "1");
const frame = document.getElementById("pp");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitReady() {
  for (let i = 0; i < 1800; i += 1) {
    const w = frame.contentWindow;
    if (w && w.__PP_READY__ === true) return;
    await sleep(100);
  }
  throw new Error("deployed site never set __PP_READY__");
}

async function main() {
  await waitReady();
  const msgs = await (await fetch("./x10_msgs.json")).json();
  const replies = [];
  for (const m of msgs) {
    const pdbText = await (await fetch(`./${m.pdb}`)).text();
    const reply = await frame.contentWindow.__PP_RUN__({ id: m.id, kind: m.kind, pdbText, numThreads, params: m.params });
    replies.push({ id: m.id, reply });
  }
  return { replies };
}

main()
  .then((res) => {
    window.__PARITY_RESULT__ = res;
    window.__PARITY_DONE__ = true;
  })
  .catch((err) => {
    window.__PARITY_ERROR__ = String((err && err.stack) || err);
    window.__PARITY_DONE__ = true;
  });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------- pure helpers (unit-tested)


def tree_digest(root: Path, exclude: frozenset[str] = frozenset()) -> str:
  """sha256 over sorted 'relpath\\0sha256\\n' lines of every file under `root` (top-level names in `exclude` are skipped)."""
  entries = []
  for path in root.rglob("*"):
    if not path.is_file():
      continue
    rel = path.relative_to(root).as_posix()
    if rel.split("/", 1)[0] in exclude:
      continue
    entries.append((rel, path))
  entries.sort(key=lambda e: e[0])
  digest = hashlib.sha256()
  for rel, path in entries:
    digest.update(f"{rel}\0{_sha256(path)}\n".encode())
  return digest.hexdigest()


def inputs_sha256(payload: dict[str, Any]) -> str:
  """The resume key of one unit: every input that could change its output, canonical JSON."""
  return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def flip_byte(data: bytes, offset: int) -> bytes:
  """`data` with the byte at `offset` XOR 0xFF."""
  buf = bytearray(data)
  buf[offset] ^= 0xFF
  return bytes(buf)


def shift_variant_names(variants: list[list[str]], symbols: list[str]) -> list[list[str]]:
  """The VOCABULARY control of X6c: every token moved to the next vocabulary symbol, modulo the vocabulary size."""
  index = {s: i for i, s in enumerate(symbols)}
  size = len(symbols)
  return [[symbols[(index[name] + 1) % size] for name in row] for row in variants]


def roll_uniforms(uniforms: list[list[float]]) -> list[list[float]]:
  """Every sample's uniform stream rolled by one position, in float32 (the X7c/X8c rolled-uniform control)."""
  return [np.roll(np.asarray(u, dtype=np.float32), 1).tolist() for u in uniforms]


def roll_flat(uniforms: list[float]) -> list[float]:
  """One flat uniform stream rolled by one position, in float32 (the sampler cells carry ONE stream of length L, not one per sample).

  Run 40a3e57a used roll_uniforms here, which on a flat list rolls each scalar by itself -- a no-op -- so the sampler PLANTED control
  could not fire (outcome instrument_unverified). This is the fix; the outcome criteria are unchanged.
  """
  return np.roll(np.asarray(uniforms, dtype=np.float32), 1).tolist()


def _pdb_rel(arm: str, cell: dict[str, Any]) -> str:
  return f"pdbs/{arm}_{Path(cell['pdb']).name}"


def translate(arm: str, cell: dict[str, Any], pdb_rel: str) -> dict[str, Any]:
  """One harness cell as the deployed hook's message (the same inputs the harness page passes to its method)."""
  if arm == "scoring":
    kind = "score"
    params: dict[str, Any] = {"labels": cell.get("labels"), "variants": cell["variants"], "bucket": cell["bucket"]}
  elif arm == "ph":
    kind = "design"
    params = {"labels": cell.get("labels"), "config": cell["config"], "bucket": cell["bucket"]}
    if cell.get("uniforms") is not None:
      params["uniforms"] = cell["uniforms"]
  else:
    kind = "sample"
    params = {"labels": None, "seed": cell.get("seed", 0), "temperature": cell["temperature"], "bucket": cell["bucket"]}
    for key in ("designed", "bias"):
      if cell.get(key) is not None:
        params[key] = cell[key]
    if cell.get("uniforms") is not None:
      params["uniforms"] = cell["uniforms"]
      params["noise"] = cell["noise"]
  return {"id": cell["id"], "kind": kind, "pdb": pdb_rel, "params": params}


def planted_cell(arm: str, cells: list[dict[str, Any]], symbols: list[str]) -> dict[str, Any] | None:
  """The one perturbed copy per arm (id suffixed PLANTED), or None when the arm has no eligible cell."""
  if arm == "scoring":
    base = cells[0] if cells else None
    if base is None:
      return None
    changed = {"variants": shift_variant_names(base["variants"], symbols)}
  elif arm == "ph":
    base = next((c for c in cells if c.get("uniforms") is not None), None)
    if base is None:
      return None
    changed = {"uniforms": roll_uniforms(base["uniforms"])}
  else:
    base = next((c for c in cells if "bias_temp" in str(c["id"]) and c.get("uniforms") is not None), None)
    if base is None:
      return None
    changed = {"uniforms": roll_flat(base["uniforms"])}
  return {**base, **changed, "id": f"{base['id']}{PLANTED}", "planted_of": base["id"]}


def _f32_equal(a: Any, b: Any) -> bool:  # noqa: ANN401
  """Bitwise float32 equality of two numeric values or lists (both cast to float32 first)."""
  x = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
  y = np.ascontiguousarray(np.asarray(b, dtype=np.float32))
  return x.shape == y.shape and bool(np.array_equal(x.view(np.uint32), y.view(np.uint32)))


def _plain_equal(a: Any, b: Any) -> bool:  # noqa: ANN401
  return a == b


def _int_equal(a: Any, b: Any) -> bool:  # noqa: ANN401
  try:
    return bool(np.array_equal(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)))
  except (TypeError, ValueError):
    return False


def compare_scoring(ref_energies: np.ndarray, ref_tokens: Any, got_energies: Any, got_tokens: Any) -> dict[str, Any]:  # noqa: ANN401
  """X6c comparator: energies as float32 with np.array_equal, tokens as integers."""
  mismatches = []
  if not np.array_equal(np.asarray(ref_energies, dtype=np.float32), np.asarray(got_energies, dtype=np.float32)):
    mismatches.append("energies")
  if not _int_equal(ref_tokens, got_tokens):
    mismatches.append("tokens")
  return {"exact": not mismatches, "mismatches": mismatches}


def compare_design(ref_designs: list[dict[str, Any]], got_designs: list[dict[str, Any]]) -> dict[str, Any]:
  """X7c comparator over every field the harness design carries (the deployed reply's extra `symbols` is not graded)."""
  if not isinstance(ref_designs, list) or not isinstance(got_designs, list) or len(ref_designs) != len(got_designs) \
      or not ref_designs:
    return {"exact": False, "mismatches": ["designs"]}
  mismatches = []
  for i, (r, g) in enumerate(zip(ref_designs, got_designs, strict=True)):
    for field in sorted(r):
      if field not in g:
        mismatches.append(f"designs[{i}].{field}")
        continue
      same = _f32_equal(r[field], g[field]) if field in FLOAT_FIELDS else _plain_equal(r[field], g[field])
      if not same:
        mismatches.append(f"designs[{i}].{field}")
  return {"exact": not mismatches, "mismatches": mismatches}


def compare_sample(ref: dict[str, Any], got: dict[str, Any]) -> dict[str, Any]:
  """X8c comparator: sequence and decoding order exact, log-probabilities float32 bitwise."""
  mismatches = []
  for field in ("sequence", "decoding_order"):
    if field not in got or not _plain_equal(ref.get(field), got.get(field)):
      mismatches.append(field)
  if "log_probs" not in got or not _f32_equal(ref.get("log_probs"), got.get("log_probs")):
    mismatches.append("log_probs")
  return {"exact": not mismatches, "mismatches": mismatches}


def _has_output(arm: str, obj: dict[str, Any]) -> bool:
  if arm == "scoring":
    return bool(obj.get("energies"))
  if arm == "ph":
    return bool(obj.get("designs"))
  return bool(obj.get("sequence")) and bool(obj.get("log_probs"))


def _env_ok(env: dict[str, Any], isolated: bool, threads: int) -> bool:
  return (bool(env.get("cross_origin_isolated")) is isolated and env.get("num_threads_effective") == threads
          and env.get("ort_version") == ORT_VERSION)


def _decide(result: dict[str, Any]) -> str:
  """The pre-registered outcome ladder (first match wins: error, instrument_unverified, fail, pass)."""
  if (
    not result.get("artifacts_verified") or not result.get("harness_pages_match_graded") or not result.get("site_built")
    or not result.get("all_units_complete") or result.get("error")
  ):
    return "error"
  if not (result.get("planted_detected") and result.get("integrity_refused") and result.get("environment_ok")):
    return "instrument_unverified"
  return "pass" if result.get("all_exact") else "fail"


# ---------------------------------------------------------------- filesystem and subprocess helpers


def _link_tree(src: Path, dst: Path) -> None:
  """Recreate `src` under `dst` with hard links (copy where linking fails). Callers must unlink before writing into it."""
  if dst.exists():
    shutil.rmtree(dst)
  for path in sorted(src.rglob("*")):
    target = dst / path.relative_to(src)
    if path.is_dir():
      target.mkdir(parents=True, exist_ok=True)
      continue
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
      os.link(path, target)
    except OSError:
      shutil.copy2(path, target)


def _same_bytes(a: Path, b: Path) -> bool:
  return a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes()


def _graded_pages_problems(arm: str, site: Path) -> list[str]:
  problems = []
  expected = {"index.html", *PAGE_FILES[arm]}
  present = {p.name for p in site.iterdir() if p.is_file() and p.suffix in (".mjs", ".html")}
  if present != expected:
    problems.append(f"{arm}: graded site pages {sorted(present)} != {sorted(expected)}")
  for name in expected & present:
    src = SCORER_DIR / (INDEX_FOR[arm] if name == "index.html" else name)
    if not _same_bytes(site / name, src):
      problems.append(f"{arm}: graded {name} differs from {src.relative_to(REPO)}")
  return problems


def _site_problems(dist: Path) -> list[str]:
  problems = []
  for name in SITE_FILES:
    if not (dist / name).is_file():
      problems.append(f"missing dist/protonpotts/{name}")
  for name in LIB_FILES:
    if not _same_bytes(dist / "lib" / name, SCORER_DIR / name):
      problems.append(f"lib/{name} is not byte-identical to browser/protonpotts-scorer/{name}")
  worker = dist / "worker.mjs"
  if worker.is_file() and "../../browser/" in worker.read_text():
    problems.append("worker.mjs still contains ../../browser/")
  return problems


def _run_p07(node: str, site: Path, out_dir: Path, threads: int, timeout_ms: int, isolate: bool) -> dict[str, Any]:  # noqa: FBT001
  """One run of browser/layer_c/run_p07.mjs; returns its harness.json. Raises only when no harness.json was written."""
  if out_dir.exists():
    shutil.rmtree(out_dir)
  out_dir.mkdir(parents=True)
  cmd = [node, str(RUNNER), "--site", str(site), "--out-dir", str(out_dir), "--num-threads", str(threads),
         "--timeout-ms", str(timeout_ms)]
  if not isolate:
    cmd.append("--no-isolation")
  with (out_dir / "run_p07.log").open("w") as log:
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=timeout_ms // 1000 + 600)  # noqa: S603
  harness_path = out_dir / "harness.json"
  if not harness_path.is_file():
    msg = f"run_p07 wrote no harness.json for {out_dir.name} (see run_p07.log)"
    raise RuntimeError(msg)
  return json.loads(harness_path.read_text())


def _reusable(out_dir: Path, inputs: str) -> dict[str, Any] | None:
  """The stored payload when the stamp is complete, its inputs match, and the unit's artifacts still hash to the stamp."""
  try:
    stamp = json.loads((out_dir / "stamp.json").read_text())
  except (OSError, ValueError):
    return None
  if stamp.get("complete") is not True or stamp.get("inputs_sha256") != inputs or not (out_dir / "harness.json").is_file():
    return None
  if tree_digest(out_dir, KEEP_OUT_OF_DIGEST) != stamp.get("artifact_sha256"):
    return None
  return json.loads((out_dir / "harness.json").read_text())


def _unit_problems(spec: dict[str, Any], payload: dict[str, Any] | None) -> list[str]:
  if payload is None:
    return ["no payload"]
  if not payload.get("harnessOk"):
    return [f"harnessOk false: {str(payload.get('harnessError'))[:300]}"]
  res = payload.get("result") or {}
  problems: list[str] = []
  if spec["kind"] == "harness":
    cells = {c.get("id"): c for c in res.get("cells", [])}
    for cid in spec["expected"]:
      cell = cells.get(cid)
      if not cell or not cell.get("ok"):
        problems.append(f"{cid}: missing or not ok")
      elif not _has_output(spec["arm"], cell):
        problems.append(f"{cid}: no output")
    return problems
  replies = {r.get("id"): r.get("reply") for r in res.get("replies", [])}
  for cid in spec["expected"]:
    rep = replies.get(cid)
    if rep is None:
      problems.append(f"{cid}: no reply")
    elif spec["kind"] == "integrity":
      if rep.get("ok") is not False:
        problems.append(f"{cid}: tampered graph was not refused")
    elif rep.get("ok") is not True or not isinstance(rep.get("result"), dict) or not _has_output(spec["arm"], rep["result"]):
      problems.append(f"{cid}: not ok or empty ({str(rep.get('error'))[:200]})")
  return problems


def _grade_reply(arm: str, ref: dict[str, Any], ref_dir: Path, reply: dict[str, Any] | None) -> dict[str, Any]:
  if reply is None or reply.get("ok") is not True:
    return {"exact": False, "mismatches": ["no_ok_reply"]}
  got = reply["result"]
  try:
    if arm == "scoring":
      ref_energies = np.fromfile(ref_dir / ref["energies"]["file"], dtype=np.float32)
      return compare_scoring(ref_energies, ref["tokens"], got["energies"], got["tokens"])
    if arm == "ph":
      return compare_design(ref["designs"], got["designs"])
    return compare_sample(ref, got)
  except Exception as exc:  # noqa: BLE001 -- a grading fault is a non-exact comparison, named as such
    return {"exact": False, "mismatches": [f"grading_error:{type(exc).__name__}"]}


def _symbols() -> list[str]:
  from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6  # noqa: PLC0415

  return list(PROTONPOTTS_V6.symbols)


# ---------------------------------------------------------------- the gate


def _run_gate(args: argparse.Namespace, result: dict[str, Any]) -> None:  # noqa: C901, PLR0912, PLR0915
  dist_root = args.dist
  dist = dist_root / "protonpotts"
  graded = {arm: args.graded_outputs / GRADED_DIRS[arm] / "work" / "site" for arm in ARMS}
  exports = {
    "scoring": (args.scoring_models, args.scoring_manifest_sha256),
    "ph": (args.ph_models, args.ph_manifest_sha256),
    "decoder": (args.decoder_models, args.decoder_manifest_sha256),
  }
  problems: list[str] = []

  # 1. artifacts: export manifests, dist and graded-site manifests byte-identical to the export's.
  for models, sha in exports.values():
    _verify_artifacts(models, sha)
  for kind in MODEL_KINDS:
    if not _same_bytes(dist / "models" / kind / "MANIFEST.json", exports[kind][0] / "MANIFEST.json"):
      problems.append(f"dist models/{kind}/MANIFEST.json differs from the {kind} export")
  for arm in ARMS:
    for rel, kind in MANIFESTS[arm]:
      if not _same_bytes(graded[arm] / rel, exports[kind][0] / "MANIFEST.json"):
        problems.append(f"graded {arm} site {rel} differs from the {kind} export")
  result["artifacts_verified"] = not problems
  if problems:
    result["error"] = "; ".join(problems)[:4000]
    return

  # 2. harness pages: git-clean against the graded commit, and the graded site's own copies identical to the repo's.
  page_problems = []
  for arm in ARMS:
    page_problems += _graded_pages_problems(arm, graded[arm])
  paths = [f"browser/protonpotts-scorer/{n}" for n in REPO_PAGE_PATHS]
  diff = subprocess.run(["git", "diff", "--quiet", args.graded_commit, "--", *paths], cwd=REPO,  # noqa: S603, S607
                        capture_output=True, check=False, timeout=120)
  if diff.returncode != 0:
    page_problems.append(f"harness pages differ from {args.graded_commit} (git diff --quiet rc={diff.returncode})")
  result["harness_pages_match_graded"] = not page_problems
  if page_problems:
    result["error"] = "; ".join(page_problems)[:4000]
    return

  # 3. the built site: files present, lib byte-identical to the repo, the worker's imports rewritten.
  site_problems = _site_problems(dist)
  result["site_built"] = not site_problems
  result["site_tree_sha256"] = tree_digest(dist) if dist.is_dir() else ""
  if site_problems:
    result["error"] = "; ".join(site_problems)[:4000]
    return

  # 4. input cells: the graded cells.json files, their sha256, the PDBs they reference.
  cells: dict[str, list[dict[str, Any]]] = {}
  cells_sha: dict[str, str] = {}
  for arm in ARMS:
    cells[arm] = json.loads((graded[arm] / "cells.json").read_text())["cells"]
    cells_sha[arm] = _sha256(graded[arm] / "cells.json")
    for cell in cells[arm]:
      if not (graded[arm] / cell["pdb"]).is_file():
        msg = f"{arm} cell {cell['id']} references missing {cell['pdb']}"
        raise RuntimeError(msg)
  result["cells_sha256"] = cells_sha
  result["graded_sites"] = {arm: str(graded[arm]) for arm in ARMS}

  node = find_node(args.node)
  if node is None:
    msg = "no node binary found (--node, $NODE_BIN, $PATH, nvm)"
    raise RuntimeError(msg)
  env = node_env_report(node)
  result["node_version"] = env.get("node") or ""
  result["onnxruntime_web_version"] = env.get("onnxruntime-web") or ""
  result["playwright_version"] = env.get("@playwright/test") or ""

  # 5. staging: graded sites (as-is), deployed root (subpath + driver + PDBs), integrity root (one flipped byte).
  work = args.work_dir
  work.mkdir(parents=True, exist_ok=True)
  for arm in ARMS:
    _link_tree(graded[arm], work / "graded" / arm)
  deployed = work / "deployed_root"
  _link_tree(dist, deployed / "aminx" / "protonpotts")
  (deployed / "index.html").write_text(DRIVER_HTML)
  pdb_dir = deployed / "pdbs"
  pdb_dir.mkdir(parents=True, exist_ok=True)
  pdb_digest: dict[str, str] = {}
  for arm in ARMS:
    lines = []
    for cell in cells[arm]:
      src = graded[arm] / cell["pdb"]
      shutil.copy2(src, pdb_dir / Path(_pdb_rel(arm, cell)).name)
      lines.append(f"{Path(cell['pdb']).name}\0{_sha256(src)}\n")
    pdb_digest[arm] = hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()
  scoring_manifest = json.loads((args.scoring_models / "MANIFEST.json").read_text())
  flip_entry = next((b for b in scoring_manifest["buckets"] if int(b["bucket"]) == INTEGRITY_BUCKET), None)
  if flip_entry is None:
    msg = f"scoring manifest has no bucket {INTEGRITY_BUCKET}"
    raise RuntimeError(msg)
  flipped_file = str(flip_entry["graphs"]["energy"]["file"])
  integrity_cells = [c for c in cells["scoring"] if int(c["bucket"]) == INTEGRITY_BUCKET]
  if not integrity_cells:
    msg = f"no scoring cell in bucket {INTEGRITY_BUCKET} to probe the tampered graph"
    raise RuntimeError(msg)
  probe = min(integrity_cells, key=lambda c: c["bucket"])
  integrity = work / "integrity_root"
  _link_tree(dist, integrity / "aminx" / "protonpotts")
  target = integrity / "aminx" / "protonpotts" / "models" / "scoring" / flipped_file
  original = target.read_bytes()
  target.unlink()
  target.write_bytes(flip_byte(original, FLIP_OFFSET))
  (integrity / "index.html").write_text(DRIVER_HTML)
  _link_tree(pdb_dir, integrity / "pdbs")

  # 6. messages per arm (cells, then the planted copy) and the per-unit specs.
  symbols = _symbols()
  messages: dict[str, list[dict[str, Any]]] = {}
  planted_ref: dict[str, str] = {}
  for arm in ARMS:
    msgs = [translate(arm, c, _pdb_rel(arm, c)) for c in cells[arm]]
    planted = planted_cell(arm, cells[arm], symbols)
    if planted is not None:
      msgs.append(translate(arm, planted, _pdb_rel(arm, planted)))
      planted_ref[arm] = planted["planted_of"]
    messages[arm] = msgs
  integrity_msg = translate("scoring", probe, _pdb_rel("scoring", probe))
  manifests = {kind: exports[kind][1] for kind in MODEL_KINDS}

  graded_digest = {arm: tree_digest(graded[arm]) for arm in ARMS}
  specs: list[dict[str, Any]] = []
  for arm in ARMS:
    for threads in THREADS:
      name = f"harness_{arm}_t{threads}"
      inputs = {"unit": name, "kind": "harness", "arm": arm, "threads": threads, "cells_sha256": cells_sha[arm],
                "graded_site_sha256": graded_digest[arm], "manifests": manifests}
      specs.append({"name": name, "kind": "harness", "arm": arm, "threads": threads, "isolate": True,
                    "expected": [c["id"] for c in cells[arm]], "inputs": inputs})
  for arm in ARMS:
    for threads, label, isolate in ((1, "pages", False), (4, "coi", True)):
      name = f"deployed_{arm}_{label}"
      msgs_sha = hashlib.sha256(json.dumps(messages[arm], sort_keys=True).encode()).hexdigest()
      inputs = {"unit": name, "kind": "deployed", "arm": arm, "threads": threads, "isolate": isolate,
                "cells_sha256": cells_sha[arm], "site_tree_sha256": result["site_tree_sha256"], "pdb_sha256": pdb_digest[arm],
                "messages_sha256": msgs_sha, "manifests": manifests}
      specs.append({"name": name, "kind": "deployed", "arm": arm, "threads": threads, "isolate": isolate,
                    "expected": [m["id"] for m in messages[arm]], "messages": messages[arm], "root": deployed,
                    "inputs": inputs})
  integrity_inputs = {"unit": "integrity_pages", "kind": "integrity", "site_tree_sha256": result["site_tree_sha256"],
                      "flipped_file": flipped_file, "flip_offset": FLIP_OFFSET, "probe": probe["id"],
                      "message_sha256": hashlib.sha256(json.dumps(integrity_msg, sort_keys=True).encode()).hexdigest(),
                      "manifests": manifests}
  specs.append({"name": "integrity_pages", "kind": "integrity", "arm": "scoring", "threads": 1, "isolate": False,
                "expected": [probe["id"]], "messages": [integrity_msg], "root": integrity, "inputs": integrity_inputs})

  # 7. run every unit (each its own process and timeout), persisting completed units with a stamp.
  payloads: dict[str, dict[str, Any] | None] = {}
  units: list[dict[str, Any]] = []
  reused_names: list[str] = []
  for spec in specs:
    name = spec["name"]
    inputs = inputs_sha256(spec["inputs"])
    out = work / "units" / name
    payload = _reusable(out, inputs)
    reused = payload is not None
    error = ""
    try:
      if not reused:
        if out.is_dir():
          (out / "stamp.json").unlink(missing_ok=True)
        if spec["kind"] == "harness":
          payload = _run_p07(node, work / "graded" / spec["arm"], out, spec["threads"], args.timeout_ms, True)  # noqa: FBT003
        else:
          (spec["root"] / "x10_msgs.json").write_text(json.dumps(spec["messages"]))
          payload = _run_p07(node, spec["root"], out, spec["threads"], args.timeout_ms, spec["isolate"])
    except Exception as exc:  # noqa: BLE001 -- a unit failure is recorded, the remaining units still run
      logger.exception("unit %s failed", name)
      error = f"{type(exc).__name__}: {exc}"[:500]
      payload = None
    unit_problems = [error] if error else _unit_problems(spec, payload)
    complete = not unit_problems
    if complete and not reused:
      out.mkdir(parents=True, exist_ok=True)
      (out / "stamp.json").write_text(json.dumps({"unit": name, "inputs_sha256": inputs, "complete": True,
                                                  "artifact_sha256": tree_digest(out, KEEP_OUT_OF_DIGEST)}, indent=2))
    if reused:
      reused_names.append(name)
    payloads[name] = payload if complete else None
    units.append({"name": name, "arm": spec["arm"], "kind": spec["kind"], "threads": spec["threads"],
                  "isolate": spec["isolate"], "inputs_sha256": inputs, "reused": reused, "complete": complete,
                  "n_expected": len(spec["expected"]), "problems": unit_problems[:10]})
    logger.info("unit %s: complete=%s reused=%s problems=%d", name, complete, reused, len(unit_problems))
  result["units"] = units
  result["units_reused"] = reused_names
  result["all_units_complete"] = all(u["complete"] for u in units)

  # 8. grading: every deployed reply against the harness output of the same arm and thread setting.
  comparisons: list[dict[str, Any]] = []
  env_ok = True
  versions: set[str] = set()
  for arm in ARMS:
    for threads, label in ((1, "pages"), (4, "coi")):
      h_name, d_name = f"harness_{arm}_t{threads}", f"deployed_{arm}_{label}"
      h_pay, d_pay = payloads.get(h_name), payloads.get(d_name)
      if h_pay is None or d_pay is None:
        continue
      h_cells = {c["id"]: c for c in h_pay["result"]["cells"]}
      h_dir = work / "units" / h_name
      replies = {r["id"]: r["reply"] for r in d_pay["result"]["replies"]}
      for cell in cells[arm]:
        comp = _grade_reply(arm, h_cells[cell["id"]], h_dir, replies.get(cell["id"]))
        comparisons.append({"arm": arm, "threads": threads, "setting": label, "cell": cell["id"], "planted": False, **comp})
      if arm in planted_ref:
        ref_id = planted_ref[arm]
        pid = f"{ref_id}{PLANTED}"
        comp = _grade_reply(arm, h_cells[ref_id], h_dir, replies.get(pid))
        comparisons.append({"arm": arm, "threads": threads, "setting": label, "cell": pid, "planted": True, **comp})
      for rep in replies.values():
        if rep and rep.get("ok") is True:
          env = rep.get("env") or {}
          versions.add(str(env.get("ort_version")))
          env_ok &= _env_ok(env, isolated=label == "coi", threads=threads)
      env_ok &= any(rep and rep.get("ok") is True for rep in replies.values())
  result["comparisons"] = comparisons
  planted_comps = [c for c in comparisons if c["planted"]]
  exact_comps = [c for c in comparisons if not c["planted"]]
  result["planted_detected"] = (
    len(planted_ref) == len(ARMS) and len(planted_comps) == len(ARMS) * len(THREADS)
    and all(not c["exact"] for c in planted_comps)
  )
  integ = payloads.get("integrity_pages")
  rep = None
  if integ is not None:
    rep = next((r["reply"] for r in integ["result"]["replies"] if r["id"] == probe["id"]), None)
  result["integrity_refused"] = bool(rep and rep.get("ok") is False and flipped_file in str(rep.get("error")))
  result["environment_ok"] = bool(env_ok and versions == {ORT_VERSION})
  result["ort_version_deployed"] = ",".join(sorted(versions))
  expected_exact = 2 * sum(len(cells[a]) for a in ARMS)
  result["all_exact"] = len(exact_comps) == expected_exact and all(c["exact"] for c in exact_comps)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--dist", type=Path, required=True, help="built dist root containing protonpotts/")
  parser.add_argument("--graded-outputs", type=Path, default=REPO / "outputs")
  parser.add_argument("--graded-commit", default="bac26b80")
  parser.add_argument("--scoring-models", type=Path, required=True)
  parser.add_argument("--scoring-manifest-sha256", required=True)
  parser.add_argument("--ph-models", type=Path, required=True)
  parser.add_argument("--ph-manifest-sha256", required=True)
  parser.add_argument("--decoder-models", type=Path, required=True)
  parser.add_argument("--decoder-manifest-sha256", required=True)
  parser.add_argument("--node", default=os.environ.get("NODE_BIN"))
  parser.add_argument("--timeout-ms", type=int, default=1_800_000, help="per unit (one process each)")
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  git_hash, git_clean = git_state(REPO)
  result: dict[str, Any] = {
    "git_hash": git_hash, "git_clean": git_clean,
    "scoring_manifest_sha256": args.scoring_manifest_sha256, "ph_manifest_sha256": args.ph_manifest_sha256,
    "decoder_manifest_sha256": args.decoder_manifest_sha256,
    "artifacts_verified": False, "harness_pages_match_graded": False, "site_built": False, "site_tree_sha256": "",
    "ort_version_deployed": "", "onnxruntime_web_version": "", "playwright_version": "", "node_version": "",
    "units": [], "units_reused": [], "all_units_complete": False, "planted_detected": False,
    "integrity_refused": False, "environment_ok": False, "all_exact": False, "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    _run_gate(args, result)
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("site gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["outcome"] = _decide(result)
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("outcome=%s", result["outcome"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
