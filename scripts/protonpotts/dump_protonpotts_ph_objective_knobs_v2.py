"""Dump upstream ProtonPottsMPNN block descent with the three objective knobs, f32 and f64 (P4m2, task 261011_p4m-objective-knobs).

P4m2 is the re-registration after P4m (`dump_protonpotts_ph_objective_knobs.py`, run b69bd83d) failed ``self_live``: this file is
a COPY of P4m's script (its reproducibility child re-executes ``__file__``, so a wrapper importing P4m would rerun P4m's
configurations) differing only in the self_weight configurations -- block_self0_T0 / block_self0_T05 (self_weight 0.0) and
block_self4_T0 (self_weight 4.0) replace P4m's self_weight 0.5 pair -- and this paragraph. Everything below this paragraph that
mentions self_weight 0.5 or seven configurations describes P4m.

Debt #2621, phase 2 (phase 1 was the visit-order knob, P4k). Pre-registration:
`scripts/protonpotts/dump_protonpotts_ph_objective_knobs_v2.bth.toml` (P4m's: `dump_protonpotts_ph_objective_knobs.bth.toml`). Same harness as the sealed P4g dump
(`dump_protonpotts_ph.py`) and P4k (`dump_protonpotts_ph_knobs.py`), both imported and reused. Seven block-descent
configurations, each P4g's block_T0 / block_T05 production criteria with sweep_order "position" and exactly one objective knob
set away from its default (dataclasses.replace on the upstream criteria): block_glob_p1_T0 / block_glob_p1_T05
(global_weight +1.0), block_glob_m1_T0 (global_weight -1.0), block_adjrep_T0 / block_adjrep_T05 (adjacent_repeat_weight 1.0),
block_self_T0 / block_self_T05 (self_weight 0.5). Run in the `aminx-oracles-protonpotts` environment; imports upstream `mpnn`,
torch and numpy only, never aminx::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/protonpotts/dump_protonpotts_ph_objective_knobs.py --checkpoint <ckpt> --encoder-dir <P4e out dir> \\
        --features-dir ~/projects/aminx-oracles-protonpotts --out <dir> --p4g-dir <P4g out dir> \\
        --cell pkad_unlabelled=<1BVC.pdb>,<chain> --cell multichain=<6m0j.pdb>,E ...

SIGN TRAP. In the CODE, ``wGlob = global_weight / sdGlob`` multiplies ``(g - g0)`` in the combined unary, so a POSITIVE
global_weight MINIMISES g = H(all binder titratable -> protonated) - H(-> deprotonated), i.e. favours PROTONATED. The upstream
docstring ("higher = our goal") reads the other way. This dump records the code's behaviour; the +1 / -1 pair makes the sign
observable (claim ``global_sign_visible``).

RECORDED, per block call (in addition to P4g's): ``sweep_order``, ``knobs`` (all five #2621 knobs plus temperature,
samples_per_site, block_size, combined_lambda, neighbour_k, max_mutations; P4k's KNOB_FIELDS), ``zscales`` (sdH, sdSel, sdGlob,
as returned by upstream ``_block_zscales``, wrapped by P4g; sdGlob is only meaningful when global_weight != 0), ``sd_glob``,
and ``final_global_dH`` (the global term g of the final sequence, upstream ``_global_protonation_dH`` as called by
``_score_design`` whatever the weight; taken from the scored ``PHDesignOutput`` the P4g harness records in ``designs``).
No upstream source is patched; every value comes from wrapping or from upstream's own scored output.

CLAIMS. engine_matches_sealed / complete / shim_consistent / f64_is_float64 / precisions_differ / reproducible / n_cells as P4k
(``complete`` additionally requires the recorded knob values to equal the configured ones, three z-scales per block call, a finite
positive sdGlob for the global configurations, and a final g per call). global_live / adjrep_live / self_live: in at least one
(cell, precision) some T=0 call of the knob's configuration yields a final sequence different from the ordinal-matched P4g
block_T0 call (P4g's sweep_order is "position"). global_sign_visible: in at least one (cell, precision) block_glob_p1_T0 and
block_glob_m1_T0 give a different final sequence. The g comparison (+1 vs -1) is recorded, not a claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
  os.environ[_var] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dump_protonpotts_ph_knobs as p4k  # noqa: E402  -- reused: KNOB_FIELDS, _knobs, _VisitRecorder, _attach_visit_orders
from dump_protonpotts_ph_knobs import p4g  # noqa: E402  -- the base dump, as imported by P4k

logger = logging.getLogger("dump_protonpotts_ph_objective_knobs")

# label -> (P4g base label, knob field, value). Exactly one knob per configuration.
CONFIGS: dict[str, tuple[str, str, float]] = {
  "block_glob_p1_T0": ("block_T0", "global_weight", 1.0),
  "block_glob_p1_T05": ("block_T05", "global_weight", 1.0),
  "block_glob_m1_T0": ("block_T0", "global_weight", -1.0),
  "block_adjrep_T0": ("block_T0", "adjacent_repeat_weight", 1.0),
  "block_adjrep_T05": ("block_T05", "adjacent_repeat_weight", 1.0),
  "block_self0_T0": ("block_T0", "self_weight", 0.0),
  "block_self0_T05": ("block_T05", "self_weight", 0.0),
  "block_self4_T0": ("block_T0", "self_weight", 4.0),
}
# claim -> (the T0 configuration(s) compared against P4g block_T0)
LIVE_CLAIMS = {
  "global_live": ("block_glob_p1_T0", "block_glob_m1_T0"),
  "adjrep_live": ("block_adjrep_T0",),
  "self_live": ("block_self0_T0", "block_self4_T0"),
}


def _knob_criteria() -> dict[str, object]:
  """The eight configurations: P4g's block criteria, sweep_order "position", one objective knob replaced."""
  base = p4g._criteria()  # noqa: SLF001 -- the P4g production values, reused unchanged
  out: dict[str, object] = {}
  for label, (ref, field, value) in CONFIGS.items():
    position = p4k._with_sweep(base[ref], "position")  # noqa: SLF001 -- same dataclass-replace helper as P4k
    out[label] = dataclasses.replace(position, **{field: value})
  return out


def _attach_objective_records(meta: dict) -> None:
  """Add sd_glob and final_global_dH (g of the final sequence) to each block call, from the recorded z-scales and designs."""
  designs = {int(d["call_index"]): d for d in meta["designs"]}
  for call in meta["calls"]:
    if call["method"] != "block":
      continue
    z = call["zscales"]
    call["sd_glob"] = None if z is None or len(z) < 3 else float(z[2])
    design = designs.get(int(call["index"]))
    g = None if design is None else design.get("global_protonation_dH")
    call["final_global_dH"] = None if g is None else float(g)


def _dump(args: argparse.Namespace, out: Path) -> dict[str, dict]:
  import mpnn.inference_engines.potts_mpnn_ph as ph
  import mpnn.pipelines.potts_mpnn as pipelines
  import mpnn.potts_inference as inference
  from mpnn.utils.inference import MPNNInferenceInput

  pipelines.get_protonation_state_transforms = lambda **_: []
  inference._PIPELINE_CACHE.clear()  # noqa: SLF001 -- trap 1 of the P4 dumper (spec §23.4)
  torch.set_num_threads(1)
  out.mkdir(parents=True, exist_ok=True)
  engine = ph.PottsMPNNPHEngine(
    checkpoint_path=str(args.checkpoint), extended_vocab="v6", out_directory=None, write_fasta=False,
    write_structures=False,
  )  # fmt: skip
  criteria = _knob_criteria()
  labels_by_id = {id(crit): name for name, crit in criteria.items()}
  recorder = p4k._VisitRecorder()  # noqa: SLF001 -- block entry point only; sweep_order position runs no ranker
  true_block = ph._block_descent  # noqa: SLF001
  report: dict[str, dict] = {}
  try:
    for cell, (pdb, chain) in args.cells.items():
      # The SAME loader prepare_potts_input uses for a path (see P4g _dump), so the engine featurises the sealed residues.
      atom_array = MPNNInferenceInput.from_atom_array_and_dict(input_dict={"structure_path": str(pdb)}).atom_array
      sealed = p4g._load(args.features_dir / p4g.FEATURES[cell] / f"{cell}.npz")  # noqa: SLF001
      entry: dict = {"binder_chain": chain, "precisions": {}}
      for name, dtype in p4g.PRECISIONS:
        tables = p4g._load(args.encoder_dir / "protonpotts_v6_encoder" / f"{cell}_{name}.npz")  # noqa: SLF001
        recorder.blocks.clear()
        ph._block_descent = recorder.block(true_block, labels_by_id)  # noqa: SLF001
        try:
          arrays, meta = p4g._run_cell(engine, ph, atom_array, chain, tables, dtype, criteria)  # noqa: SLF001
        finally:
          ph._block_descent = true_block  # noqa: SLF001
        p4k._attach_visit_orders(meta, recorder.blocks)  # noqa: SLF001
        _attach_objective_records(meta)
        np.savez(out / f"{cell}_{name}.npz", **arrays)
        (out / f"{cell}_{name}.json").write_text(json.dumps(meta, sort_keys=True, default=str), encoding="utf-8")
        entry["precisions"][name] = {
          "content_sha256": p4g._content_hash(arrays),  # noqa: SLF001
          "meta_sha256": hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest(),
          "n_calls": len(meta["calls"]),
          "dtypes": sorted({str(a.dtype) for k, a in arrays.items() if k.startswith("design_")}),
        }  # fmt: skip
      entry["sealed_L"] = int(sealed["S"].shape[-1])
      report[cell] = entry
  finally:
    ph._block_descent = true_block  # noqa: SLF001
  return report


def _child(args: argparse.Namespace) -> dict[str, str]:
  """A second run in a fresh process, same as P4g / P4k, using this script."""
  with tempfile.TemporaryDirectory() as tmp:
    result = Path(tmp) / "hashes.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint),
               "--encoder-dir", str(args.encoder_dir), "--features-dir", str(args.features_dir),
               "--out", str(Path(tmp) / "child"), "--child-hashes-to", str(result)]  # fmt: skip
    for name, (pdb, chain) in args.cells.items():
      command += ["--cell", f"{name}={pdb},{chain}"]
    subprocess.run(command, check=True, capture_output=True)
    return json.loads(result.read_text(encoding="utf-8"))


def _differs(arrs_a: dict, calls_a: list[dict], arrs_b: dict, calls_b: list[dict]) -> tuple[int, int]:
  """(n ordinal pairs, n pairs whose final sequences differ) between two lists of block calls."""
  n = min(len(calls_a), len(calls_b))
  diff = sum(
    not np.array_equal(arrs_a[f"c{a['index']}_S_out"], arrs_b[f"c{b['index']}_S_out"])
    for a, b in zip(calls_a[:n], calls_b[:n])
  )
  return n, int(diff)


def _mean_g(calls: list[dict]) -> float | None:
  values = [c["final_global_dH"] for c in calls if c.get("final_global_dH") is not None]
  return float(np.mean(values)) if values else None


def _compare(args: argparse.Namespace, directory: Path) -> tuple[dict[str, dict], dict[str, bool]]:
  """Per cell and precision: each configuration vs the P4g default designs, and glob +1 vs -1. Returns (rows, claim flags).

  The live and sign flags are claims, so they are computed from these comparisons (P4k's _vs_p4g machinery, ordinal-matched
  against the P4g block_T0 calls of the same cell and precision)."""
  ref_dir = args.p4g_dir / "protonpotts_v6_ph"
  rows: dict[str, dict] = {}
  claims = {"global_live": False, "adjrep_live": False, "self_live": False, "global_sign_visible": False}
  for cell in args.cells:
    rows[cell] = {}
    for p in ("f32", "f64"):
      meta = json.loads((directory / f"{cell}_{p}.json").read_text(encoding="utf-8"))
      arrs = p4g._load(directory / f"{cell}_{p}.npz")  # noqa: SLF001
      ref_meta = json.loads((ref_dir / f"{cell}_{p}.json").read_text(encoding="utf-8"))
      ref_arrs = p4g._load(ref_dir / f"{cell}_{p}.npz")  # noqa: SLF001
      by_label = {k: [c for c in meta["calls"] if c["label"] == k] for k in CONFIGS}
      per: dict[str, dict] = {}
      for label, (ref_label, _field, _value) in CONFIGS.items():
        rc = [c for c in ref_meta["calls"] if c["label"] == ref_label]
        n, diff = _differs(arrs, by_label[label], ref_arrs, rc)
        per[label] = {"p4g_label": ref_label, "n_ordinal_pairs": n, "n_differ_from_p4g": diff,
                      "n_calls_knob": len(by_label[label]), "n_calls_p4g": len(rc),
                      "mean_final_global_dH": _mean_g(by_label[label])}  # fmt: skip
      for claim, labels in LIVE_CLAIMS.items():
        claims[claim] |= any(per[lb]["n_differ_from_p4g"] > 0 for lb in labels)
      n_pm, diff_pm = _differs(arrs, by_label["block_glob_p1_T0"], arrs, by_label["block_glob_m1_T0"])
      claims["global_sign_visible"] |= diff_pm > 0
      per["glob_p1_vs_m1_T0"] = {
        "n_ordinal_pairs": n_pm, "n_differ": diff_pm,
        "mean_g_p1": _mean_g(by_label["block_glob_p1_T0"]), "mean_g_m1": _mean_g(by_label["block_glob_m1_T0"]),
      }  # fmt: skip
      rows[cell][p] = per
  return rows, claims


def _knobs_match(call: dict) -> bool:
  """The recorded knobs equal the configured ones: the one knob at its value, the other two objective knobs at defaults."""
  _ref, field, value = CONFIGS[call["label"]]
  defaults = {"global_weight": 0.0, "adjacent_repeat_weight": 0.0, "self_weight": 1.0}
  knobs = call["knobs"]
  return (
    all(k in knobs for k in p4k.KNOB_FIELDS)
    and all(float(knobs[f]) == (value if f == field else d) for f, d in defaults.items())
    and knobs["sweep_order"] == "position"
    and knobs["zscale_mode"] == "block"
  )  # fmt: skip


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--encoder-dir", type=Path, required=True)
  parser.add_argument("--features-dir", type=Path, required=True)
  parser.add_argument("--cell", action="append", default=[], metavar="NAME=PDB,CHAIN")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--p4g-dir", type=Path, default=None,
    help="P4g out dir (holding protonpotts_v6_ph/); required: the live claims compare against the P4g designs",
  )  # fmt: skip
  parser.add_argument("--child-hashes-to", type=Path, default=None)
  args = parser.parse_args()
  if args.p4g_dir is None and args.child_hashes_to is None:
    parser.error("--p4g-dir is required (the knob-live claims compare against the P4g designs)")
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args.cells = {}
  for item in args.cell:
    name, _, rest = item.partition("=")
    pdb, _, chain = rest.rpartition(",")
    args.cells[name] = (Path(pdb), chain)

  directory = args.out / "protonpotts_v6_ph"
  report = _dump(args, directory)
  if args.child_hashes_to is not None:
    args.child_hashes_to.write_text(json.dumps(p4g._flatten(report)), encoding="utf-8")  # noqa: SLF001
    return 0

  reproducible = _child(args) == p4g._flatten(report)  # noqa: SLF001
  complete = shim_consistent = f64_ok = precisions_differ = engine_matches_sealed = True
  token_agreement: dict[str, float] = {}
  labels = set(CONFIGS)
  for cell in args.cells:
    metas = {p: json.loads((directory / f"{cell}_{p}.json").read_text(encoding="utf-8")) for p in ("f32", "f64")}
    arrs = {p: p4g._load(directory / f"{cell}_{p}.npz") for p in ("f32", "f64")}  # noqa: SLF001
    for p in ("f32", "f64"):
      seen = {c["label"] for c in metas[p]["calls"]}
      blocks = [c for c in metas[p]["calls"] if c["method"] == "block"]
      complete &= seen == labels and all(
        c["zscales"] is not None and len(c["zscales"]) == 3 and _knobs_match(c) and c["final_global_dH"] is not None
        for c in blocks
      )  # fmt: skip
      complete &= all(
        c["sd_glob"] is not None and math.isfinite(c["sd_glob"]) and c["sd_glob"] > 0
        for c in blocks if CONFIGS[c["label"]][1] == "global_weight"
      )  # fmt: skip
      for c in metas[p]["calls"]:
        used = arrs[p][f"c{c['index']}_uniforms"].shape[0]
        shim_consistent &= (used == 0) if c["temperature"] <= 0 else (used > 0)
    sealed = p4g._load(args.features_dir / p4g.FEATURES[cell] / f"{cell}.npz")  # noqa: SLF001
    tables = p4g._load(args.encoder_dir / "protonpotts_v6_encoder" / f"{cell}_f32.npz")  # noqa: SLF001
    for p in ("f32", "f64"):
      engine_matches_sealed &= bool(
        arrs[p]["ctx_S_native"].shape == sealed["S"][0].shape
        and np.array_equal(arrs[p]["ctx_S_native"], sealed["S"][0])
        and np.array_equal(arrs[p]["ctx_eidx"], tables["E_idx"])
      )
    f64_ok &= arrs["f64"]["design_final_potts_energy"].dtype == np.float64
    precisions_differ &= bool(
      np.abs(arrs["f64"]["design_final_potts_energy"] - arrs["f32"]["design_final_potts_energy"].astype(np.float64)).max()
      > 0.0
    )
    same = [
      bool(np.array_equal(arrs["f32"][f"c{i}_S_out"], arrs["f64"][f"c{i}_S_out"]))
      for i in range(min(len(metas["f32"]["calls"]), len(metas["f64"]["calls"])))
    ]
    token_agreement[cell] = float(np.mean(same)) if same else float("nan")
  comparison, claims = _compare(args, directory)
  flags = {
    "engine_matches_sealed": bool(engine_matches_sealed),
    "complete": bool(complete),
    "shim_consistent": bool(shim_consistent),
    "f64_is_float64": bool(f64_ok),
    "precisions_differ": bool(precisions_differ),
    "reproducible": bool(reproducible),
    "global_live": bool(claims["global_live"]),
    "adjrep_live": bool(claims["adjrep_live"]),
    "self_live": bool(claims["self_live"]),
    "global_sign_visible": bool(claims["global_sign_visible"]),
    "n_cells": len(report),
  }
  (args.out / "ph_objective_knobs_manifest.json").write_text(
    json.dumps(
      {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
       "torch_version": torch.__version__, "seed": p4g.SEED, "cells": {n: [str(p), c] for n, (p, c) in args.cells.items()},
       "configs": {k: list(v) for k, v in CONFIGS.items()},
       "report": report, "f32_vs_f64_same_final_tokens": token_agreement, "flags": flags,
       "p4g_and_sign_comparison": comparison},
      indent=2, sort_keys=True),
    encoding="utf-8",
  )  # fmt: skip
  logger.info("flags %s agreement %s", flags, token_agreement)
  results = os.environ.get("BTH_RESULTS_PATH")
  if results:
    Path(results).write_text(json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8")
  else:
    logger.warning("BTH_RESULTS_PATH unset: this run will record outcome 'unknown'")
  return 0


if __name__ == "__main__":
  sys.exit(main())
