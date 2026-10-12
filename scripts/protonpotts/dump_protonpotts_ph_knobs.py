"""Dump upstream ProtonPottsMPNN block descent with the visit-order knob (sweep_order knn / energy), f32 and f64 (P4k).

Pre-registration: `scripts/protonpotts/dump_protonpotts_ph_knobs.bth.toml` (#2621). Same harness as the sealed P4g dump
(`scripts/protonpotts/dump_protonpotts_ph.py`, imported and reused here), with four block-descent configurations that
differ only in the visit-order knob: block_knn_T0 / block_knn_T05 (sweep_order "knn", upstream _knn_rank) and
block_energy_T0 / block_energy_T05 (sweep_order "energy", upstream _centre_energy_rank). Run in the
`aminx-oracles-protonpotts` environment; imports upstream `mpnn`, torch and numpy only, never aminx::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/protonpotts/dump_protonpotts_ph_knobs.py --checkpoint <ckpt> --encoder-dir <P4e out dir> \\
        --features-dir ~/projects/aminx-oracles-protonpotts --out <dir> \\
        --cell pkad_unlabelled=<1BVC.pdb>,<chain> --cell multichain=<6m0j.pdb>,E ... [--p4g-dir <P4g out dir>]

VISIT ORDER RECORDING. ``ph._knn_rank`` and ``ph._centre_energy_rank`` are wrapped (module attributes, which
``_block_descent`` resolves as globals) so every returned order is recorded inside the block-descent call that
produced it. ``ph._block_descent`` is wrapped too, so each block-descent invocation logs its own criteria (id-mapped to
its label) before it runs; these invocations are matched 1:1, in order, to the block calls ``_run_cell`` records. Every
wrapper is restored in ``finally``.

RECORDED, per block call (in addition to P4g's): ``sweep_order``, ``knobs`` (sweep_order, temperature, samples_per_site,
block_size, combined_lambda, neighbour_k, max_mutations), ``visit_order`` (the order the ranker returned, or None when
no ranking ran or it ran more than once), ``visit_recordings`` and ``neigh_ascending``.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import logging
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
import dump_protonpotts_ph as p4g  # noqa: E402  -- reused: _run_cell, _Stream, _load, _flatten, _jsonable, constants

logger = logging.getLogger("dump_protonpotts_ph_knobs")

KNOB_FIELDS = (
  "sweep_order", "global_weight", "zscale_mode", "adjacent_repeat_weight", "self_weight",  # the five #2621 knobs
  "temperature", "samples_per_site", "block_size", "combined_lambda", "neighbour_k", "max_mutations",
)  # fmt: skip
# The P4g block-descent cells, compared ordinal-for-ordinal against the P4g dump when --p4g-dir is given.
P4G_REFERENCE = {
  "block_knn_T0": "block_T0", "block_knn_T05": "block_T05", "block_energy_T0": "block_T0", "block_energy_T05": "block_T05",
}  # fmt: skip


def _with_sweep(crit: object, sweep_order: str) -> object:
  """A copy of a P4g criteria object with ``sweep_order`` replaced (dataclass replace, else copy + setattr)."""
  if dataclasses.is_dataclass(crit):
    return dataclasses.replace(crit, sweep_order=sweep_order)
  out = copy.copy(crit)
  setattr(out, "sweep_order", sweep_order)  # noqa: B010
  return out


def _knob_criteria() -> dict[str, object]:
  """The four configurations: P4g's block_T0 / block_T05 production criteria with sweep_order knn or energy."""
  base = p4g._criteria()  # noqa: SLF001 -- the P4g production values, reused unchanged
  return {
    "block_knn_T0": _with_sweep(base["block_T0"], "knn"),
    "block_knn_T05": _with_sweep(base["block_T05"], "knn"),
    "block_energy_T0": _with_sweep(base["block_T0"], "energy"),
    "block_energy_T05": _with_sweep(base["block_T05"], "energy"),
  }


def _knobs(crit: object) -> dict[str, object]:
  return {field: p4g._jsonable(getattr(crit, field, None)) for field in KNOB_FIELDS}  # noqa: SLF001


class _VisitRecorder:
  """Wraps the block-descent entry point and the two rankers; logs one entry per block-descent invocation."""

  def __init__(self) -> None:
    self.current: list[dict] | None = None
    self.blocks: list[dict] = []

  def block(self, original, labels: dict[int, str]):  # noqa: ANN001, ANN201
    def run(ctx, crit, valid_mask, pins, neigh, initial_sequence, rng, trajectory=None):  # noqa: ANN001, ANN202
      entry = {
        "label": labels.get(id(crit)), "sweep_order": str(crit.sweep_order), "knobs": _knobs(crit), "orders": [],
      }  # fmt: skip
      self.blocks.append(entry)
      previous, self.current = self.current, entry["orders"]
      try:
        return original(ctx, crit, valid_mask, pins, neigh, initial_sequence, rng, trajectory=trajectory)
      finally:
        self.current = previous

    return run

  def ranker(self, name: str, original):  # noqa: ANN001, ANN201
    def rank(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
      order = original(*args, **kwargs)
      if self.current is not None:
        self.current.append({"ranker": name, "order": [int(x) for x in order]})
      return order

    return rank


def _attach_visit_orders(meta: dict, blocks: list[dict]) -> None:
  """Match the engine's block calls to the recorder's invocations (1:1, in order) and attach the visit order."""
  calls = [c for c in meta["calls"] if c["method"] == "block"]
  if len(calls) != len(blocks):
    msg = f"block calls {len(calls)} != recorded block-descent invocations {len(blocks)}"
    raise RuntimeError(msg)
  for call, entry in zip(calls, blocks):
    if call["label"] != entry["label"] or call["temperature"] != float(entry["knobs"]["temperature"]):
      msg = f"visit-order recorder out of step at call {call['index']} (label {call['label']})"
      raise RuntimeError(msg)
    call["sweep_order"] = entry["sweep_order"]
    call["knobs"] = entry["knobs"]
    call["visit_recordings"] = len(entry["orders"])
    call["visit_order"] = None
    if len(entry["orders"]) == 1 and entry["orders"][0]["ranker"] == entry["sweep_order"]:
      call["visit_order"] = entry["orders"][0]["order"]
    call["neigh_ascending"] = call["neigh"] == sorted(call["neigh"])


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
  recorder = _VisitRecorder()
  true_block, true_knn, true_energy = ph._block_descent, ph._knn_rank, ph._centre_energy_rank  # noqa: SLF001
  ph._knn_rank = recorder.ranker("knn", true_knn)  # noqa: SLF001
  ph._centre_energy_rank = recorder.ranker("energy", true_energy)  # noqa: SLF001
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
        _attach_visit_orders(meta, recorder.blocks)
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
    ph._knn_rank = true_knn  # noqa: SLF001
    ph._centre_energy_rank = true_energy  # noqa: SLF001
  return report


def _child(args: argparse.Namespace) -> dict[str, str]:
  """A second run in a fresh process, same as P4g, using this script."""
  with tempfile.TemporaryDirectory() as tmp:
    result = Path(tmp) / "hashes.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint),
               "--encoder-dir", str(args.encoder_dir), "--features-dir", str(args.features_dir),
               "--out", str(Path(tmp) / "child"), "--child-hashes-to", str(result)]  # fmt: skip
    for name, (pdb, chain) in args.cells.items():
      command += ["--cell", f"{name}={pdb},{chain}"]
    subprocess.run(command, check=True, capture_output=True)
    return json.loads(result.read_text(encoding="utf-8"))


def _vs_p4g(args: argparse.Namespace, directory: Path) -> dict[str, dict]:
  """Recorded, not required: per cell and precision, how many ordinal-matched block designs are identical to P4g's."""
  ref_dir = args.p4g_dir / "protonpotts_v6_ph"
  rows: dict[str, dict] = {}
  for cell in args.cells:
    rows[cell] = {}
    for p in ("f32", "f64"):
      meta = json.loads((directory / f"{cell}_{p}.json").read_text(encoding="utf-8"))
      arrs = p4g._load(directory / f"{cell}_{p}.npz")  # noqa: SLF001
      ref_meta = json.loads((ref_dir / f"{cell}_{p}.json").read_text(encoding="utf-8"))
      ref_arrs = p4g._load(ref_dir / f"{cell}_{p}.npz")  # noqa: SLF001
      per_knob: dict[str, dict] = {}
      for knob, ref in P4G_REFERENCE.items():
        kc = [c for c in meta["calls"] if c["label"] == knob]
        rc = [c for c in ref_meta["calls"] if c["label"] == ref]
        n = min(len(kc), len(rc))
        same = sum(
          bool(np.array_equal(arrs[f"c{k['index']}_S_out"], ref_arrs[f"c{r['index']}_S_out"]))
          for k, r in zip(kc[:n], rc[:n])
        )
        per_knob[knob] = {"p4g_label": ref, "n_ordinal_pairs": n, "n_identical": int(same),
                          "n_calls_knob": len(kc), "n_calls_p4g": len(rc)}  # fmt: skip
      rows[cell][p] = per_knob
  return rows


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--encoder-dir", type=Path, required=True)
  parser.add_argument("--features-dir", type=Path, required=True)
  parser.add_argument("--cell", action="append", default=[], metavar="NAME=PDB,CHAIN")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--p4g-dir", type=Path, default=None,
    help="optional P4g out dir (holding protonpotts_v6_ph/) for the recorded-not-required design comparison",
  )  # fmt: skip
  parser.add_argument("--child-hashes-to", type=Path, default=None)
  args = parser.parse_args()
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
  labels = set(_knob_criteria())
  live = {"knn": False, "energy": False}
  for cell in args.cells:
    metas = {p: json.loads((directory / f"{cell}_{p}.json").read_text(encoding="utf-8")) for p in ("f32", "f64")}
    arrs = {p: p4g._load(directory / f"{cell}_{p}.npz") for p in ("f32", "f64")}  # noqa: SLF001
    for p in ("f32", "f64"):
      seen = {c["label"] for c in metas[p]["calls"]}
      complete &= seen == labels and all(
        c["zscales"] is not None and c["visit_order"] is not None
        for c in metas[p]["calls"] if c["method"] == "block"
      )  # fmt: skip
      for c in metas[p]["calls"]:
        used = arrs[p][f"c{c['index']}_uniforms"].shape[0]
        shim_consistent &= (used == 0) if c["temperature"] <= 0 else (used > 0)
        if c["method"] == "block" and c["visit_order"] is not None and c["visit_order"] != sorted(c["neigh"]):
          live[c["sweep_order"]] = True
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
  flags = {
    "engine_matches_sealed": bool(engine_matches_sealed),
    "complete": bool(complete),
    "shim_consistent": bool(shim_consistent),
    "f64_is_float64": bool(f64_ok),
    "precisions_differ": bool(precisions_differ),
    "reproducible": bool(reproducible),
    "knn_order_live": bool(live["knn"]),
    "energy_order_live": bool(live["energy"]),
    "n_cells": len(report),
  }
  p4g_rows = _vs_p4g(args, directory) if args.p4g_dir is not None else None
  (args.out / "ph_knobs_manifest.json").write_text(
    json.dumps(
      {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
       "torch_version": torch.__version__, "seed": p4g.SEED, "cells": {n: [str(p), c] for n, (p, c) in args.cells.items()},
       "report": report, "f32_vs_f64_same_final_tokens": token_agreement, "flags": flags,
       "p4g_design_comparison_recorded_not_graded": p4g_rows},
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
