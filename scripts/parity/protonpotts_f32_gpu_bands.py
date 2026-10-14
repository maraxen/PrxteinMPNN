"""Measure how far the aminx float32 replays of the ph_block and ph_methods waves sit from the dumps on THIS JAX backend.

The two waves grade float32 against 10x an in-run spread with a 1e-6 floor. On GPU (run e2a185de) ph_block missed one
score and ph_methods missed an energy by 2.46x the band, with float64 exact to 1e-15 on both. This script re-runs the SAME
replay code the waves use (imported, not copied) and records every graded float32 difference as a raw number, so a band is
chosen from the measurement and not from the failure. It grades nothing about the port; the waves do that.

Run once with JAX_PLATFORMS=cpu and once with JAX_PLATFORMS=cuda. AMINX_EXPECT_BACKEND (cpu|gpu) is checked against
jax.default_backend(), so a silent fallback to CPU cannot masquerade as a GPU measurement.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

logger = logging.getLogger("protonpotts_f32_gpu_bands")


def _module(name: str):  # noqa: ANN202
  spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def block_rows(block, encoder_dir: Path, ph_dir: Path) -> list[dict]:  # noqa: ANN001
  """Every (cell, precision, call) comparison of the ph_block wave, with the raw scale-relative differences."""
  import dataclasses  # noqa: PLC0415

  base = block.PHDesignConfig()
  rows: list[dict] = []
  for cell_name in block.CELLS:
    cells = {p: block.Cell(encoder_dir, ph_dir, cell_name, p) for p in ("f64", "f32")}
    for precision, cell in cells.items():
      other = cells["f32" if precision == "f64" else "f64"]
      for call in cell.block_calls():
        config = dataclasses.replace(base, temperature=float(call["temperature"]))
        got = block.run_call(cell, call, config)
        verdict = block.compare_call(cell, other, call, got, f32=precision == "f32")
        design = cell.designs[call["index"]]
        _energy, gap = block.scores(cell, got[0], np.asarray(got[2].seq))
        sel = design["selective_energy"]
        rows.append({
          "cell": cell_name, "precision": precision, "label": call["label"], "index": call["index"], "ok": bool(verdict["ok"]),
          "z_rel": verdict["zscale_rel_diff"], "e_rel": verdict["energy_rel_diff"],
          "s_rel": abs(gap - sel) / max(1.0, abs(sel)),
          "floor_e_rel": abs(design["final_potts_energy"] - other.designs[call["index"]]["final_potts_energy"])
          / max(1.0, abs(design["final_potts_energy"])),
        })  # fmt: skip
        logger.info("block %s/%s/%s ok=%s z=%.3g e=%.3g s=%.3g", cell_name, precision, call["label"], rows[-1]["ok"],
                    rows[-1]["z_rel"], rows[-1]["e_rel"], rows[-1]["s_rel"])  # fmt: skip
  return rows


def methods_rows(methods, state_npz: Path, methods_dir: Path, features_dir: Path) -> list[dict]:  # noqa: ANN001
  """Every (cell, precision, label) comparison of the ph_methods wave, with the raw differences and the in-run floors."""
  import jax.numpy as jnp  # noqa: PLC0415

  model = methods._load_converter()(methods._npz(state_npz))  # noqa: SLF001
  directory = methods_dir / "protonpotts_v6_methods"
  rows: list[dict] = []
  for cell in methods.CELLS:
    features = methods._npz(features_dir / methods.FEATURES / f"{cell}.npz")  # noqa: SLF001
    units = {}
    for precision, dtype in (("f64", jnp.float64), ("f32", jnp.float32)):
      up = methods._npz(directory / f"{cell}_{precision}.npz")  # noqa: SLF001
      meta = json.loads((directory / f"{cell}_{precision}.json").read_text(encoding="utf-8"))
      units[precision] = methods.Unit(model, features, dtype, up, meta)
    u64, u32 = units["f64"], units["f32"]
    for label in methods.LABELS:
      m64 = methods.measure(u64, u64.calls[label], u64.up, methods.Override())
      m32 = methods.measure(u32, u32.calls[label], u32.up, methods.Override())
      m64_on_32 = methods.measure(u64, u32.calls[label], u32.up, methods.Override(), free=False)
      floors = methods.f32_floors(m32, m64_on_32)
      for precision, m, f in (("f64", m64, None), ("f32", m32, floors)):
        v = methods.verdict(m, f)
        place = m["placement"]
        rows.append({
          "cell": cell, "precision": precision, "label": label, "ok": bool(v["ok"]), "worst_over_band": v["worst_over_band"],
          "probs": m["probs_diff"], "placement": place["diff"], "zscales": m.get("zscales_diff", 0.0),
          "energies": m.get("energies_diff", float("inf")), "floors": f,
        })  # fmt: skip
        logger.info("methods %s/%s/%s ok=%s over_band=%.3g", cell, precision, label, v["ok"], v["worst_over_band"])
  return rows


def _worst(rows: list[dict], precision: str, key: str) -> float:
  values = [r[key] for r in rows if r["precision"] == precision and np.isfinite(r[key])]
  return float(max(values, default=0.0))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--oracle-root", type=Path, default=None)
  parser.add_argument("--state-npz", type=Path, default=None)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

  oracles = (args.oracle_root or Path(os.environ.get("AMINX_PROTONPOTTS_ORACLES", "~/projects/aminx-oracles-protonpotts"))).expanduser()
  state_npz = (args.state_npz or Path(os.environ.get("AMINX_PROTONPOTTS_STATE", "~/scratch/v6_state.npz"))).expanduser()

  block = _module("protonpotts_ph_block_parity")
  methods = _module("protonpotts_ph_methods_parity")
  import jax  # noqa: PLC0415

  backend = jax.default_backend()
  expected = os.environ.get("AMINX_EXPECT_BACKEND", "")
  device = str(jax.devices()[0])
  logger.info("backend %s (%s), expected %r", backend, device, expected)

  b_rows = block_rows(block, oracles / "features_v6_p4e", oracles / "features_v6_p4g")
  m_rows = methods_rows(methods, state_npz, oracles / "features_v6_p4j3", oracles)
  f64_ok = all(r["ok"] for r in b_rows + m_rows if r["precision"] == "f64")
  results = {
    "backend": backend, "device": device, "expected_backend": expected, "backend_matches_request": bool(expected) and backend == expected,
    "n_block_rows": len(b_rows), "n_methods_rows": len(m_rows), "f64_arm_in_band": bool(f64_ok),
    "n_f32_rows_outside_current_band": sum(not r["ok"] for r in b_rows + m_rows if r["precision"] == "f32"),
    "block_f32_worst_z_rel": _worst(b_rows, "f32", "z_rel"), "block_f32_worst_e_rel": _worst(b_rows, "f32", "e_rel"),
    "block_f32_worst_s_rel": _worst(b_rows, "f32", "s_rel"),
    "methods_f32_worst_probs": _worst(m_rows, "f32", "probs"), "methods_f32_worst_placement": _worst(m_rows, "f32", "placement"),
    "methods_f32_worst_zscales": _worst(m_rows, "f32", "zscales"), "methods_f32_worst_energies": _worst(m_rows, "f32", "energies"),
    "methods_f64_worst_energies": _worst(m_rows, "f64", "energies"), "block_f64_worst_e_rel": _worst(b_rows, "f64", "e_rel"),
    "block_rows": b_rows, "methods_rows": m_rows,
  }  # fmt: skip
  path = os.environ.get("BTH_RESULTS_PATH")
  if not path:
    msg = "protonpotts_f32_gpu_bands requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
  logger.info("backend=%s f64_ok=%s f32_outside_current=%s", backend, f64_ok, results["n_f32_rows_outside_current_band"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
