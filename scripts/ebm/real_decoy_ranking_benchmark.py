"""Real Rosetta decoy-ranking accuracy check for ProteinEBM (backlog node E5, paper target 0.838).

The design spec's own headline decoy-ranking target (Spearman(E, TM-score) = 0.838, App. 8.2) was
never measured anywhere in this repo -- `aminx.ebm.decoy_ranking`'s own module docstring says so
explicitly, and `tests/ebm/test_decoy_ranking.py` only exercises a synthetic coordinate-noise proxy.
This script is the first real measurement: it scores the actual Rosetta decoy set (133 native
structures, ~957 decoys each, `~/repos/ProteinEBM/eval_data/decoys/`) with the real ported checkpoint
and correlates against the real TM-score labels (`~/repos/ProteinEBM/eval_data/decoy_data/tmscore.txt`).

It does **not** reimplement any scoring math -- it is a real-data harness on top of the already-built
:func:`aminx.ebm.decoy_ranking.rank_decoys_over_noise_time` (E5's real application logic, never
exercised against real data until now).

**Two numbers are reported per native, not one** (avoid cherry-picking across the t-grid): the
Spearman at the design spec's own pinned MVP time (t=0.05) and the best-|correlation| time in the
sweep. Both are averaged across natives; the paper's target (0.838) is compared against the t=0.05
number, since that is the pre-registered comparison, not a post-hoc best-of-sweep number.

Requires: the real Rosetta decoy set + TM-score labels downloaded to ``--decoys-dir``/``--tmscore``
(not committed to the repo -- see this script's own README note / the audit report's provenance
section for the exact source URLs), and the orbax-ported checkpoint (``--orbax-model``, built by
``scripts/ebm/checkpoint_parity_check.py``).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from aminx.ebm.decoy_ranking import DEFAULT_NOISE_TIME_GRID, rank_decoys_over_noise_time
from aminx.ebm.model import ProteinEBMModel

log = logging.getLogger("real_decoy_ranking_benchmark")

DEFAULT_DECOYS_DIR = Path("~/repos/ProteinEBM/eval_data/decoys").expanduser()
DEFAULT_TMSCORE = Path("~/repos/ProteinEBM/eval_data/decoy_data/tmscore.txt").expanduser()
DEFAULT_ORBAX_MODEL = Path("/tmp/proteinebm_weights/ported_jax_model")
PINNED_T = 0.05
PAPER_TARGET = 0.838

# Matches scripts/ebm/lpla_biasing_check.py's real-checkpoint config (verified against the real
# checkpoint's hyper_parameters.config.model, not defaults -- see that script's module docstring).
CKPT_TOKEN_S = 256
CKPT_TOKEN_Z = 128
CKPT_DIM_FOURIER = 256
CKPT_CONDITIONING_TRANSITION_LAYERS = 2
CKPT_TRANSFORMER_DEPTH = 16
CKPT_TRANSFORMER_HEADS = 8
CKPT_NUM_CONTACT_EMBEDDINGS = 3

COORDINATE_SCALING = 0.1  # aminx.ebm.diffusion.DEFAULT_COORDINATE_SCALING


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--decoys-dir", type=Path, default=DEFAULT_DECOYS_DIR)
  parser.add_argument("--tmscore", type=Path, default=DEFAULT_TMSCORE)
  parser.add_argument("--orbax-model", type=Path, default=DEFAULT_ORBAX_MODEL)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--max-natives", type=int, default=None, help="Cap the number of native structures scored (default: all 133).")
  parser.add_argument("--max-decoys-per-native", type=int, default=None, help="Cap decoys per native (default: all, typically ~957).")
  parser.add_argument("--out", type=Path, required=True)
  return parser.parse_args()


def _load_tmscore_index(path: Path) -> dict[str, dict[str, float]]:
  """Parse ``tmscore.txt`` (``native decoy_filename tm_ref value``) into ``{native: {file: tm}}``."""
  index: dict[str, dict[str, float]] = {}
  with path.open(encoding="utf-8") as fh:
    for line in fh:
      parts = line.split()
      if len(parts) != 4:  # noqa: PLR2004
        continue
      native, decoy_file, _tag, value = parts
      index.setdefault(native, {})[decoy_file] = float(value)
  return index


def _restore_model(orbax_dir: Path, seed: int) -> ProteinEBMModel:
  template = ProteinEBMModel(
    token_s=CKPT_TOKEN_S,
    token_z=CKPT_TOKEN_Z,
    dim_fourier=CKPT_DIM_FOURIER,
    conditioning_transition_layers=CKPT_CONDITIONING_TRANSITION_LAYERS,
    transformer_depth=CKPT_TRANSFORMER_DEPTH,
    transformer_heads=CKPT_TRANSFORMER_HEADS,
    num_contact_embeddings=CKPT_NUM_CONTACT_EMBEDDINGS,
    key=jax.random.PRNGKey(seed),
  )
  manager = ocp.CheckpointManager(
    orbax_dir.resolve(), options=ocp.CheckpointManagerOptions(max_to_keep=1),
    item_handlers={"model": ocp.PyTreeCheckpointHandler()},
  )
  step = manager.latest_step()
  if step is None:
    msg = f"No orbax checkpoint found under {orbax_dir}"
    raise FileNotFoundError(msg)
  # Current orbax requires explicit per-leaf sharding on restore (the legacy ``items=`` path
  # leaves it None and raises "sharding ... Got None"). Derive it from the concrete template
  # arrays (a SingleDeviceSharding on the current device) via construct_restore_args.
  restore_args = ocp.checkpoint_utils.construct_restore_args(template)
  restored = manager.restore(
    step,
    args=ocp.args.Composite(model=ocp.args.PyTreeRestore(item=template, restore_args=restore_args)),
  )
  return restored["model"]


def _score_one_native(
  model: ProteinEBMModel,
  native_id: str,
  decoys_dir: Path,
  tm_by_file: dict[str, float],
  max_decoys: int | None,
) -> dict | None:
  from aminx.io.parsing import parse_structure  # noqa: PLC0415

  native_path = decoys_dir / "natives" / f"{native_id}.pdb"
  if not native_path.exists():
    log.warning("missing native structure for %s, skipping", native_id)
    return None
  native = parse_structure(str(native_path))
  n_res = int(np.asarray(native.aatype).shape[0])
  aatype = jnp.asarray(np.asarray(native.aatype), dtype=jnp.int32)
  mask = jnp.asarray(np.asarray(native.mask), dtype=bool) if native.mask is not None else jnp.ones((n_res,), dtype=bool)

  decoy_dir = decoys_dir / native_id
  decoy_files = sorted(f for f in tm_by_file if (decoy_dir / f).exists())
  if max_decoys is not None:
    decoy_files = decoy_files[:max_decoys]
  if len(decoy_files) < 3:  # noqa: PLR2004 -- Spearman needs >= 3 points to be meaningful
    log.warning("native %s: only %d usable decoys, skipping", native_id, len(decoy_files))
    return None

  coords_list: list[np.ndarray] = []
  tmscores: list[float] = []
  n_skipped_mismatch = 0
  for decoy_file in decoy_files:
    decoy = parse_structure(str(decoy_dir / decoy_file))
    if int(np.asarray(decoy.aatype).shape[0]) != n_res:
      n_skipped_mismatch += 1
      continue
    ca = np.asarray(decoy.coordinates)[:, 1, :] * COORDINATE_SCALING
    coords_list.append(ca)
    tmscores.append(tm_by_file[decoy_file])

  if len(coords_list) < 3:  # noqa: PLR2004
    log.warning("native %s: only %d length-matched decoys after alignment check, skipping", native_id, len(coords_list))
    return None
  if n_skipped_mismatch:
    log.info("native %s: dropped %d/%d decoys for residue-count mismatch vs. native", native_id, n_skipped_mismatch, len(decoy_files))

  coords = jnp.asarray(np.stack(coords_list, axis=0))
  # Center each decoy (subtract its CA centroid): the model is not translation-equivariant, so the
  # reference (score_decoys.py) applies center_random_augmentation before scoring. Without this the
  # absolute position of each decoy leaks into its energy. (Rotation augmentation is omitted for a
  # deterministic score.)
  coords = coords - jnp.mean(coords, axis=1, keepdims=True)
  # external_contacts = ones: the checkpoint has num_contact_embeddings == 3, so the reference's
  # compute_energy defaults external_contacts to ONES (ebm.py:167-169), not zeros.
  # NOTE: as of the Track-B trunk change, model.energy(contacts=None) now ALSO resolves to ones for
  # the num=3 checkpoint, so this explicit array is redundant; kept as a defensive, self-documenting
  # pin so the harness stays correct regardless of the library default.
  contacts = jnp.ones((n_res,), dtype=jnp.int32)
  quality = np.asarray(tmscores)

  result = rank_decoys_over_noise_time(model, coords, aatype, mask, quality, contacts=contacts)
  spearman_by_t = dict(zip((f"{t:.3f}" for t in result.t_values), result.spearman_by_t, strict=True))
  spearman_at_pinned_t = spearman_by_t.get(f"{PINNED_T:.3f}")

  return {
    "native_id": native_id,
    "n_residues": n_res,
    "n_decoys_scored": len(coords_list),
    "n_decoys_skipped_length_mismatch": n_skipped_mismatch,
    "spearman_by_t": spearman_by_t,
    "spearman_at_pinned_t": spearman_at_pinned_t,
    "best_t": result.best_t,
    "best_spearman": result.best_spearman,
  }


def main() -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
  args = _parse_args()

  if not args.decoys_dir.exists():
    log.error("Decoys dir not found: %s", args.decoys_dir)
    return 1
  if not args.tmscore.exists():
    log.error("tmscore.txt not found: %s", args.tmscore)
    return 1

  tm_index = _load_tmscore_index(args.tmscore)
  natives = sorted(tm_index)
  if args.max_natives is not None:
    natives = natives[: args.max_natives]
  log.info("Scoring %d native structures (grid t=%s)", len(natives), DEFAULT_NOISE_TIME_GRID)

  log.info("Restoring ported checkpoint from %s", args.orbax_model)
  model = _restore_model(args.orbax_model, args.seed)

  # Incremental checkpoint: append each native's result to a sibling .partial.jsonl as soon as it is
  # scored, so a walltime timeout (or crash) mid-run does not discard every completed native. The
  # final aggregate JSON is still written atomically at the end; the partial file is a recovery log.
  args.out.parent.mkdir(parents=True, exist_ok=True)
  partial_path = args.out.parent / f"{args.out.stem}.partial.jsonl"

  per_native: list[dict] = []
  t0 = time.time()
  for i, native_id in enumerate(natives):
    result = _score_one_native(model, native_id, args.decoys_dir, tm_index[native_id], args.max_decoys_per_native)
    if result is not None:
      per_native.append(result)
      with partial_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")
      log.info(
        "[%d/%d] %s: n_decoys=%d spearman(t=%.2f)=%.3f best=%.3f@t=%.2f (%.1fs elapsed)",
        i + 1, len(natives), native_id, result["n_decoys_scored"], PINNED_T,
        result["spearman_at_pinned_t"] or float("nan"), result["best_spearman"], result["best_t"],
        time.time() - t0,
      )

  pinned_vals = [r["spearman_at_pinned_t"] for r in per_native if r["spearman_at_pinned_t"] is not None]
  best_vals = [r["best_spearman"] for r in per_native]
  mean_pinned = float(np.mean(pinned_vals)) if pinned_vals else float("nan")
  mean_best = float(np.mean(best_vals)) if best_vals else float("nan")
  # Sign-convention note (see aminx.ebm.decoy_ranking.spearman_energy_quality_correlation's own
  # docstring): lower energy = more native-like, but higher TM-score = more native-like, so the
  # PHYSICALLY EXPECTED raw Spearman(E, TMscore) sign is NEGATIVE. The paper's own 0.838 figure is
  # reported positive -- consistent with an implicit absolute-value convention. Report both,
  # explicitly, so this is never silently conflated: compare |mean_pinned| against the paper target.
  abs_mean_pinned = float(np.mean(np.abs(pinned_vals))) if pinned_vals else float("nan")
  abs_mean_best = float(np.mean(np.abs(best_vals))) if best_vals else float("nan")

  log.info(
    "=== RESULT: n_natives=%d  mean Spearman @ t=%.2f (pre-registered, signed) = %.4f  "
    "(|.|=%.4f vs paper target %.3f)  |  mean best-of-sweep (signed) = %.4f (|.|=%.4f) ===",
    len(per_native), PINNED_T, mean_pinned, abs_mean_pinned, PAPER_TARGET, mean_best, abs_mean_best,
  )

  payload = {
    "n_natives_scored": len(per_native),
    "n_natives_total": len(natives),
    "pinned_t": PINNED_T,
    "paper_target": PAPER_TARGET,
    "mean_spearman_at_pinned_t_signed": mean_pinned,
    "mean_spearman_at_pinned_t_abs": abs_mean_pinned,
    "mean_best_spearman_signed": mean_best,
    "mean_best_spearman_abs": abs_mean_best,
    "sign_convention_note": (
      "Raw Spearman(E, TMscore) is expected NEGATIVE (lower energy = more native-like; higher "
      "TMscore = more native-like). Compare mean_spearman_at_pinned_t_abs against paper_target, "
      "not the signed value."
    ),
    "per_native": per_native,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2))
  log.info("Wrote results to %s", args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
