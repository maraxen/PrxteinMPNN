"""Real ProteinGym/Tsuboyama ddG-stability accuracy check for ProteinEBM (E6, paper target 0.686).

The design spec's headline ΔΔG-stability target (Spearman >= 0.686, App. 8.2) was never measured --
`aminx.ebm.ddg_stability`'s own module docstring says so explicitly, citing the ProteinGym/Tsuboyama
megascale data as "not downloaded in this environment." This script is the first real measurement:
it scores the real Tsuboyama-2023 cDNA-display proteolysis stability assays (64 proteins,
`~/repos/ProteinEBM/eval_data/proteingym/DMS_ProteinGym_substitutions/*Tsuboyama_2023*.csv`, matched
AF2 structures) with the real ported checkpoint, using the real unfolded-ensemble generator
(:func:`aminx.ebm.ddg_stability.generate_real_unfolded_ensemble`, a new NeRF-based port added
alongside -- not replacing -- the existing synthetic random-walk substitute), and correlates against
the real experimental ``DMS_score`` per protein -- mirroring
``~/repos/ProteinEBM/notebooks/ddg_prediction.ipynb``'s actual protocol.

Does **not** reimplement any scoring math: reuses ``aminx.ebm.dispatch.score_mutant_ensemble`` (E4)
and ``aminx.ebm.ddg_stability.unfolded_state_correction`` (E6) directly. Builds mutant sequences from
each CSV row's ``mutated_sequence`` column directly (the real full mutant sequence is already given
by ProteinGym -- no position-based mutation parsing/validation needed, unlike
``scripts/ebm/lpla_biasing_check.py``'s point-mutation-label spot-check, which was needed there only
because that CSV gives a label, not the full sequence).

Requires: the real ProteinGym/Tsuboyama data (``--proteingym-dir``, see the audit report's
provenance section for source + the TLS-chain workaround needed to fetch it), and the orbax-ported
checkpoint (``--orbax-model``).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
from scipy.stats import spearmanr

from aminx.ebm.ddg_stability import (
  generate_real_unfolded_ensemble,
  load_ca_backbone_from_pdb,
  unfolded_state_correction,
)
from aminx.ebm.dispatch import score_mutant_ensemble
from aminx.ebm.model import ProteinEBMModel
from aminx.utils.aa_convert import protein_sequence_to_string, string_to_protein_sequence

log = logging.getLogger("real_ddg_stability_benchmark")

DEFAULT_PROTEINGYM_DIR = Path("~/repos/ProteinEBM/eval_data/proteingym").expanduser()
DEFAULT_ORBAX_MODEL = Path("/tmp/proteinebm_weights/ported_jax_model")
PINNED_T = 0.05
PAPER_TARGET = 0.686
N_UNFOLDED_ENSEMBLE = 32  # matches ddg_prediction.ipynb's num_unfolded

CKPT_TOKEN_S = 256
CKPT_TOKEN_Z = 128
CKPT_DIM_FOURIER = 256
CKPT_CONDITIONING_TRANSITION_LAYERS = 2
CKPT_TRANSFORMER_DEPTH = 16
CKPT_TRANSFORMER_HEADS = 8
CKPT_NUM_CONTACT_EMBEDDINGS = 3


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--proteingym-dir", type=Path, default=DEFAULT_PROTEINGYM_DIR)
  parser.add_argument("--orbax-model", type=Path, default=DEFAULT_ORBAX_MODEL)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--max-assays", type=int, default=None, help="Cap the number of Tsuboyama assay files scored (default: all 64).")
  parser.add_argument("--max-mutants-per-assay", type=int, default=None, help="Cap mutant rows per assay (default: all).")
  parser.add_argument("--out", type=Path, required=True)
  return parser.parse_args()


def _restore_model(orbax_dir: Path, seed: int) -> ProteinEBMModel:
  template = ProteinEBMModel(
    token_s=CKPT_TOKEN_S, token_z=CKPT_TOKEN_Z, dim_fourier=CKPT_DIM_FOURIER,
    conditioning_transition_layers=CKPT_CONDITIONING_TRANSITION_LAYERS,
    transformer_depth=CKPT_TRANSFORMER_DEPTH, transformer_heads=CKPT_TRANSFORMER_HEADS,
    num_contact_embeddings=CKPT_NUM_CONTACT_EMBEDDINGS, key=jax.random.PRNGKey(seed),
  )
  manager = ocp.CheckpointManager(
    orbax_dir.resolve(), options=ocp.CheckpointManagerOptions(max_to_keep=1),
    item_handlers={"model": ocp.PyTreeCheckpointHandler()},
  )
  step = manager.latest_step()
  if step is None:
    msg = f"No orbax checkpoint found under {orbax_dir}"
    raise FileNotFoundError(msg)
  return manager.restore(step, items={"model": template})["model"]


def _find_tsuboyama_assays(proteingym_dir: Path) -> list[Path]:
  sub_dir = proteingym_dir / "DMS_ProteinGym_substitutions"
  return sorted(sub_dir.glob("*Tsuboyama_2023*.csv"))


def _score_one_assay(
  model: ProteinEBMModel,
  csv_path: Path,
  af2_dir: Path,
  max_mutants: int | None,
  seed: int,
) -> dict | None:
  protein_prefix = "_".join(csv_path.stem.split("_")[:2])
  pdb_path = af2_dir / f"{protein_prefix}.pdb"
  if not pdb_path.exists():
    log.warning("no AF2 structure for %s (expected %s), skipping", csv_path.name, pdb_path)
    return None

  wildtype = load_ca_backbone_from_pdb(pdb_path)
  n_res = int(wildtype.aatype.shape[0])

  df = pd.read_csv(csv_path)
  df = df[df["mutated_sequence"].str.len() == n_res]
  if max_mutants is not None:
    df = df.iloc[:max_mutants]
  if len(df) < 3:  # noqa: PLR2004 -- Spearman needs >= 3 points
    log.warning("%s: only %d length-matched mutants, skipping", csv_path.name, len(df))
    return None

  mutant_aatype = jnp.stack([string_to_protein_sequence(seq) for seq in df["mutated_sequence"]])
  wt_sequence = protein_sequence_to_string(wildtype.aatype)

  raw_ddg = score_mutant_ensemble(
    model, wildtype.coords, mutant_aatype, jnp.asarray(PINNED_T), wildtype.mask,
    wildtype_aatype=wildtype.aatype,
  )
  unfolded_coords = generate_real_unfolded_ensemble(wt_sequence, N_UNFOLDED_ENSEMBLE, seed)
  correction = unfolded_state_correction(
    model, wildtype.aatype, jnp.asarray(PINNED_T), wildtype.mask, unfolded_coords,
  )
  ddg = np.asarray(raw_ddg - correction)
  dms_score = df["DMS_score"].to_numpy()

  result = spearmanr(ddg, dms_score)
  return {
    "assay": csv_path.stem,
    "protein": protein_prefix,
    "n_residues": n_res,
    "n_mutants_scored": len(df),
    "spearman_r": float(result.statistic),
    "spearman_p": float(result.pvalue),
  }


def main() -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
  args = _parse_args()

  assays = _find_tsuboyama_assays(args.proteingym_dir)
  if args.max_assays is not None:
    assays = assays[: args.max_assays]
  if not assays:
    log.error("No Tsuboyama assay CSVs found under %s", args.proteingym_dir)
    return 1
  log.info("Scoring %d Tsuboyama stability assays at t=%.2f", len(assays), PINNED_T)

  model = _restore_model(args.orbax_model, args.seed)

  # Incremental checkpoint: append each assay's result to a sibling .partial.jsonl as soon as it is
  # scored, so a walltime timeout (or crash) mid-run does not discard every completed assay. The
  # final aggregate JSON is still written atomically at the end; the partial file is a recovery log.
  args.out.parent.mkdir(parents=True, exist_ok=True)
  partial_path = args.out.parent / f"{args.out.stem}.partial.jsonl"

  per_assay: list[dict] = []
  t0 = time.time()
  for i, csv_path in enumerate(assays):
    result = _score_one_assay(model, csv_path, args.proteingym_dir / "ProteinGym_AF2_structures", args.max_mutants_per_assay, args.seed)
    if result is not None:
      per_assay.append(result)
      with partial_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")
      log.info(
        "[%d/%d] %s: n=%d spearman=%.3f (p=%.3g)  (%.1fs elapsed)",
        i + 1, len(assays), result["protein"], result["n_mutants_scored"],
        result["spearman_r"], result["spearman_p"], time.time() - t0,
      )

  rs = [r["spearman_r"] for r in per_assay]
  mean_r = float(np.mean(rs)) if rs else float("nan")
  # Sign-convention note (same discipline as real_decoy_ranking_benchmark.py): whether ddG and
  # DMS_score should correlate positively or negatively depends on DMS_score's own orientation
  # (higher = more stable, in most ProteinGym fitness assays) crossed with ddG's sign convention
  # (E(mutant) - E(wildtype) - unfolded_correction; a destabilizing mutation raising E). Report
  # both signed and absolute means; compare |mean| against the paper's target, not the raw signed
  # value, exactly as for decoy-ranking.
  abs_mean_r = float(np.mean(np.abs(rs))) if rs else float("nan")
  log.info(
    "=== RESULT: n_assays=%d  mean Spearman (ddG, DMS_score), signed = %.4f  "
    "(|.|=%.4f vs paper target %.3f) ===",
    len(per_assay), mean_r, abs_mean_r, PAPER_TARGET,
  )

  payload = {
    "n_assays_scored": len(per_assay),
    "n_assays_total": len(assays),
    "pinned_t": PINNED_T,
    "n_unfolded_ensemble": N_UNFOLDED_ENSEMBLE,
    "paper_target": PAPER_TARGET,
    "mean_spearman_signed": mean_r,
    "mean_spearman_abs": abs_mean_r,
    "sign_convention_note": (
      "ddG-vs-DMS_score sign depends on DMS_score's own stability-direction convention crossed "
      "with ddG's E(mutant)-E(wildtype)-correction convention. Compare mean_spearman_abs against "
      "paper_target, not the signed value."
    ),
    "per_assay": per_assay,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2))
  log.info("Wrote results to %s", args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
