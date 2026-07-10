"""Real ``eval_data/lpla.csv`` validation for conformational biasing (backlog node E7).

Computes ``ΔE = E(closed, mutant_seq) - E(open, mutant_seq)`` (see
``aminx.ebm.conformational_biasing``'s module docstring for the exact sign
convention) for real LplA point mutants from
``~/repos/ProteinEBM/eval_data/lpla.csv`` against the two real LplA
conformational states (PDB ``1x2g`` = closed, ``3a7r`` = open; Cavanaugh et
al., fetched from the ``ConformationalBiasing`` GitHub repo -- see this
node's task brief for the authorized source URLs), using the real
checkpoint-ported ``ProteinEBMModel`` weights at
``/tmp/proteinebm_weights/ported_jax_model``. Reports the actual Spearman
rank correlation between ``ΔE`` and the CSV's real experimental
``Promiscuous Activity`` readout -- **whatever number comes out**, not a
targeted one. Design spec §8.2's own pinned expectation for this
application is qualitative ("sign/rank of ΔE ... positive corr. w/
activity") and explicitly weaker than the paper's ProteinMPNN-Bayes decoy/
ΔΔG numbers (§0), so a modest correlation here is not itself evidence of a
bug.

Deliberately **not** part of the fast pytest suite (per the ``ephemeral_
scripts``/E7 task-brief discipline): this script does real network-fetched-
PDB parsing, loads a ~1GB restored JAX pytree, and runs ~100 full forward
passes of an 85M-parameter transformer on CPU (a few minutes wall-clock) --
see ``tests/ebm/test_conformational_biasing.py`` for the fast, synthetic,
torch/checkpoint-free regression coverage of the same wiring logic.

Subset choice
-------------
``lpla.csv`` has 6403 rows total, but only **101** have a non-null
``Promiscuous Activity`` value (the real experimental readout) -- the other
~6300 rows are ESM-IF/Frame2Seq/ThermoMPNN/ProteinMPNN model-score-only rows
with no ground truth to correlate against. This script uses **all 101**
labeled rows by default (not an arbitrary smaller hand-picked slice): 101
forward passes complete in a few minutes on CPU with ``eqx.filter_jit``
compilation reused across rows, and using the full available real-label set
gives the most statistically robust correlation estimate available, rather
than introducing an arbitrary sampling choice on top of an already-small
n=101. ``--max-rows`` is available to cap this for a quick smoke run.

Sequence/structure alignment (the CRITICAL check this node's brief requires)
-----------------------------------------------------------------------------
``1x2g`` (closed) resolves only 331/337 residues (a 6-residue disordered loop,
PDB residue numbers 177-182, is missing from that crystal structure; verified
identical amino-acid identity with ``3a7r`` at all 331 shared positions).
``aminx.ebm.conformational_biasing.align_conformational_states`` handles this
via an explicit intersection mask (see that module's docstring) rather than
silently misaligning the two coordinate arrays index-by-index. Separately,
this script verifies -- BEFORE scoring anything -- that every selected
mutant's ``Sequence`` (337 residues) has the same length as the aligned
structures' canonical residue count, and spot-checks that each row's single
point mutation (from the ``Mutant (WT Context)`` label, e.g. ``"T57L"``) is
consistent with the ``3a7r``-derived reference sequence at that exact
position (all other positions must match). Any row failing either check is
dropped and reported, never silently scored.

Coordinate-frame limitation (read before citing the correlation number)
---------------------------------------------------------------------------
See ``aminx.ebm.conformational_biasing``'s module docstring: this script
centers each conformational state at its own centroid (matching half of the
reference's ``center_random_augmentation``) but applies **no** random SO(3)
rotation, for reproducibility. ``ProteinEBMModel`` is documented
non-equivariant, so a real amount of rotation-dependent noise in the absolute
``ΔE`` values is a known, un-quantified limitation of this MVP run -- not
claimed away.

Requires: real network access (already used once to fetch the two PDBs into
``/tmp/proteinebm_weights/lpla_structures/`` -- this script does not re-fetch
if the files already exist), ``~/repos/ProteinEBM/eval_data/lpla.csv``
(already on disk), and the orbax-ported checkpoint at
``/tmp/proteinebm_weights/ported_jax_model`` (built by
``scripts/ebm/checkpoint_parity_check.py``, backlog node E3.5).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
from scipy.stats import spearmanr

from aminx.ebm.conformational_biasing import (
  ConformationalStates,
  load_conformational_states,
  score_conformational_bias,
  sequence_to_af_aatype,
)
from aminx.ebm.model import ProteinEBMModel
from aminx.utils.aa_convert import AF_ALPHABET

log = logging.getLogger("lpla_biasing_check")

DEFAULT_LPLA_CSV = Path("~/repos/ProteinEBM/eval_data/lpla.csv").expanduser()
DEFAULT_STRUCTURES_DIR = Path("/tmp/proteinebm_weights/lpla_structures")
DEFAULT_CLOSED_PDB = "1x2g.pdb"  # index 0 -> ΔE = E(closed) - E(open)
DEFAULT_OPEN_PDB = "3a7r.pdb"
DEFAULT_ORBAX_MODEL = Path("/tmp/proteinebm_weights/ported_jax_model")
DEFAULT_DIFFUSION_TIME = 0.05  # ProteinEBM-x MVP target t (design spec §9 / ckpt_config.log eval_time)

# Real checkpoint config -- verified against /tmp/proteinebm_weights/ckpt_config.log
# and aminx.ebm.checkpoint's module docstring. These happen to coincide with
# aminx.ebm.model's defaults EXCEPT num_contact_embeddings (checkpoint uses 3,
# default is 2) -- do not assume defaults carry over to a different checkpoint.
CKPT_TOKEN_S = 256
CKPT_TOKEN_Z = 128
CKPT_DIM_FOURIER = 256
CKPT_CONDITIONING_TRANSITION_LAYERS = 2
CKPT_TRANSFORMER_DEPTH = 16
CKPT_TRANSFORMER_HEADS = 8
CKPT_NUM_CONTACT_EMBEDDINGS = 3

_MUTATION_LABEL_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--lpla-csv", type=Path, default=DEFAULT_LPLA_CSV)
  parser.add_argument("--structures-dir", type=Path, default=DEFAULT_STRUCTURES_DIR)
  parser.add_argument("--closed-pdb", type=str, default=DEFAULT_CLOSED_PDB)
  parser.add_argument("--open-pdb", type=str, default=DEFAULT_OPEN_PDB)
  parser.add_argument("--orbax-model", type=Path, default=DEFAULT_ORBAX_MODEL)
  parser.add_argument("--diffusion-time", type=float, default=DEFAULT_DIFFUSION_TIME)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--max-rows",
    type=int,
    default=None,
    help="Cap the number of labeled rows scored (default: all 101 non-null rows).",
  )
  parser.add_argument("--out", type=Path, default=None, help="Optional JSON results path.")
  return parser.parse_args()


def _restore_ported_model(orbax_dir: Path, seed: int) -> ProteinEBMModel:
  """Build a template matching the real checkpoint's config, then orbax-restore into it."""
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
    orbax_dir.resolve(),
    options=ocp.CheckpointManagerOptions(max_to_keep=1),
    item_handlers={"model": ocp.PyTreeCheckpointHandler()},
  )
  step = manager.latest_step()
  if step is None:
    msg = f"No orbax checkpoint found under {orbax_dir}"
    raise FileNotFoundError(msg)
  restored = manager.restore(step, items={"model": template})
  return restored["model"]


def _select_labeled_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
  """Rows with a real (non-null) ``Promiscuous Activity`` and a full ``Sequence``."""
  sub = df[df["Promiscuous Activity"].notna() & df["Sequence"].notna()].copy()
  if max_rows is not None:
    sub = sub.iloc[:max_rows]
  return sub


def _drop_length_mismatches(rows: pd.DataFrame, expected_length: int) -> pd.DataFrame:
  """CRITICAL check: drop (and report) rows whose Sequence length != structure length."""
  seq_lengths = rows["Sequence"].str.len()
  mismatched = rows[seq_lengths != expected_length]
  if len(mismatched) > 0:
    log.warning(
      "Dropping %d/%d rows whose Sequence length != aligned structure length (%d): %s",
      len(mismatched),
      len(rows),
      expected_length,
      mismatched["Mutant (WT Context)"].tolist()[:10],
    )
  return rows[seq_lengths == expected_length]


def _spot_check_mutation_labels(rows: pd.DataFrame, wt_from_struct: str) -> tuple[int, int, list[str]]:
  """Verify each row's single-point-mutation label against the structure-derived WT sequence.

  Returns ``(n_checked, n_consistent, inconsistent_labels)``. A row is
  "consistent" iff its ``Sequence`` differs from ``wt_from_struct`` at
  EXACTLY the labeled position, with the labeled WT/mutant residues matching
  exactly.
  """
  n_checked = 0
  n_consistent = 0
  inconsistent: list[str] = []
  for _, row in rows.iterrows():
    label = str(row["Mutant (WT Context)"])
    match = _MUTATION_LABEL_RE.match(label)
    if match is None:
      continue
    n_checked += 1
    wt_res, pos_str, mut_res = match.group(1), match.group(2), match.group(3)
    pos = int(pos_str)
    seq = row["Sequence"]
    diffs = [i for i in range(len(seq)) if seq[i] != wt_from_struct[i]]
    ok = (
      diffs == [pos - 1]
      and wt_from_struct[pos - 1] == wt_res.upper()
      and seq[pos - 1] == mut_res.upper()
    )
    if ok:
      n_consistent += 1
    else:
      inconsistent.append(label)
  return n_checked, n_consistent, inconsistent


def main() -> int:
  # force=True: jax/orbax configure the root logger's handlers as a side
  # effect of import (visible above as the CUDA-fallback warning's
  # ``ERROR:jax._src.xla_bridge:...``-formatted line) -- without ``force``,
  # basicConfig silently no-ops if a handler already exists, which drops
  # every log.info() call below the WARNING default level.
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
  args = _parse_args()

  if not args.lpla_csv.exists():
    log.error("lpla.csv not found: %s", args.lpla_csv)
    return 1
  closed_path = args.structures_dir / args.closed_pdb
  open_path = args.structures_dir / args.open_pdb
  if not (closed_path.exists() and open_path.exists()):
    log.error("Structure files not found: %s / %s", closed_path, open_path)
    return 1

  log.info("Loading + aligning conformational states: closed=%s open=%s", closed_path, open_path)
  states: ConformationalStates = load_conformational_states([closed_path, open_path])
  n_canonical = int(states.mask.shape[0])
  n_shared = int(np.asarray(states.mask).sum())
  log.info(
    "Canonical residue count=%d; shared (scored) residues=%d (%d excluded, e.g. loop(s) missing in one state)",
    n_canonical,
    n_shared,
    n_canonical - n_shared,
  )

  wt_from_struct = "".join(AF_ALPHABET[i] for i in np.asarray(states.reference_aatype))

  df = pd.read_csv(args.lpla_csv)
  rows = _select_labeled_rows(df, args.max_rows)
  log.info("Selected %d/%d rows with non-null Promiscuous Activity", len(rows), len(df))

  rows = _drop_length_mismatches(rows, n_canonical)
  if len(rows) == 0:
    log.error("No rows remain after the sequence/structure length alignment check.")
    return 1

  n_checked, n_consistent, inconsistent = _spot_check_mutation_labels(rows, wt_from_struct)
  log.info(
    "Mutation-label alignment spot-check: %d/%d rows consistent with the 3a7r-derived WT sequence "
    "at their labeled position",
    n_consistent,
    n_checked,
  )
  if inconsistent:
    log.warning("Dropping %d label-inconsistent rows: %s", len(inconsistent), inconsistent[:10])
    rows = rows[~rows["Mutant (WT Context)"].astype(str).isin(inconsistent)]
  if len(rows) == 0:
    log.error("No rows remain after the mutation-label alignment spot-check.")
    return 1

  log.info("Restoring ported checkpoint model from %s", args.orbax_model)
  model = _restore_ported_model(args.orbax_model, args.seed)

  t = jnp.asarray(args.diffusion_time)

  @eqx.filter_jit
  def _score(model: ProteinEBMModel, aatype: jax.Array) -> jax.Array:
    return score_conformational_bias(model, states, aatype, t)

  delta_es: list[float] = []
  activities: list[float] = []
  labels: list[str] = []
  t0 = time.time()
  for i, (_, row) in enumerate(rows.iterrows()):
    aatype = sequence_to_af_aatype(row["Sequence"])
    delta_e = _score(model, aatype)
    delta_es.append(float(delta_e))
    activities.append(float(row["Promiscuous Activity"]))
    labels.append(str(row["Mutant (WT Context)"]))
    if (i + 1) % 10 == 0 or (i + 1) == len(rows):
      log.info("Scored %d/%d mutants (elapsed %.1fs)", i + 1, len(rows), time.time() - t0)

  delta_es_arr = np.asarray(delta_es)
  activities_arr = np.asarray(activities)
  result = spearmanr(delta_es_arr, activities_arr)
  log.info(
    "=== RESULT: n=%d  Spearman(ΔE=E(closed)-E(open), Promiscuous Activity) = %.4f  (p=%.4g) ===",
    len(delta_es_arr),
    result.statistic,
    result.pvalue,
  )

  if args.out is not None:
    payload = {
      "n_mutants": len(delta_es_arr),
      "diffusion_time": args.diffusion_time,
      "spearman_r": float(result.statistic),
      "spearman_p": float(result.pvalue),
      "n_canonical_residues": n_canonical,
      "n_shared_residues": n_shared,
      "mutation_label_alignment": {"n_checked": n_checked, "n_consistent": n_consistent},
      "mutants": [
        {"label": lbl, "delta_e": de, "promiscuous_activity": act}
        for lbl, de, act in zip(labels, delta_es, activities, strict=True)
      ],
    }
    args.out.write_text(json.dumps(payload, indent=2))
    log.info("Wrote results to %s", args.out)

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
