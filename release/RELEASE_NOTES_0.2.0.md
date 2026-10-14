# aminx 0.2.0

**Beta, still largely experimental.** This is the first release off alpha. The core `sample()` / `score()` API is functional, but parity with the upstream references is measured per path and is not complete, and the API may change between releases.

## What's new

- **Three new model families:** PottsMPNN, LASErMPNN and ProtonPottsMPNN. Select one with `--model-family`, or let the `checkpoint_id` prefix choose it. Which purposes each family runs itself, and which fall back to stock MPNN, is in `docs/MODEL_FAMILIES.md`.
- **Redsox parity gate and per-family ledger.** The gate passed by record at `8004a04f` (run `cc98ddb1`, 1707 ids; `verify_wave` 135/135).
- **Parity dashboard** (`scripts/dashboard/`). Regenerate its JSON against the current ledger before quoting numbers from it.
- **Browser export and gates** for PottsMPNN and ProtonPottsMPNN.
- Debt sweep, spec JSON `schema_version` 1, seed 0 now means seed 0, bucketed residue lengths by default, xtrax pinned to `0.4.0a11`. Full list in `CHANGELOG.md`.

## Read before upgrading

- **Seeded sample outputs change** (length bucketing is on by default; pass `--no-length-bucketing` to reproduce a pre-bucketing run). Agent `spec_sha256` values change. Checkpoints written by the old trainer cannot be resumed.
- **No Hugging Face weights for the new families.** `maraxen/aminx` holds the ProteinMPNN, SolubleMPNN, LigandMPNN and membrane checkpoints only. For PottsMPNN, ProtonPottsMPNN and LASErMPNN, pass a checkpoint with `--model-local-path`: the upstream `.pt` (needs `torch`; PottsMPNN's on-the-fly conversion needs a source checkout) or a converted `.eqx`. The `weights/pottsmpnn/*.eqx.zst` files in the source tree are not loadable PottsMPNN artifacts; do not pass them.
- **LASErMPNN needs the `laser` extra:** `pip install 'aminx[laser]'`.
- **Known parity gaps remain** (ligand, membrane and conditional-scoring defects from the 2026-09-26 audit; the global grade there is FAIL). See the README table.

## Install

```bash
pip install aminx==0.2.0
pip install 'aminx[laser]==0.2.0'   # for LASErMPNN
```
