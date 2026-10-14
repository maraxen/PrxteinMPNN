# aminx 0.2.1

Beta, still largely experimental. This release makes the 0.2.0 model families usable from a plain `pip install`.

## What's new

- **Checkpoints for PottsMPNN, LASErMPNN and ProtonPottsMPNN are on the Hugging Face repo** and resolve by id:

  ```bash
  pip install aminx==0.2.1
  aminx run --checkpoint-id pottsmpnn_vanilla_20 score --output-kind energy --inputs complex.pdb
  ```

  Nine are published: `pottsmpnn_{vanilla_20,vanilla_30,sol_20,sol_30,ft}`, `lasermpnn_{nothing_heldout,noise_ligandmpnn_split,soluble_65000}` and `protonpottsmpnn_v6_30`. Every file is checked against a sha256 pinned in the wheel; a changed or truncated download is refused. `--model-local-path` and `AMINX_WEIGHTS_DIR` still override.
- **Licences.** The weights are converted copies of MIT-licensed upstream weights; notices are in `NOTICE-weights.md` and on the model card.

## Fixes

- LASErMPNN loaded a serialised checkpoint with `build_hydrogens` off although every upstream checkpoint turns it on. It is now read from the checkpoint's manifest.
- The 0.2.0 notes suggested the `weights/pottsmpnn/*.eqx.zst` files in the source tree. They do not load as PottsMPNN; ignore that advice.

## Read before upgrading

- No numerical change for any existing path. The Hub pin moves to a revision that only adds files; the existing MPNN checkpoints are byte-identical.
- LASErMPNN still needs the `laser` extra: `pip install 'aminx[laser]==0.2.1'`.
- Not published: the MSA PottsMPNN variants (not supported yet) and any checkpoint without parity evidence.
- Parity with the upstream references is measured per path and is not complete; see the README table.
