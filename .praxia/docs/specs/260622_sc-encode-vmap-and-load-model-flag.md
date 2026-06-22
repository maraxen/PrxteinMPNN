# aminx task spec — fix side-chain-context inference + expose SC flag

**Repo:** `/home/marielle/projects/aminx` · **Version:** 0.1.0a6 · **Date:** 2026-06-22
**Requested by:** tev_design Stage 3 conditioning audit (`tev_design/.praxia/docs/audits/260622_stage3-conditioning-audit-and-techdebt.md`)
**Scope:** two surgical fixes + a differential regression test. Backward compatible. Do **not** touch
the in-flight run_spec migration (RS-6b). Minimal diff.

---

## Background (what's broken)

Side-chain context (`use_side_chain_context`) is **non-functional** through the inference path, for two
independent reasons. Verified empirically against `ligandmpnn_v_32_020_25`:
- The **weights are fine** — building `PrxteinLigandMPNN(ligand_mpnn_use_side_chain_context=True)` loads
  the existing checkpoint (static field, no extra params) and the encoder **responds** to side chains
  (node-feature max|Δ| = 1.16 between reveal vs no-reveal when called directly).
- But the **inference encode path crashes**, and **`load_model` can't enable the flag**.

Reference ground truth (`~/repos/LigandMPNN`): SC atoms = atom37 indices `5:37` (32 atoms; CB=3, O=4 in
this atom order); side chains are revealed where the per-residue **designability mask == 0** (i.e. at
*fixed* residues). In aminx that mask is `chain_mask = 1 - fixed_mask`. (`chain_mask` is a misnomer — it
is a designability mask, unrelated to PDB chain / `chain_index`. Consider renaming to `design_mask` with
a deprecation alias as part of this work, or file a follow-up.)

---

## Bug 1 — `chain_mask` is not threaded through the encode vmap/scan  *(blocker)*

`src/aminx/inference/encode.py`. In **both** encoders, `encode_one` closes over
`bundle.conditioning.fixed_mask` (shape `(S, N)`) and computes `chain_mask = 1.0 - …` **inside** the
per-state function, but `chain_mask` is **not** a mapped input:

- `_ParallelEncode`: `chain_mask` built at **line 90**, used line 93; `in_axes` at **line 99** has no
  entry for it; vmap call lines 100-113.
- `_ScanEncode`: same pattern at **lines 175/178**; `scan_xs` tuple lines 191-204 / 207-211.

Inside the per-state `vmap`/`scan`, `chain_mask` keeps its full `(S, N)` shape while `atom_37_mask` is
per-state `(N, 37)`. Downstream `model/ligand_features.py:365`
(`atom_37_mask * (1.0 - chain_mask_in[:, None])`) then fails to broadcast:
`Incompatible shapes (N,37) vs (1,1,N)`. The SC path cannot run for any input.

**Fix:** make the per-residue designability mask a **per-state mapped input**, not a closure capture.
- Compute `chain_mask_stack = 1.0 - bundle.conditioning.fixed_mask` (`(S, N)`) **before** the
  vmap/scan, alongside `xyz37` / `xyz37_m`.
- Add a `chain_mask` parameter to `encode_one`; inside, use the passed `(N,)` value (do **not**
  recompute from `bundle`).
- `_ParallelEncode`: append `chain_mask` to the vmap call args and add its `in_axes` entry:
  `None if xyz37 is None else 0` (mirrors `xyz37`/`xyz37_m`). When SC is off, pass `None` / `in_axes None`.
- `_ScanEncode`: add `chain_mask_stack` to the SC-active `scan_xs` tuple and unpack it in the
  SC-active `scan_body`; the non-SC `scan_body` arity stays unchanged.
- Gate identically to `xyz37` (only present when side chains active) so the non-SC paths are untouched.

## Bug 2 — `load_model` cannot enable side-chain context  *(blocker)*

`src/aminx/io/weights.py` — `load_model` (def line 151) builds `PrxteinLigandMPNN(...)` (line ~213)
**without** `ligand_mpnn_use_side_chain_context`, so it defaults `False` and there is no way to turn it on.

**Fix:** add a keyword `use_side_chain_context: bool = False` to `load_model`; thread it into the
`PrxteinLigandMPNN(..., ligand_mpnn_use_side_chain_context=use_side_chain_context)` construction (ligand
model branch only). No-op for non-ligand model types. The flag is `eqx.field(static=True)` and adds no
parameters, so the **same checkpoint deserializes** into the SC-enabled skeleton — verified.

---

## Acceptance criteria (all must pass)

Add a test module (e.g. `tests/inference/test_side_chain_context.py`). Use `ligandmpnn_v_32_020_25` and
a small fixture structure (reuse an existing test PDB/fixture; ≥2 states for the multistate check).

1. **Loads with flag:** `load_model("ligandmpnn_v_32_020_25", use_side_chain_context=True)` succeeds and
   the model reports `ligand_mpnn_use_side_chain_context is True`. Default (no kwarg) stays `False`.
2. **Multistate SC kernel runs (regression for Bug 1):** `score_unconditional.kernel` (and the scan
   path) run with `atom_37`/`atom_37_mask` provided and `S ≥ 2` **without** the broadcast error. Run via
   both `_ParallelEncode` and `_ScanEncode`.
3. **Differential — SC changes output (the core gate):** with the SC-enabled model, logits with a
   reveal mask (`fixed_mask = 1` at a subset of residues → those side chains visible) differ from logits
   with `fixed_mask = 0` everywhere: `max|Δ logits| > 1e-3`. And SC-on-with-no-reveal ≈ SC-off
   (`max|Δ| < 1e-5`). This is the assertion that would have caught the original silent no-op.
4. **Ligand differential (lock existing behavior):** ligand-on vs ligand-off logits differ
   (`max|Δ| > 1e-3`) for the ligand model. (Already works; pin it so it can't regress.)
5. **No regressions:** existing `uv run pytest` suite stays green.

## Deliverables / build

- The two fixes + the test module.
- Bump version `0.1.0a6 → 0.1.0a7` (`pyproject.toml`).
- `uv run pytest` green locally in the aminx repo.
- tev_design's venv already installs aminx **editable from this source** (`[tool.uv.sources] aminx =
  { path = "/home/marielle/projects/aminx", editable = true }`), so the fix is live for tev_design local
  smoke immediately on save — no publish needed for local validation.
- For the tev_design **cluster** re-run: either publish 0.1.0a7 and bump the pin, or rsync this source to
  engaging + editable-install there (decided on the tev_design side).

## Out of scope

run_spec migration; renaming `chain_mask` repo-wide (optional within this task if low-risk, else file a
follow-up); any sampling-API side-chain surface (separate item — `aminx.sample()` has no SC args today).
