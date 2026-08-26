# SEED FINDINGS — task 260826_chain-selection-vendor-superset-audit

Phase 1 (recon/freeze) only. Nothing below is an ASSERTED verdict yet — that requires
Phase 2's AST reachability checkers + differential bit-identity probes, per the F001-F005
methodology this sprint is extending. This document freezes what a manual read found so
Phase 2 has a hashable starting seam list rather than re-discovering it.

## Axis A — chain-selection reachability across ALL runner surfaces

**Confirmed structural finding: two different, disconnected wiring mechanisms exist for
"fixed_mask" today, and most surfaces show no evidence of reading it at all.**

| surface | fixed_mask / chain_mask wiring found | file:line |
|---|---|---|
| `runner.inspect` (unconditional_logits branch) | `fixed_mask=spec.fixed_mask` passed directly into `build_inference_bundle` | `host/runner.py:949` |
| `runner.inspect` (conditional_logits branch) | not found in this pass | `host/runner.py:958+` (untraced) |
| `runner.sample` | derives `chain_mask = 1 - fixed_mask` in `_sampling_helper.py:343-365`, but that value is only read inside `if ligand_context is not None` guards in `kernel_dispatch.py:251,334,441,520` — i.e. it may only affect ligand-context/exposure computation, not necessarily which residues get sampled vs held fixed during autoregressive decode | `_sampling_helper.py:343-365`, `kernel_dispatch.py:251,334,441,520` |
| `runner.score` | not found in this pass | untraced |
| `runner.jacobian` | not found in this pass | untraced |
| `chain_id` (prep-time chain filter) | consumed once at `host/prep.py:96`, upstream of all runner surfaces — structurally reaches everything by construction, but not yet differentially confirmed to actually restrict the parsed structure | `host/prep.py:96` |
| `tie_group_map`/`tied_positions`/`pass_mode` (symmetry family) | untraced entirely | — |

**This is exactly the shape F002/F005 found for `state_position_map`**: a field that exists
on the base spec and is silently, inconsistently honored across surfaces. Unlike F002/F005,
this has NOT been AST-hit-tested or differential-probed yet — the above is a manual grep/read
pass and must not be treated as an adjudicated verdict. The single highest-priority Phase 2
action is a differential bit-identity probe on `runner.sample`: fix a known residue subset,
sample, and check whether the fixed positions are held literal in the output. If they are not,
`fixed_mask` is decorative on the sample path and this becomes the sprint's F001-equivalent —
a correctness bug, not a documentation gap.

## Axis B — vendor superset audit vs LigandMPNN

**Vendor pin is stale.** `tests/parity/parity_assets.json` records `dauparas/LigandMPNN@3870631`
for the ProteinMPNN checkpoint; that SHA is unreachable in the upstream repo's current history
(only 15 commits total exist there). This should be filed as its own debt item independent of
this audit — a parity test that resolves a reference commit by SHA and gets a wrong/missing
checkout is a silent-failure risk for the parity suite generally, not just this audit.
Substituted `HEAD` (`26ec57ac976ade5379920dbd43c7f97a91cf82de`) for this pass.

**Candidate real gap: no chain-letter-level design/fix convenience in aminx.**
Vendor's `--chains_to_design "A,B"` (run.py:886) has no aminx equivalent. The closest thing —
`chain_mask_fixed` from orphaned PR #1881 — is a raw residue-index array, not a chain-ID
selector; a caller still has to hand-build the per-residue mapping. `--parse_these_chains_only`
(run.py:893, structure-level filter) maps reasonably to aminx's `chain_id` field. The
`--symmetry_residues`/`--symmetry_weights`/`--homo_oligomer` family maps conceptually to
aminx's `tie_group_map`/`tied_positions`/`pass_mode`, but that mapping is unverified — Phase 2
needs to actually trace tie_group_map's consumers the way this pass traced fixed_mask's.

Full vendor flag inventory (chain-selection subset): `vendor_flags.json`.
Full aminx-side field inventory + wiring trace: `aminx_chain_fields.json`.

## Recommended Phase 2 (not yet started — compute cost warning below)

1. AST-derive both seam lists programmatically (reuse `.claude/skills/jax-deep-audit/checkers/derive_seams.py`'s
   walker, scoped to the field names above) and hash them as the immutable denominator, same
   discipline as `seams.json`'s sha256 in the F001-F005 audit.
2. Build `checkers/chain_reachability_checker.py` (Axis A, U1-shaped: AST zero-reference test +
   differential bit-identity probe per seam) and `checkers/v1_vendor_superset_checker.py` (Axis B:
   alias-mapped coverage matrix).
3. Differential probes require loading real model weights and running `runner.sample`/`score`/
   `jacobian`/`inspect` — this is real compute (not free like the AST pass), and needs disk
   headroom for weights + JAX compilation cache. Current free space: ~17GB after this session's
   cleanup — check before Phase 2 if weights aren't already cached locally.
4. File the stale vendor-pin issue as its own debt item (`tests/parity/parity_assets.json`'s
   `dauparas/LigandMPNN@3870631` reference) — orthogonal to this audit, but discovered by it.

## Independence note

This seed was derived by direct grep/Read of `src/aminx/` and a fresh clone of
`dauparas/LigandMPNN`, without reading the prior F001-F005 audit's specific findings first —
only its methodology (seam-freeze, AST+differential adjudication, closure-gate ledger,
blast×silence ranking, honest shortfall reporting) was carried forward by design, per the
user's explicit request to reuse "lessons learned."
