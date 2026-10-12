---
title: Backbone noise-level axis for design and score
description: Every family honours a multi-level backbone_noise sweep in both sample and score, with one shared key scheme; replaces the #214 refusal
task_id: 261011_noise-axis
status: active
---

# Backbone noise-level axis for design and score (261011)

User direction (261011): "we need to have a noise-level axis available for both design and score".
This supersedes PR #214's decision to *refuse* nonzero `backbone_noise` on plain score (debt #2509).

## Current state (recon 261011, spot-checked)

| Path | Noise applied? | Axis in output? |
|---|---|---|
| stock `runner.sample` Path A | yes, per level | yes: `(S, N, D, T, L)` (`host/kernel_dispatch.py:661`) |
| stock `runner.sample` Path B (encoding fusion) | yes, fused over D | no (by design) |
| stock `runner.score` plain, encode-once | no (`scoring/score.py:279` hard-codes 0.0) | no, `(S, C)` |
| stock `runner.score` plain, per-candidate | no (`score_fn` not passed noise) | no |
| stock `_score_fused_multistate` | no | no, `(1, C)` |
| stock averaged (`average_node_features`) | yes, fused over D | no (by design); D>=2 crashes (#2517) |
| pottsmpnn (sample + score) | level[0] only, extra levels silently dropped | no |
| lasermpnn (sample + score) | level[0] only, extra levels silently dropped | no |
| protonpottsmpnn (all purposes) | never (silently ignored) | no |

Design derives the per-structure noise key as `fold_in(_base_sampling_key(spec), structure_idx)`
(`kernel_dispatch.py:236,562`); the one draw is shared by every level, temperature and sample.

## Decisions

- **N-D1 shape.** Score returns `(S, C)` when the spec has exactly one noise level (any value, including
  nonzero), and `(S, C, D)` when it has two or more. `C` before `D` mirrors design's samples-before-noise
  order. The shape depends only on the spec, never on data. Fused multi-state: `(1, C)` / `(1, C, D)`.
  Logits, when returned, gain the same axis in the same position. The levels are always reported
  (`noise_levels`) in the result metadata.
- **N-D2 key.** Score draws noise with exactly design's key: `fold_in(_base_sampling_key(spec), structure_idx)`,
  shared across candidates and levels. So scoring a design's output at the same spec/seed sees the identical
  noised backbone. The per-(structure, candidate) keys that drive decoding order are unchanged, so noise-0
  scores stay bit-identical to today.
- **N-D3 encode-once.** Encode once per (structure, level) and score every candidate against it; the per-candidate
  branch passes `backbone_noise=level` with the N-D2 noise key (not the candidate key).
- **N-D4 averaged path.** Unchanged semantics (it fuses D). Fix #2517 (move the R3 host check out of the vmap)
  if it is local; otherwise leave it and keep D>=2 refused there with a clear error.
- **N-D5 driver families.** pottsmpnn, lasermpnn and protonpottsmpnn take every level. Each family keeps its
  existing noise rule and key (`0x9077` Potts, `0x6262` LASEr); the level index is NOT folded into the key, so
  levels share one draw scaled by the level (as stock design does) and a single-level run is bit-identical to
  today. Output gains a noise axis only when D >= 2, after the sample/candidate axis, declared in `sample_axes`
  / `result_schema`. ProtonPotts reuses `apply_potts_backbone_noise` (upstream PottsMPNN-pH featurizes through
  the same `structure_noise` path; its pH engine passes 0.0, `potts_mpnn_ph.py:1119`) with its own key constant,
  applied to featurized coordinates before any table/energy/pH-design computation.
- **N-D6 sinks.** `agent/shaping.py::_score_structures` accepts the 3-D form and emits `noise_index`;
  `SCORING_SCHEMA_VERSION` bumps to `scoring_v2` (adds `noise_levels`). Docstrings fixed (runner.sample still says
  `(B*N, L)`).
- **N-D7 out of scope here.** Browser pages (no backbone noise exists in any JS page today; it needs a
  coordinate-noise step before the encoder graph and can only be distributionally, not bitwise, graded).
  Multistate PoE campaign path (`sampling/multistate_poe.py`) keeps its own key scheme.

## Acceptance

1. Noise 0 (default) is bit-identical to main for every path and family (pin tests + the 9-row re-wave).
2. Score with D >= 2 returns `(S, C, D)`; slice `[..., d]` equals a single-level run at level `d`.
3. For the stock family, the noised backbone (or encoder output) at (structure s, level d) is identical between
   `runner.sample` and `runner.score` for the same spec/seed.
4. Nonzero noise changes the score (control: the old silent-drop bug would fail this).
5. Each driver family: D >= 2 yields the noise axis; level 0 slice equals the noise-free run; a level > 0 changes it.
6. #214's refusal test file is replaced by these tests.

Ledger: `host/runner.py`, `scoring/score.py`, the family drivers are in row closures, so all 9 rows go stale;
one re-wave after the branch is complete, ledger last.
