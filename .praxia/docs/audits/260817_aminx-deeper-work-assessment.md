---
title: 'aminx deeper-work assessment: EPIC #1541 remaining scope, RS-track stall, and a silent ruff/jaxtyping corruption'
description: 'State-of-play assessment against actual code: P4''s remaining work is gate measurement not migration; the standing T2.GATE was re-run against xtrax 0.4.0a5 and its throughput leg found unusable (noise straddles all three outcome bands); RS-1 blocks T4.1 on an unmade decision; ruff UP037 + fix=true was silently corrupting jaxtyping shape specs.'
status: complete
task_id: 260817_aminx-deeper-work-assessment
date: '260817'
verdict: ''
base_sha: ''
---

# aminx deeper-work assessment (2026-08-17)

**Scope:** what deeper work aminx actually needs, measured against code rather than against
status docs. Prompted by a stash triage that surfaced `stash@{12}`'s `heterogeneous_axes` work;
that item was reconciled separately (commit `c7a96c6e`) and is not repeated here.

**Method:** every claim below was checked against current source. Where a status document and the
code disagreed, the code won and the disagreement is recorded as its own finding. Two claims I
initially made in-session were wrong and are corrected in place (§2.2, §5) rather than quietly
dropped.

---

## 1. Headline

Three things, in descending urgency:

1. **A live silent corruption was found in the working tree** — `ruff` was rewriting jaxtyping
   shape specs on every `check` invocation. Fixed (`9af689e5`).
2. **EPIC #1541's remaining work is measurement, not refactoring.** P4's migration content was
   re-scoped to near-zero and its concrete items are already done. What is missing is gate
   evidence — and the standing gate turns out to be a broken instrument (§3).
3. **RS-1 has blocked T4.1 for six weeks on an unmade decision**, not on engineering (§4).

---

## 2. The ruff/jaxtyping corruption (FIXED)

### 2.1 What happened

`UP037` (*quoted-annotation*) strips quotes from any annotation whose contents parse as a Python
expression. For jaxtyping that is destructive:

| before | after | consequence |
|---|---|---|
| `Float[Array, "..."]` | `Float[Array, ...]` | `Ellipsis`, not jaxtyping's any-shape spec |
| `Float[Array, "C"]` | `Float[Array, C]` | `C` undefined → latent `NameError` |

The corruption is **silent and selective**. Multi-word specs (`"S C"`, `"... L V"`, `"top_k L"`)
survive only because they are not valid expressions, so the diff reads as a tidy-up. And
`from __future__ import annotations` (e.g. `sampling/mbr_consensus.py:17`) defers evaluation, so
the module still imports — the shape contract simply stops being one until something calls
`get_type_hints()`.

Affected, uncommitted, when found: `cli.py`, `host/campaign.py`, `sampling/mbr_consensus.py`,
`types/stages.py` — 7 lines.

### 2.2 Root cause, and a correction

**`pyproject.toml:106` sets `fix = true`.** A bare `ruff check` therefore rewrites source with no
`--fix` and no opt-in. Reproduced against HEAD: `ruff check --select UP037` on
`sampling/mbr_consensus.py` reports *"Found 3 errors (3 fixed)"* and mutates the file.

*Correction to an in-session claim:* I first estimated ~54 at-risk sites from a grep. That was
wrong — the grep counted docstring occurrences (`weights : Float[Array, "S"]` in Parameters
blocks), which UP037 does not touch. The true exposure was the 7 lines already mangled; `ruff
--select UP037` reports clean on `src/` afterwards precisely *because* they were already
rewritten.

### 2.3 Remediation

- `9af689e5` — ignore `UP037`, placed beside the pre-existing `"F722", # jaxtyping strings`
  entry, which is the same class of friction. asr carries a manual `# noqa: F821` on
  `Bool[Array, "n_nodes"]` (`src/asr/dataset.py:58`) for a third instance.
- `90027c96` — the other two files' diffs were **not** UP037: import sorting (`I001`) plus one
  genuinely-unused import (`F401`, `CAMPAIGN_OWNED_KEYS`, verified not re-exported). Committed
  separately as semantically-inert hygiene.

### 2.4 Still open

**`fix = true` + `select = ["ALL"]` is the amplifier, and it is still armed.** Any future ruff
release shipping a new autofix rewrites this codebase on the next lint run. UP037 is the instance
that was caught, not the class. Options: drop `fix = true`, or pin ruff and review autofix
additions on upgrade. This is a config decision, deliberately left to the owner.

A jaxlint rule ("never unquote a jaxtyping shape spec") would catch the class rather than the
instance — the repo already selects `["JL","JD","JM"]`.

---

## 3. EPIC #1541: the gates, not the code

### 3.1 What is actually done

- **P0–P3: done.** **T2.GATE: passed** (`260611_aminx-xtrax-refactor.md:243`) — bit-for-bit
  golden, exact recompile count, cluster throughput within 0.2% on production shapes
  (L=208, TEV protease).
- **P4: re-scoped to near-zero** by the 260706 adversarial review. Boundary types were already
  xtrax-sourced; concrete sinks are protein-domain-specific and stay local; `StageSet` **cannot**
  adopt `StageBundle` at all (two independent, verified blockers — permanent by design).
- **P4's two remaining concrete items are DONE** (verified this pass, contradicting
  `INDEX.md:19`'s "Remaining P4 scope"): the dead vendored `BoundedCallbackHandler` is removed and
  `async_indexed_stream` renamed; `utils/_vendored_callbacks.py` is now a 2KB stub documenting it.

So the epic is not short of migration work.

### 3.2 The standing gate was stale — and is a broken instrument

The throughput gate is explicitly **standing**, "not a one-shot at flip". That is the stated
reason the legacy `make_axis_dispatch` and `iterator.py` are deliberately retained as the
comparison baseline (`:246`). But the last `gate:T2.GATE`-tagged run was **2026-07-06 against
xtrax 0.4.0a1**, while `pyproject.toml` now pins `xtrax[io]==0.4.0a5` (via a4). The dual
implementation's carrying cost was being paid across two upgrades without the benefit collected.

**Re-run 2026-08-17 against 0.4.0a5:**

| leg | result |
|---|---|
| bit-for-bit golden (`test_t2_gate_bitforbit_golden.py`) | **6/6 pass** |
| `all_decision_parity` | **true** (5/5 observations) |
| `all_recompile_parity` | **true** (5/5 observations) |
| `max_adapter_vs_legacy_throughput_ratio` | **unusable — see below** |
| GPU / production-shape leg | **not run** — no CUDA jaxlib locally; needs cluster |

The correctness legs revalidate cleanly on 0.4.0a5. The throughput leg does not, because it
cannot:

| observation | max_ratio | sidecar band |
|---|---|---|
| 2026-07-06 (0.4.0a1) | 1.341 | marginal |
| 2026-08-17 tracked | 1.637 | **fail** (`is_residual`) |
| repeat 1 | 1.032 | pass |
| repeat 2 | 1.479 | marginal |
| repeat 3 | 1.012 | pass |

Identical inputs, same machine, minutes apart, **straddling all three of the sidecar's own outcome
bands**. The gate keys on `max()` across ~9 cases whose absolute medians are 11–100 µs, so a
single jittery median sets the verdict. In a quiet run every case sits at **0.90–1.01** — i.e. the
adapter overhead is genuinely ~0–1%, comfortably inside the 1.10 pass bar.

**Consequence:** the July "marginal 1.34" verdict was a coin flip, and so is today's "fail 1.64".
Under bathos discipline the `fail` branch is `is_residual`, so this gate can block a flip at
random. *This corrects an intermediate claim I made in-session that a1→a5 showed a throughput
regression — it does not; the difference is inside the noise.*

**Fix before this gate is trusted again:** key on a robust statistic (median or geometric mean
across cases) instead of `max()`; raise per-case work so timings are not microseconds; and/or
repeat-and-aggregate within the script. Until then, treat only the parity legs as load-bearing.

### 3.3 Incidental finding: `bth ls` and `bth find` do not surface runs that exist

**CORRECTED 260817 (same day).** This section first claimed `bth run` records nothing in aminx.
That was wrong, and the error is instructive: it was inferred from `bth ls` alone.

What is actually true — the write path is fine, the read path is not:

```
bth sql "SELECT ... FROM runs WHERE project_slug='aminx' ORDER BY timestamp DESC"
  -> 2026-08-17 15:22:40+00:00  completed  outcome=fail   <-- the run IS there (22 aminx rows)

bth ls   --project aminx --limit 40   -> 0 rows from 2026-08; newest shown is 2026-07-26
bth find --filter "project_slug='aminx'"  -> does not return it either
```

So `bth run` recorded the run correctly. `bth ls` and `bth find` both fail to surface it while
`bth sql` against the same `bathos.db` sees it — a read-path defect, present before and after
`bth compact`.

This is worse than the original mistaken diagnosis rather than better. `bth ls` is the
human-facing "did my run get tracked?" check, and it answered *no* when the truth was *yes*. Any
agent or human using it to confirm tracking gets a false negative, which is how a standing gate
quietly stops being registered. **Not yet filed** as bathos debt and not fixable from this repo —
it belongs against bathos, whose read path (`ls`/`find` vs `sql`) is where the divergence lives.

**Second-order problem this exposes:** the catalog now holds `outcome=fail` for a `gate:T2.GATE`
run whose `fail` verdict is pure noise (§3.2). A future reader querying gate history sees a failed
gate. That verdict needs annotating, not deleting — the run happened, its threshold comparison is
just not meaningful at this bench's resolution.

---

## 4. RS track: one unmade decision, blocking T4.1

`260611_runspec-unification` is INDEX-listed as *"Active — blocks T4.1"*. The 260707 gap audit
called it "stalled, partially inverted"; **every finding still holds today**:

- **27 of 72** `RunSpec` sub-config fields are populated at construction and never read.
  Spot-checked six across all five affected sub-configs (`BatchingConfig`, `TiedPositionsConfig`,
  `AveragingConfig`, `GridLineageConfig`, `LigandConfig`): zero read sites each.
- **`260614_runspec-migration-map.md` is still `status: complete`** while being wrong in *both*
  directions — all 22 "already migrated" rows stale, 6+ "needs migration" rows already done.

The audit's own status line reads *"findings complete, remediation not chosen."* The blocker is
not engineering: it is the choice to **fix the RS-1 sub-config migration or formally abandon it**.
Until it is made, anyone scoping T4.1 reads a document that is confidently backwards.

**Minimum action even if the decision is deferred:** flip that map's frontmatter to
`status: outdated`. It is cheap and it stops the misinformation.

---

## 5. Documentation drift is now a hazard, not untidiness

Four dead or inverted references, all load-bearing for someone planning this work:

| location | claim | reality |
|---|---|---|
| `tiling/dispatch.py` docstring | `make_axis_dispatch_via_xtrax` is "not yet wired into any call site", "the flip target once T2.GATE passes" | It is the **only** production path (8 call sites); legacy has **zero**; the gate passed 6 weeks ago |
| `260614_runspec-migration-map.md` | `status: complete` | Wrong in both directions (§4) |
| `INDEX.md:19` | "Remaining P4 scope: delete X, rename Y" | Both done (§3.1) |
| `tiling/dispatch.py` comment | `_HETEROGENEOUS_AXIS_NAMES` lives in `tiling/carry.py` | That module was deleted; it is in `host/plan.py` — **fixed in `c7a96c6e`** |

*Correction to an in-session claim:* I described the legacy `make_axis_dispatch` as "slated to be
replaced", having believed its own docstring. It is retained **permanently and deliberately** as
the standing gate's reference implementation. The conclusion drawn from the wrong premise still
holds — do not widen its signature — but for a better reason: a reference implementation exists to
mirror production, so it must read the same constant and must not grow parameters production
lacks, or the comparison weakens.

---

## 6. Deferred xtrax adoption, still open

- **`utils/safe_map.py` / `safe_scan.py` are local forks.** The cardinality bug was fixed
  2026-07-07 (`n_samples_override`), but the *mechanism that let it fail silently* is untouched:
  `safe_map.py:49` still carries a `batch_size == 0` "always vmap" branch that xtrax's canonical
  version does not have. Against xtrax's `safe_map`, the same disconnected-cardinality input
  raises `ZeroDivisionError` instead of quietly vmapping an oversized array. Adopting it is real
  defense-in-depth; blast radius is every `aminx.utils.safe_map` call site.
- **`training/trainer.py` (939 lines) hand-rolls its loop** instead of `xtrax.engine.Engine`,
  while already importing `xtrax.training`'s `ResumableState`/`make_optimizer` and
  `xtrax.checkpoint.orbax` into that same loop. Real duplicated orchestration; never assessed as
  a safe swap (custom eval cadence, `setup_mixed_precision`).
- **10+ ad-hoc epsilon guards** vs `xtrax.safety.safe_norm`/`safe_reciprocal` — consolidation
  only; none were found to have a wrong or missing guard.

aminx imports from 6 of xtrax's 15 subpackages.

---

## 7. Recommended sequencing

1. ~~Revert the UP037 damage and neutralize the rule~~ — **done** (`9af689e5`, `90027c96`).
2. ~~Re-run the standing gate against 0.4.0a5~~ — **done**; parity legs pass, **throughput leg
   found unusable**. Next: repair the bench's statistic (§3.2) before trusting it, and run the
   GPU/production-shape leg on the cluster (the only leg that can speak to the real DoD).
3. **Decide `fix = true`** (§2.4) — small, and it is the amplifier for the whole class.
4. **Make the RS-1 call** (§4) — unblocks T4.1. Owner decision, not an engineering task. Flip the
   migration map to `status: outdated` regardless.
5. Fix the four drifted doc references (§5) — cheap, and each one currently misleads.
6. Then: `xtrax.transforms.safe_map` adoption, and a real assessment of `Engine.fit()` against
   trainer.py's requirements.

## Provenance

- Gate re-run output: `outputs/results/xtrax_vs_aminx_tiling_260817.json`, and the tracked run
  **is** in the bathos catalog at `2026-08-17 15:22:40+00:00` (reachable via `bth sql`, though not
  via `bth ls`/`bth find` — §3.3). Its recorded `outcome=fail` is a noise artifact, annotated by
  postmortem rather than removed.
- Commits from this pass: `c7a96c6e` (heterogeneous-axes reconciliation), `9af689e5` (UP037),
  `90027c96` (import hygiene).
- Prior art relied on and re-verified: `260707_xtrax-migration-gap-audit-runspec-scaffolding.md`,
  `260706_epic1541-p4-runner-hostsinks-scoping.md`, `260611_aminx-xtrax-refactor.md` §Status
  Update.
