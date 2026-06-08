# Sprint: Phase 3 — Pytree payloads + composed RunSpec

| Field | Value |
| :--- | :--- |
| **task_id** | `refactor-phase3-sprint-20260505` |
| **Roadmap** | `.agents/REFACTOR_ROADMAP.md` §296–316, §6 Phase 3 row, §11 DoD items 8–9 |
| **Plan audit** | **NEEDS_WORK → accepted with amendments** (2026-05-05): added explicit **scripts/spec audit PR**; narrowed **PR1** to harness-only; **§2** half-day Proxide spike with written go/no-go; per-PR DoD and commands; **engaging** corpus caveat for this checkout. |
| **OODA (Praxia)** | Same **task_id**; `recon_id` `260505_prxteinmpnn_phase3`; `plan_id` `plan-phase3-20260505`; `audit_id` `audit-plan-phase3-20260505` (logged under `.praxia/*.jsonl`). |

## Preconditions

- Phase 2 complete per roadmap §14 (`refactor-phase2-sprint-20260505`).
- Phase 0a SPIKE still **not** required for Phase 3 PRs 1–6 (SPIKE gates Phase 4 unification only).

## Amendment (plan-auditor)

1. **PR6 (scripts audit)** is mandatory in this sprint plan: roadmap §11 #8 and Phase 3 tasks require an inventory with per-file **updated / deferred / out-of-scope** notes. Baseline at HEAD (2026-05-05): `rg 'Specification\\(' scripts/**/*.py` hits `scripts/collect_parity_evidence.py`, `scripts/260410/verify_massive_sampling.py`, `scripts/overfit/overfit_check.py` (subclass ctors, not `RunSpecification(`). Extend patterns in the audit to all `*Specification` and any future `RunSpecification(` under `scripts/`.
2. **`scripts/engaging/`** is absent in this repository checkout; **representative pickle corpus** for `migrate_run_spec.py` may live under parent `tev_design` or cluster docs. PR5 DoD: either (a) add fixtures under `tests/` from sanitized pickles, or (b) document **blocker path** with owner + link to corpus location before merge of pickle-dependent behavior.
3. **PR1** must be **scaffold only**: prove `eqx.tree_at`-style `replace()` on a **minimal in-test `eqx.Module`**, without asserting final field names of the eight roadmap payloads (avoids churn when PR2 lands real types).
4. **§2 (prep wiring):** schedule a **≤1 day** spike: read `proxide.ops.dataset.create_protein_dataset` signature and `TECHNICAL_DEBT.md` §2; record **go** (thread kwargs) or **no-go** (leave `compute_resource_allocation` invoked only to compute *effective* caps logged / stored on `RunSpec` for later use, with TODO keyed to upstream issue). No silent “minimal forever” without a written decision in the sprint PR or `TECHNICAL_DEBT.md`.

## Work packages (merge order)

### PR1 — Replace harness (tests only)

1. Add `tests/payloads/test_replace_roundtrip.py` with a tiny dummy `eqx.Module` + tests that `replace(**kwargs)` (via `eqx.tree_at` or the same helper PR2 will use) round-trips under `jax.tree_util.tree_leaves` / equality checks appropriate for arrays + static ints.
2. **DoD:** `PYTHONPATH=src uv run pytest tests/payloads/test_replace_roundtrip.py -q`; `uv run pytest tests/parity -m parity_fast -q`; `uv run ty check` on any new helper module if placed under `src/` (prefer **no** new `src/` in PR1 — keep helpers in-test).

### PR2 — `payloads.py` (eight Equinox modules)

1. Add `src/prxteinmpnn/payloads.py`: `MultistateStackPayload`, `LigandStack`, `LigandContext`, `SamplingControls`, `MultistateContext`, `EncodedFeatures`, `SampleResult`, `GridLineage` per roadmap §3.2 (field names derived from current tuple call-sites, not the roadmap snippet alone).
2. Shared `replace(self, **kw)` pattern (module-level helper or method) using `eqx.tree_at`.
3. **DoD:** PR1 tests extended or duplicated to cover **at least one** real payload type + `uv run ty check src/prxteinmpnn/payloads.py`; `ruff check` on touched files; `parity_fast` green.

### PR3 — Composed `RunSpec` + shim

1. Add `src/prxteinmpnn/run/spec.py` (or agreed path): nine sub-configs from roadmap §3.5 + composed `RunSpec`.
2. Refactor `run/specs.py` so `RunSpecification` / task subclasses **build or delegate to** `RunSpec` (kwargs shim + `DeprecationWarning` per §13 Q4 when old kwargs appear).
3. **DoD:** `uv run ty check` on `run/specs.py`, `run/spec.py`, importers; `parity_fast` green; targeted `pytest` for any spec construction tests if present.

### PR4 — `PrecisionConfig` + resource alignment

1. Introduce `PrecisionConfig` on `RunSpec` (or nested module); route `training/trainer.py` dtype / mixed-precision policy to read from spec (closes TECHNICAL_DEBT §1 alignment with roadmap).
2. Execute **§2 spike** (see Amendment); minimally **call** `compute_resource_allocation` from `prep_protein_stream_and_model` (or single prep entry) and attach results to the object graph **or** pass through to `create_protein_dataset` **only if** spike go — otherwise record no-go + TODO owner.
3. **DoD:** `ty check` on touched training + run modules; `parity_fast` green; spike decision **one paragraph** in PR description or `.agents/TECHNICAL_DEBT.md` delta.

### PR5 — Pickle migration

1. Add `scripts/migrate_run_spec.py` per roadmap (old pickle → new `RunSpec` pickle).
2. Tests: corpus-driven round-trip under `tests/` (or documented waiver — see Amendment §2).
3. **DoD:** script `uv run python scripts/migrate_run_spec.py --help` works; tests pass; roadmap §11 #9 language reflected in PR description (“representative engaging pickle corpus” or documented substitute).

### PR6 — Scripts / spec audit (governance)

1. Run `rg -l 'Specification\\(|RunSpecification\\(' scripts/` (refresh patterns as constructors evolve); **annotate every path** in PR description table: updated / deferred / OOS with one-line reason.
2. Update call-sites in scope (likely the three files at HEAD + any new matches) to use the new construction path if PR3 requires it.
3. **DoD:** `rg` output pasted or linked in PR; parity_fast green on CI-relevant paths.

## Deferred (re-entry triggers)

| Item | Re-enter after |
| :--- | :--- |
| `MultistateStackPayload` at **all** `state_vmap_exact` + `DesignArrayRecordWriter` sites | PR3 merged (typed carrier stable) |
| Full **writer** tuple unpack migration | Same |
| **`parity_heavy`** | Phase 3 tag / release gate per roadmap §10 |

## Verification commands (every PR)

```bash
uv run pytest tests/parity -m parity_fast -q
uv run ty check <modules-touched>
uv run ruff check <paths-touched>
```

Targeted sampling tests (when touching sampling/scoring):

```bash
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/model/test_ligand_wave_parallel.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -q
```

## Definition of Done (whole sprint)

- PR1–PR6 merged or explicitly split with carry recorded in roadmap §14.
- Roadmap §11 checklist items **8** (scripts audit) and **9** (migration script + corpus story) **not** silently dropped.
- `parity_fast` green at sprint end; `ty` clean on all touched `src/` modules.
