# Sprint: Phase 4 entry + Phase 0a GO gate (prep)

| Field | Value |
| :--- | :--- |
| **task_id** | `refactor-phase4-entry-20260505` |
| **Roadmap** | `.agents/REFACTOR_ROADMAP.md` §14, §227–238 (Phase 0a), §320–345 (Phase 4), §11 checklist #10 (split acceptance) |
| **Prior sprint (closed, retained)** | `.agents/SPRINT_refactor-phase3b-20260506.md` — Phase 3b PR1–PR5; portable RunSpec JSON v2 + `spec.py` import/exception hygiene signed off in §14 (2026-05-05) |
| **Plan audit carryover** | Plan-auditor amendments below apply to **this** slice; Phase 4 implementation PRs still require a fresh audit pass when code lands. |

## Objectives

1. **Phase 0a — GO artifact checklist (gating Phase 4 on `main`):** ensure the spike PR on `main` records numeric agreement at `get_tolerances("float32")`, HLO narrative / byte stats (per roadmap), explicit **go / no-go** text, and (when run) `parity_heavy` notes where `REFERENCE_PATH` is available — so Phase 4 registry/unify PRs are not merged under a missing gate (roadmap §227–238, §11 #10).
2. **Phase 4 prep:** keep the registry / `_COMBINE_INDEX` / multistate / `state_vmap_exact` unify-vs-route design ready to execute **immediately after** the GO record exists on `main`; no speculative closure of tuple→payload work outside the agreed deferrals.

## WP1 — Cluster spike smoke (Engaging)

| Item | Detail |
| :--- | :--- |
| **Script path** | `scripts/engaging/submit_phase0a_state_vmap_spike.sh` |
| **Intent** | `sbatch` entrypoint for Phase 0a spike tests + `parity_fast` smoke on cluster hardware; complements local `pytest` DoD. |

*Script may not exist in-tree until a follow-up PR adds it; this sprint doc names the canonical path for when it lands.*

## Verification DoD (local; every PR touching `src/` or `tests/`)

Per `AGENTS.md` (full suite where applicable):

```bash
uv run ty check
uv run ruff check .
uv run ruff format .
uv run pytest
```

**Spike (Phase 0a, `parity_fast`):**

```bash
PYTHONPATH=src uv run pytest tests/sampling/spikes/test_state_vmap_exact_spike.py -m parity_fast -q
```

Use `pytest -W default` locally when HLO narrative warnings must appear in logs.

**Parity subset (`parity_fast`):**

```bash
PYTHONPATH=src uv run pytest tests/parity -m parity_fast -q
```

**Sampling bundle** (paths adjusted for this repo root; parent `tev_design/CLAUDE.md` uses `prxteinmpnn/tests/...`):

```bash
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/model/test_ligand_wave_parallel.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -q
```

**`parity_heavy`:** manual release gate only (`REFERENCE_PATH`); not default CI — roadmap §10.

**Verification Visibility Protocol:** redirect critical verification to a log file; commit a **filtered** excerpt under `.agents/verification_logs/` when the PR touches parity or spike paths.

## Non-goals (this doc-only bookkeeping PR)

- No **`registry.py`**, **`_COMBINE_INDEX`**, **`MULTISTATE_MODES`**, **`SAMPLERS`**, or deletion / unification of **`state_vmap_exact`** in the same change set as this documentation-only PR.
- No claim that tuple→payload migration is complete on **STE / straight_through** or on **`multistate_stack is None`** tuple paths unless a separate code PR has already landed and been verified.

## Plan-auditor amendments (carried into Phase 4 execution)

| Topic | Amendment |
| :--- | :--- |
| **§11 #10 split** | Spike evidence + recorded go/no-go may precede Phase 4 merges; *“matching Phase 4 implementation”* completes only when registry/unification (or routing-on-no-go) PRs merge. |
| **Tuple → payload** | **STE / straight_through** call chains are **excluded** from the mandatory tuple→payload slice for this bookkeeping gate; treat as a **follow-up PR** unless already done elsewhere. |
| **`multistate_stack is None`** | Tuple paths here remain **deferred** to a follow-up PR unless already eliminated; do not mark closed without proof. |
| **WP4 / JSON extensions** | Further **`RunSpec` JSON** field groups ship as **phased extensions** after the v2 `io` portable slice; avoid bundling unrelated schema churn with registry PR1. |

---

*This sprint document is the active Phase 4 entry pointer from roadmap §14; prior Phase 3b sprint file remains in-tree for history.*
