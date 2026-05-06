# Sprint: Phase 0a GO checklist + PR2 tuple follow-on (unconditional factories)

| Field | Value |
| :--- | :--- |
| **task_id** | `refactor-phase4-pr2-20260506` (OODA log) / `refactor-phase3c-0a-pr2-20260506` (sprint label) |
| **Roadmap** | `.agents/REFACTOR_ROADMAP.md` §227–238 (Phase 0a), §296–316 (Phase 3 payloads), §320–345 (Phase 4 **blocked** until 0a **GO** on `main`), §571–579 (§14) |
| **Prior sprint** | `.agents/SPRINT_refactor-phase4-entry-20260505.md` |
| **Plan-auditor** | Verdict **NEEDS_WORK** → **PASS with amendments** (pinned DoD below; WP3 sequencing; canonical pytest one-liners). |

## Sprint intent

Land **parity-pinned** refactors and **documentation** so Phase 4 registry work has a clear **GO** bar and PR2 tuple pressure drops on **non-deferred** hot paths. **STE / straight_through** and **portable JSON v3 code** stay out of scope.

---

## WP1 — Phase 0a: GO artifact (process + optional fixture hardening)

| Item | Detail |
| :--- | :--- |
| **Objective** | Record a **mergeable GO** narrative on `main`: numeric agreement at `get_tolerances("float32")`, HLO warning or stats captured per roadmap §230, explicit **go / no-go** text in PR + filtered log under `.agents/verification_logs/`. |
| **In scope (this sprint)** | Spike tests under `tests/sampling/spikes/test_state_vmap_exact_spike.py`; optional **reuse** of shared parity helpers **without** requiring LigandMPNN in the fast path unless cheap. |
| **Out of scope** | LigandMPNN spike **requirement** (document as follow-up if not done); `parity_heavy` unless `REFERENCE_PATH` is set (manual). |
| **GO checklist (DoD)** | 1) `PYTHONPATH=src uv run pytest tests/sampling/spikes/test_state_vmap_exact_spike.py -m parity_fast -q` green. 2) Same with `PYTEST_ADDOPTS=-W default` when HLO body must appear in CI log. 3) Filtered verification excerpt committed. 4) PR description states **GO** or **NO-GO** with reason. 5) **LigandMPNN**: explicitly **in** or **out** of this PR’s claims. |
| **Non-goals** | `registry.py`, `_COMBINE_INDEX`, deleting `state_vmap_exact`. |

---

## WP2 — PR2: `make_unconditional_logits_state_vmap_fn` → payload delegation

| Item | Detail |
| :--- | :--- |
| **Objective** | Replace internal calls to `score_unconditional_state_vmap_exact(...)` positional tuples with **`unconditional_state_vmap_logits_from_payload`** (or direct `score_unconditional_state_vmap_exact_from_payload`) so JIT factories align with the payload-first path; **numerical parity** vs pre-change tuple calls. |
| **Files** | `src/prxteinmpnn/sampling/unconditional_logits.py`; tests: extend `tests/sampling/test_state_vmap_payload_logits.py` if new edge cases; existing parity tests remain the bar. |
| **DoD** | Global commands below; **explicit** assertion that factory output matches `unconditional_state_vmap_logits_from_payload` on the same tensors (existing test already encodes this — keep green). |
| **Deferred (separate PR)** | `multistate_stack is None` branches in `sampling/sample.py` (larger surface); **STE** (`ste_optimize.py`). |

---

## WP3 — `TECHNICAL_DEBT.md` scripts / engaging audit

| Item | Detail |
| :--- | :--- |
| **Objective** | Reconcile PR6 table with **current** `scripts/engaging/` (shell entrypoints count as in-tree; Python ctors still zero). **Sequence:** refresh **after** WP1/WP2 if those add scripts; otherwise mechanical `rg` now + one-line “post-GO” note if GO narrative lands same PR. |
| **Out of scope (unless listed)** | Full `docs/TODO_BLOCKED_MODULES.md` sweep — **either** include in WP3 **or** open a follow-up ticket (plan-auditor: avoid fig-leaf `rg`). |

---

## WP4 — Portable RunSpec JSON: next slices (**planning only**)

Backlog order (separate future PRs each with version/tests): **ligand** → **tied** → **grid** → **batching** → **averaging** (rationale: align with `RunSpec` sub-configs in `run/spec.py`; do not bundle with registry PR1).

---

## WP5 — Roadmap §14 pointer

After WP1–WP3: update §14 **Last update**, **Still open** (Phase 4 still blocked until 0a **GO** on `main`; tuple/STE deferrals explicit), **Plan** row → this file as active sprint body.

---

## Canonical verification (repo root `prxteinmpnn/`)

```bash
uv run ty check
uv run ruff check .
uv run ruff format .
PYTHONPATH=src uv run pytest tests/parity -m parity_fast -q
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/model/test_ligand_wave_parallel.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -q
PYTHONPATH=src uv run pytest tests/sampling/spikes/test_state_vmap_exact_spike.py -m parity_fast -q
PYTHONPATH=src uv run pytest tests/sampling/test_state_vmap_payload_logits.py -q
```

**Heavy (manual):** `REFERENCE_PATH=... pytest ... -m parity_heavy` — not default CI.

---

## Global non-goals

- No **Phase 4** registry implementation until Phase 0a **GO** on `main`.
- No **STE** tuple migration.
- No **portable JSON v3** schema code in this sprint artifact’s default scope.

---

*Plan-auditor amendments integrated 2026-05-06.*
