# Sprint: Phase 0a GO closure + PR2 `sample.py` tuple→payload (revised 2026-05-07)

| Field | Value |
| :--- | :--- |
| **task_id (OODA)** | `refactor-sprint-20260507-phase0a-go-pr2-sample` |
| **Roadmap** | `.agents/REFACTOR_ROADMAP.md` §227–238 (Phase 0a: numeric + **HLO byte counts + op-count summaries** §230); §571–579 (§14); Q6 §13 |
| **Prior sprint** | `.agents/SPRINT_refactor-phase3c-0a-pr2-20260506.md` — **WP2 (unconditional factories)** complete; **PR2b** (`sample.py` loose-stack branches) remains open (see **JIT note** below). |

## Plan-auditor amendments (integrated)

- **WP1:** PR body must record **HLO byte counts and op-count summaries** per §230 (not warnings alone). **LigandMPNN** scope explicitly **in** or **out**. Prefer **separate PR** for GO narrative vs behavioral refactors.
- **WP2:** List targeted pytest subsets for any `sample.py` change; commit **filtered** verification logs per AGENTS.md.
- **WP2b / naming:** Call completed unconditional work **PR2a**; **`multistate_stack is None` + loose kwargs** path is **PR2b** to avoid ambiguous “PR2”.
- **task_id:** OODA logs may use `refactor-sprint-20260507-ooda` for this cycle; sprint execution uses the table `task_id` above.

## Sprint intent

1. **Close Phase 0a process DoD** on `main`: spike tests green, explicit **GO** or **NO-GO** in merge PR, filtered log, LigandMPNN claim scope, numeric bar at `get_tolerances("float32")`, HLO stats per roadmap.
2. **Advance PR2b** (`sample.py`): route `state_vmap_exact` loose-stack calls through `sample_autoregressive_state_vmap_exact_from_payload` **without** breaking `jax.jit` (see JIT note).
3. **Governance:** optional `TECHNICAL_DEBT.md` / §14 bump after merges.

---

## JIT note — why PR2b is not a one-line swap inside `sample_sequences`

`MultistateStackPayload` carries **static** `n_flat`, `n_states`, `n_canonical` (`eqx.field(static=True)`). Inside `@jax.jit` on `make_sample_sequences`, values such as `int(jnp.max(state_flat_rows))` are **not concretizable** (JAX `ConcretizationTypeError`). `int(x.shape[0])` is fine for **leading** dimensions but **does not** recover `n_flat` from flat-row indices.

**Allowed PR2b strategies (pick one in implementation PR):**

- **A (preferred for API stability):** Callers that today pass `coords_stack=`, …, `state_flat_rows=` also pass `multistate_stack=` built **on host** via `multistate_stack_payload_from_prep_numpy` / prep helpers (already parity-tested vs loose kwargs in `tests/sampling/test_state_vmap_exact_jit.py`).
- **B:** Extend `sample_sequences` with **static** kwargs (e.g. `n_flat: int | None = None`, `flat_row_offsets` optional) and add the needed names to `static_argnames` so `n_flat` is concrete at compile time (recompile per shape; document).
- **C:** Keep tuple `sample_autoregressive_state_vmap_exact` inside JIT until A or B is chosen; do not ship a fake `n_flat` guess.

**Non-goals this sprint:** guessing `n_flat` from padded `coords_stack` shape; Phase 4 registry; STE; portable JSON v3.

---

## Default sequencing

**WP1 first** (GO evidence + logs + PR narrative), then **PR2b** on a **separate branch/PR** after GO merges or with strict bisect boundaries. Parallel work is OK only with **separate PRs**.

---

## WP1 — Phase 0a: GO process DoD

| | |
| :--- | :--- |
| **Objective** | Mergeable **GO** or **NO-GO** on `main` with numeric + HLO evidence per §227–236; **§230** satisfied in PR text (byte counts + op-count summaries, e.g. from spike `UserWarning` bodies or `export_hlo`). |
| **Files** | `tests/sampling/spikes/test_state_vmap_exact_spike.py`; `.agents/verification_logs/*.txt` (filtered); PR description (not committed — author fills at merge). |
| **DoD** | 1) `PYTHONPATH=src uv run pytest tests/sampling/spikes/test_state_vmap_exact_spike.py -m parity_fast -q` green. 2) Optional `PYTEST_ADDOPTS='-W default'` when CI must show HLO warning bodies. 3) Filtered excerpt committed under `.agents/verification_logs/`. 4) PR lists **LigandMPNN in/out**. |
| **Non-goals** | Phase 4 implementation; claiming §11 #10 “matching Phase 4 implementation”. |

---

## WP2 — PR2b: `sample.py` loose-stack → payload path

| | |
| :--- | :--- |
| **Objective** | Eliminate duplicate tuple call sites per roadmap §296–308 **subject to JIT note** (strategy A/B/C above). |
| **Files** | `src/prxteinmpnn/sampling/sample.py`; tests: extend `test_state_vmap_exact_jit.py` / contract tests for any new kwargs. |
| **DoD** | `parity_fast`, CLAUDE.md sampling bundle, `test_state_vmap_payload_logits.py` as in prior sprint; **no** `ConcretizationTypeError`; filtered verification log for this PR. |
| **Non-goals** | `_COMBINE_INDEX` / registry migration; STE. |

---

## WP3 — Optional: `TECHNICAL_DEBT.md` scripts row

Mechanical `rg` refresh **only if** `scripts/engaging/` or ctor counts change.

---

## WP4 — Optional: Roadmap §14

Bump **Last update**, **Plan** → this file, **Still open** (Phase 4 blocked until 0a **GO**; PR2b status; STE deferral).

---

## Canonical verification (`prxteinmpnn/` repo root)

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

**Heavy (manual):** `REFERENCE_PATH=... pytest ... -m parity_heavy`.

---

## Global non-goals

- Phase 4 (`registry.py`, `_COMBINE_INDEX`, `MULTISTATE_MODES`, unify vs route).
- STE / `ste_optimize.py` tuple migration.
- Portable RunSpec JSON v3 interchange code in this track.

---

*Plan gate: PASS with nits (2026-05-07). OODA `task_id` for logging: `refactor-sprint-20260507-ooda`.*
