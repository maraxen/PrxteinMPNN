# Sprint plan: `refactor-sprint-phase0-20260505`

**Praxia:** `ooda_log` (`append_recon` / `append_plan` / `append_audit`) was not invoked — Praxia MCP tools are not available in this Cursor session. This file is the parent-owned **CreatePlan** artifact after recon → planner → plan-auditor (NEEDS_WORK, amended).

**Task ID:** `refactor-sprint-phase0-20260505`  
**Roadmap:** `.agents/REFACTOR_ROADMAP.md` Phase 0 + Phase 0a + §13 Q5/Q8  
**Plan-auditor verdict:** NEEDS_WORK → **accepted with amendments** (below).

## Amendments from plan-auditor

1. **HLO / CI semantics:** Raw StableHLO text files under `tests/profiling/baseline_hlo/` are **review artifacts only**. CI **must not** fail on arbitrary byte-for-byte HLO diff vs baseline. CI may assert: (a) HLO export succeeds (smoke), (b) optional byte-count vs `hlo_allowlist.toml` **maximum** slack only where explicitly allowlisted, (c) `assert_zero_copy_overhead` only when comparing two intentional variants (or self-check `f,f` as wiring smoke), never blind baseline equality.
2. **Phase 0a DoD:** Spike test must cover **numeric** `get_tolerances("float32")` agreement, **HLO byte/op summary** hooks for PR narrative, **`parity_fast`** always; **`parity_heavy`** when `REFERENCE_PATH` is set (skip otherwise). PR template records **go/no-go** for Phase 4.
3. **PR labels:** Map slices to roadmap bookkeeping: Phase 0.1 (tooling), 0.2 (vendor+Q5), 0.3 (profiling), 0.4 (0a spike).
4. **jaxbeans / CI:** Isolated `prxteinmpnn` CI checkout does not include sibling `jaxbeans`. **Mitigation:** Vendor `export_hlo`, `analyze_memory`, `assert_zero_copy_overhead` from jaxbeans `src/jaxbeans/core/profiling.py` into `src/prxteinmpnn/profiling/hlo_tools.py` with upstream path + commit hash in file header. **Optional later:** add `jaxbeans` workspace dep when root workspace includes it. **`jaxlint`:** Not on PyPI and not planned for PyPI; add `[tool.jaxlint]` to `pyproject.toml` for local runs only. **No CI** that installs jaxlint from PyPI.

## Objectives (sprint)

- Tooling: `ty.toml` `allowed-unresolved-imports`; `[tool.jaxlint] select = ["JL"]` (local).
- Vendored: `get_tolerances`, `PRXTEINMPNN_VERIFY` (env `PRXTEINMPNN_VERIFY`), jaxbeans-attributed HLO/zero-copy helpers.
- Q5: `tests/parity/conftest.py` + README note.
- Q8: `tests/profiling/hlo_allowlist.toml` with rationale fields.
- Profiling tests: smoke + allowlist-aware checks per §1 above.
- Phase 0a: `tests/sampling/spikes/test_state_vmap_exact_spike.py`.

## Verification

```bash
uv run ty check
uv run ruff check .
uv run pytest tests/profiling -q
uv run pytest tests/sampling/spikes -q
uv run pytest tests/parity -m parity_fast -q
```

## Definition of Done

Sprint complete when items above merge, `parity_fast` green, ty/ruff green, roadmap § “Sprint status” updated with outcome and pointer to this file.
