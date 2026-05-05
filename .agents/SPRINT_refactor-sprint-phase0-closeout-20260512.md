# Sprint plan: Phase 0 closeout (CreatePlan, post plan-auditor)

**task_id:** `refactor-sprint-phase0-20260505` (Praxia / roadmap continuity; this file is the §14 sprint artifact for the closeout slice).  
**Roadmap:** `.agents/REFACTOR_ROADMAP.md` §14 + Phase 0 / 0a.  
**Plan-auditor verdict:** NEEDS_WORK → **amended plan** (this document).

## Decisions (single outcome each — plan-auditor amendments)

1. **Jaxbeans / isolated CI:** **Vendored-only for Phase 0.** No `jaxbeans` entry in `[tool.uv.sources]` until a published jaxbeans wheel (≥0.1.0) or an org-approved monorepo layout is documented. `hlo_tools.py` remains the supported path for profiling in CI; README states this explicitly.
2. **Jaxlint:** **Not on PyPI and not planned for PyPI.** No CI workflow installs jaxlint. `[tool.jaxlint]` remains for **local** runs when the jaxlint executable is available (e.g. monorepo / developer `PATH`). Roadmap §7.1 documents the policy.
3. **Baseline HLO DoD:** Each `tests/profiling/baseline_hlo/{name}.txt` is **StableHLO text** captured once per name with **JAX/Equinox versions recorded in the file header**. **Regeneration:** documented in `tests/profiling/README.md` (command or entrypoint). **CI:** continues to assert existence + smoke only (no text diff). **Stability:** re-capture when intentionally changing parity-pinned lowers; PR must mention “baseline refresh” if bytes change materially for review.
4. **0a spike hardening:** **Numeric:** same `get_tolerances(jnp.float32)` as today for unconditional path. **HLO narrative:** emit **line count + count of substring `custom-call`** (cheap op-ish proxy) alongside byte count in `UserWarning`. **`parity_heavy`:** when `REFERENCE_PATH` is set **and** that path is a directory, run a **strictly larger** synthetic stack (e.g. more states / residues) with the same numeric+HLO summary assertions; when unset, `pytest.skip` before any work. **CI:** heavy remains excluded from default CI (existing marker policy); local/README documents `REFERENCE_PATH` for heavy.
5. **§14 roadmap success (one sentence):** “Phase 0 is closed when §14 lists no remaining *blocking* closeout items for scaffolding, or each remaining item is explicitly deferred with owner/trigger.” This sprint moves jaxbeans-dep, jaxlint policy clarity, baseline capture, and 0a extension toward that bar.
6. **Prolix lockfile / isolated clone:** **Deferrable.** Trigger: open an issue if `uv sync` fails on a GitHub-only clone; this sprint adds at most a **README note** if a concrete repro exists (optional).

## Traceability matrix (plan-auditor)

| ID | Primary deliverable | Verification |
|:---|:--------------------|:-------------|
| P0C-DOC | This file + `tests/profiling/README.md` | Doc review |
| P0C-JAXLINT | ~~`.github/workflows/jaxlint-advisory.yml`~~ **superseded** — jaxlint is not on PyPI; policy in roadmap §7.1 + `pyproject.toml` only |
| P0C-PY | `pyproject.toml` comment (jaxbeans scope) | `uv run ty check` |
| P0C-SPIKE | `tests/sampling/spikes/test_state_vmap_exact_spike.py` | `uv run pytest tests/sampling/spikes -q` |
| P0C-BASE | `tests/profiling/baseline_hlo/*.txt` headers + content where generated | `uv run pytest tests/profiling -q` |
| P0C-RM | `.agents/REFACTOR_ROADMAP.md` §14 | Doc review |

## Non-goals

Phase 1 hygiene; blind HLO diff CI gates; Phase 4 `state_vmap_exact` unification implementation.

## Verification bundle (fixer)

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn
uv run ty check
uv run ruff check .
uv run pytest tests/profiling -q
uv run pytest tests/sampling/spikes -q
uv run pytest tests/parity -m parity_fast -q
```

Log filtered output to `.agents/verify_logs/` if the Verification Visibility Protocol applies; otherwise attach key pass lines in the PR/commit message.
