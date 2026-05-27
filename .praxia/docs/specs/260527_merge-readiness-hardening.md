# Specification: Merge-Readiness Hardening (refactor-full → main)

**Task ID:** `260527_merge-readiness-audit`  
**Date:** 2026-05-27  
**Branch:** `refactor-full`  
**Status:** ORACLE-REVIEWED — PASS (15/18) — fixes applied 2026-05-27

---

## Executive Summary

Harden the `refactor-full` branch against 8 concrete packaging, documentation, and debt-closure items before merging to `main`. The goal is a library that is PEP 561-compliant, RTD-ready, has zero tracked open debt that would confuse future contributors, and passes a clean docs build. No new features; no architectural changes.

Sprint 6 is sealed (443 tests green, Risk D-2 STE parity held, driver.py at 104 lines). This spec covers only the pre-merge hygiene gap.

---

## Scope

### In Scope
- `src/prxteinmpnn/py.typed` marker creation (PEP 561)
- `[project.urls]` addition to `pyproject.toml`
- `.readthedocs.yml` creation at repo root
- Ruff auto-fix pass (`ruff check --fix`) — automated, not manual
- `driver.decode` deprecation comment verify (zero-to-minimal touch; already correct)
- `decoder.py:8` stale TODO comment update
- `host/campaign.py:25-31` dead-import comment removal
- Debt #16 open-reference closure check

### Out of Scope
- Training module (`NotImplementedError` intentionally; entire training module deferred)
- `tiling/planner.py:16` bucketing TODO (future work; `BatchPlanner.plan()` is production-ready)
- COMP-533: `sample.py:184` `make_stage_set` direct call (deferred per SamplerFn protocol constraint)
- IREE/WASM downstream targets
- Manual ruff error fixes beyond the `--fix` auto-fixable subset
- New test coverage

---

## Task Decomposition

### Wave A — No Dependencies (6 parallel tasks)

#### MR-01: Add py.typed marker + update package-data
**Files:** `src/prxteinmpnn/py.typed` (create), `pyproject.toml` (modify `[tool.setuptools.package-data]`)  
**Action:**
1. Create an empty `src/prxteinmpnn/py.typed` file. PEP 561 requires only existence in source.
2. Update `pyproject.toml` at `[tool.setuptools.package-data]` (currently `prxteinmpnn = ["model_params/*.eqx.zst"]`) to include the marker so it ships in the built wheel:
```toml
[tool.setuptools.package-data]
prxteinmpnn = ["model_params/*.eqx.zst", "py.typed"]
```
Without step 2, `py.typed` exists in source but is not included in the built wheel — PEP 561 compliance requires it to be in the installed package.

**Success criterion:**
```bash
test -f src/prxteinmpnn/py.typed && echo "source ok"
uv build && python -c "
import zipfile, pathlib
whl = next(pathlib.Path('dist').glob('*.whl'))
z = zipfile.ZipFile(whl)
assert any('py.typed' in n for n in z.namelist()), 'py.typed not in wheel'
print('wheel ok')
"
```
**Dependencies:** none  
**LOC:** 0 (empty file) + 1 line changed in pyproject.toml

---

#### MR-02: Add [project.urls] to pyproject.toml
**Files:** `pyproject.toml` (modify)  
**Action:** After the `[project.scripts]` section (or after `[project.optional-dependencies]` if no scripts section), add:
```toml
[project.urls]
repository = "<git remote get-url origin>"
documentation = "https://prxteinmpnn.readthedocs.io"
```
Fixer must run `git remote get-url origin` first and use the actual remote URL. If no remote is configured, use a placeholder + `# TODO: update` comment.  
**Success criterion:** `uv run python -c "import importlib.metadata; m = importlib.metadata.metadata('prxteinmpnn'); print(m['Project-URL'])"` prints URL lines without error.  
**Dependencies:** none  
**LOC:** ~4

---

#### MR-03: Create .readthedocs.yml
**Files:** `.readthedocs.yml` (create at `prxteinmpnn/` repo root)  
**Mandatory prerequisite (hard step — not optional):** Run `uv run sphinx-build -W docs/source docs/_build/html` before writing the YAML. Record the exit code:
- Exit 0 → include `fail_on_warning: true` in the YAML (clean build confirmed)
- Exit non-zero → fix warnings first if they are trivial (import errors, orphaned refs); OR drop `fail_on_warning: true` from the YAML, write the YAML without it, and add a row to the Deferral Register: `"sphinx fail_on_warning | docs/source/conf.py | N warnings on sphinx-build -W; fix in follow-up before RTD goes live"`

Known potential warning sources:
- `docs/source/conf.py` uses `main_doc = "index"` instead of `master_doc` — Sphinx 4+ may warn on the non-standard key
- jaxtyping forward references during autodoc
- MyST non-standard config keys

**Action (after prerequisite passes):** Create:
```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.12"

sphinx:
  configuration: docs/source/conf.py
  fail_on_warning: true   # remove if sphinx-build -W exit non-zero; see above

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
```
**Success criterion:** `python3 -c "import yaml; yaml.safe_load(open('.readthedocs.yml'))"` parses without error. If `fail_on_warning: true` was included, also verify `uv run sphinx-build -W docs/source docs/_build/html` exits 0.  
**Dependencies:** none  
**LOC:** ~15

---

#### MR-04: Ruff auto-fix pass
**Files:** `src/prxteinmpnn/**/*.py` (bulk automated)  
**Action:**
1. `uv run ruff check src/ --fix`
2. `uv run ruff format src/`
3. `uv run ruff check src/` — note residual count in commit body
4. `git diff --stat` — review changes in `inference/`, `types/`, `host/` for any `type: ignore` removals that might expose real type errors; if any surface, restore the ignore with a note
5. Commit: `style(ruff): apply auto-fix pass (closes lint known-debt)`

**Success criterion:** `uv run ruff check src/ 2>&1 | grep 'Found.*fixable' | wc -l` returns 0 (no auto-fixable errors remain). Non-auto-fixable residual is acceptable; document count in commit message body.  
**Dependencies:** none  
**LOC:** 0 manual; automated

---

#### MR-05: Clean up decoder.py stale TODO comment
**Files:** `src/prxteinmpnn/model/decoder.py` (modify, line 8)  
**Action:** Replace:
```python
# TODO(tech-debt): `.agents/TECHNICAL_DEBT.md` §6 — docstring / public API audit.
```
With:
```python
# NOTE: Public API surface audited Sprint 3 Wave 5. Docstrings complete as of refactor-full.
```
**Success criterion:** `grep -c 'TODO.*TECHNICAL_DEBT.*§6' src/prxteinmpnn/model/decoder.py` returns `0`.  
**Dependencies:** none  
**LOC:** 1 changed

---

#### MR-06: Clean up campaign.py dead-import comment
**Files:** `src/prxteinmpnn/host/campaign.py` (modify, lines 25-31)  
**Action:**
1. Run the **exact** pattern below (do NOT use the substring `campaign_manifest` — that matches function names like `plan_campaign_manifest` and will false-positive):
   ```bash
   grep -rE 'from prxteinmpnn\.run\.campaign_manifest|import prxteinmpnn\.run\.campaign_manifest' tests/ src/
   ```
   Expected: zero matches. If any matches exist, stop and escalate — something actually imports the dead module.
2. Remove the dead-import comment block:
```python
# TODO(TASK-2): wire up after bundle_builder move (TASK-5)
# from prxteinmpnn.run.campaign_manifest import (
#   build_manifest_row,
#   load_manifest,
#   validate_manifest_rows,
#   write_manifest,
# )
```
Replace with one line:
```python
# campaign manifest functions are implemented in this module (see build_manifest_row et al.)
```
**Success criterion:** `grep -c 'TODO(TASK-2)' src/prxteinmpnn/host/campaign.py` returns `0`.  
**Dependencies:** none  
**LOC:** net -5

---

### Wave B — After Wave A (2 sequential tasks)

#### MR-07: Verify driver.decode deprecation completeness
**Files:** `src/prxteinmpnn/inference/driver.py` (modify only if docstring is insufficient)  
**Action:**
1. Confirm `__all__` does NOT contain `"decode"` (recon confirmed; verify again after MR-04 ruff pass).
2. Read the deprecation docstring at lines ~80-96. If it does not show a concrete migration example, append:
```python
# Migration example:
#   plan = make_inference_plan(model, spec)
#   result = plan.decode(enc, bundle, key, config)
```
3. Do NOT change the `raise NotImplementedError(...)` — intentional.

**Success criterion:**
```bash
python -c "from prxteinmpnn.inference import driver; assert 'decode' not in driver.__all__"
python -c "
from prxteinmpnn.inference.driver import decode
try: decode(None,None,None,None,None,None,None)
except NotImplementedError: pass
else: raise AssertionError('expected NotImplementedError')
"
```
Both assertions pass.  
**Dependencies:** MR-04 (ruff may touch driver.py; apply docstring edit on top of ruff-formatted file)  
**LOC:** 0-5

---

#### MR-08: Close Debt #16 open reference
**Files:** any file containing `Debt.*#16` or `debt-16` or `#16.*UnconditionalDecode`  
**Action:**
1. `grep -r 'Debt.*16\|debt-16\|#16.*UnconditionalDecode' src/ tests/ .praxia/ --include='*.py' --include='*.md'`
2. If open references found, add `RESOLVED:` annotation or remove them.
3. If no references found, this task is a no-op — commit a one-line note under the "Closures" section below confirming the closure.

**Background:** Debt #16 suspected `UnconditionalDecodeStep` passed invalid inference kwargs to `Decoder`. Recon confirmed `src/prxteinmpnn/types/stages.py:219-253` is correct — kwargs match `Decoder.__call__` exactly. Debt was never formally closed.

**Success criterion:** `grep -rc 'Debt.*#16' src/ tests/` — every matching file returns 0 count OR shows a RESOLVED annotation.  
**Dependencies:** MR-05 (decoder.py might be a source; clean it first)  
**LOC:** 0-3

---

## Wave Summary

```
Wave A (parallel, no deps — 6 tasks):
  MR-01  src/prxteinmpnn/py.typed — create (0 LOC)
  MR-02  pyproject.toml [project.urls] — add (~4 LOC)
  MR-03  .readthedocs.yml — create (~15 LOC)
  MR-04  ruff auto-fix — automated (~0 manual LOC)
  MR-05  decoder.py:8 stale TODO — replace (1 LOC)
  MR-06  campaign.py:25-31 dead-import — remove (net -5 LOC)

Wave B (after Wave A — 2 tasks):
  MR-07  driver.decode deprecation verify (0-5 LOC)
  MR-08  Debt #16 open reference closure (0-3 LOC)
```

**Total:** 8 tasks, 2 waves, ~20 LOC net change (mostly automated).

---

## Test Gates

### After Wave A:
```bash
uv run pytest tests/ -q --ignore=tests/parity -x 2>&1 | tail -5
```
Expected: all pass, no collection errors.

### After Wave B (full gate):
```bash
# Tests
uv run pytest tests/ -q --ignore=tests/parity -x 2>&1 | tail -5
# Lint
uv run ruff check src/ 2>&1 | tail -3
# Type check (catches any type: ignore removals by MR-04 that exposed real gaps)
uv run ty check src/ 2>&1 | tail -10
# Docs build
uv run sphinx-build -W docs/source docs/_build/html 2>&1 | tail -10
# driver.__all__ invariant
python -c "from prxteinmpnn.inference import driver; assert 'decode' not in driver.__all__"
# py.typed in wheel
uv build && python -c "
import zipfile, pathlib
whl = next(pathlib.Path('dist').glob('*.whl'))
z = zipfile.ZipFile(whl)
assert any('py.typed' in n for n in z.namelist()), 'py.typed not in wheel'
print('wheel ok')
"
# project.urls in metadata
uv run python -c "import importlib.metadata; m = importlib.metadata.metadata('prxteinmpnn'); urls = [v for k,v in m.items() if k=='Project-URL']; assert urls, 'no Project-URL'; print(urls)"
```

---

## Risk Table

| Risk | Severity | Mitigation |
|------|----------|-----------|
| `ruff --fix` removes a `type: ignore` suppressing a real gap | Medium | Run `git diff` after fix; review `inference/`, `types/`, `host/` changes; restore ignore + note if a type error surfaces. `uv run ty check src/` in Wave B gate will catch any escaping type errors. |
| `.readthedocs.yml` `fail_on_warning: true` fails RTD build | Medium | MR-03 requires running `sphinx-build -W` before writing the YAML; if non-zero, drop `fail_on_warning` and defer. `conf.py` uses `main_doc` (non-standard key) — may warn on Sphinx 4+. |
| `py.typed` not in built wheel (ships in source but not in wheel) | High | MR-01 explicitly updates `[tool.setuptools.package-data]` to include `"py.typed"`; Wave B gate runs `uv build` and verifies it via zipfile inspection. |
| `[project.urls]` uses placeholder URL | Low | Fixer runs `git remote get-url origin` first; placeholder + `# TODO` acceptable. |
| `campaign.py` grep pattern false-positives on function names | Medium | MR-06 uses exact import-path pattern (`from prxteinmpnn.run.campaign_manifest`) not substring — this avoids matching `plan_campaign_manifest` and `write_campaign_manifest` function names in tests. |
| MR-07 docstring edit conflicts with MR-04 ruff changes | Low | Do MR-04 before MR-07; apply on top of ruff-formatted file. |

---

## Deferral Register

| Item | File:Line | Rationale |
|------|-----------|-----------|
| COMP-533: `make_stage_set` in `sample.py` | `sampling/sample.py:184` | SamplerFn protocol constraint; cannot move without changing protocol signature. Deferred post-merge. |
| Training module `NotImplementedError` | `training/**` | Entire training module intentionally unimplemented on this branch. |
| Bucketing TODO | `tiling/planner.py:16` | Future work; `BatchPlanner.plan()` production-ready as-is. |
| IREE/WASM targets | downstream | Requires `jax.export` artifact; not a merge gate. |
| Non-auto-fixable ruff residual | `src/**` | Manual annotation drift; acceptable for merge; count documented in MR-04 commit. |
| Debt #17 (`driver.decode` router) | `inference/driver.py:71` | Intentional deprecation; `decode` raises `NotImplementedError` and is excluded from `__all__`. A router would silently mask the deprecation and mislead callers migrating to `InferencePlan.decode()`. |

---

## Closures

- **Debt #16 (UnconditionalDecodeStep kwargs):** Confirmed RESOLVED. Recon 260527 verified `types/stages.py:219-253` kwargs match `Decoder.__call__` exactly. No open references found in src/ or tests/.

---

## Done Criteria

- [ ] `src/prxteinmpnn/py.typed` exists at HEAD
- [ ] `py.typed` present in built wheel (verified by zipfile inspection via `uv build`)
- [ ] `pyproject.toml` has `[project.urls]` with at least `repository` and `documentation` keys
- [ ] `.readthedocs.yml` exists at repo root, valid YAML; contains `fail_on_warning: true` if `sphinx-build -W` exited 0
- [ ] `uv run ruff check src/` reports zero auto-fixable errors (or residual count documented in MR-04 commit)
- [ ] `uv run ty check src/` exits 0 (no new type errors introduced by MR-04)
- [ ] `uv run pytest tests/ -q --ignore=tests/parity -x` exits 0
- [ ] `uv run sphinx-build -W docs/source docs/_build/html` exits 0 (or `fail_on_warning` deferred with follow-up filed)
- [ ] `decoder.py:8` no longer references `.agents/TECHNICAL_DEBT.md §6`
- [ ] `campaign.py:25` `TODO(TASK-2)` comment removed
- [ ] Debt #16 has no open tracking reference in `src/` or `tests/`
- [ ] All tasks MR-01 through MR-08 committed with task IDs in commit messages

---

## References

- `src/prxteinmpnn/inference/driver.py:71-96` — `decode()` deprecation shim; `__all__` at ~line 99
- `src/prxteinmpnn/host/campaign.py:25-31` — dead-import TODO block
- `src/prxteinmpnn/tiling/planner.py:16` — bucketing TODO (deferred)
- `src/prxteinmpnn/model/decoder.py:8` — stale TODO reference to worktree-only debt doc
- `pyproject.toml` — `[project]` section; missing `[project.urls]`
- `docs/source/conf.py` — Sphinx config exists; RTD explicit config missing
- `tests/tiling/test_planner_phase0.py` — BatchPlanner.plan() test coverage
- CLAUDE.md known-debt: 403 ruff fixable errors in `src/`
