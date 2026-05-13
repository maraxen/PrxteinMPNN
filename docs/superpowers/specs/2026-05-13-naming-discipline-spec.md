# Naming Discipline Refactor — Design Spec

**Date**: 2026-05-13
**Kind**: refactor (7 fixer tasks)
**Related**: `docs/superpowers/specs/2026-05-13-codebase-analysis-design.md`, `memory/feedback_naming.md`

---

## Motivation

`state_vmap_exact` appearing in module names and function names is the same class of error as `Payload` appearing in public API method names: it leaks a JAX implementation detail (vmap-over-states-with-full-attention) into the conceptual surface of the API. Users and contributors think in two orthogonal axes:

| Axis | Values |
|---|---|
| Operation | `sample` / `score` |
| Strategy | `exact` (full multistate attention, implemented via vmap over states) / `scan` (sequential `lax.scan`) |
| Modality | implicit in file name: protein (no suffix) / ligand (`_ligand`) |

The verb `vmap` must disappear from every public symbol name. It may remain in internal docstrings as an implementation note.

`multi_state_sampling.py` is inconsistent with the already-landed `multistate_stack.py` — the `multi_state_` underscore is a separate naming error this refactor corrects in the same pass.

## Goal

Rename five model-layer files and their exported functions so that every public-facing name describes what the operation does (`sample_ar_exact`, `score_exact_ligand`) rather than how it does it (`state_vmap_exact`), with zero broken import sites at HEAD after each commit lands.

## Scope

**In scope:**
1. Rename 4 inference files, drop `mpnn_` prefix and `state_vmap_exact` infix, rename their primary exported functions
2. Rename `multi_state_sampling.py` → `multistate_sampling.py` (underscore inconsistency)
3. Delete `features_direct.py` and its test (confirmed unused in production code)
4. Document `DecoderLayerJ` with a one-line docstring tracing the `J` suffix
5. Add a two-axis conceptual documentation block to `model/__init__.py`
6. Update every import site in source, tests, scripts, and data files

**Out of scope:** see Out of Scope section below.

---

## Complete Rename Table

### File renames

| Current path (`src/prxteinmpnn/`) | New path | Rationale |
|---|---|---|
| `model/mpnn_autoregressive_scan.py` | `model/ar_scan.py` | Drops `mpnn_` (redundant inside model package); `ar_scan` = operation+strategy |
| `model/mpnn_autoregressive_state_vmap_exact.py` | `model/ar_exact.py` | Drops `mpnn_` and `state_vmap_` infix; `ar_exact` = autoregressive + exact strategy |
| `model/mpnn_autoregressive_state_vmap_exact_ligand.py` | `model/ar_exact_ligand.py` | Same + ligand modality suffix |
| `model/mpnn_scoring_state_vmap_exact_ligand.py` | `model/score_exact_ligand.py` | Operation is `score` not `ar`; `exact_ligand` = strategy + modality |
| `model/multi_state_sampling.py` | `model/multistate_sampling.py` | Consistency with already-landed `multistate_stack.py` |

### Function renames (inside renamed files)

| Current name | New name | File |
|---|---|---|
| `run_autoregressive_scan` | `run_sample_ar_scan` | `ar_scan.py` |
| `run_sample_autoregressive_state_vmap_exact` | `run_sample_ar_exact` | `ar_exact.py` |
| `run_sample_autoregressive_state_vmap_exact_ligand` | `run_sample_ar_exact_ligand` | `ar_exact_ligand.py` |
| `run_score_unconditional_state_vmap_exact_ligand` | `run_score_unconditional_exact_ligand` | `score_exact_ligand.py` |
| `run_score_conditional_state_vmap_exact_ligand` | `run_score_conditional_exact_ligand` | `score_exact_ligand.py` |

### Private method renames (on model classes)

| Class | Current | New | File |
|---|---|---|---|
| `PrxteinMPNN` | `_run_autoregressive_scan` | `_run_sample_ar_scan` | `model/mpnn.py` |
| `PrxteinLigandMPNN` | `_run_autoregressive_scan` | `_run_sample_ar_scan` | `model/ligand_mpnn.py` |

These methods are `_`-prefixed so no external callers exist. Keeping the old name on the private method while the module function is renamed creates exactly the asymmetry being fixed.

### Deletions

| Path | Action |
|---|---|
| `src/prxteinmpnn/model/features_direct.py` | Delete — no production import site (see Discovery D6) |
| `tests/model/test_features_direct.py` | Delete — test for deleted module |

### Note on `test_state_vmap_exact_jit.py`

`tests/sampling/test_state_vmap_exact_jit.py` contains `state_vmap_exact` in its name but tests the JIT boundary / host prep logic in `sampling/state_vmap_prep.py`, not the renamed model functions. Renaming this test file is a **follow-on task, not part of this spec**, because: (a) `CLAUDE.md` fast-test commands reference it by exact name, and (b) its content must be coordinated with the eventual `sampling/state_vmap_prep.py` rename. Do not rename this file here.

---

## Discovery Work (Implementer Must Run Before Each Task)

### D1 — References to `mpnn_autoregressive_scan`

```bash
grep -r "mpnn_autoregressive_scan\|run_autoregressive_scan\|_run_autoregressive_scan" \
  src/ tests/ scripts/ docs/ --include="*.py" --include="*.json" --include="*.md" -l
```

Expected files: `model/mpnn_autoregressive_scan.py`, `model/mpnn.py`, `model/ligand_mpnn.py`, doc files.

### D2 — References to `mpnn_autoregressive_state_vmap_exact` (protein only)

```bash
grep -r "mpnn_autoregressive_state_vmap_exact\b\|run_sample_autoregressive_state_vmap_exact\b" \
  src/ tests/ scripts/ --include="*.py" -l
```

Expected files: `model/mpnn_autoregressive_state_vmap_exact.py`, `model/mpnn.py`.

### D3 — References to `mpnn_autoregressive_state_vmap_exact_ligand`

```bash
grep -r "mpnn_autoregressive_state_vmap_exact_ligand\|run_sample_autoregressive_state_vmap_exact_ligand" \
  src/ tests/ scripts/ --include="*.py" -l
```

Expected files: `model/mpnn_autoregressive_state_vmap_exact_ligand.py`, `model/ligand_mpnn.py` (lazy import inside method body at ~line 1217).

### D4 — References to `mpnn_scoring_state_vmap_exact_ligand`

```bash
grep -r "mpnn_scoring_state_vmap_exact_ligand\|run_score_unconditional_state_vmap_exact_ligand\|run_score_conditional_state_vmap_exact_ligand" \
  src/ tests/ scripts/ --include="*.py" -l
```

Expected files: `model/mpnn_scoring_state_vmap_exact_ligand.py`, `model/ligand_mpnn.py` (two lazy imports at ~lines 1039 and ~1122).

### D5 — References to `multi_state_sampling`

```bash
grep -r "multi_state_sampling\|from prxteinmpnn\.model\.multi_state_sampling" \
  src/ tests/ scripts/ docs/ \
  --include="*.py" --include="*.json" --include="*.md" --include="*.html" --include="*.ipynb" -l
```

Expected files: `model/multi_state_sampling.py`, `model/_shared.py`, `tests/model/test_multi_state_sampling.py`, `tests/parity/parity_matrix.json`, `scripts/260410/verify_weighted_multistate.py`, and docs.

### D6 — Confirm `features_direct.py` has no production import site (gate before deletion)

```bash
grep -r "features_direct\|ProteinFeaturesDirect" src/ tests/ scripts/ --include="*.py" -l
```

**Gate for deletion:** Output must be exactly these two lines:
```
src/prxteinmpnn/model/features_direct.py
tests/model/test_features_direct.py
```

If ANY other file appears, stop. Do not delete. File a follow-on task.

---

## Fixer Tasks

### Fixer 1 — Rename `ar_scan.py` + update import sites (~30 LOC)

**Files:** `model/mpnn_autoregressive_scan.py` (rename), `model/mpnn.py`, `model/ligand_mpnn.py`

Changes:
- `git mv src/prxteinmpnn/model/mpnn_autoregressive_scan.py src/prxteinmpnn/model/ar_scan.py`
- In `ar_scan.py`: `def run_autoregressive_scan(` → `def run_sample_ar_scan(`; update module docstring to remove `mpnn_` mention
- In `mpnn.py`: `from prxteinmpnn.model.mpnn_autoregressive_scan import run_autoregressive_scan` → `from prxteinmpnn.model.ar_scan import run_sample_ar_scan`; `def _run_autoregressive_scan(` → `def _run_sample_ar_scan(`; update the `:mod:` docstring reference; `return run_autoregressive_scan(` → `return run_sample_ar_scan(`; all `self._run_autoregressive_scan(` → `self._run_sample_ar_scan(`
- In `ligand_mpnn.py`: `def _run_autoregressive_scan(` → `def _run_sample_ar_scan(`; all `self._run_autoregressive_scan(` → `self._run_sample_ar_scan(`

Gate:
```bash
cd /home/marielle/projects/tev_design/prxteinmpnn
find src tests scripts -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
grep -r "mpnn_autoregressive_scan\|run_autoregressive_scan\b\|_run_autoregressive_scan" src/ tests/ scripts/ --include="*.py" && exit 1 || echo "CLEAN"
PYTHONPATH=src uv run pytest tests/sampling/test_sample.py tests/sampling/test_state_vmap_exact_jit.py -q
```

Commit: `refactor(model): rename ar_scan module, run_autoregressive_scan → run_sample_ar_scan`

### Fixer 2 — Rename `ar_exact.py` + update import sites (~20 LOC)

**Files:** `model/mpnn_autoregressive_state_vmap_exact.py` (rename), `model/mpnn.py`

Changes:
- `git mv src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact.py src/prxteinmpnn/model/ar_exact.py`
- In `ar_exact.py`: `def run_sample_autoregressive_state_vmap_exact(` → `def run_sample_ar_exact(`; module docstring title line: `Stacked wave-parallel autoregressive sampler for ProteinMPNN (exact strategy).`; add implementation note: `Implemented via jax.vmap over states with full attention.`
- In `mpnn.py`: replace import block with `from prxteinmpnn.model.ar_exact import run_sample_ar_exact`; delegation call → `return run_sample_ar_exact(`

Gate: as above with corresponding greps. Commit: `refactor(model): rename ar_exact module, run_sample_ar_exact (protein)`

### Fixer 3 — Rename `ar_exact_ligand.py` + update import sites (~20 LOC)

**Files:** `model/mpnn_autoregressive_state_vmap_exact_ligand.py` (rename), `model/ligand_mpnn.py`

Changes: `git mv` to `ar_exact_ligand.py`; function rename `run_sample_autoregressive_state_vmap_exact_ligand` → `run_sample_ar_exact_ligand`; update lazy import block at ~line 1217 in `ligand_mpnn.py`.

Commit: `refactor(model): rename ar_exact_ligand module, run_sample_ar_exact_ligand`

### Fixer 4 — Rename `score_exact_ligand.py` + update import sites (~25 LOC)

**Files:** `model/mpnn_scoring_state_vmap_exact_ligand.py` (rename), `model/ligand_mpnn.py`

Changes: `git mv` to `score_exact_ligand.py`; rename `run_score_unconditional_state_vmap_exact_ligand` → `run_score_unconditional_exact_ligand`; rename `run_score_conditional_state_vmap_exact_ligand` → `run_score_conditional_exact_ligand`; update both lazy import blocks in `ligand_mpnn.py` (~lines 1039, 1122).

Commit: `refactor(model): rename score_exact_ligand module, drop state_vmap from scoring fns`

### Fixer 5 — Rename `multistate_sampling.py` + update all sites (~15 LOC, ~12 doc lines)

**Files:**
- `model/multi_state_sampling.py` → `multistate_sampling.py`
- `model/_shared.py` (import update)
- `tests/model/test_multi_state_sampling.py` (import line update; filename NOT changed here)
- `scripts/260410/verify_weighted_multistate.py` (import line)
- `tests/parity/parity_matrix.json` (`code_paths` array)
- Docs: `MULTI_STATE_IMPLEMENTATION.md`, `CURRENT_STATUS.md`, `codebase_analysis/*.md`, `superpowers/plans/2026-05-08-*.md`, `parity/parity_report.md`, `parity/parity_report.html`

Commit: `refactor(model): multi_state_sampling → multistate_sampling (consistency with multistate_stack)`

### Fixer 6 — Delete `features_direct.py` and its test (~200 LOC removed)

**Precondition:** Run D6 grep. Output must be exactly the two expected files.

Changes:
- `git rm src/prxteinmpnn/model/features_direct.py`
- `git rm tests/model/test_features_direct.py`

Commit: `refactor(model): delete ProteinFeaturesDirect — unused in production code`

### Fixer 7 — Document `DecoderLayerJ` + add two-axis block in `model/__init__.py` (~30 LOC added)

**Part A — `model/decoder.py`** — replace `DecoderLayerJ` docstring with documentation tracing the `J` suffix to the upstream LigandMPNN `DecLayerJ` reference; describe it as a decoder layer for ligand-atom (Y) context encoding, used as `y_context_encoder_layers` in `Packer`. Operates on a 3-D node tensor `[L, M, D]` and edge tensor `[L, M, M, D]` where `M` is the ligand context neighborhood size.

**Part B — `model/__init__.py`** — replace module docstring with:

```
Neural network architectures for PrxteinMPNN.

Public inference functions are organized along two conceptual axes:

    Operation × Strategy

    Operation: sample  — autoregressive sequence generation
               score   — log-probability evaluation of a given sequence

    Strategy:  exact   — full multistate attention over all conformational
                         states simultaneously (implemented via vmap over states)
               scan    — sequential per-position decode using lax.scan
                         (lower peak memory; single-state only)

    Modality:  implicit in the file name suffix:
               (no suffix) — protein-only model (PrxteinMPNN)
               _ligand     — ligand-aware model (PrxteinLigandMPNN)

File-to-concept mapping:
    model/ar_scan.py              sample × scan  × protein
    model/ar_exact.py             sample × exact × protein
    model/ar_exact_ligand.py      sample × exact × ligand
    model/score_exact_ligand.py   score  × exact × ligand

The verb ``vmap`` does not appear in any public name; it is an implementation
detail documented in the module-level docstrings of the individual files.
```

Commit: `docs(model): document DecoderLayerJ provenance; add two-axis conceptual block to __init__`

---

## Commit Sequence

Each commit must leave the fast test suite green. The file rename and its import updates are **always in the same commit** — no commit may leave a broken `HEAD~1`.

```
1. refactor(model): rename ar_scan module, run_autoregressive_scan → run_sample_ar_scan
2. refactor(model): rename ar_exact module, run_sample_ar_exact (protein)
3. refactor(model): rename ar_exact_ligand module, run_sample_ar_exact_ligand
4. refactor(model): rename score_exact_ligand module, drop state_vmap from scoring fns
5. refactor(model): multi_state_sampling → multistate_sampling (consistency with multistate_stack)
6. refactor(model): delete ProteinFeaturesDirect — unused in production code
7. docs(model): document DecoderLayerJ provenance; add two-axis conceptual block to __init__
```

Commits 1–4 are independent of each other and of 5–7. Commit 6 must be preceded by a passing D6 grep. Commit 7 may land in any order but reads best last.

---

## Risks

| Risk | Detection | Mitigation |
|---|---|---|
| `tests/parity/parity_matrix.json` `code_paths` not updated (JSON, not Python — linters miss it) | `grep "multi_state_sampling" tests/parity/parity_matrix.json` returns a match | D5 surfaces this file explicitly; Fixer 5 gate tests it |
| `ligand_mpnn.py` has two separate lazy import blocks for the scoring file | Full test suite exercises both unconditional and conditional scoring paths | D4 with `-l` confirms both in scope |
| `__pycache__` shadowing renamed modules | `ImportError` at test time | Each gate prefaced with `find ... -name __pycache__ -exec rm -rf {} +` |
| `features_direct.py` acquired a new import site since spec authoring | D6 grep returns more than 2 files | Re-run D6 immediately before `git rm` |
| `DecoderLayerJ` etymology incorrect | Reader review | Verify with `git log --follow`; fall back to conservative wording + TODO |

---

## Final Merge Gates

All must pass before the PR is merged:

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn

# 1. No stale module names in code, tests, scripts, or shipped data
grep -rE "mpnn_autoregressive_scan|mpnn_autoregressive_state_vmap_exact|mpnn_scoring_state_vmap_exact|multi_state_sampling" \
  src/ tests/ scripts/ --include="*.py" --include="*.json" --include="*.ipynb" && exit 1 || echo "PASS"

# 1b. Doc cleanup (separate gate so historical plan notes don't false-positive)
# Exhaustively list every doc with a stale reference; Fixer 5 must update them all:
grep -rln "multi_state_sampling\|mpnn_autoregressive\|mpnn_scoring_state_vmap" docs/

# 2. No stale function names. Both the module-level fn AND the private method must be caught.
# Note: \b after underscore does NOT bound (underscore is a word char), so we list both forms.
grep -rE "(^|[^_])run_autoregressive_scan\b|_run_autoregressive_scan|run_sample_autoregressive_state_vmap_exact|run_score_unconditional_state_vmap_exact|run_score_conditional_state_vmap_exact" \
  src/ tests/ scripts/ --include="*.py" && exit 1 || echo "PASS"

# 3. parity_matrix.json updated
grep "multi_state_sampling" tests/parity/parity_matrix.json && exit 1 || echo "PASS"

# 4. Deleted files are gone
test ! -f src/prxteinmpnn/model/features_direct.py && test ! -f tests/model/test_features_direct.py || exit 1

# 5. Fast suite passes
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/model/test_ligand_wave_parallel.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  tests/model/test_multi_state_sampling.py \
  -q
```

---

## Out of Scope

- Moving files into `model/_inference/` subdirectory — separate spec
- Renaming `sampling/state_vmap_prep.py`, `sampling/state_vmap_payload_logits.py`, or `build_state_vmap_exact_stacks` — separate sampling-layer naming spec
- Renaming `tests/sampling/test_state_vmap_exact_jit.py` — CI implications
- Renaming `tests/model/test_multi_state_sampling.py` — follow-on
- Any changes to `run/sampling.py` or its decomposition
- The protocol seam refactor (`protocols.py`)
- Any behavior changes or API surface changes beyond symbol names
- Backwards-compatibility shims (sole maintainer; none required)

---

**Key verified facts:**

- `ligand_mpnn.py` uses **lazy imports** (inline `from ... import` inside method bodies) for all three ligand inference files — Fixers 3 and 4 must update inside method bodies.
- `DecoderLayerJ` is used as `y_context_encoder_layers: tuple[DecoderLayerJ, ...]` in `model/packer.py` — confirming Y-context etymology.
- `ProteinFeaturesDirect` grep returns exactly two files: implementation and test. Deletion safe.
- `tests/parity/parity_matrix.json` contains the string `"src/prxteinmpnn/model/multi_state_sampling.py"` in a `code_paths` data field — most likely miss for a fixer working from Python imports alone.
