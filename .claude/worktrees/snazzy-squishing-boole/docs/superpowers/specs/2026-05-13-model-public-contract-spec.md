# `prxteinmpnn.model` Public Contract — Design Spec

**Date**: 2026-05-13
**Kind**: refactor (5 fixer tasks)
**Related**: `2026-05-13-naming-discipline-spec.md` (DEPENDENCY — must land first; see §7)

---

## 1. Motivation

`src/prxteinmpnn/model/` contains 20 Python files but no explicit statement of which are public contract and which are implementation details. The consequence is two-fold:

1. **Refactor unsafety.** Any rename or restructuring of an internal file risks breaking callers that were never supposed to import it — or introduces conservatism that leaves internal files untouched because their import graph is unclear.
2. **Onboarding opacity.** A contributor cannot determine from the directory listing or `__init__.py` whether `mpnn_core.py`, `ligand_tiling.py`, or `_shared.py` are API surfaces or private machinery.

This spec makes the boundary explicit and permanent: one `__init__.py` with a declared `__all__`, internal machinery in `model/_inference/`, and a `model/README.md` that explains the rule.

---

## 2. Goal

Establish `model/__init__.py` as the single, documented source of truth for the public API of `prxteinmpnn.model`, moving internal-only implementation files into `model/_inference/` and resolving all import leaks from external code.

---

## 3. Module Triage Table

**Classification key:**
- PUBLIC — part of the stable contract; re-exported from `model/__init__.py`
- INTERNAL — used only within `model/` siblings; not for external callers
- LEAKED — currently imported by code outside `model/` that should not be; resolution noted
- INTERNAL-DEAD — no production imports in `src/`

Evidence was gathered by running:
```bash
grep -rn "from prxteinmpnn\.model\." src/ tests/ scripts/
grep -rn "from prxteinmpnn\.model import" src/ tests/ scripts/
# Also notebooks (low frequency but they reference public symbols):
grep -rn "prxteinmpnn\.model" --include="*.ipynb" .
```

**Confirmed script-layer callers (must NOT be missed when moving files):**

| Script | Imports | Action on `_inference/` move |
|---|---|---|
| `scripts/convert_weights.py` | `Packer`, `ProteinFeaturesLigand` | None — both stay at `model/` root |
| `scripts/diag_packer_parity.py` | `Packer` | None — `Packer` stays |
| `scripts/collect_parity_evidence.py` | `Packer` | None |
| `scripts/260410/verify_weighted_multistate.py` | `multistate_sampling` functions | **Update import path** to `prxteinmpnn.model._inference.multistate_sampling` |
| `colab_training_test.ipynb` | `DiffusionPrxteinMPNN` | Optional cleanup: prefer `from prxteinmpnn.model import DiffusionPrxteinMPNN` after Fixer 1 |

| File | In `__all__`? | Classification | Evidence / Resolution |
|---|---|---|---|
| `__init__.py` | n/a | n/a | The file to be updated |
| `mpnn.py` | Yes (`PrxteinMPNN`) | PUBLIC | Imported by `sampling/`, `run/`, `scoring/`, `io/weights.py`, `training/`, `utils/types.py` (TYPE_CHECKING), and ~25 test files |
| `ligand_mpnn.py` | Yes (`PrxteinLigandMPNN`) | PUBLIC | Imported by `sampling/conditional_logits.py`, `sampling/unconditional_logits.py`, `sampling/state_vmap_payload_logits.py`, `scoring/score.py`, ~8 tests |
| `diffusion_mpnn.py` | **No** | PUBLIC — promote | Imported by `io/weights.py`, `training/trainer.py`, `training/train_diffusion.py`, tests |
| `packer.py` | **No** | PUBLIC — promote | Imported by `io/weights.py`, `tests/parity/test_parity_ci.py`, `tests/parity/test_packer_parity.py` |
| `encoder.py` | Yes (`Encoder`, `EncoderLayer`) | PUBLIC | `EncoderLayer` used in `packer.py`; imported by `tests/parity/test_jax_pytorch_parity.py`; `encoder_forward_with_int_neighbors` leaks to one test |
| `decoder.py` | Yes (`Decoder`, `DecoderLayer`) | PUBLIC | `DecoderLayer` used in `packer.py`; imported by tests |
| `features.py` | Yes (`ProteinFeatures`) | PUBLIC | Used in model encoding path; `top_k` accessed by `tests/debug_proxide_mpnn_mismatch.py` (debug script) |
| `capabilities.py` | Yes | PUBLIC | Imported by `run/averaging.py`, tests |
| `multistate_stack.py` | Yes | PUBLIC | Imported by `sampling/`, `scoring/score.py`, `_inference/mpnn_scoring_state_vmap_exact_ligand.py`, tests |
| `mpnn_autoregressive_scan.py` | No | INTERNAL | Only `mpnn.py`. Move to `_inference/ar_scan.py` |
| `mpnn_autoregressive_state_vmap_exact.py` | No | INTERNAL | Only `mpnn.py`. Move to `_inference/ar_exact.py` |
| `mpnn_autoregressive_state_vmap_exact_ligand.py` | No | INTERNAL | Only `ligand_mpnn.py` (inline PLC0415). Move to `_inference/ar_exact_ligand.py` |
| `mpnn_scoring_state_vmap_exact_ligand.py` | No | INTERNAL | Only `ligand_mpnn.py` (inline). Move to `_inference/scoring_exact_ligand.py` |
| `mpnn_core.py` | No | INTERNAL | All callers within `model/`. Move to `_inference/mpnn_core.py` |
| `_shared.py` | No | INTERNAL | All callers within `model/`. Move to `_inference/_shared.py` |
| `multi_state_sampling.py` | No | LEAKED | Test imports it; move to `_inference/` + update test import path |
| `ligand_features.py` | No | LEAKED | Test imports `ProteinFeaturesLigand`; keep at root, annotate test |
| `ligand_tiling.py` | No | LEAKED | Test imports `map_chunks_axis0`; keep at root, annotate test |
| `features_direct.py` | No | INTERNAL-DEAD | Zero `src/` imports. Move to `_inference/`; update test |

**Leaked-symbol resolution:**

| Symbol | Leaked to | Resolution |
|---|---|---|
| `encoder_forward_with_int_neighbors` (`encoder.py`) | `tests/pipeline/test_encoder_state_fn.py` | Refactor test to invoke through `PrxteinMPNN` public interface; do NOT promote |
| `multi_state_sampling` functions | `tests/model/test_multi_state_sampling.py` | Update test import to `_inference/multi_state_sampling` |
| `ProteinFeaturesLigand` (`ligand_features.py`) | `tests/model/test_ligand_feature_tiling.py` | Keep direct import; add `# internal import` comment |
| `map_chunks_axis0` (`ligand_tiling.py`) | `tests/model/test_ligand_feature_tiling.py` | Same — comment only |
| `ProteinFeaturesDirect` (`features_direct.py`) | `tests/model/test_features_direct.py` | Update test import |
| `top_k` (`features.py`) | `tests/debug_proxide_mpnn_mismatch.py` | Debug script; add comment; do NOT promote |

---

## 4. Target Directory Structure

```
src/prxteinmpnn/model/
├── __init__.py              # canonical public API; __all__ = 15 symbols
├── README.md                # explains public/internal rule
├── capabilities.py          # PUBLIC
├── decoder.py               # PUBLIC
├── diffusion_mpnn.py        # PUBLIC (added to __all__)
├── encoder.py               # PUBLIC
├── features.py              # PUBLIC
├── ligand_features.py       # internal-only (NOT in __all__); kept at root
├── ligand_mpnn.py           # PUBLIC
├── ligand_tiling.py         # internal-only (NOT in __all__); kept at root
├── mpnn.py                  # PUBLIC
├── multistate_stack.py      # PUBLIC
├── packer.py                # PUBLIC (added to __all__)
└── _inference/
    ├── __init__.py          # contains: __all__ = []  (explicit-empty contract)
    ├── _shared.py
    ├── ar_exact.py
    ├── ar_exact_ligand.py
    ├── ar_scan.py
    ├── mpnn_core.py
    ├── multistate_sampling.py    # post-rename name; NOT multi_state_sampling
    └── score_exact_ligand.py     # matches function names; NOT scoring_exact_ligand
```

**Filename reconciliation with `2026-05-13-naming-discipline-spec.md`:**

- The internal file is `score_exact_ligand.py` (NOT `scoring_exact_ligand.py`). This matches the function names `run_score_unconditional_exact_ligand` and `run_score_conditional_exact_ligand`.
- The internal file is `multistate_sampling.py` (NOT `multi_state_sampling.py`). The naming-discipline spec already standardized on `multistate_*` (no underscore in prefix); moving it into `_inference/` does not revert that decision.
- `features_direct.py` is **not** in `_inference/` — the naming-discipline spec (Fixer 6) deletes it outright as unused. This spec defers to that deletion.

`ligand_features.py` and `ligand_tiling.py` remain at `model/` root because moving them requires restructuring `ligand_mpnn.py`'s imports — separate follow-on. The boundary rule is: **not in `__all__` = not public, regardless of physical location.**

`_inference/__init__.py` is intentionally empty.

---

## 5. Public API Declaration

After Task 1, `model/__init__.py` will contain:

```python
"""Public API for prxteinmpnn.model.

Public contract (everything in __all__):
- Model classes: PrxteinMPNN, PrxteinLigandMPNN, DiffusionPrxteinMPNN, Packer
- Encoder/decoder layers: Encoder, EncoderLayer, Decoder, DecoderLayer
- Feature extraction: ProteinFeatures
- Capability introspection: ModelCapabilities, PRXTEIN_MPNN_CAPABILITIES, PRXTEIN_LIGAND_MPNN_CAPABILITIES
- Multistate helpers: gather_flat_to_stack, scatter_stack_to_flat

Internal: files in model/_inference/ are implementation details.
Importing from prxteinmpnn.model._inference.* outside of model/ is unsupported.
"""

from __future__ import annotations

from .capabilities import (
    PRXTEIN_LIGAND_MPNN_CAPABILITIES,
    PRXTEIN_MPNN_CAPABILITIES,
    ModelCapabilities,
)
from .decoder import Decoder, DecoderLayer
from .diffusion_mpnn import DiffusionPrxteinMPNN
from .encoder import Encoder, EncoderLayer
from .features import ProteinFeatures
from .ligand_mpnn import PrxteinLigandMPNN
from .mpnn import PrxteinMPNN
from .multistate_stack import gather_flat_to_stack, scatter_stack_to_flat
from .packer import Packer

__all__ = [
    "PrxteinMPNN", "PrxteinLigandMPNN", "DiffusionPrxteinMPNN", "Packer",
    "Encoder", "EncoderLayer", "Decoder", "DecoderLayer",
    "ProteinFeatures",
    "ModelCapabilities", "PRXTEIN_MPNN_CAPABILITIES", "PRXTEIN_LIGAND_MPNN_CAPABILITIES",
    "gather_flat_to_stack", "scatter_stack_to_flat",
]
```

15 symbols total (up from 11).

---

## 6. Fixer Tasks

### Fixer 1 — Promote `DiffusionPrxteinMPNN` and `Packer` to `__all__` (~25 LOC)

Add imports and `__all__` entries. Replace docstring.

Gate:
```bash
uv run python -c "
from prxteinmpnn.model import (
    PrxteinMPNN, PrxteinLigandMPNN, DiffusionPrxteinMPNN, Packer,
    Encoder, EncoderLayer, Decoder, DecoderLayer,
    ProteinFeatures, ModelCapabilities,
    PRXTEIN_MPNN_CAPABILITIES, PRXTEIN_LIGAND_MPNN_CAPABILITIES,
    gather_flat_to_stack, scatter_stack_to_flat,
)
print('all 15 public symbols importable')
"
uv run ruff check src/prxteinmpnn/model/__init__.py
```

### Fixer 2 — Resolve test leaks and annotate internal imports (~40 LOC across 4 files)

- `tests/pipeline/test_encoder_state_fn.py` — refactor to call through `PrxteinMPNN` public method; fall back to `# internal import: not public API` annotation if no public method covers the path
- `tests/model/test_ligand_feature_tiling.py` — add `# Internal imports: not public API` block comment
- `tests/debug_proxide_mpnn_mismatch.py` — add `# internal symbol` comment inline
- Test import paths for `multi_state_sampling` and `features_direct` get updated in Fixer 3 (path moves happen there)

### Fixer 3 — Create `model/_inference/` and move 8 internal files (~80 LOC of import edits, 8 git mv)

**Precondition:** Naming-discipline spec has landed.

Git mv operations (assuming naming-discipline spec has landed; filenames as renamed there):
```bash
mkdir -p src/prxteinmpnn/model/_inference
printf '"""Internal inference machinery. Not public API."""\n\n__all__: list[str] = []\n' \
  > src/prxteinmpnn/model/_inference/__init__.py

git mv src/prxteinmpnn/model/ar_scan.py            src/prxteinmpnn/model/_inference/ar_scan.py
git mv src/prxteinmpnn/model/ar_exact.py           src/prxteinmpnn/model/_inference/ar_exact.py
git mv src/prxteinmpnn/model/ar_exact_ligand.py    src/prxteinmpnn/model/_inference/ar_exact_ligand.py
git mv src/prxteinmpnn/model/score_exact_ligand.py src/prxteinmpnn/model/_inference/score_exact_ligand.py
git mv src/prxteinmpnn/model/mpnn_core.py          src/prxteinmpnn/model/_inference/mpnn_core.py
git mv src/prxteinmpnn/model/_shared.py            src/prxteinmpnn/model/_inference/_shared.py
git mv src/prxteinmpnn/model/multistate_sampling.py src/prxteinmpnn/model/_inference/multistate_sampling.py
# features_direct.py is NOT moved — it was deleted by the naming-discipline spec.
```

Update import paths in (full list, do NOT skip `scripts/`):
- `src/prxteinmpnn/model/mpnn.py` — lines for `_shared`, `ar_scan`, `ar_exact`, `mpnn_core`
- `src/prxteinmpnn/model/ligand_mpnn.py` — top imports + inline PLC0415 imports at ~1039, 1122, 1217
- All moved files in `_inference/` — fix relative imports among them
- `tests/model/test_multi_state_sampling.py` (if not already renamed)
- `scripts/260410/verify_weighted_multistate.py` — `from prxteinmpnn.model.multistate_sampling import ...` → `from prxteinmpnn.model._inference.multistate_sampling import ...`
- Any test file the naming-discipline spec did not already update

Discovery grep the fixer must run BEFORE the move:
```bash
grep -rEn \
  "from prxteinmpnn\.model\.(ar_scan|ar_exact|ar_exact_ligand|score_exact_ligand|mpnn_core|_shared|multistate_sampling)\b" \
  src/ tests/ scripts/
# Every match outside src/prxteinmpnn/model/ must be in the update list above.
# If a match appears in src/prxteinmpnn/<other-package>/, STOP — that is a leak
# that must be resolved before moving (either promote symbol to public, or refactor caller).
```

Gate (after the move):
```bash
# Save matches to a file rather than relying on grep's exit code in a pipe.
TMP=$(mktemp)
grep -rEn \
  "from prxteinmpnn\.model\.(ar_scan|ar_exact|ar_exact_ligand|score_exact_ligand|mpnn_core|_shared|multistate_sampling)\b" \
  src/ tests/ scripts/ 2>/dev/null \
  | grep -v "src/prxteinmpnn/model/_inference/" > "$TMP" || true
if [ -s "$TMP" ]; then
  echo "FAIL — stale imports of moved modules:"
  cat "$TMP"
  exit 1
fi
rm -f "$TMP"
echo "PASS — no stale imports"

# _inference/ exports nothing
PYTHONPATH=src uv run python -c "
import prxteinmpnn.model._inference as p
assert p.__all__ == [], f'_inference exports: {p.__all__}'
print('_inference/__all__ empty: ok')
"

PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -q
```

### Fixer 4 — Add `model/README.md` and finalize `__init__.py` docstring

Create `src/prxteinmpnn/model/README.md`:

```markdown
# prxteinmpnn.model — Public/Internal Boundary

## Public API

Import only from `prxteinmpnn.model` (the package). Everything in `__all__` is stable.

## Internal: `model/_inference/`

Implementation details of `PrxteinMPNN` and `PrxteinLigandMPNN`.
**Do not import from `_inference/` outside of `model/`.**

## Internal: model root files not in `__all__`

- `ligand_features.py` — ligand geometry feature extraction
- `ligand_tiling.py` — tiling helpers used internally

## The rule

If a symbol is not in `model/__init__.__all__`, it is not public API.
```

### Fixer 5 — Add `scripts/check_model_boundary.sh` CI gate (~20 LOC)

```bash
#!/usr/bin/env bash
# Exits non-zero if any file outside model/ imports from prxteinmpnn.model._inference.
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
VIOLATIONS=$(grep -rn "from prxteinmpnn\.model\._inference" \
  "${REPO}/src" "${REPO}/tests" --include="*.py" \
  | grep -v "src/prxteinmpnn/model/" || true)
if [[ -n "$VIOLATIONS" ]]; then
  echo "ERROR: External import of prxteinmpnn.model._inference:"
  echo "$VIOLATIONS"
  exit 1
fi
echo "Model public boundary: OK"
```

---

## 7. Dependency Note

**This spec requires the naming-discipline spec (`2026-05-13-naming-discipline-spec.md`) to land before Fixer 3.** The rename spec changes the source filenames; Fixer 3 here `git mv`s from the renamed paths into `_inference/`. If renames have NOT landed, Fixer 3 must use original long names (`mpnn_autoregressive_scan.py`, etc.) and a follow-up rename is needed.

**Do not dispatch Fixer 3 simultaneously with the naming-discipline rename fixer** — same files, merge conflict.

Fixers 1, 2, 4, 5 have no dependency and can land immediately.

---

## 8. Risks and Rollback

| Risk | Mitigation |
|---|---|
| Inline PLC0415 imports in `ligand_mpnn.py` reference old paths | Gate command greps for old names including inline forms |
| TYPE_CHECKING-only imports create cycles to type checkers | All moved files carry `from __future__ import annotations` |
| `_inference/__init__.py` accidentally populated with re-exports | Gate asserts file is empty |
| `_shared.py` and `multi_state_sampling.py` moved in same commit with stale cross-imports | Both moved and updated in single atomic commit |

**Rollback:** Each fixer is a single atomic commit; `git revert <sha>` undoes cleanly.

---

## 9. Pre-merge Gates

```bash
# Gate 1: All 15 public symbols importable
uv run python -c "from prxteinmpnn.model import (PrxteinMPNN, PrxteinLigandMPNN, DiffusionPrxteinMPNN, Packer, Encoder, EncoderLayer, Decoder, DecoderLayer, ProteinFeatures, ModelCapabilities, PRXTEIN_MPNN_CAPABILITIES, PRXTEIN_LIGAND_MPNN_CAPABILITIES, gather_flat_to_stack, scatter_stack_to_flat); print('ok')"

# Gate 2: No moved-file imports outside _inference (use -E for alternation; save matches to assert empty)
TMP=$(mktemp)
grep -rEn "from prxteinmpnn\.model\.(ar_scan|ar_exact|ar_exact_ligand|score_exact_ligand|mpnn_core|_shared|multistate_sampling)\b" \
  src/ tests/ scripts/ 2>/dev/null \
  | grep -v "src/prxteinmpnn/model/_inference/" > "$TMP" || true
if [ -s "$TMP" ]; then cat "$TMP"; rm -f "$TMP"; exit 1; fi
rm -f "$TMP"

# Gate 3: No external imports of _inference
bash scripts/check_model_boundary.sh

# Gate 4: Fast test suite
PYTHONPATH=src uv run pytest tests/sampling/test_sample.py tests/model/test_ligand_wave_parallel.py tests/sampling/test_state_vmap_exact_jit.py tests/sampling/test_sample_call_kw_contract.py -q

# Gate 5: __all__ covers every intended public symbol — but ignore submodule attributes
# that appear in dir(m) by virtue of being imported (capabilities, decoder, etc. are modules,
# not API symbols). We test the inverse: every name in __all__ resolves.
uv run python -c "
import prxteinmpnn.model as m
for name in m.__all__:
    assert hasattr(m, name), f'__all__ lists {name!r} but it is not bound on the module'
print(f'__all__ ({len(m.__all__)} symbols) all resolve')
"

# Gate 6: _inference/__init__.py exports nothing (assert by import, not byte count)
PYTHONPATH=src uv run python -c "
import prxteinmpnn.model._inference as p
assert p.__all__ == [], f'_inference exports: {p.__all__}'
"

# Gate 7: README present
test -f src/prxteinmpnn/model/README.md
```

---

## 10. Out of Scope

- Renaming files — handled by naming-discipline spec
- Relocating `ligand_features.py` / `ligand_tiling.py` to `_inference/` — separate follow-on
- The protocol seam (separate spec)
- Deduplicating `mpnn.py` / `ligand_mpnn.py` (separate spec)
- Deleting `features_direct.py` (handled by naming-discipline spec)
- Adding compatibility shims (sole maintainer)
- `parity/`, `training/`, `ensemble/` internal organization
