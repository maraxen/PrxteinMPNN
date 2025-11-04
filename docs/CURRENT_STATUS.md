# Equinox Migration: Clean Replacement Strategy

**Date**: November 4, 2025  
**Branch**: `eqx_migration`  
**Status**: 🚀 Ready for clean replacement

## TL;DR

We have a **complete, tested Equinox implementation** in `eqx_new.py`. Instead of gradual migration, we're doing a **clean replacement**: delete legacy code, reorganize Equinox model into proper modules, update all interfaces.

## Current State

### ✅ What's Working
- `src/prxteinmpnn/eqx_new.py` - Complete Equinox implementation
  - ProteinFeatures, Encoder, Decoder, PrxteinMPNN classes
  - All 3 decoding modes (unconditional, conditional, autoregressive)
  - Numerically equivalent to legacy model
  - All equivalence tests passing
- `src/prxteinmpnn/utils/ste.py` - Straight-through estimator (moved from model/)
- `src/prxteinmpnn/utils/types.py` - Type definitions (DecodingApproach, MaskedAttentionType, decoder signatures)

### ⚠️ Legacy Code (TO BE DELETED)
- `src/prxteinmpnn/model/` - Old PyTree-based model
- `src/prxteinmpnn/functional/` - Old functional operations
- Various imports in `sampling/`, `scoring/`, `run/` modules

## Goal: Clean Replacement

**What we're doing:**

### 1. New `prxteinmpnn.model` submodule with Equinox components

Split `eqx_new.py` into modular structure:
```
src/prxteinmpnn/model/
  ├── __init__.py        # Re-exports: PrxteinMPNN, Encoder, Decoder, etc.
  ├── features.py        # ProteinFeatures class
  ├── encoder.py         # EncoderLayer, Encoder classes  
  ├── decoder.py         # DecoderLayer, Decoder classes
  └── mpnn.py            # PrxteinMPNN top-level class
```

All using Equinox modules, **NO** PyTree ModelParameters pattern.

### 2. Update ALL interfaces to use Equinox exclusively

- **`sampling/*`** - Update to work with Equinox modules
  - Remove adapter logic for PyTree models (clean break)
  - Update initialize, sample, conditional_logits, unconditional_logits, etc.
- **`scoring/*`** - Update to use Equinox
- **`run/*`** - Update to use Equinox
- **Remove files** that only work with legacy architecture

### 3. All tests passing

- Update test imports to new model structure
- Remove tests for deleted modules
- Ensure `tests/test_eqx_equivalence.py` still passes
- Full test suite: `pytest tests/`

### 4. Clean documentation

- ✅ Old migration docs deleted
- Update README with new architecture
- Update AGENTS.md and .github/copilot-instructions.md
- This file serves as the "Migration Complete" summary

## Implementation Plan

### Phase 1: Reorganize Equinox Model ⏳
1. **Delete** `src/prxteinmpnn/model/` directory entirely
2. **Create** new modular structure from `eqx_new.py`:
   ```bash
   # Split eqx_new.py into:
   model/features.py   # ProteinFeatures
   model/encoder.py    # EncoderLayer, Encoder
   model/decoder.py    # DecoderLayer, Decoder  
   model/mpnn.py       # PrxteinMPNN
   model/__init__.py   # Re-exports
   ```
3. **Delete** `eqx_new.py` (contents moved)
4. **Test**: `from prxteinmpnn.model import PrxteinMPNN` works

### Phase 2: Delete Legacy Functional ⏳
1. **Delete** `src/prxteinmpnn/functional/` directory entirely
2. **Remove** any imports/references to `functional` in codebase

### Phase 3: Update High-Level APIs ⏳
1. **`sampling/`** module:
   - Update `adapter.py` - remove PyTree logic, use Equinox only
   - Update `initialize.py` - use model.features directly
   - Update `conditional_logits.py`, `unconditional_logits.py`
   - Update `sample.py`, `sampling_step.py`, `ste_optimize.py`
2. **`scoring/`** module:
   - Update `score.py` to use Equinox model
3. **`run/`** module:
   - Update any files importing from old model/

### Phase 4: Fix Tests ⏳
1. Update test imports to `prxteinmpnn.model.*`
2. Remove `tests/functional/` (testing deleted code)
3. Ensure `tests/test_eqx_equivalence.py` passes
4. Fix any broken tests in `tests/sampling/`, `tests/scoring/`
5. Run full suite: `uv run pytest tests/`

### Phase 5: Clean Docs ⏳
1. ✅ Remove old PHASE_*.md migration docs (DONE)
2. Update `README.md` with new architecture
3. Update `AGENTS.md` with new structure
4. Update `.github/copilot-instructions.md`

## Key Technical Details

### Type Imports
**All types are in `prxteinmpnn.utils.types`**, including:
- `DecodingApproach` - Literal["conditional", "autoregressive", "unconditional"]
- `MaskedAttentionType` - Literal["none", "cross", "conditional"]
- Decoder function signatures (RunDecoderFn, RunConditionalDecoderFn, etc.)

**Example:**
```python
from prxteinmpnn.utils.types import DecodingApproach, MaskedAttentionType
```

### Equinox Model Structure
The classes in `eqx_new.py`:
- **ProteinFeatures** - Extracts and projects node/edge features
- **EncoderLayer** - Single encoder layer with attention + feedforward
- **Encoder** - Stack of encoder layers
- **DecoderLayer** - Single decoder layer with attention + feedforward
- **Decoder** - Stack of decoder layers
- **PrxteinMPNN** - Top-level model with encoder + decoder + 3 decoding methods

### No Adapter Pattern
We're **NOT** maintaining compatibility with PyTree models. This is a clean break:
- ❌ No `make_encoder()`, `make_decoder()` factory functions
- ❌ No `ModelParameters` PyTree pattern
- ❌ No adapter logic to detect model type
- ✅ Direct use of Equinox modules everywhere

## What NOT to Do

- ❌ Don't create compatibility layers for old PyTree models
- ❌ Don't gradually update imports file-by-file (delete and replace)
- ❌ Don't support both architectures
- ❌ Don't create "_pure" versions or legacy wrappers

## Success Criteria

- [ ] No `prxteinmpnn.model.encoder`, `.model.decoder` legacy imports
- [ ] No `prxteinmpnn.functional.*` imports
- [ ] New `prxteinmpnn.model` uses only Equinox modules
- [ ] All tests passing: `uv run pytest tests/`
- [ ] Type checking passes: `uv run pyright`
- [ ] Linting passes: `uv run ruff check src/`
- [ ] Documentation updated

## Key Files Reference

- `src/prxteinmpnn/eqx_new.py` - Complete working Equinox implementation (to be split)
- `src/prxteinmpnn/utils/ste.py` - Straight-through estimator utility
- `src/prxteinmpnn/utils/types.py` - All type definitions
- `tests/test_eqx_equivalence.py` - Must keep passing (validates numerical equivalence)

## Time Estimate

**4-6 hours** for complete clean replacement (vs. weeks for gradual migration)

## Next Action

**START Phase 1**: Delete old `model/` directory and reorganize `eqx_new.py` into new modular structure.

```bash
# Step 1: Delete old model
rm -rf src/prxteinmpnn/model/

# Step 2: Create new structure (split eqx_new.py)
mkdir -p src/prxteinmpnn/model/
# ... split files ...

# Step 3: Delete eqx_new.py
rm src/prxteinmpnn/eqx_new.py

# Step 4: Test imports
uv run python -c "from prxteinmpnn.model import PrxteinMPNN; print('✅ Success')"
```
