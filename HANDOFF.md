# Equinox Migration - Handoff Document

**Date**: November 4, 2025  
**Current State**: Phase 1 partially complete  
**Next Agent**: Continue from here

## What Just Happened

1. ✅ **Cleaned up documentation**: Deleted 14+ old migration docs (PHASE_*.md, etc.)
2. ✅ **Created fresh CURRENT_STATUS.md**: Clear strategy for clean replacement
3. ✅ **Deleted legacy code**: Removed old `model/` and `functional/` directories
4. ✅ **Started new model structure**: 
   - Created `model/features.py` with ProteinFeatures class
   - Created `model/encoder.py` with EncoderLayer and Encoder classes
5. ✅ **Committed**: Clean commit with all cleanup done

## Current Blocker

**Import errors**: Files in `sampling/`, `scoring/`, `run/` still try to import from the deleted `model/` directory, causing cascading import failures.

**Example**:
```python
# sampling/initialize.py line 13
from prxteinmpnn.model.features import extract_features, project_features
# ❌ These functions don't exist - ProteinFeatures is now an Equinox class
```

## What Needs to Happen Next

### Immediate (Complete Phase 1)

1. **Create `model/decoder.py`** - Extract from `eqx_new.py` lines 289-545:
   - DecoderLayer class
   - Decoder class (with 3 methods: `__call__`, `call_conditional`, autoregressive scan)

2. **Create `model/mpnn.py`** - Extract from `eqx_new.py` lines 548-end:
   - PrxteinMPNN class (top-level model)
   - All 3 decoding modes

3. **Create `model/__init__.py`** - Re-export all classes:
   ```python
   from .features import ProteinFeatures
   from .encoder import Encoder, EncoderLayer
   from .decoder import Decoder, DecoderLayer
   from .mpnn import PrxteinMPNN
   
   __all__ = ["ProteinFeatures", "Encoder", "EncoderLayer", "Decoder", "DecoderLayer", "PrxteinMPNN"]
   ```

4. **Delete `eqx_new.py`** - Contents fully migrated

5. **Test imports**:
   ```bash
   uv run python -c "from prxteinmpnn.model import PrxteinMPNN; print('✅')"
   ```

### Then (Phase 2 - Easiest to just delete and skip for now)

**Delete `functional/` directory entirely** - Already done!  ✅

### Then (Phase 3 - Update High-Level APIs)

This is the big one. Need to update ~20+ files:

#### Strategy: Comment Out Broken Imports First

To unblock development, temporarily comment out broken imports and add TODO markers:

```python
# TODO: Update after Equinox migration
# from prxteinmpnn.model.features import extract_features
```

Then systematically fix each module:

1. **`sampling/initialize.py`**:
   - Old: `extract_features(...)` function call
   - New: `model.features(...)` Equinox module call

2. **`sampling/conditional_logits.py`**, **`unconditional_logits.py`**:
   - Old: `make_encoder()`, `make_decoder()` factory functions
   - New: Use `model.encoder`, `model.decoder` directly

3. **`sampling/adapter.py`**:
   - Keep Equinox path, remove PyTree path entirely

4. **`scoring/score.py`**:
   - Update to use Equinox model methods

### Files with Import Errors (From Grep Earlier)

**Priority 1 - Core API:**
- `sampling/initialize.py` - imports `extract_features`, `project_features`
- `sampling/conditional_logits.py` - imports `make_encoder`, `make_decoder`, `final_projection`
- `sampling/unconditional_logits.py` - same as conditional
- `sampling/adapter.py` - imports `DecodingApproach`, `MaskedAttentionType` (✅ types moved to utils.types)
- `scoring/score.py` - imports `extract_features`, `project_features`, `final_projection`

**Priority 2 - Tests:**
- `tests/functional/test_decoder.py`
- `tests/functional/test_encoder.py`
- `tests/scoring/test_score.py`
- `tests/sampling/test_sampling_step.py`

## Source Material

**All code is in `src/prxteinmpnn/eqx_new.py`** - This is the complete, working Equinox implementation.

**Line numbers:**
- Lines 1-63: Imports + constants
- Lines 64-173: ProteinFeatures ✅ (done in features.py)
- Lines 174-288: EncoderLayer ✅ (done in encoder.py)
- Lines 289-375: DecoderLayer ⏳ (TODO in decoder.py)
- Lines 376-419: Encoder ✅ (done in encoder.py)
- Lines 420-547: Decoder ⏳ (TODO in decoder.py)
- Lines 548-955: PrxteinMPNN ⏳ (TODO in mpnn.py)

## Quick Commands

```bash
# Check what still imports from old model
grep -r "from prxteinmpnn.model" src/ --include="*.py" | grep -v "__pycache__"

# Check what still imports from functional
grep -r "from prxteinmpnn.functional" src/ --include="*.py" | grep -v "__pycache__"

# Test equivalence tests still pass (they won't until Phase 1 is done)
uv run pytest tests/test_eqx_equivalence.py -v

# Run all tests
uv run pytest tests/ -v
```

## Recommended Next Steps

**Option 1: Complete Phase 1 First (Recommended)**
1. Create decoder.py and mpnn.py from eqx_new.py
2. Create __init__.py with re-exports
3. Delete eqx_new.py
4. Test that `from prxteinmpnn.model import PrxteinMPNN` works
5. Then tackle Phase 3 (updating imports)

**Option 2: Quick Unblock**
1. Comment out all broken imports with TODO markers
2. This allows the package to at least import
3. Then methodically fix each module

## Files to Reference

- `docs/CURRENT_STATUS.md` - The main strategy document
- `src/prxteinmpnn/eqx_new.py` - Complete working implementation (source of truth)
- `src/prxteinmpnn/utils/types.py` - All type definitions
- `tests/test_eqx_equivalence.py` - Must keep passing

## Success Criteria for Phase 1 Complete

- [ ] `model/decoder.py` exists with DecoderLayer and Decoder
- [ ] `model/mpnn.py` exists with PrxteinMPNN
- [ ] `model/__init__.py` re-exports all classes
- [ ] `eqx_new.py` deleted
- [ ] Can import: `from prxteinmpnn.model import PrxteinMPNN`
- [ ] Lint passes on new files: `uv run ruff check src/prxteinmpnn/model/`

## Time Estimate

- Phase 1 completion: 30-60 minutes (just file creation from existing code)
- Phase 3 (update imports): 2-4 hours (requires understanding each module)
- Total remaining: 3-5 hours

## Notes

- The Equinox implementation in `eqx_new.py` is **complete and tested**
- All equivalence tests were passing before we started this cleanup
- We're just reorganizing the code, not changing functionality
- Once Phase 1 is done, Phase 3 becomes much clearer

---

**Ready to proceed!** Start with creating `model/decoder.py` from `eqx_new.py` lines 289-547.
