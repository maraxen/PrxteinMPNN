# Proxide Integration Handoff

## Work Accomplished This Session

### 1. Core Data Structure Migration ✅

- **`src/prxteinmpnn/utils/data_structures.py`**: Replaced local `ProteinTuple` (NamedTuple) and `Protein` (dataclass) with imports from `proxide.core.containers.Protein`
- Added `ProteinTuple = Protein` alias for backward compatibility
- Removed helper functions `from_tuple`, `from_tuple_numpy`, `none_or_jnp`, `none_or_numpy`

### 2. Residue Constants Migration ✅

- **`src/prxteinmpnn/utils/residue_constants.py`**: Now imports `atom_order` from `proxide.core.containers` and `restypes` from `proxide.chem.residues`

### 3. Parsing Module Overhaul ✅

**Removed 9 deprecated files:**

- `biotite.py` - PDB/CIF parsing (now via `proxide.parse_structure`)
- `mdcath.py` - MD-CATH H5 parsing (now via `proxide.parse_mdcath_*`)
- `mdtraj.py` - MDTraj H5 parsing (now via `proxide.parse_mdtraj_h5_*`)
- `utils.py` - Static feature extraction (now in `proxide.Protein`)
- `coords.py` - Coordinate reshaping (handled by proxide)
- `mappings.py` - Residue/atom mappings (now via `proxide.chem.conversion`)
- `physics_utils.py` - Force field params (handled by proxide)
- `pqr.py` - PQR parsing (now via `proxide.parse_pqr`)
- `structures.py` - ProcessedStructure dataclass (not needed)

**Rewrote:**

- `proxide.py` - Now the primary parsing interface using proxide exclusively
- `dispatch.py` - Simplified routing layer to proxide.py
- `__init__.py` - Clean re-exports

### 4. Consumer Updates (Partial) 🟡

- **`preprocess.py`**: Simplified to use `parse_structure()` with physics via OutputSpec
- **`foldcomp_utils.py`**: Updated to use `proxide.chem.conversion.string_to_protein_sequence`
- **`download_and_process.py`**: Same import fix

### 5. Test Cleanup ✅

**Removed deprecated tests:**

- `tests/io/parsing/` directory (8 test files)
- `tests/io/test_physics_integration.py`
- `tests/io/test_solvent_and_ff.py`

---

## Current Test Failure

The remaining test failure is in `array_record_source.py`:

```
test_array_record_source_robustness - ProteinTuple constructor mismatch
```

The `_record_to_protein_tuple()` function uses old ProteinTuple fields that don't match `proxide.Protein`:

- `atom_mask` → should be `full_atom_mask`
- `source` → not a field in proxide.Protein
- Missing: `one_hot_sequence`, `mask`, `mapping`, etc.

---

## Next Agent Task: Deep Integration Analysis

### Primary Objective

Deeply inspect both `prxteinmpnn` and `proxide` to determine the optimal path to full deprecation of redundant PrxteinMPNN code.

### Key Investigation Areas

#### 1. Proxide Internals to Inspect

```bash
uv run python -c "
import proxide
# Check array_record capabilities
print([x for x in dir(proxide) if 'record' in x.lower() or 'grain' in x.lower()])

# Check OutputSpec fields
import inspect
print(inspect.signature(proxide.OutputSpec))

# Check Protein dataclass fields
from proxide.core.containers import Protein
import dataclasses
print([f.name for f in dataclasses.fields(Protein)])
"
```

#### 2. Files Still Needing Updates

| File | Issue | Action Needed |
|------|-------|---------------|
| `io/array_record_source.py` | Uses old ProteinTuple fields | Rewrite `_record_to_protein_tuple` to construct `Protein` |
| `io/operations.py` | References `atom_mask`, `_replace` | Update to use `full_atom_mask`, `dataclasses.replace` |
| `io/process.py` | Type hints reference ProteinTuple | Update type hints |
| `io/dataset.py` | Same issue | Update type hints |

#### 3. Proxide Features to Investigate

- Does proxide have array_record/grain integration? (`proxide.data` or similar)
- Does proxide handle coordinate transformations for model input?
- Does proxide provide batching/collation utilities?
- Does proxide expose force field loading directly?

#### 4. Potential Proxide Enhancements

If proxide doesn't have these, consider adding:

- `Protein.from_dict()` classmethod for array_record deserialization
- Grain-compatible data source wrapper
- Batch collation utilities

### Questions for Decision Making

1. Should `array_record_source.py` be moved to proxide entirely?
2. Should `operations.py` (padding, truncation, collation) live in proxide?
3. Can we eliminate the `prxteinmpnn.io` module entirely?

### Verification Steps

After updates, run:

```bash
uv run pytest tests/ -x --ignore=tests/training -q
```

---

## File Locations Summary

### Modified Files (This Session)

- `src/prxteinmpnn/utils/data_structures.py`
- `src/prxteinmpnn/utils/residue_constants.py`
- `src/prxteinmpnn/io/parsing/proxide.py` (rewritten)
- `src/prxteinmpnn/io/parsing/dispatch.py` (simplified)
- `src/prxteinmpnn/io/parsing/__init__.py` (simplified)
- `src/prxteinmpnn/training/dataloading/preprocess.py`
- `src/prxteinmpnn/utils/foldcomp_utils.py`
- `src/prxteinmpnn/training/data/download_and_process.py`

### Files Still Needing Work

- `src/prxteinmpnn/io/array_record_source.py` - Critical: constructor mismatch
- `src/prxteinmpnn/io/operations.py` - field names
- `src/prxteinmpnn/io/process.py` - type hints
- `src/prxteinmpnn/io/dataset.py` - type hints

### Proxide Key Modules

- `proxide.core.containers` - Protein, ProteinBatch, ProteinStream
- `proxide.chem.residues` - restypes
- `proxide.chem.conversion` - string_to_protein_sequence
- `proxide.physics.features` - compute_*_node_features
- `proxide.physics.force_fields` - load_force_field
- `proxide` top-level - parse_structure, parse_mdcath_*, OutputSpec
