---
title: Proxide Adapter Surface and Aminx CLI Integration
issue: "#1203"
task_id: "260618_autonomous-loop"
date: "2026-06-22"
status: "RESEARCH_COMPLETE"
---

## Objective

Support heterogeneous input types in aminx CLI `--inputs` flag via proxide-compatible URIs (SMILES, FASTA, PDB accession codes, HDF5 bundles, remote structures). This research note documents the proxide adapter surface, aminx's current input path, and the integration seam required for the spec.

---

## A. Proxide Adapter Surface

### A.1 Public API Overview

**proxide version:** 0.1.0 (installed as hard dependency in `pyproject.toml: line 23`)

**Top-level exports:**
- `parse_structure(file_path: str | Path, spec=None, use_jax: bool = True) -> proxide.core.containers.Protein`
- `fetch_rcsb(pdb_id: str, output_dir: str = '.', format_type: str = 'mmcif') -> str`
- `fetch_afdb(uniprot_id: str, output_dir: str = '.', version: int = 4) -> str`
- `fetch_md_cath(md_cath_id: str, output_dir: str = '.') -> str`
- `iterload(file_path, chunk_size=100)` — trajectory streaming
- `chem`, `core`, `io` modules

### A.2 High-Level Loaders (Primary Integration Points)

**proxide.io.parse_input** (wrapper; most flexible)
```python
def parse_input(
    file_path: str | pathlib.Path | IO[str],
    file_format: str | None = None,  # auto-detect if None
    chain_id: str | Sequence[str] | None = None,
    return_type: str = 'Protein',
    **kwargs: Any,
) -> Any  # Protein, ProteinStream, etc.
```

**proxide.io.load_structure** (alias for parse_input)
```python
def load_structure(
    file_path: str | pathlib.Path | IO[str],
    file_format: str | None = None,
    chain_id: str | Sequence[str] | None = None,
    return_type: str = 'Protein',
    **kwargs: Any,
) -> Any
```

**proxide.io.fetch_rcsb** (PDB ID accession lookup)
```python
def fetch_rcsb(pdb_id: str, output_dir: str = '.', format_type: str = 'mmcif') -> str
```

**proxide.io.fetch_afdb** (AlphaFold Structure DB)
```python
def fetch_afdb(uniprot_id: str, output_dir: str = '.', version: int = 4) -> str
```

**proxide.io.fetch_md_cath** (MD-CATH trajectory)
```python
def fetch_md_cath(md_cath_id: str, output_dir: str = '.') -> str
```

### A.3 Return Type: proxide.core.containers.Protein

```python
@dataclass
class Protein:
  coordinates: StructureAtomicCoordinates      # shape (N_res, 37, 3) — Atom37 format
  aatype: ProteinSequence                      # shape (N_res,)
  residue_index: ResidueIndex                  # shape (N_res,)
  chain_index: ChainIndex                      # shape (N_res,)
  source: str | None  # origin label
  # ~50 optional fields for physics, bonds, charges, etc.
```

### A.4 Supported Input Modalities

| Modality | Entry Point | Return Type | Notes |
| --- | --- | --- | --- |
| **Local PDB/CIF** | `parse_structure()` | `Protein` | Current aminx usage |
| **PDB accession** | `fetch_rcsb(pdb_id)` | `str` (file path) | Downloads, returns path |
| **UniProt (AlphaFold)** | `fetch_afdb(uniprot_id)` | `str` (file path) | AlphaFold v4 default |
| **MD-CATH trajectory** | `fetch_md_cath(md_cath_id)` | `str` (file path) | .h5 trajectory |
| **HDF5 trajectory** | `parse_structure(..., file_format='h5')` | `Protein` | Via iterload |
| **FoldComp DB** | `create_protein_dataset(..., foldcomp_database)` | Integrated | Via proxide.ops.dataset |
| **SMILES** | Not in proxide v0.1.0a9 | N/A | Future (rdkit) |
| **FASTA** | Not in proxide v0.1.0a9 | N/A | Future (ESMFold/OmegaFold) |

### A.5 URI Scheme Convention

Proxide **does NOT define standard URI schemes** in public API. Callers must detect and dispatch manually.

---

## B. Aminx CLI Current Input Path

### B.1 Input Expansion (cli.py lines 96–164)

```python
def _expand_inputs(inputs: list[str], *, fail_fast: bool = False) -> list[str]:
  """Expand --inputs to flat, deduped list of structure file paths."""
```

**Key constants:**
- `_STRUCTURE_EXTS = frozenset({".pdb", ".cif"})` (line 92)
- `_GLOB_CHARS = frozenset("*?[")` (line 93)

**Entry points (all run subcommands):**
- `run_sample()` line 408: calls `_expand_inputs()` at line 465
- `run_score()` line 524: calls `_expand_inputs()` at line 559
- `run_jacobian()` line 602: calls `_expand_inputs()` at line 639
- `run_inspect()` line 677: calls `_expand_inputs()` at line 715

### B.2 Spec Creation

**File:** `/home/marielle/projects/aminx/src/aminx/run/specs.py`

```python
spec = SamplingSpecification(
    inputs=inputs,  # list[str] from _expand_inputs
    **_base_spec_kwargs(b),
    ...
)
```

**RunSpecification field:**
```python
@dataclass
class RunSpecification:
  inputs: Sequence[str | TextIO]
```

### B.3 Runner Loading (prep.py lines 68–149)

```python
protein_iterator = create_protein_dataset(
    _loader_inputs(spec.inputs),  # line 103
    batch_size=spec.batch_size,
    parse_kwargs=parse_kwargs,  # chain_id, model, altloc, topology
    ...
)
```

**create_protein_dataset signature:**
```python
def create_protein_dataset(
    inputs: str | Path | Sequence[str | Path | IO[str]],
    batch_size: int,
    parse_kwargs: dict | None = None,
    foldcomp_database: str | None = None,
    ...
) -> IterDataset
```

---

## C. Integration Seam & Gaps

### C.1 Where Heterogeneous Input Dispatch Must Slot In

**Current pipeline:**
```
CLI --inputs flag
  ↓
_expand_inputs(inputs)  [glob/dir expansion]
  ↓
SamplingSpecification(inputs=expanded_list)
  ↓
prep_protein_stream_and_model(spec)
  ↓
create_protein_dataset(_loader_inputs(spec.inputs))
  ↓
[proxide internals]
```

**Option A (Recommended):** Add resolver at CLI stage
- Detect URI schemes (`pdb://`, `afdb://`, `mdcath://`, etc.)
- Call proxide fetch functions immediately
- Return list of local file paths
- Preserves spec serialization

**Option B:** Defer to proxide.ops.dataset
- Requires proxide team coordination
- Less clean separation

**Recommendation:** Option A — resolve at CLI stage.

### C.2 Downstream Interface

**create_protein_dataset() expects:**
```python
inputs: str | Path | Sequence[str | Path | IO[str]]
```

**Proxide returns (Protein dataclass):**
```python
Protein(
    coordinates: (N_res, 37, 3) array,  # Atom37 format
    aatype: (N_res,) int array,
    residue_index: (N_res,) int array,
    chain_index: (N_res,) int array,
    ...
)
```

**No conversion shim needed** — adapters already return compatible Protein objects.

---

## D. Dependency Gate & Optional Extras

### D.1 Current Proxide Dependency

**pyproject.toml line 23:**
```toml
dependencies = [
    "proxide>=0.1.0a9",
]
```

**Status:** Hard dependency; already installed.

### D.2 Optional Extras in Proxide v0.1.0a9

Proxide **does NOT yet export SMILES/FASTA adapters**. Current stable loaders:
- PDB/mmCIF (file system)
- Remote fetches (fetch_rcsb, fetch_afdb, fetch_md_cath)
- HDF5 trajectories (iterload)
- FoldComp database

**Future (not in v0.1.0a9):**
- SMILES → 3D (rdkit)
- FASTA → predicted (ESMFold/OmegaFold)

**Current status:** Do NOT add SMILES/FASTA in initial integration.

### D.3 Offline Cluster Consideration

**Critical risk:** HPC cluster nodes have no outbound internet.

**Mitigation:**
- `fetch_rcsb()`, `fetch_afdb()`, `fetch_md_cath()` fail on offline nodes
- **Solution:** Fetch + cache remotely before sbatch, OR validate offline mode
- Add flag: `--input-source { file | pdb | afdb | mdcath }` for strict offline mode

---

## E. Open Questions & Risks

### E.1 URI Scheme Design

**Question:** Custom URI scheme (e.g., `pdb://1A3A`) or rely on extensions?

**Risk:** Ambiguity if user has local file `pdb://1A3A.pdb`

**Recommendation:**
- `file:///path/to/structure.pdb` → local file
- `pdb://1A3A` → fetch from RCSB
- `pdb://1A3A.cif` → fetch as mmCIF
- `afdb://P12345` → fetch from AlphaFold DB
- `mdcath://some_id` → fetch from MD-CATH
- Bare path → treat as local (backward compatible)

### E.2 Fetch Caching

**Question:** Cache fetched structures? Where?

**Risk:** Repeated runs hit network repeatedly

**Recommendation:**
- Cache in `$output_dir/.proxide_cache/` or `~/.cache/proxide/` default
- Add `--cache-fetches` flag (enabled by default)
- Add `--cache-dir` override

### E.3 Accession Validation

**Question:** Validate accession codes before fetch?

**Risk:** User typos (e.g., `pdb://1A3X` doesn't exist) → timeout

**Recommendation:**
- Add `--strict-accessions` flag (disabled by default)
- If enabled, validate PDB ID format (4 alphanumeric) before fetching
- Catch RCSB 404 errors gracefully

### E.4 Mixed Local/Remote Inputs

**Question:** Mix `--inputs pdb://1A3A /tmp/my.pdb afdb://P12345` in one run?

**Risk:** Spec serialization and reproducibility

**Recommendation:** Always resolve to local paths at CLI stage, store metadata in `inputs_metadata.json` for tracing.

---

## F. Recommended Spec Outline (Task Decomposition)

### Phase 1: Detection & Validation (F1)

**File:** `src/aminx/cli.py`

Create `_normalize_and_resolve_inputs()`:
- Parse URI scheme (`pdb://`, `afdb://`, etc.)
- Validate accession format if `--strict-accessions`
- Detect local vs. remote
- Preserve backward compat

### Phase 2: Fetching & Caching (F2)

**File:** `src/aminx/io/proxide_adapters.py` (new)

```python
def resolve_pdb_accession(pdb_id: str, cache_dir: Path) -> Path:
def resolve_afdb_accession(uniprot_id: str, cache_dir: Path) -> Path:
def resolve_mdcath_accession(mdcath_id: str, cache_dir: Path) -> Path:
```

- Wrap proxide.fetch_* with error handling
- Implement cache logic
- Handle network errors

### Phase 3: CLI Integration (F3)

**File:** `src/aminx/cli.py`

- Add `--input-source` enum flag (auto, local, remote, mixed)
- Add `--cache-fetches` bool (default True)
- Add `--cache-dir` Path override
- Replace `_expand_inputs()` with `_normalize_and_resolve_inputs()`
- Store metadata in spec

### Phase 4: Spec Serialization (F4)

**File:** `src/aminx/run/specs.py`

- Add `inputs_metadata: dict | None` field to RunSpecification
- Update serialization/deserialization

### Phase 5: Runner Awareness (F5)

**File:** `src/aminx/host/prep.py`

- Emit warning if offline mode + remote URIs
- Log input resolution trace

### Phase 6: Tests & Docs (F6)

- Unit tests for URI parsing
- Integration tests (fetch + sample)
- Offline mode tests
- CLI docstring examples

---

## G. Summary Table: Proxide Adapters Per Modality

| Modality | Function | Signature | Return | Integration |
| --- | --- | --- | --- | --- |
| **Local PDB/CIF** | `parse_structure()` | `(file_path: str)` | `Protein` | Current; no change |
| **PDB Accession** | `fetch_rcsb(pdb_id, output_dir)` | `(pdb_id: str) -> str` | File path | F2.1: wrap + cache |
| **AlphaFold (UniProt)** | `fetch_afdb(uniprot_id, output_dir)` | `(uniprot_id: str) -> str` | File path | F2.1: wrap + cache |
| **MD-CATH (Trajectory)** | `fetch_md_cath(md_cath_id, output_dir)` | `(md_cath_id: str) -> str` | File path | F2.1: wrap + cache |
| **HDF5 Trajectory** | `parse_structure(..., file_format='h5')` | `(file_path: str)` | `Protein` | Current; no change |
| **FoldComp DB** | `create_protein_dataset(..., foldcomp_database)` | kwarg | Integrated | Current; no change |
| **SMILES (future)** | TBD in ≥0.2.0 | TBD | TBD | Out of scope |
| **FASTA (future)** | TBD in ≥0.2.0 | TBD | TBD | Out of scope |

---

## H. Confidence & Next Steps

**Confidence:** 0.92 (high)
- Proxide 0.1.0a9 public API is stable and documented
- Aminx integration point (create_protein_dataset) is clear
- No hidden blockers; risks are UX (URI design) + offline mitigation

**Next steps:**
1. Write implementation spec (task_id: 260622_1203_spec)
2. Implement F1–F2 (adapter + resolution layer)
3. Review URI scheme with team before F3
4. Implement F3–F6 (CLI, serialization, tests)
