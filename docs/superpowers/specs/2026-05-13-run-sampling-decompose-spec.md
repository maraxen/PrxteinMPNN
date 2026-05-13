# run/sampling.py Decomposition — Design Spec

**Date**: 2026-05-13
**Kind**: refactor (3 fixer tasks + 1 pre-task)
**Related**: `docs/superpowers/specs/2026-05-07-phase6-batch-layout-design.md`

---

## Problem

`src/prxteinmpnn/run/sampling.py` is 1964 LOC and bundles five unrelated concerns: grid-lineage identity hashing, ligand context preparation, I/O streaming dispatch hooks, averaged-mode sampling, and the core batch-dispatch orchestration. Every change to any one concern requires reading 2000 lines. The actual refactor risk is concentrated in three specific places (the monkeypatched symbols, the grid lineage hash chain, and `_broadcast_per_structure` placement).

## Goal

Split `run/sampling.py` into three focused sibling files under `src/prxteinmpnn/run/`, keeping `sampling.py` as the orchestrator that owns `sample()`, `_sample_batch`, `_sample_streaming`, `_sample_streaming_arrayrecord`, and the four I/O hooks — extracting grid-lineage, ligand-prep, and averaged-mode concerns into dedicated modules with **zero behavior change**.

---

## Discovery Requirement (mandatory before code change)

Before touching any file, the fixer must read `run/sampling.py` end-to-end and validate the responsibility map below against the live file. Specifically:

1. Confirm `_broadcast_per_structure` is called at exactly three sites: inside `_prepare_ligand_context`, `_prepare_fixed_controls`, and directly inside `_sample_batch` (for `structure_mapping`). If a fourth call site has been added, stop and surface it.
2. Confirm no private symbol from `run/sampling.py` is imported by `run/campaign.py` other than `sample`. Gate: `grep -n "from.*run.sampling import\|from .sampling import" src/prxteinmpnn/run/campaign.py` must return only the `sample` line.
3. Confirm the four I/O hook names are patched by tests using the dotted path `prxteinmpnn.run.sampling.<name>`. Gate: `grep -r "_noop_sampling\|_dispatch_sampling" tests/` must show no path other than `prxteinmpnn.run.sampling.*`.

## Responsibility Map (verified at spec authorship)

| Function | LOC | Concern | Destination |
|---|---|---|---|
| `_canonical_structure_id` | 12 | Grid lineage | `_sampling_grid_lineage.py` |
| `_canonical_structure_ids_for_spec` | 10 | Grid lineage | `_sampling_grid_lineage.py` |
| `_structure_ids_for_batch` | 10 | Grid lineage | `_sampling_grid_lineage.py` |
| `_resolve_grid_lineage` | 20 | Grid lineage | `_sampling_grid_lineage.py` |
| `_grid_sample_indices` | 5 | Grid lineage | `_sampling_grid_lineage.py` |
| `_grid_iteration_arrays` | 20 | Grid lineage | `_sampling_grid_lineage.py` |
| `_canonical_float_strings` | 3 | Grid serialization | `_sampling_grid_lineage.py` |
| `_canonical_json_bytes` | 8 | Grid serialization | `_sampling_grid_lineage.py` |
| `_grid_manifest_row_hash` | 15 | Grid lineage | `_sampling_grid_lineage.py` |
| `_grid_job_seed_hash` | 15 | Grid lineage | `_sampling_grid_lineage.py` |
| `_seed_words_from_manifest_hash` | 8 | Grid lineage | `_sampling_grid_lineage.py` |
| `_base_sampling_key` | 12 | Grid lineage | `_sampling_grid_lineage.py` |
| `_split_ligand_payload_key` | 10 | Ligand prep | `_sampling_ligand_prep.py` |
| `_normalize_keyed_ligand_array` | 10 | Ligand prep | `_sampling_ligand_prep.py` |
| `_normalize_ligand_tensor` | 30 | Ligand prep | `_sampling_ligand_prep.py` |
| `_load_ligand_context_file` | 70 | Ligand prep | `_sampling_ligand_prep.py` |
| `_prepare_ligand_context` | 60 | Ligand prep | `_sampling_ligand_prep.py` |
| `_prepare_fixed_controls` | 55 | Ligand prep | `_sampling_ligand_prep.py` |
| `_broadcast_per_structure` | 25 | **Shared helper** | `_sampling_helpers.py` |
| `_noop_sampling_chunk_io` | 12 | I/O hooks | **STAYS in `sampling.py`** (monkeypatched) |
| `_noop_sampling_structure_batch_io` | 15 | I/O hooks | **STAYS** |
| `_noop_sampling_tensor_batch_io` | 15 | I/O hooks | **STAYS** |
| `_dispatch_sampling_tensor_batch_io` | 15 | I/O hooks | **STAYS** |
| `_make_sampling_planner` | 20 | Batch planning | STAYS |
| `_sample_batch` | 230 | Core dispatch | STAYS |
| `sample` | 90 | Public entrypoint | STAYS |
| `_sample_streaming` | 120 | HDF5 streaming | STAYS |
| `_sample_streaming_arrayrecord` | 130 | ArrayRecord streaming | STAYS |
| `_create_decode_wrapper` | 30 | Averaged mode | `_sampling_averaged.py` |
| `_sample_averaged_mode` | 40 | Averaged mode | `_sampling_averaged.py` |
| `_internal_sample_averaged` | 35 | Averaged mode | `_sampling_averaged.py` |
| `_compute_logits_averaged` | 35 | Averaged mode | `_sampling_averaged.py` |
| `_sample_batch_averaged` | 95 | Averaged mode | `_sampling_averaged.py` |
| `_sample_streaming_averaged` | 50 | Averaged mode streaming | `_sampling_averaged.py` |

**Post-extraction target size of `sampling.py`**: ~740 LOC (from 1964).

---

## Layout Decision: Flat-With-Underscore (NOT Subpackage)

**Monkeypatch constraint is decisive.** Tests patch `prxteinmpnn.run.sampling._noop_sampling_chunk_io` and three companion symbols. Python's `unittest.mock.patch` binds into `sys.modules["prxteinmpnn.run.sampling"].__dict__`. Moving these hooks to `_io_dispatch.py` and re-exporting them would NOT preserve patch semantics — the live call inside `_sample_batch` would call the original unpatched function. Solution: **these 4 hooks stay in `sampling.py`**. Flat layout makes this non-controversial.

Additionally: `campaign.py`, tests, and public `__init__.py` patch other names at `prxteinmpnn.run.sampling.*` (`prep_protein_stream_and_model`, `SamplingDriver.build_sampler_fn`, `make_encoding_sampling_split_fn`, `get_averaged_encodings`). These are imported names in `sampling.py`'s namespace. The injection pattern (see Fixer 3) preserves them.

---

## Proposed Target Structure

```
src/prxteinmpnn/run/
  sampling.py                  # ENTRYPOINT — sample(), _sample_batch,
                               # _sample_streaming, _sample_streaming_arrayrecord,
                               # _make_sampling_planner, 4 noop/dispatch hooks
  _sampling_grid_lineage.py    # NEW — grid hashing, seed derivation
  _sampling_ligand_prep.py     # NEW — ligand context normalization
  _sampling_helpers.py         # NEW — _broadcast_per_structure
  _sampling_averaged.py        # NEW — averaged-mode sampling
```

---

## Per-Module Charter

### `sampling.py` (orchestrator)

**Owns:** `sample()`, `_sample_batch`, `_sample_streaming`, `_sample_streaming_arrayrecord`, `_make_sampling_planner`, all four noop/dispatch hooks, module-level constants (`GRID_SCHEMA_VERSION`, `SAMPLING_SCHEMA_VERSION`, `RANK_WITH_TEMPERATURE`).

**Imports from siblings:**
```python
from ._sampling_grid_lineage import (
    _canonical_structure_id, _canonical_structure_ids_for_spec,
    _structure_ids_for_batch, _resolve_grid_lineage, _grid_sample_indices,
    _grid_iteration_arrays, _grid_manifest_row_hash, _base_sampling_key,
)
from ._sampling_ligand_prep import (
    _prepare_ligand_context, _prepare_fixed_controls,
    AMINO_ACID_VOCAB_SIZE, LIGAND_CONTEXT_KEYS, LIGAND_PLACEHOLDER_ATOMS,
)
from ._sampling_helpers import _broadcast_per_structure
from ._sampling_averaged import _sample_averaged_mode as _sample_averaged_mode_impl
```

**Re-export rule for patches:** `sampling.py` continues to import `make_encoding_sampling_split_fn` and `get_averaged_encodings` from `prxteinmpnn.run.averaging` at module level. `_sample_averaged_mode_impl` receives them as injected callables (see Fixer 3).

### `_sampling_grid_lineage.py` (~150 LOC)

**Owns:** All grid identity, seeding, and hashing functions.

**Exports:** the 12 functions listed in the responsibility map.

**Imports:** `hashlib`, `json`, `pathlib.Path`, `numpy as np`, `jax`, `jax.numpy as jnp`. `SamplingSpecification` under `TYPE_CHECKING`. No imports from other `run/` modules.

### `_sampling_helpers.py` (~25 LOC)

**Owns:** `_broadcast_per_structure` — pure tensor shape-normalizer.

**Imports:** `jax`, `jax.numpy as jnp`. No `run/` imports.

**Rationale for separate file:** Consumed by `_sampling_ligand_prep.py`, `sampling.py`, and potentially future modules. Placing in `_sampling_ligand_prep.py` creates false coupling.

### `_sampling_ligand_prep.py` (~240 LOC)

**Owns:** Ligand context normalization and fixed-position controls.

**Exports:** Constants `LIGAND_CONTEXT_KEYS = ("Y", "Y_t", "Y_m")`, `LIGAND_PLACEHOLDER_ATOMS = 1`, `AMINO_ACID_VOCAB_SIZE = 21`, plus the 6 functions (`_split_ligand_payload_key`, `_normalize_keyed_ligand_array`, `_normalize_ligand_tensor`, `_load_ligand_context_file`, `_prepare_ligand_context`, `_prepare_fixed_controls`).

**Imports:** `pathlib.Path`, `numpy as np`, `jax`, `jax.numpy as jnp`, `collections.abc.Sequence`, `_broadcast_per_structure` from `._sampling_helpers`. Types under `TYPE_CHECKING`.

### `_sampling_averaged.py` (~290 LOC)

**Owns:** Averaged-mode sampling path (6 functions).

**Critical: monkeypatch preservation.** Tests patch `prxteinmpnn.run.sampling.make_encoding_sampling_split_fn` and `prxteinmpnn.run.sampling.get_averaged_encodings`. To preserve this:

```python
# In _sampling_averaged.py:
def _sample_averaged_mode(
    spec, protein_iterator, model,
    *,
    make_split_fn,  # injected from sampling.py's namespace
    get_encodings,  # injected from sampling.py's namespace
) -> dict[str, Any]: ...
```

```python
# In sampling.py — sample() function:
if spec.average_node_features:
    return _sample_averaged_mode_impl(
        spec, protein_iterator, model,
        make_split_fn=make_encoding_sampling_split_fn,  # resolved from this namespace
        get_encodings=get_averaged_encodings,            # patches intercept here
    )
```

Patches at `sampling.py` win because injection always uses sampling.py's bindings.

**Imports:** `functools.partial`, `jax`, `jax.numpy as jnp`, `h5py`, `numpy as np`, `prxteinmpnn.run.averaging.{get_averaged_encodings, make_encoding_sampling_split_fn}` (as fallback defaults), `StreamingBatchHost`, `resolve_tie_groups`, `DecodingOrderFn`, `random_decoding_order`.

---

## Determinism Regression Test (Pre-task, MANDATORY)

Before any code is moved, add `tests/run/test_sampling_grid_lineage.py` pinning `_base_sampling_key` output.

### Pre-step A — verify `SamplingSpecification` field signature

The fixer MUST run this first to confirm the constructor kwargs are valid:

```bash
PYTHONPATH=src uv run python -c "
from prxteinmpnn.run.specs import SamplingSpecification
print(sorted(SamplingSpecification.__dataclass_fields__.keys()))
"
```

Adjust the test's `SamplingSpecification(...)` call to match the actual field names returned. Below is a TEMPLATE — actual field names must match HEAD.

### Pre-step B — write the test

```python
# tests/run/test_sampling_grid_lineage.py
import jax  # noqa: F401
import numpy as np
from prxteinmpnn.run.sampling import _base_sampling_key, _resolve_grid_lineage
from prxteinmpnn.run.specs import SamplingSpecification

# Captured by fixer in pre-step C. Replace this LINE with an assignment:
#   PINNED_KEY_BYTES = b"\\x..."
# Do NOT keep this as a bare annotation — test module import will NameError.
PINNED_KEY_BYTES = b""  # FIXER MUST REPLACE before committing

def test_base_sampling_key_determinism():
    """Pin PRNGKey bytes for a fixed spec + grid lineage.

    If this fails after refactor, hash chain changed and in-flight grid
    sampling jobs will produce different sequences.
    """
    spec = SamplingSpecification(
        # Adjust kwargs to match actual SamplingSpecification fields from pre-step A.
        inputs=["/tmp/dummy.pdb"],
        random_seed=42,
        num_samples=10,
        # ... grid-related fields the fixer fills in based on the actual signature
    )
    lineage = _resolve_grid_lineage(spec)
    key = _base_sampling_key(spec, grid_lineage=lineage)
    key_bytes = np.asarray(key).tobytes()
    assert key_bytes == PINNED_KEY_BYTES, (
        f"PRNGKey bytes changed: {key_bytes.hex()} != {PINNED_KEY_BYTES.hex()}. "
        "Grid lineage hash chain was altered; bump GRID_SCHEMA_VERSION if intentional."
    )
```

### Pre-step C — capture pinned bytes

Run the test once with `PINNED_KEY_BYTES = b""` and observe the assertion's error message containing the actual hex. Replace the `b""` literal with `b"\\x<hex from error>"`. Commit. The test must pass cleanly before any extraction begins.

### Pre-step D — discover all monkeypatch targets

Beyond the well-known ones in the API Stability Assertion section, the fixer must verify nothing else in tests patches a private symbol that this refactor will move:

```bash
grep -rn "prxteinmpnn\.run\.sampling\." tests/ scripts/
# Every dotted name found must be either:
#  - in §"API Stability Assertion" (preserved at sampling.py), OR
#  - explicitly added to this list (the spec must be revised to preserve it)
```

If the grep returns a name the spec does not enumerate, STOP and revise the spec before proceeding.

---

## API Stability Assertion

Public symbols whose signatures must NOT change:

- `prxteinmpnn.run.sampling.sample`
- `prxteinmpnn.run.__init__.sample` (re-export)
- `prxteinmpnn.run.sampling_driver.SamplingDriver`

Patchable names at `prxteinmpnn.run.sampling.*` that must remain:
- `_noop_sampling_chunk_io`, `_noop_sampling_structure_batch_io`, `_noop_sampling_tensor_batch_io`
- `prep_protein_stream_and_model`
- `SamplingDriver`
- `make_encoding_sampling_split_fn`
- `get_averaged_encodings`

Verification:
```bash
grep -r "patch.*prxteinmpnn.run.sampling\." tests/ | grep -v "^Binary"
# Every line must reference a name still in sampling.py's namespace after refactor
```

---

## Migration Sequence (Atomic Commits)

### Pre-task: Determinism regression test

Create `tests/run/test_sampling_grid_lineage.py`. Capture and paste `PINNED_KEY_BYTES`. Commit.

```bash
PYTHONPATH=src uv run pytest tests/run/test_sampling_grid_lineage.py -v
```

### Fixer 1 — Extract Grid Lineage (~150 LOC)

Create `_sampling_grid_lineage.py` with 12 functions verbatim. Update `sampling.py` to import from it. Remove `hashlib`/`json` from `sampling.py` if no longer used.

Gate:
```bash
PYTHONPATH=src uv run pytest tests/run/test_sampling_grid_lineage.py tests/run/test_sampling_grid.py tests/run/test_sampling.py -v
```

### Fixer 2 — Extract Helper + Ligand Prep (~265 LOC)

Create `_sampling_helpers.py` and `_sampling_ligand_prep.py`. Move constants. Update `sampling.py` imports.

Gate:
```bash
PYTHONPATH=src uv run pytest tests/run/ tests/streaming/test_sampling_chunk_io_callback.py tests/streaming/test_sampling_tensor_batch_io_callback.py -v
grep -n "_noop_sampling_chunk_io\|_noop_sampling_structure_batch_io\|_noop_sampling_tensor_batch_io\|_dispatch_sampling_tensor_batch_io" src/prxteinmpnn/run/sampling.py | grep "def "
# Must print 4 lines
```

### Fixer 3 — Extract Averaged Mode (~290 LOC)

Create `_sampling_averaged.py` with 6 functions + injection signature. Replace call sites in `sampling.py`.

Gate:
```bash
PYTHONPATH=src uv run pytest tests/run/ tests/streaming/ tests/parity/test_averaging_pipeline_parity.py -v
python -c "
import prxteinmpnn.run.sampling as m
for n in ['_noop_sampling_chunk_io', '_noop_sampling_structure_batch_io', '_noop_sampling_tensor_batch_io', 'make_encoding_sampling_split_fn', 'get_averaged_encodings']:
    assert hasattr(m, n), f'missing {n}'
print('patch surface OK')
"
wc -l src/prxteinmpnn/run/sampling.py  # must be < 900
```

### Final Gate: parity_heavy

```bash
export REFERENCE_PATH=/absolute/path/to/ligandmpnn_reference_assets
PYTHONPATH=scripts:src uv run pytest tests/parity tests/model/test_ligandmpnn_equivalence.py -m parity_heavy -v
```

---

## Risks

| Risk | Mitigation |
|---|---|
| Grid lineage hash chain altered | Pre-task determinism regression test pins `_base_sampling_key` bytes |
| Monkeypatch tests fail because hooks moved | Spec forbids moving the 4 hooks; gate verifies presence |
| `make_encoding_sampling_split_fn`/`get_averaged_encodings` patch broken after averaged extraction | Injection pattern preserves resolution through `sampling.py`'s namespace |
| Circular import via `_broadcast_per_structure` | Both consumers import from `_sampling_helpers.py`; no cycles |
| Constants silently duplicated | Single definition in `_sampling_ligand_prep.py`; grep gate verifies |

---

## Out of Scope

- `run/campaign.py` (1402 LOC) — separate concern
- `run/scoring.py` — parallel refactor if desired (separate spec)
- New tests beyond determinism regression
- Behavior changes — none
- Removing `_sample_streaming` (deprecated HDF5 path) — stays
- Converting any private function to public API
- Protocol seam, naming discipline, public-contract — parallel specs, no merge conflict here

---

## Coordination

**Merge-conflict-prone files:** Only `run/sampling.py`. If a concurrent branch modifies it, rebase rather than merge.

**Parallel-safe specs:** Naming discipline, protocol seam, public contract — all operate on `model/` layer; no overlap.

**Ordering:** This decomposition has no prerequisite. Can land before, after, or in parallel.
