# Benchmark Suite Spec (Follow-up): Ligand-Conditioned Inference
## aminx JAX + LigandMPNN PyTorch Reference

**Spec Version:** 2 (ligand-conditioned follow-up to 260601_benchmark-spec.md)
**Date:** 2026-06-02
**Status:** DRAFT — not yet oracle-reviewed
**Blocking Prereq:** Acquire protein-ligand PDB fixture suitable for both adapters (§10)

---

## 1. Motivation

The approved no-ligand benchmark (260601) establishes baseline throughput and latency for
unconditional ProteinMPNN in both frameworks. This follow-up measures the **cost of ligand
conditioning**: Does aminx JAX achieve comparable per-residue throughput to LigandMPNN
PyTorch when ligand atoms are present?

Both implementations support ligand-conditioned sequence design:
- **aminx JAX** via `build_inference_bundle(ligand_coords, ligand_atom_types, ligand_mask)`;
  LigandBundle traces into JIT kernels; checkpoint `ligandmpnn_v_32_010_25_converted.eqx`
- **LigandMPNN PyTorch** via `featurize(..., model_type="ligand_mpnn")` which computes Y/Y_t/Y_m
  nearest-neighbor features; checkpoint `ligandmpnn_v_32_010_25.pt`

**Key difference vs no-ligand bench:** featurize() performs offline nearest-neighbor lookup for
the nearest `number_of_ligand_atoms=16` atoms per protein residue CB, shaping Y/Y_t/Y_m into
(L, 16) arrays. The JAX adapter must replicate this host-side computation before calling
build_inference_bundle.

---

## 2. Design Constraints & Open Decisions

### 2.1 Ligand Fixture Acquisition (BLOCKING — §10)

No protein-ligand complex PDB currently in `tests/data/`. Options:

| Option | PDB | Residues | Ligand | Pros | Cons |
|---|---|---|---|---|---|
| **A** | 1BC8 chain C | 93 | DNA (chains A/B) | Canonical LigandMPNN example | DNA, not small molecule |
| **B** | 5awl | 10 | Water | Already in tests/data | Trivially small; HOH not meaningful |
| **C (preferred)** | TBD from RCSB/PDBbind | 50–150 | Small molecule | Realistic drug-design scenario | Requires download + verification |

**Recommendation:** Option C for final runs; Option A (1BC8) acceptable for adapter debugging.

### 2.2 Ligand Atom Types Mapping

**PyTorch:** `element_dict = {elem: idx for idx, elem in enumerate(element_list)}` from
`data_utils.py`; Y_t shape `(L, 16)` nearest atoms per residue CB.

**JAX:** LigandBundle expects `ligand_atom_types: Int[Array, "S L_lig A"]` where A=16.
No element mapping currently defined — must align with PyTorch's element_list ordering.

**Decision:** JAX adapter defines element_dict locally (avoids REFERENCE_PATH runtime dep).

### 2.3 Nearest-Neighbor: Host-Side vs JIT

**Recommendation for this bench:** host-side with scipy.spatial (mirrors PyTorch approach).
Deferred JIT kernel is a future optimisation.

---

## 3. Benchmark Matrix

### 3.1 Core Matrix

| Dimension | Levels | Notes |
|---|---|---|
| seq_len | TBD (target 50–150) | Once ligand fixture acquired |
| batch_size | 1, 4, 16 | Same as no-ligand; test ligand overhead at scale |
| ligand_conditioning | True | Both adapters use ligand checkpoint + ligand data |
| precision | bf16 (primary), fp32 (secondary) | Same as 260601 |
| framework | aminx (JAX), ligandmpnn (PyTorch) | Subprocess isolation |
| task | score_conditional, ar_sample | Both support ligand conditioning |

**Fixed:** axis_strategy=Vmap; num_nearest_ligand_atoms=16; model_type="ligand_mpnn" (PyTorch).

### 3.2 Comparison Axes (ligand cost quantification)

| Metric | Comparison | Expected |
|---|---|---|
| latency_per_residue_us | ligand=False vs True, same seq_len + framework | +15–25% |
| throughput_seq_per_s (batch scaling) | batch=1 vs 4 vs 16 | Earlier plateau with ligand |
| peak_gpu_memory_gb | batch=1, ligand=True vs False | +0.1–0.3 GB |

---

## 4. JSON Output Schema Extension (v2)

All v1 fields retained. New fields:

```json
{
  "schema_version": "2",
  "ligand_conditioning": true,
  "ligand_atom_count": 47,
  "ligand_nearest_neighbors": 16
}
```

---

## 5. Adapter Changes Required

### 5.1 JAX Adapter (`bench_aminx_jax.py`)

1. **Extend `_PDB_MAP`** to include new ligand PDB once acquired.
2. **Extend `load_pdb_as_arrays()`** to return ligand atoms `(N_lig, 3)` + element symbols.
3. **Add `compute_ligand_features_jax()`** — host-side nearest-neighbor (scipy.spatial.distance):

   ```python
   def compute_ligand_features_jax(
       cb_coords: np.ndarray,         # (L, 3)
       ligand_coords: np.ndarray,      # (N_lig, 3)
       ligand_elements: list[str],     # element symbols
       num_neighbors: int = 16,
   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
       # → (Y_knn, Y_t_knn, Y_m_knn)  shapes (L, 16, 3), (L, 16), (L, 16)
   ```

4. **Extend `benchmark_cell()`** with `ligand_enabled` flag; populate `build_inference_bundle`
   with ligand arrays when enabled; load `ligandmpnn_v_32_010_25_converted.eqx` checkpoint.
5. **Add `--ligand` CLI flag** (default off; no-ligand matrix unchanged).

### 5.2 PyTorch Adapter (`bench_ligandmpnn_pytorch.py`)

1. **Extend `_PDB_MAP`** (same fixture as JAX).
2. **Add `extract_ligand_features_from_pdb()`** — extract Y/Y_t/Y_m from PDB ligand atoms.
3. **Populate `protein_dict`** with `"Y"`, `"Y_t"`, `"Y_m"` tensors when `--ligand-conditioning`;
   switch model_type to `"ligand_mpnn"` and checkpoint to `ligandmpnn_v_32_010_25.pt`.
4. **Add `--ligand-conditioning` CLI flag.**

---

## 6. Timing Methodology (Ligand-Specific)

### 6.1 JAX

Host-side nearest-neighbor is NOT timed. Cold compile and warm latency measure only
the JIT-compiled inference path:

```python
# Host-side (not timed)
Y_knn, Y_t_knn, Y_m_knn = compute_ligand_features_jax(cb_coords, lig_coords, lig_elems)

# Cold compile (timed): full JIT with ligand arrays present in bundle
jax.config.update("jax_enable_compilation_cache", False); jax.clear_caches()
t0 = time.perf_counter()
result = plan.sample(bundle_with_ligand, key, config)
jax.block_until_ready(result)
compile_time_cold_s = time.perf_counter() - t0
```

### 6.2 PyTorch

Cold overhead **includes** featurize() nearest-neighbor computation (intentional — measures
total pipeline latency from raw arrays to result, matching production usage):

```python
t0 = time.perf_counter()
feature_dict = featurize(protein_dict_with_ligand,
                          model_type="ligand_mpnn", number_of_ligand_atoms=16)
with torch.no_grad():
    out = model.sample(feature_dict)
torch.cuda.synchronize()
cold_overhead_s = time.perf_counter() - t0
```

---

## 7. Expected Calibration

| Metric | No-Ligand (from 260601) | With-Ligand (expected) |
|---|---|---|
| latency_median_ms (L~93, B=1, A100) | ~9 ms | ~11 ms |
| latency_per_residue_us | ~97 | ~118 |
| peak_gpu_memory_gb (B=1) | ~0.6 | ~0.8 |

If JAX ligand latency exceeds PyTorch by >2×, investigate LigandBundle re-tracing
(static field issue with ligand arrays — see debt item #68).

---

## 8. Implementation Roadmap

### Wave 0 — Fixture (no GPU)
- [ ] W0-1: Identify PDB — confirm molecule type, residue count, ligand atom count
- [ ] W0-2: Fetch + push to cluster via rsync; verify Bio.PDB + REFERENCE_PATH/parse_PDB both succeed

### Wave 1 — JAX Adapter Extension
- [ ] W1-1: Extend bench_aminx_jax.py (load ligand, compute features, build_inference_bundle)
- [ ] W1-2: L1 dry-run; L2 smoke test (L~93, B=1, n_warmup=1, n_timed=3)

### Wave 2 — PyTorch Adapter Extension
- [ ] W2-1: Extend bench_ligandmpnn_pytorch.py (ligand PDB load, Y/Y_t/Y_m extraction)
- [ ] W2-2: L1 dry-run; L2 smoke test

### Wave 3 — Harness + Cross-Check
- [ ] W3-1: Add `--ligand` dispatch path to bench_suite.py
- [ ] W3-2: Verify both adapters produce valid schema_version=2 JSON
- [ ] W3-3: Side-by-side run on same GPU, check isolation

### Wave 4 — Cluster Validation
- [ ] W4-1: Submit to A100/H200/Blackwell with ligand flags
- [ ] W4-2: Collect and analyse; write §3.2 comparison table
- [ ] W4-3: Gate: ligand overhead within expected range → approve for pre-merge report

---

## 9. Open Questions

1. **Element indexing:** Define locally in JAX adapter, or import from REFERENCE_PATH/data_utils.py?
   (Recommend: local, for hermeticity.)
2. **Fixture source:** 1BC8 (DNA) acceptable placeholder for Wave 1 debugging?
3. **Batch>1 ligand in PyTorch:** Does LigandMPNN support multiple ligand contexts per batch, or
   does each sequence share the same ligand? Check REFERENCE_PATH/run.py lines 400–460.
4. **build_inference_bundle batching with ligand:** ligand_coords/atom_types shape for batch>1 —
   `(B, N_lig, 3)` or broadcast `(N_lig, 3)`? Needs investigation before Wave 1. (See debt #68.)

---

## 10. Blocking Prerequisite

**Cannot begin Wave 1 until** a protein-ligand PDB is acquired and verified parseable by both
Bio.PDB (JAX adapter) and REFERENCE_PATH/data_utils.py parse_PDB (PyTorch adapter).

Target: 50–150 protein residues; 5–50 ligand atoms; small molecule (not DNA/RNA/water).

Suggested source: RCSB PDB full-text search, filter resolution ≤ 2.5 Å, single chain,
chain length 50–150. Or use an example from the LigandMPNN paper's test set.
