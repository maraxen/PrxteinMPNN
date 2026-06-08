---
title: Benchmark Suite Spec — prxteinmpnn vs LigandMPNN PyTorch Reference
task_id: 260601_benchmark-staging
date: 260601
status: draft
---

# Benchmark Suite Spec: prxteinmpnn JAX vs LigandMPNN PyTorch

Pre-merge benchmark suite to produce authoritative GPU throughput, latency, and memory numbers for prxteinmpnn across five hardware targets. Compares our JAX+Equinox implementation against the dauparas/LigandMPNN PyTorch reference (commit 3870631).

## 1. Scope and Goals

**Primary goal:** Produce GPU throughput and latency tables for the pre-merge report. The original LigandMPNN paper reports 0.9s/100-res CPU only — no GPU batch throughput numbers exist in the literature, so this benchmark generates the first calibrated numbers.

**Secondary goal:** Characterize compile overhead, memory footprint, and protein-length scaling to inform deployment decisions.

**Reference note on "ColabDesign":** The dauparas/LigandMPNN repo (commit 3870631) includes both the unconditional ProteinMPNN path and the LigandMPNN ligand-conditioned path. It is the complete comparison target. A separate ColabDesign/sokrypton checkout is not available locally and is not required for this benchmark.

---

## 2. Hardware Targets

| Target | Partition | Memory BW | Notes |
|---|---|---|---|
| A100 | `mit_normal_gpu` or `pi_so3` | 1.555 TB/s HBM2e | Baseline |
| H100 | `pi_so3` | 3.35 TB/s HBM3 | ~6× compute vs A100 |
| H200 | `pi_so3` | 4.89 TB/s HBM3e | Best bandwidth; same compute as H100 |
| L40 | check `sinfo` | 864 GB/s GDDR6 | Bandwidth-limited for AR decode |
| Blackwell SM120 | `pi_so3` | HBM3 | node4007/node4008; XLA flags mandatory (see §7) |

Autoregressive decode is **bandwidth-bound**, so the expected performance order for per-sequence latency is: H200 > H100 > A100 >> L40. Blackwell results are architectural data, not a fair comparison without autotuning enabled.

---

## 3. Benchmark Matrix

### 3.1 Core (headline comparison: JAX vs PyTorch)

| Dimension | Levels | Notes |
|---|---|---|
| seq_len | 76, 150, 300, 500 | 76 = 1ubq.pdb; 150/300/500 from RCSB or truncation |
| batch_size | 1, 4, 16 | For throughput (seq/s) measurement |
| ligand_conditioning | False, True | Both modes present in LigandMPNN reference |
| precision | bf16 (primary), fp32 (secondary) | BF16 = native Tensor Core on all 5 targets |
| framework | prxteinmpnn (JAX), ligandmpnn (PyTorch) | Separate subprocess per framework |

**Config fixed for core matrix:** AxisStrategy=Vmap, average_encoding_mode=inputs_and_noise, sidechain_conditioning=False (baseline), temperature sampling.

### 3.2 Secondary (JAX-only axis sweep)

| Dimension | Levels | Fixed |
|---|---|---|
| AxisStrategy | Vmap, SafeMap(tile=8), SafeMap(tile=1) | seq_len=300, batch=4, ligand=False, bf16 |
| average_encoding_mode | inputs, noise_levels, inputs_and_noise | Vmap, batch=4, ligand=False |

### 3.3 Blackwell SM120: two-run protocol

Blackwell (SM120) requires special treatment due to the XLA shard-autotuning hang. Run two distinct jobs on node4007/node4008:

| Run label | XLA flags set | Purpose |
|---|---|---|
| `blackwell_representative` | `--xla_gpu_shard_autotuning=false` only | Performance-representative numbers; autotuning may complete or timeout silently |
| `blackwell_reproducible` | `--xla_gpu_shard_autotuning=false --xla_gpu_autotune_level=0` | Reproducible compile times; autotuning disabled; performance may be 10–40% lower than representative |

Both runs use identical sequence lengths, batch sizes, and precision. Report them as separate rows in all benchmark tables. Do not merge or average.

### 3.4 Latency decomposition (JAX-only)

Measure encode-only, decode-only, and full encode+decode separately at seq_len=[76, 300, 500]:
- `plan.encode(bundle, key, config)` → encode latency
- `plan.decode(enc, bundle, key, config)` → decode latency
- `runner.sample(...)` → full wall-clock (uses io_callback streaming path → `jax.effects_barrier()`)

---

## 4. Metrics

Report per `(model, hardware, seq_len, batch_size, precision, ligand_conditioning)` cell:

| Metric | Unit | Notes |
|---|---|---|
| `compile_time_cold_s` | s | First call; XLA compilation included; cache disabled |
| `compile_time_warm_s` | s | AOT-compiled warm call overhead; should be ≈0 |
| `latency_median_ms` | ms | Median of n_timed=20 warm calls |
| `latency_p95_ms` | ms | 95th percentile |
| `latency_per_residue_us` | μs/res | latency_median_ms × 1000 / seq_len |
| `throughput_seq_per_s` | seq/s | batch_size / (latency_median_ms / 1000) |
| `peak_gpu_memory_gb` | GB | Peak allocated during timed run |

**Calibration anchor:** LigandMPNN paper CPU baseline is 0.9s/100-res single-sequence. At seq_len=76 batch=1 on A100, the PyTorch adapter should be substantially faster as a sanity check.

---

## 5. Script Architecture

```
scripts/benchmarks/
├── bench_prxteinmpnn_jax.py       # JAX adapter (extends commit 132eca7 template)
├── bench_ligandmpnn_pytorch.py    # PyTorch adapter (dauparas ref)
├── bench_colabdesign_jax.py       # JAX adapter (ColabDesign ProteinMPNN, no-ligand only)
├── bench_suite.py                 # Subprocess harness — no JAX/torch import
└── bench_report.py                # JSON → comparison table (markdown + CSV)

scripts/engaging/
├── submit_benchmark_a100.sh
├── submit_benchmark_h100.sh
├── submit_benchmark_h200.sh
├── submit_benchmark_l40.sh
└── submit_benchmark_blackwell.sh
```

### 5.1 JSON output contract (all adapters)

```json
{
  "schema_version": "1",
  "model": "prxteinmpnn_jax | ligandmpnn_pytorch",
  "hardware": "A100 | H100 | H200 | L40 | Blackwell_SM120",
  "seq_len": 100,
  "batch_size": 1,
  "precision": "bf16 | fp32",
  "ligand_conditioning": false,
  "axis_strategy": "Vmap | SafeMap_tile8 | Scan | null",
  "average_encoding_mode": "inputs_and_noise | inputs | noise_levels | null",
  "compile_time_cold_s": 12.3,
  "compile_time_warm_s": 0.001,
  "compile_time_note": "JAX: XLA compilation. PyTorch (eager, no torch.compile): CUDA kernel warmup only; value is first_call overhead, not compiler overhead.",
  "latency_median_ms": 8.4,
  "latency_p95_ms": 9.1,
  "latency_per_residue_us": 84.0,
  "throughput_seq_per_s": 119.0,
  "peak_gpu_memory_gb": 0.6,
  "n_warmup": 10,
  "n_timed": 20,
  "jax_version": null,
  "torch_version": null,
  "cuda_version": null,
  "timestamp_utc": null
}
```

---

## 6. Timing Methodology

### 6.1 JAX adapter

**Cold compile time:**
```python
import os
os.environ.setdefault("XLA_FLAGS",
    "--xla_gpu_autotune_level=0 --xla_gpu_shard_autotuning=false")
import jax
jax.config.update("jax_enable_compilation_cache", False)
jax.clear_caches()
t0 = time.perf_counter()
result = plan.decode(enc, bundle, key, config)
jax.block_until_ready(result)
compile_time_cold_s = time.perf_counter() - t0
```

**Warm throughput (AOT path):**
```python
compiled = jax.jit(plan.decode).lower(
    abstract_enc, abstract_bundle, abstract_key, abstract_config
).compile()
for _ in range(n_warmup):
    jax.block_until_ready(compiled(enc, bundle, key, config))
times = []
for _ in range(n_timed):
    t0 = time.perf_counter()
    jax.block_until_ready(compiled(enc, bundle, key, config))
    times.append(time.perf_counter() - t0)
```

**Streaming path:** Use `jax.effects_barrier()` instead of `block_until_ready` — runner.sample() uses io_callback which is an effect.

### 6.2 PyTorch adapter

```python
import torch.utils.benchmark as benchmark

# Cold compile time
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
with torch.no_grad():
    out = model(inputs)  # first call: CUDA kernel warmup (LigandMPNN ref is eager — no torch.compile)
torch.cuda.synchronize()
first_call_overhead_s = time.perf_counter() - t0  # CUDA warmup, NOT compiler overhead (eager model)

# Warm throughput
timer = benchmark.Timer(
    stmt="model(inputs)",
    globals={"model": model, "inputs": inputs},
    num_threads=1,
)
result = timer.blocked_autorange(min_run_time=2.0)
latency_median_ms = result.median * 1e3

peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / 1e9
```

**Process isolation:** bench_suite.py launches each adapter as a separate `subprocess.run()` call. Neither script imports the other framework. This prevents GPU memory pool cross-contamination — both JAX and PyTorch preallocate GPU memory on import.

---

## 7. Blackwell SM120 XLA Flags

Set in SLURM script **before** any `uv run` invocation:

```bash
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false --xla_gpu_autotune_level=0"
    echo "Blackwell SM120 XLA flags active: ${XLA_FLAGS}"
fi
```

`--xla_gpu_autotune_level=0` disables kernel autotuning for reproducible compile times. **This changes performance, not just reproducibility** — see §3.4 for the two-run protocol (representative vs reproducible). Also set `XLA_PYTHON_CLIENT_PREALLOCATE=false` in the SLURM environment for all benchmark jobs to prevent JAX GPU memory preallocation from exhausting device memory when subprocesses run sequentially.

---

## 8. Protein Length Fixtures

| seq_len | Source | Notes |
|---|---|---|
| 76 | `tests/data/1ubq.pdb` | Available locally |
| 150 | Pad 1ubq | Acceptable — minority masked |
| 300 | Pad 1ubq | Acceptable — minority masked |
| 500 | `tests/data/1SMD.pdb` (PDB 1SMD, human salivary amylase, 495 residues) ✅ | **Do NOT pad 1ubq to L=500** — 424/500 residues would be masked zeros, producing ~20–40% under-reported latency. 1SMD needs only 5 pad residues. Added 2026-06-01. |

Cluster compute nodes have no outbound internet. Fetch all structures locally first, push via rsync.

The `create_bundle_and_plan_from_real_structure` function in commit 132eca7 implements truncation + zero-padding. Reuse truncation for all seq_len variants. **Padding is acceptable for L=150 and L=300 (minority of residues masked); avoid for L=500 where padding would dominate.**

---

## 9. SLURM Script Template

Base each `submit_benchmark_<gpu>.sh` on `scripts/engaging/submit_parity_heavy_ligand.sh`:

```bash
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/slurm/benchmark_%j.out

export REFERENCE_PATH="${REFERENCE_PATH:-/home/maarxaru/repos/LigandMPNN}"
export JAX_PLATFORMS=cuda

uv run python scripts/benchmarks/bench_suite.py \
    --hardware "${GPU_TARGET}" \
    --output-dir outputs/results/benchmarks/ \
    --seq-lens 76 150 300 500 \
    --batch-sizes 1 4 16 \
    --n-warmup 10 \
    --n-timed 20
```

---

## 10. Implementation Waves

```
Wave 0 (fixtures — no GPU needed):
  [ ] B-0a  Fetch 2KHO and 4ZYP locally; push to cluster
  [ ] B-0b  Verify padding logic from commit 132eca7 handles all target seq_lens

Wave 1 (JAX adapter — extends 132eca7):
  [ ] B-1a  bench_prxteinmpnn_jax.py: cold/warm timing, JSON output, --seq-lens, --batch-sizes, --precision
  [ ] B-1b  Local L1 dry-run + L2 smoke on CPU

Wave 2 (PyTorch adapter):
  [ ] B-2a  bench_ligandmpnn_pytorch.py: torch.utils.benchmark.Timer, same JSON contract
  [ ] B-2b  Local L1 dry-run + L2 smoke on CPU
  [ ] B-2c  Verify REFERENCE_PATH imports work correctly

Wave 3 (harness + report):
  [ ] B-3a  bench_suite.py: subprocess dispatch, JSON aggregation
  [ ] B-3b  bench_report.py: markdown table + CSV output

Wave 4 (cluster submission):
  [ ] B-4a  submit_benchmark_a100.sh — L3 smoke (array=0-0)
  [ ] B-4b  submit_benchmark_h100.sh, h200, l40
  [ ] B-4c  submit_benchmark_blackwell.sh (with XLA flags, node4007/8)

Wave 5 (report):
  [ ] B-5a  Collect all JSON outputs, run bench_report.py
  [ ] B-5b  Write benchmark section of pre-merge report
```

---

## 11. Open Questions Before Wave 1

> **Status (2026-06-01):** All six questions below are resolved. Wave 2 is unblocked.

1. **Ligand coordinates**: ✅ **Resolved.** `bench_ligandmpnn_pytorch.py` will use `REFERENCE_PATH/inputs/1BC8.pdb` directly on the cluster — the LigandMPNN reference repo includes this structure as the canonical example for its `ligand_mpnn` model type. No local copy needed. For the JAX adapter's ligand=True path (post-Wave 2), a synthetic dummy ligand will be generated in `prepare_fixtures.py`.

2. **ColabDesign:** ✅ **Resolved — in scope.** `sokrypton/ColabDesign` is a widely-used JAX implementation of ProteinMPNN. It is not installed locally or on the cluster — being added as a `benchmark` dependency group. ColabDesign covers the no-ligand ProteinMPNN path only (no ligand conditioning). The comparison matrix is:

   | Path | prxteinmpnn (JAX+Equinox) | LigandMPNN (PyTorch) | ColabDesign (JAX) |
   |---|---|---|---|
   | No ligand | ✓ | ✓ | ✓ |
   | With ligand | ✓ | ✓ | — |

   A third adapter script `bench_colabdesign_jax.py` is needed. Since ColabDesign is JAX, the same `block_until_ready` + `clear_caches` timing methodology applies.

3. **Model size:** ✅ **Resolved** — use production weights from REFERENCE_PATH checkpoints (`ligandmpnn_v_32_010_25_converted.eqx` for JAX, `ligandmpnn_v_32_010_25.pt` for PyTorch). Both adapters load the same weights for a fair head-to-head comparison.

4. **AOT abstract shapes**: `jax.jit(...).lower()` requires fixed shapes. For variable seq_len, compile once per length. This is correct for benchmarks but means 4 compiled artifacts per configuration. Use `jax.eval_shape` to introspect leaf shapes rather than hand-rolling abstract arg builders — add this to Wave 1 scope.

5. **PyTorch batching:** ✅ **Resolved (2026-06-01).** `${REFERENCE_PATH}/run.py` line 402 sets `feature_dict["batch_size"] = args.batch_size` and the randn shape is `[batch_size, L]`, so the model IS vectorized over a true sequence batch dim. The comment "batch size should be 1 for now" (line 403) refers to the structure batch (B=1, always one PDB at a time); `batch_size` controls simultaneous sequence generation. The PyTorch adapter benchmark matrix can use `batch_size=[1, 4, 16]` as planned, measuring batched-forward throughput.

6. **ColabDesign SHA:** ✅ **Resolved (2026-06-01).** Pinned to `e31a56fe1d9b4de25c8697f3a28b75892941cc72` (2025-10-23, "Update mapping.py") in `pyproject.toml [dependency-groups].benchmark`. Install: `uv sync --extra cuda --group dev --group benchmark`.
