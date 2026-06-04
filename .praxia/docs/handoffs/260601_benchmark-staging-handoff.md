---
task_id: 260601_benchmark-staging
session_id: 88e9cc8f-6185-4aa1-9b07-1ed6dbbbe33f
status: in_progress
phase: Wave 1 complete — Wave 2 prereqs + implementation next
date: 260601
---

# Handoff: Benchmark Suite Staging (260601_benchmark-staging)

## Goal

Pre-merge GPU benchmark suite comparing three implementations of protein sequence design:
- **aminx** (JAX + Equinox, ours)
- **dauparas/LigandMPNN** (PyTorch reference, commit 3870631)
- **sokrypton/ColabDesign** (JAX ProteinMPNN, no ligand conditioning)

Hardware targets: A100, H100, H200, L40, Blackwell SM120 (node4007/node4008).

## What Was Done This Session

| Commit | Description |
|---|---|
| `d976a12` | Benchmark spec written: `.praxia/docs/specs/260601_benchmark-spec.md` |
| `42b94ac` | ColabDesign added to comparison matrix; pyproject.toml benchmark dep group; open questions resolved |
| `c14f8c9` | Wave 0: `scripts/benchmarks/prepare_fixtures.py` (pad/truncate 1ubq to L=[76,150,300,500]) |
| `cae498f` | Oracle required changes applied: Blackwell two-run protocol, PyTorch eager label, L=500 real fixture, ColabDesign git-SHA pin prereq |
| `78bc404` | Cosmetic: spec §3.3/§3.4 reordered (Blackwell before Latency decomp) |
| `2c34e69` | Wave 1: `scripts/benchmarks/bench_aminx_jax.py` (645 lines, full CLI, cold/warm timing, JSON §5.1 contract) |
| `10402f0` | Fix: batch dim was jnp.repeat → jnp.stack; block_until_ready unconditional |

**Spec status:** Oracle APPROVED (two-pass review). Safe to implement.

**Wave 1 status:** DONE. L1 (dry-run) and L2 (smoke) both pass. JSON output matches §5.1 schema exactly.

## Immediately Relevant Files

1. **`.praxia/docs/specs/260601_benchmark-spec.md`** — oracle-approved spec. Read in full before starting any Wave. Key sections:
   - §3.1 core matrix (seq_len, batch_size, ligand, precision, framework)
   - §3.4 Blackwell two-run protocol (representative vs reproducible)
   - §5.1 JSON output contract for all adapters
   - §6.1 JAX timing methodology (cold/warm/AOT/effects_barrier)
   - §6.2 PyTorch timing (eager, no torch.compile — first_call_overhead not compile_time)
   - §8 fixture table (L=500 must use real structure, NOT padded 1ubq)
   - §11 six open questions/prereqs before Wave 2

2. **`scripts/benchmarks/prepare_fixtures.py`** — Wave 0 complete. Generates `outputs/benchmark_fixtures/structure_L{76,150,300,500}.npz`. Coords are (L,4,3) backbone. **Note: L=500 slot uses padded 1ubq — must be replaced with a real ≥400-residue structure before cluster runs** (spec §8 forbids this padding because 424/500 masked residues under-reports latency by ~20–40%).

3. **`pyproject.toml`** lines 214–230 — `[dependency-groups].benchmark = ["colabdesign"]`. Currently unpinned PyPI ref. **Must pin to git SHA before Wave 2.** Install cmd: `uv sync --extra cuda --group dev --group benchmark`

4. **`scripts/engaging/submit_parity_heavy_ligand.sh`** lines 29–80 — SLURM template to adapt for benchmark scripts. Has correct env setup: `REFERENCE_PATH`, `JAX_PLATFORMS=cuda`, `uv sync --extra cuda --group dev`.

5. **`git:132eca7:scripts/benchmarks/bench_inference_plan_latency.py`** — Wave 1 JAX adapter template. NOT on main (lives in the parity-merge branch). 396 lines. Has correct `jax.block_until_ready()` + warmup + timed + median pattern. Extend this for the full benchmark adapter.

## Next Steps (Ordered)

### Before Wave 1 starts
- [x] **Cosmetic fix**: §3.3/§3.4 reordered in spec (commit 78bc404).

### Wave 1: JAX adapter
- [x] `scripts/benchmarks/bench_aminx_jax.py` written and L1/L2 verified (commits 2c34e69, 10402f0).
  - **Known gap**: `_BenchmarkSpec.average_node_features = False` was added to avoid InferencePlan error — verify this field is correct for the no-ligand/no-fusion benchmark path before cluster runs.
  - **Known gap**: Precision (`bf16`/`fp32`) is recorded in JSON but not yet applied to model params. Both runs use fp32 weights. Fix before final cluster submission if bf16 performance numbers are needed.
  - **GPU memory** is 0.0 placeholder — add pynvml or nvidia-smi query for real values on cluster.

### Before Wave 2 starts (prereqs)
- [ ] **PyTorch batching**: SSH to cluster, `grep -n "num_seq\|batch\|for.*seq" ${REFERENCE_PATH}/run.py | head -30`. If loop-only (no vectorized batch dim), restrict PyTorch matrix to `batch_size=1` and update spec §3.1 accordingly.
- [ ] **ColabDesign SHA**: Look up latest commit on sokrypton/ColabDesign GitHub. Update `pyproject.toml` benchmark group from `"colabdesign"` to `"colabdesign @ git+https://github.com/sokrypton/ColabDesign.git@<sha>"`.
- [ ] **L=500 fixture**: Find a real single-chain ≥400-residue PDB (e.g. search RCSB for a monomeric protein ~450–550 residues). Fetch locally: `wget https://files.rcsb.org/download/<ID>.pdb -O tests/data/<ID>.pdb`. Push to cluster with rsync.
- [ ] **Ligand fixture**: For ligand=True benchmark path — 1ubq has no ligand, 5awl.pdb is only 10 residues. Options: (a) use a real ligand-protein complex PDB from the parity corpus (check `REFERENCE_PATH/inputs/` for examples), or (b) synthetic dummy ligand (small random coordinate array). Resolve before Wave 2.

### Wave 2: PyTorch + ColabDesign adapters
- [ ] Write `scripts/benchmarks/bench_ligandmpnn_pytorch.py` — `torch.utils.benchmark.Timer.blocked_autorange()`, load REFERENCE_PATH model, `first_call_overhead_s` (not compile_time), JSON output
- [ ] Write `scripts/benchmarks/bench_colabdesign_jax.py` — ColabDesign ProteinMPNN, no-ligand path only, same `block_until_ready` timing methodology as JAX adapter

### Wave 3: Harness + Report
- [ ] `bench_suite.py` — subprocess harness (no JAX/torch import), launches each adapter, collects JSON
- [ ] `bench_report.py` — JSON → markdown + CSV comparison table

### Wave 4: SLURM scripts
Write 6 scripts in `scripts/engaging/`:
- `submit_benchmark_a100.sh` — `GPU_TARGET=A100`
- `submit_benchmark_h100.sh` — `GPU_TARGET=H100`
- `submit_benchmark_h200.sh` — `GPU_TARGET=H200`
- `submit_benchmark_l40.sh` — `GPU_TARGET=L40`
- `submit_benchmark_blackwell_representative.sh` — `GPU_TARGET=Blackwell_SM120`, `--nodelist=node4007,node4008`, only `--xla_gpu_shard_autotuning=false`
- `submit_benchmark_blackwell_reproducible.sh` — same but adds `--xla_gpu_autotune_level=0`

All scripts must set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
Template from: `scripts/engaging/submit_parity_heavy_ligand.sh`.

### Wave 5: Results + report
- [ ] Submit Wave 4 SLURM scripts (L3 smoke with `--array=0-0` first)
- [ ] Collect JSON from `outputs/results/benchmarks/`
- [ ] Run `bench_report.py`
- [ ] Write benchmark section of pre-merge report

## Known Issues / Failed Approaches

**Do not cherry-pick from worktree branches that predate the target file's creation.**
When worktrees branch from a commit before a file was created on main, cherry-pick produces an add/add conflict even though the content is correct.
→ Instead: `git checkout <worktree-branch> -- <file> && git add <file> && git commit`

## Hardware Notes

- **Blackwell SM120** (node4007/node4008): XLA shard-autotuning HANGS without `--xla_gpu_shard_autotuning=false`. Two-run protocol per spec §3.4.
- **L40**: GDDR6 (864 GB/s), NOT HBM. Will be bandwidth-limited for autoregressive decode. Report separately with GDDR6 footnote.
- **Bandwidth order** (bandwidth-bound AR decode): H200 > H100 > A100 >> L40

## Open Questions

1. Does `${REFERENCE_PATH}/run.py` support native batch dim or loop-over-sequences?
2. What real ≥400-residue PDB to use for L=500 fixture?
3. What ColabDesign git SHA to pin?
4. Ligand fixture for ligand=True benchmark path?
