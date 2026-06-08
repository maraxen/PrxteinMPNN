---
task_id: 260601_benchmark-staging
session_id: bg-wave2-prereqs-260601
status: in_progress
phase: Wave 2 prereqs complete — merge worktree then write two Wave 2 adapters
date: 260601
---

# Handoff: Benchmark Suite — Wave 2 Prereqs Complete (260601_benchmark-staging)

## Goal

Pre-merge GPU benchmark suite comparing three implementations of protein sequence design:
- **prxteinmpnn** (JAX + Equinox, ours)
- **dauparas/LigandMPNN** (PyTorch reference, commit 3870631)
- **sokrypton/ColabDesign** (JAX ProteinMPNN, no ligand conditioning)

Hardware targets: A100, H100, H200, L40, Blackwell SM120 (node4007/node4008).

## What Was Done This Session

| Commit | Description |
|---|---|
| `feat(benchmark-wave2-prereqs)` | pyproject.toml: pin ColabDesign SHA e31a56fe; tests/data/1SMD.pdb added; prepare_fixtures.py uses 1SMD for L=500 |
| `docs(benchmark-spec)` | spec §11 all 6 prereqs marked resolved; §8 fixture table updated with 1SMD |

**Worktree**: `.claude/worktrees/benchmark-wave2-prereqs` — 2 commits ahead of main, ready to merge.

## Four Prereqs Resolved

| Prereq | Resolution |
|---|---|
| PyTorch batch dim | `run.py` line 402 sets `feature_dict["batch_size"]` and randn shape is `[batch_size, L]` — vectorized over sequence generation. Comment "batch size should be 1 for now" refers to structure batch (B=1), NOT sequence batch. batch_size=[1,4,16] valid for PyTorch adapter. |
| ColabDesign SHA | `e31a56fe1d9b4de25c8697f3a28b75892941cc72` (2025-10-23). Pinned in `pyproject.toml` benchmark dep group. |
| L=500 fixture | `tests/data/1SMD.pdb` (human salivary amylase, 495 residues). Only 5 pad residues at L=500, vs old 1ubq approach (424/500 masked). `prepare_fixtures.py` updated. |
| Ligand fixture | `bench_ligandmpnn_pytorch.py` will use `REFERENCE_PATH/inputs/1BC8.pdb` directly on cluster. LigandMPNN's own canonical ligand_mpnn example. No local copy needed. |

## Immediately Relevant Files

1. **`.claude/worktrees/benchmark-wave2-prereqs/`** — working branch with Wave 2 prereq commits. Merge before starting Wave 2.
2. **`pyproject.toml` line 225** — `benchmark = ["colabdesign @ git+...@e31a56fe..."]`. Install: `uv sync --extra cuda --group dev --group benchmark`
3. **`tests/data/1SMD.pdb`** — 495-residue salivary amylase for L=500 benchmark fixture.
4. **`scripts/benchmarks/prepare_fixtures.py`** — now produces correct L=500 from 1SMD with 1ubq fallback+warning.
5. **`scripts/benchmarks/bench_prxteinmpnn_jax.py`** — Wave 1 adapter (645 lines). `ligand_conditioning=False` hardcoded; `_BenchmarkSpec.average_node_features=False`.
6. **`.praxia/docs/specs/260601_benchmark-spec.md`** — oracle-approved, §11 all resolved. Key sections: §3.1 matrix, §5.1 JSON contract, §6.2 PyTorch timing, §8 fixture table.
7. **`REFERENCE_PATH=/home/maarxaru/repos/LigandMPNN`** on cluster — has `inputs/1BC8.pdb` (ZN ligand, 113 residues) for ligand=True path.

## Next Steps (Ordered)

### Merge worktree
```bash
git -C /home/marielle/projects/tev_design/prxteinmpnn merge --no-ff worktree-benchmark-wave2-prereqs \
  -m "feat(benchmark-wave2-prereqs): ColabDesign SHA pin, 1SMD L500 fixture, spec §11 resolved"
```

### Regenerate fixtures (from tev_design root — workspace uv.lock lives there)
```bash
cd /home/marielle/projects/tev_design
uv run python prxteinmpnn/scripts/benchmarks/prepare_fixtures.py \
  --fixture-dir prxteinmpnn/outputs/benchmark_fixtures/ --verbose
```
This will regenerate all 4 NPZ files, now with structure_L500.npz from 1SMD instead of padded 1ubq.

### Push to cluster
```bash
just -g cluster-push-submit tev_design scripts/benchmarks/prepare_fixtures.py
```
Or just rsync if not submitting a SLURM job yet.

### Wave 2A: `scripts/benchmarks/bench_ligandmpnn_pytorch.py`
Key spec requirements (§6.2):
- Use `torch.utils.benchmark.Timer.blocked_autorange()` for timing
- Load model: `REFERENCE_PATH/model_params/ligandmpnn_v_32_010_25.pt` (NOT `.eqx`)
- Load structure via `featurize()` from LigandMPNN's `data_utils.py`
- `first_call_overhead_s` field (not `compile_time_cold_s`) — PyTorch has no JIT compile
- `torch_version` in JSON, `jax_version: null`
- ligand=True path: use `REFERENCE_PATH/inputs/1BC8.pdb`, `model_type="ligand_mpnn"`
- `ligand=False` path: use structure fixture npz (adapt featurize to accept raw coords)
- JSON output must match §5.1 schema exactly

### Wave 2B: `scripts/benchmarks/bench_colabdesign_jax.py`
Key spec requirements:
- Install via `uv sync --group benchmark` (ColabDesign pinned SHA)
- No-ligand path only (ColabDesign ProteinMPNN has no ligand conditioning)
- Same `jax.block_until_ready()` + `jax.clear_caches()` timing methodology as Wave 1
- Same §5.1 JSON contract (`model: "colabdesign_jax"`)

## Known Issues / Failed Approaches

**uv run from prxteinmpnn worktree subdirectory fails.** The uv.lock lives at `tev_design/`, one level up. Always run:
```bash
cd /home/marielle/projects/tev_design && uv run python prxteinmpnn/scripts/...
```

**Do not cherry-pick from worktrees that predate a target file's creation.**
→ Instead: `git checkout <branch> -- <file> && git add <file> && git commit`

## Open Questions

1. **`_BenchmarkSpec.average_node_features=False`** — added to `bench_prxteinmpnn_jax.py` to prevent InferencePlan rejection. Confirm this is the correct field name and semantics before cluster runs (no-fusion = inputs only, no arithmetic mean across encoding states).
2. **1BC8.pdb has only ZN ion as ligand** — sufficient for performance benchmarking (exercises ligand conditioning code path) but doesn't represent real organic drug-protein workloads. Note in benchmark report.
3. **GPU memory reporting** — `_get_gpu_memory_gb()` returns 0.0 placeholder. Add `pynvml` to benchmark dep group before cluster runs: `pynvml>=11.5.0`.
