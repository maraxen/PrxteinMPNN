---
task_id: 260601_benchmark-staging
session_id: bg-wave2-260601
status: in_progress
phase: Wave 2 adapters written — Wave 3 harness next
date: 260601
---

# Handoff: Benchmark Suite — Wave 2 Complete (260601_benchmark-staging)

## State

Branch: `worktree-benchmark-wave2-prereqs` (worktree off main)

Recent commits:
- `c1fefc9` feat(benchmark-wave2): add PyTorch and ColabDesign adapters
- `f1718a1` docs(handoff): wave2-prereqs complete
- `67adcfa` docs(benchmark-spec): spec §11 resolved
- `db18e15` feat: ColabDesign SHA pin, 1SMD L500 fixture

## What was done this session

| File | Status |
|---|---|
| `scripts/benchmarks/bench_ligandmpnn_pytorch.py` | ✅ Written, dry-run passes |
| `scripts/benchmarks/bench_colabdesign_jax.py` | ✅ Written, dry-run passes, smoke confirmed (CPU: cold=1.4s, warm=124ms) |

## Wave 2A — bench_ligandmpnn_pytorch.py

**Usage:**
```bash
# No ligand (protein_mpnn):
cd /home/marielle/projects/tev_design
uv run python aminx/.claude/worktrees/benchmark-wave2-prereqs/scripts/benchmarks/bench_ligandmpnn_pytorch.py \
  --seq-lens 76 150 300 500 --batch-sizes 1 4 16 \
  --precision bf16 --hardware A100 --output-json results/pt.json

# With ligand (partially wired — see gaps):
... --ligand ...
```

**Key design decisions:**
- Imports `featurize`/`ProteinMPNN` from `REFERENCE_PATH` via `sys.path.insert(0, ...)`
- NPZ fixture → manual `protein_dict` → `featurize()` for no-ligand path
- `torch.utils.benchmark.Timer.blocked_autorange(min_run_time=2.0)` for warm timing
- `first_call_overhead_s` stored in `compile_time_cold_s` field with explanatory note
- p95 computed as `np.percentile(measurement.times / number_per_run, 95)`
- `n_warmup=0` in JSON (Timer handles warmup internally); `n_timed` = actual count
- `--ligand` flag sets `model_type="ligand_mpnn"` and loads checkpoint with `atom_context_num`

**Known gap — `--ligand` path incomplete:**
`featurize()` for `model_type="ligand_mpnn"` requires `Y`/`Y_t`/`Y_m` keys in `protein_dict`
(ligand atom coordinates). NPZ fixtures don't have these. `--ligand` will fail at `featurize()`.

Fix (~20 lines): Add a `load_ligand_structure(reference_path, device)` function that calls
`parse_PDB(str(reference_path / "inputs/1BC8.pdb"), device=device)`. In `benchmark_cell`,
when `model_type == "ligand_mpnn"`, skip NPZ loading and call `parse_PDB` instead. The
`parse_PDB` return already has `Y`/`Y_t`/`Y_m` for ligand_mpnn structures.

## Wave 2B — bench_colabdesign_jax.py

**Usage:**
```bash
cd /home/marielle/projects/tev_design
uv run python aminx/.claude/worktrees/benchmark-wave2-prereqs/scripts/benchmarks/bench_colabdesign_jax.py \
  --seq-lens 76 500 --batch-sizes 1 4 16 \
  --hardware A100 --pdb-dir aminx/tests/data \
  --output-json results/cd.json
```

**Key design decisions:**
- ColabDesign SHA `e31a56fe`; `from colabdesign.mpnn.model import mk_mpnn_model`
- `mk_mpnn_model(model_name="v_48_020", backbone_noise=0.0, dropout=0.0, seed=42, weights="original")`
- `model.prep_inputs(pdb_filename=str(pdb_file))` → `model.sample(num=batch_size, temperature=1.0)`
- `jax.block_until_ready(jax.tree_util.tree_leaves(result))` for synchronization
- pdb_map: `{76: 1ubq.pdb, 150: 1ubq.pdb, 300: 1ubq.pdb, 500: 1SMD.pdb}`
- `actual_len` reported from loaded PDB (1SMD → ~496, not 500); reported as `seq_len` in JSON
- No-ligand path only (ColabDesign has no ligand conditioning)

**Known gap — L=150/300 map to 1ubq (76 residues):**
These cells run at L=76, making `latency_per_residue_us` identical to L=76 cells. For a fair
benchmark, simply skip L=150/300 for ColabDesign (they don't add information), or update
the benchmark suite's default `--seq-lens` for ColabDesign to `76 500` only.

## Remaining gaps (all three adapters)

| Gap | Severity | Fix |
|---|---|---|
| Wave 2A `--ligand` path broken at `featurize()` | Medium | Add `parse_PDB(1BC8.pdb)` path in `benchmark_cell` for `ligand_mpnn` |
| Wave 2B L=150/300 cells map to 1ubq (L=76) | Low | Remove L=150/300 from pdb_map, or skip in bench_suite |
| GPU memory: 0.0 placeholder in all adapters | Low | Add `pynvml>=11.5.0` to benchmark dep group |

## Wave 3 next — bench_suite.py + bench_report.py

See spec §5 for full architecture. Key requirements:
- `bench_suite.py`: subprocess dispatch only, no JAX/torch imports (GPU memory isolation)
  - `subprocess.run([sys.executable, adapter_path, "--output-json", tmp_json, ...], check=True)`
  - Merge JSON outputs from all three adapters
- `bench_report.py`: JSON → markdown table + CSV
  - Group by `(seq_len, batch_size, precision, ligand_conditioning)`
  - Side-by-side columns: `aminx_jax | ligandmpnn_pytorch | colabdesign_jax`
  - Report speedup ratios relative to PyTorch baseline

## Merge instructions

Worktree `worktree-benchmark-wave2-prereqs` has 4 commits ahead of main.
From the main repo:
```bash
git -C /home/marielle/projects/tev_design/aminx merge --no-ff worktree-benchmark-wave2-prereqs \
  -m "feat(benchmark-wave2): merge Wave 2 adapters + prereqs"
```
