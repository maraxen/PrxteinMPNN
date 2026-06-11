# aminx → xtrax Refactor: Codebase Model (Wave-1 synthesis)

**task_id:** 260611_xtrax-refactor-spec
**Date:** 2026-06-11
**Status:** Wave-1 recon complete; Wave-2 validation pending before adversarial spec cycle.

## Goal

Refactor aminx to leverage `xtrax` (high-performance composable JAX library, now on PyPI v0.2.0).
Move domain-agnostic primitives — bundles, inference plans, high-performance output sinks, tiling
primitives — into xtrax, and rebuild aminx's training implementation on xtrax training primitives.
During the worktree refactor, pin a **local editable** xtrax (`../xtrax`) to catch xtrax bugs; production
uses the PyPI release.

## Hard facts (Wave-1, anchored)

1. **Zero current coupling.** No `import xtrax` anywhere in aminx `src/` or `tests/`. Only backlog items
   1480–1483 mention it. This is a greenfield integration, not a migration.
2. **xtrax v0.2.0 published to PyPI** 2026-06-10 (hatchling dynamic version, `src/xtrax/__init__.py`).
   53 public symbols via lazy `__getattr__`. Requires **Python ≥3.13**. aminx requires **≥3.12**.
3. **aminx deps** (pyproject `[project].dependencies`): jax≥0.4.35, equinox≥0.13.2, optax≥0.2, grain≥0.2,
   array-record, h5py, proxide, typer, huggingface-hub, … No xtrax. **uv.sources** currently pins
   `proxide = {index="pypi"}`, `torch = {index="pypi"}`. Local xtrax would add
   `xtrax = { path = "../xtrax", editable = true }`.
4. **Banned-API landmine** (aminx pyproject `[tool.ruff.lint.flake8-tidy-imports.banned-api]`): imports of
   `aminx.inference.decode`, `aminx.host.plan`, `aminx.types.stages`, `aminx.inference.logits` are banned
   (ADR `260605_potts-parallel-not-stageset`). Moving any of these into xtrax, or having xtrax import them,
   needs ADR review.

## xtrax public surface (landing zones)

| xtrax module | Key symbols | Maturity |
| --- | --- | --- |
| `training` | Trainer, SafetyTrainStep, create_train_step, ResumableState, Callback, LossFunction, WeightedLoss, MultiTaskLoss, make_optimizer, adamw_with_schedule | Production |
| `engine` | Engine (async fit/eval), BoundedCallbackHandler, async_indexed_stream | Production |
| `tiling` | AxisSpec, AxisDecision, BatchPlan, BatchPlanner, Vmap/SafeMap/DedupGather/Scan, make_axis_dispatch | Production but **thinner than aminx** |
| `sparse` | SparseConfig, SparsePolicy, sparsify_model, make_sparse_forward_fn, sparse_filter_jit | Production |
| `transforms` | safe_map, safe_scan | Production |
| `data` | DataModule, create_distributed_pipeline | Production |
| `distributed` | init_dist, is_distributed, ShardingPolicy/LogicalMesh, get_device_mesh | Production |
| `checkpoint` | get_checkpoint_manager, save_checkpoint, load_checkpoint (orbax, PyTreeCheckpointHandler) | Production |
| `safety` | safe_norm, safe_reciprocal | Production |
| `stages` | TransformFn, RollingFn, FuseFn (tier-1 protocols), StageBundle | Production |

## aminx MOVE / SPLIT / KEEP classification

### Tiling (`src/aminx/tiling/`) — aminx ~2× richer than xtrax
- **MOVE (generic):** strategy.py, iterator.py, dispatch.py, dedup.py, carry_shape.py, planner.py, bucketing.py, errors.py
- **KEEP (protein-specific):** axes.py (10 named protein axes registry), buckets.py (LENGTH_BUCKETS, pad_to_bucket), pad.py (pad_bundle over InferenceBundle), carry.py (_HETEROGENEOUS_AXIS_NAMES)
- **xtrax must be upgraded first**: add `Scan.init`, enrich `DedupSpec` (unique_indices/index_map/k/dedup_fn/gather_fn/to_dedup_gather), `CarrySpec`+`CarryShape`, `AxisSpec.axis_index`, 5-phase `BatchPlanner` (phases 0/0b/1/2/3), `bucketing.py`.
- **Naming collision**: `make_axis_dispatch` exists in both with incompatible signatures (aminx=factory→iterator; xtrax=inline executor→result). MUST reconcile.

### Host/run inference (`src/aminx/host/`, `src/aminx/run/`, `src/aminx/types/`)
- **MOVE (generic):** types/bundles.py (Geometry/Conditioning/Ligand/WaveSchedule/InferenceBundle — pure PyTrees), types/protocols.py (ConditionalLogitsFn, UnconditionalLogitsFn, SamplerFn, ScoreFn, DesignSink, Pipeline), host/output_sinks.py (StreamingTensorStagingSink + io_callback staging), host/streaming_host.py (StreamingBatchHost, sink_barrier, iter_chunks), host/plan.py (make_sampling_planner, memory budgeting), kernel_dispatch._dispatch_axis, stages tier-1 protocols.
- **SPLIT:** runner.py (sample/score/inspect/jacobian — ~60% control flow MOVE, ~40% kernel KEEP), streaming.py (HDF5/ArrayRecord chunk loop MOVE, schema attrs KEEP), specs.py/spec.py (RunSpec/IOConfig/ResourceConfig/BatchingConfig/PrecisionConfig MOVE; MultistateConfig/LigandConfig/TiedPositions/GridLineage/Averaging KEEP), kernel_dispatch._sample_batch (dispatch MOVE, kernel KEEP), types/stages.py (tier-1 MOVE/already-in-xtrax, tier-2 MPNN aliases KEEP), types/protocols.ModelProtocol (structure MOVE, encoder/decoder fields KEEP).
- **KEEP:** model loading (prep_protein_stream_and_model), build_inference_bundle, MPNN/LigandMPNN kernels.

### Training (`src/aminx/training/`) — ~50% rebuildable on xtrax
- **REBUILD-ON-XTRAX:** state container → ResumableState; create_optimizer → adamw_with_schedule/make_optimizer; checkpoint.py → xtrax.checkpoint.orbax; grad accumulation → xtrax.training.grad.accumulate_grads (pre-stacked shape differs); optional SafetyTrainStep.
- **KEEP (domain):** losses.py (cross_entropy/perplexity/sequence_recovery), metrics.py (or adopt dict), diffusion.py + train_diffusion.py (NoiseSchedule — GAP in xtrax), dataloading/preprocess.py (physics features), specs.py.
- **GAPS in xtrax:** diffusion/noise scheduling; multi-checkpoint (rolling+permanent) management; metrics container vs plain dict.

## Inference-path perf hotspots (no-regression targets)
1. Device→host tensor drain via io_callback (`take_staging_sequences_logits`) — once per batch after `effects_barrier`.
2. Batch planning memory estimation (`estimate_memory_theoretical`, hardcoded `activation_multiplier=2.5`).
3. ArrayRecord async writer vs deprecated HDF5 path.
4. JIT recompilation on static bundle field changes (`n_states`, `structure_mapping`).
5. Axis-strategy dispatch isinstance branching (negligible).

## Test / parity / cluster infra
- aminx `tests/`: host, run, tiling, inference (+decode), model, parity, potts, sampling, scoring, cli, benchmarks.
- Fixtures: root conftest (markers + structure fixtures 1ubq/1a00, mock_model_parameters, minimal_bundle_fixture); host conftest (minimal_model); parity conftest (AMINX_VERIFY autouse).
- Markers: slow, potts, potts_slow, parity_fast/heavy/targeted/audit, phase0a_spike, requires_weights. Default `addopts = -m 'not parity_heavy and not slow'`.
- Heavy parity deps: torch, colabdesign, biopython, dm-tree, scipy. `require_heavy_parity_prereqs()`.
- Benchmarks: bathos-sidecar scripts (`scripts/benchmarks/*.py` + `.bth.toml`), subprocess-validated (--dry-run/--smoke). Cluster sbatch wrappers per GPU (a100/h200/l40s/blackwell) under scripts/engaging + scripts/cluster. Blackwell SM120 XLA flag workaround.
- xtrax benchmarks: pytest-benchmark (bench_tiling, bench_training_step, bench_grad_accum).

## Wave-2 validation RESOLUTIONS (authoritative)
- **A1 — Python floor (VALIDATED):** aminx must raise `requires-python` to `>=3.13` AND update both `[tool.uv.environments]` markers to `python_version >= '3.13'`. Under `>=3.12`, `uv add --editable ../xtrax` fails the *whole* lock (uv resolves all env splits; 3.12 split unsatisfiable). After the bump it resolves and `import xtrax` → 0.2.0. Cluster scripts use `uv run python` (no version pin) → 3.13 fine. Pin source explicitly: `xtrax = { path = "../xtrax", editable = true }` (do not trust `uv add`'s cwd-relative path).
- **A2 — make_axis_dispatch collision (REFUTED/UNIFY):** aminx = iterator factory `(strategy,*,axis)→iterator`; xtrax = eager executor `(strategy,fn,xs,init)→result`. Inverse abstractions, same name. Decision: adopt ONE pattern (xtrax eager is more JAX-idiomatic) and rename the other or layer a factory over the executor.
- **A4 — banned-API (VALIDATED+nuance):** ruff TID251 bans are literal `aminx.*` path strings (global; only `potts/designer.py` exempt). `xtrax.*` is inherently unbanned, so moving host/plan + types/stages to xtrax will NOT trip ruff — but that silently bypasses ADR 260605's intent (keep potts parallel-family decoupled from the stageset pipeline). Spec MUST add an equivalent boundary lint on the new `xtrax.*` import paths.
- **A5 — xtrax tiling maturity (BLOCKER):** xtrax `Scan` lacks `init` (blocker); no `CarrySpec`/`CarryShape`; `DedupSpec` split (k_bucket/max_unique) vs aminx rich `DedupGather`. xtrax tiling must be UPGRADED to aminx parity before aminx can depend on it. (`AxisSpec.axis_index` already exists in xtrax.)
- **A6 — bundles (RESOLVED → KEEP in aminx):** all 5 bundle classes are protein-coded (atom_37 = 37-atom PDB standard, residue_index, chain_index, tie_group_map, ligand_atom_types, wave scheduling). DAG-clean (no domain *imports*) but semantically protein. KEEP in aminx; optionally extract a generic skeleton (StructureBundle/FeatureBundle/IterationBundle) to xtrax only if cross-project reuse is wanted. The host/run recon's "MOVE bundles" was wrong on semantics.
- **A7 — StageBundle (INCOMPATIBLE):** aminx StageSet fields `encoder_sink: tuple[...]`, `decoder_sink: tuple[...]`, `axis_boundaries: dict[...]` violate xtrax StageBundle's Optional[Callable]-only `__init_subclass__` validator. Resolve by loosening xtrax StageBundle (allow tuple[Callable,...]/Mapping) OR wrap aminx fields in callable containers.
- **A3 — DAG (VALIDATED):** all 12 MOVE candidates are clean to move; every protein/model reference sits behind `TYPE_CHECKING`. No aminx→xtrax→aminx cycle.
- **A3b — inverse (VALIDATED):** xtrax imports nothing from aminx. Dependency is strictly unidirectional.
- **A8 — checkpoint parity (NUANCED/BREAKING):** aminx `Composite(StandardSave model, StandardSave opt_state, JsonSave metrics)` vs xtrax single `PyTreeCheckpointHandler` over `ResumableState`. On-disk layouts incompatible; metrics must move into `ResumableState.extras`; existing aminx checkpoints need migration tooling to be restored post-refactor. Neither side uses a dual rolling/permanent manager.

## (historical) Open questions raised in Wave-1 — now resolved above
- A1. **Python 3.13 vs 3.12**: must aminx bump to 3.13? Do cluster nodes / production support 3.13? Does the editable install even resolve under aminx's current `requires-python`?
- A2. **make_axis_dispatch collision**: confirm exact signatures in both repos; decide rename/reconcile strategy.
- A3. **Bidirectional import DAG**: do candidate MOVE modules (bundles, protocols, plan) import any protein-specific aminx symbols that would create a cycle if hosted in xtrax?
- A4. **Banned-API**: would moving types/stages or host/plan trip the ruff ban or the ADR?
- A5. **xtrax tiling upgrade scope**: confirm xtrax really lacks Scan.init/CarrySpec/CarryShape/5-phase planner (the two recon passes agreed, but verify against canonical xtrax repo, not worktrees).
- A6. **bundles ownership**: xtrax-surface recon says aminx data bundles are protein-specific (KEEP); host/run recon says they are pure PyTrees (MOVE). **Direct contradiction** — resolve.
- A7. **StageBundle vs aminx stage sets**: can aminx stage sets subclass xtrax.stages.StageBundle cleanly?
- A8. **orbax checkpoint parity**: aminx uses Composite(Standard+Json); xtrax uses single PyTree. Does moving lose the metrics-as-json layout, and do existing checkpoints still restore?
