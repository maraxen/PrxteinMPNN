# ProteinEBM Parity: State of the Port

- **task_id**: `260716_ebm_parity_report`
- **status**: UPDATED 2026-07-19 — §7's open "decoy vs. ddg torch.compile asymmetry" question is resolved (root cause: `aminx.training`'s stub `__getattr__` raised `NotImplementedError` instead of `AttributeError`, breaking `pickle.whichmodule`'s `sys.modules` scan during Inductor's `FxGraphCache` pickling; see §7 for the full trace). Fix in PR #118 (not yet merged). Prior status (2026-07-18): a real, previously-undiscovered bug meant the PyTorch reference model/tensors never moved to GPU across all 6 benchmark scripts (decoy/ddg/biasing/langevin/langevin_annealing/heterogeneous_batch), despite every result JSON claiming `"device": "cuda"`. Every JAX-vs-PyTorch throughput number this report (and the epic's own founding claim) had ever cited was actually GPU-vs-**CPU**, not the same-hardware comparison it was presented as. Fixed (commit `3faeee6`) and re-run in full on real GPU-vs-GPU hardware (jobs `18284162` + `18293715`, Engaging pi_so3/Blackwell) — see the rewritten §1 for the corrected numbers. Prior status line (2026-07-17): real accuracy-vs-paper data acquired + preliminary correlations measured (§6, unaffected by this bug); `heterogeneous_batch_benchmark`'s JAX crash closed as an open Blackwell/SM120 compiler limitation, confirmed version-independent (jax/jaxlib 0.11.0 retest, job `18161115`) — see §7, still standing, now with an added side-effect noted (the same crash's corrupted CUDA context also takes down PyTorch's two strategies now that PyTorch shares the GPU).
- **author**: orchestrator session (background job)
- **date**: 2026-07-16
- **branch**: `worktree-ebm-parity-report`
- **companion artifact**: interactive HTML report with charts (published separately)

## Summary

ProteinEBM (Roney, Ou & Ovchinnikov, bioRxiv 2025.12.09.693073) was ported into aminx as a new
forward energy/score composition path (EPIC #3294, 14 backlog nodes + a follow-on epic, completed
and merged to `main` 2026-07-09 through 2026-07-12; see
[`plans/260709_proteinebm-epic-backlog-dag.md`](../plans/260709_proteinebm-epic-backlog-dag.md)).
This report reviews what "parity" actually means for that port today. Sections 1–5 were written
under an explicit decision to *not* attempt the paper's own headline benchmarks (Rosetta decoy set,
ProteinGym/Tsuboyama stability data) — that decision was **reversed later the same day** per direct
follow-up request ("let's also set up the proteingym and rosetta analysis"); §6 documents what was
actually acquired and measured as a result. Sections 1–5 below are otherwise unchanged from the
original review and still describe the state of the *epic itself* (throughput, numerical port
parity, the LplA real-world check) accurately.

Three distinct kinds of "parity" are in play, and they are in very different states:

| Kind | Status | Evidence |
| :-- | :-- | :-- |
| **Throughput / speed parity** (JAX vs. PyTorch, same computation) | Real, positive, but **corrected down from a bugged 11–92× to ~2–7×** (device-placement bug, fixed 2026-07-18) | §1 |
| **Numerical port parity** (does the JAX model compute what the real PyTorch reference model computes?) | Strong, freshly re-confirmed today | §2 |
| **Real-world accuracy** (does it correlate with real experimental data?) | One real result (E7 LplA) | §3 |
| **Accuracy vs. the paper's own headline numbers** (Spearman 0.838 decoy, 0.686 ΔΔG) | **Not measured** | §5 |

## 1. Throughput parity — CORRECTED 2026-07-18 (was a bugged 11–92×, is really ~2–7×)

**A real, previously-undiscovered bug: PyTorch never ran on GPU, in any of the 6 benchmark scripts,
ever.** Every one of `decoy_benchmark.py`, `ddg_benchmark.py`, `biasing_benchmark.py`,
`langevin_benchmark.py`, `langevin_annealing_benchmark.py`, and `heterogeneous_batch_benchmark.py`
loaded its checkpoint with `torch.load(..., map_location="cpu")` and never called `.cuda()`/`.to()`
on the resulting model or any input tensor. The `"device"` field every one of these scripts wrote to
its result JSON was derived purely from `torch.cuda.is_available()` — which only checks whether a
GPU *exists* on the machine, not whether anything was *placed* on it. Confirmed directly on cluster
hardware (GPU available, `next(ref_model.parameters()).device` reporting `cpu`): every PyTorch number
this report — and the epic's own founding throughput claim below — had ever cited was silently a CPU
number, despite every JSON and every table in this document claiming `"cuda"`.

This means the original claim in this section ("JAX beats PyTorch by 11–92× across every tested
length and all four applications," sourced from
[`plans/260709_proteinebm-epic-backlog-dag.md`](../plans/260709_proteinebm-epic-backlog-dag.md)
§7/§9/§11 — the same four scripts, run earlier on titanix/Turing and engaging/Blackwell) was
comparing **JAX-on-GPU against PyTorch-on-CPU**, not the same-hardware comparison it was presented
as. Those titanix/Blackwell numbers predate this fix and have **not** been re-verified against it
(titanix is a separate workstation remote, out of scope for this correction) — they are left in the
plan doc as a historical record, not deleted, but should not be cited as evidence of GPU-vs-GPU
speedup until someone re-runs them with the fix.

**Fixed (commit `3faeee6`)**: each script's model-construction path now takes an explicit `device`
parameter and calls `.to(device)` before returning; every `torch.tensor`/`zeros`/`full`/`arange` call
in every PyTorch timing path now passes `device=` explicitly, using the same value reported in the
JSON's `"device"` field (so the label and the placement can never diverge again). Two follow-on
scripts (`langevin_annealing_benchmark.py`, `heterogeneous_batch_benchmark.py`) had no device field
at all before this fix; both now report one.

**Corrected result (fresh re-run on real GPU-vs-GPU hardware, jobs `18284162` + `18293715`, Engaging
pi_so3/Blackwell, jaxlib 0.9.2-pinned decoy/ddg + project-default jaxlib for biasing/langevin,
`n_repeats=10`, batch=4 for the apples-to-apples row below — see §7's reproducibility table for the
full per-batch-size sweep):**

| Script | L=64 | L=128 | L=256 | L=512 |
| :-- | :-- | :-- | :-- | :-- |
| decoy (energy evals/s) | 3.17× | 2.59× | 2.21× | 1.96× |
| ddg (energy evals/s) | 3.29× | 2.57× | 2.20× | 1.96× |
| biasing (energy evals/s) | 6.83× | 4.92× | 3.29× | 2.20× |
| langevin (steps/s) | 3.70× | 2.61× | 2.15× | 1.86× |

JAX is still faster than PyTorch at every length and every application — the direction of the claim
survives — but by **~2–7×**, not 11–92×, and the margin now shrinks with length rather than growing
(the old numbers had JAX's advantage *increasing* at L=512, which never made physical sense for a
GPU-bound comparison; that pattern was itself a symptom of PyTorch being memory- and cache-bound on
CPU at large batch, not a real GPU-vs-GPU trend). Every one of these four scripts also now hits a
genuine PyTorch-side `CUDA out of memory` at its single largest (length, batch_size) cell — JAX and
PyTorch now compete for the same GPU memory, which never happened when PyTorch silently ran on CPU
with effectively unlimited host RAM headroom. This is expected, not a bug: each script's incremental
per-cell write (added alongside this fix, see below) catches it cleanly and every other cell's data
is unaffected. Full per-length, per-batch-size tables are in §7 and the companion artifact.

**A second, related bug surfaced by the corrected re-run and also fixed**: `run_cluster_benchmarks.sh`'s
`run_per_length_group` ran its three length-group invocations (L64-128, L256, L512) as a single unit
under the wrapper script's own `set -euo pipefail` — a crash in the *first* invocation silently
aborted the other two as well, even though they never failed themselves. This is why the first
corrected re-run (job `18284162`) produced *zero* new output files for decoy/ddg despite only the
L64-128 invocation actually OOM-ing. Fixed by isolating each length group in its own error-tolerant
subshell (commit `75fd14e3`), verified via a standalone simulation before the second re-run
(job `18293715`) confirmed it end-to-end.

This was logged as a lesson (praxia lesson `#273`) and as tech debt in the source repos for
`using-bathos` (`#794` — extend the pipeline-verification guidance to cover device placement),
`using-xtrax` (`#795` — a related but separate finding, see §7), and `using-jax`/jaxlint (`#796`).

Along the way, a real jaxlib regression was found, root-caused, and worked around: `-jax.grad(energy)`
(the conservative score, used by decoy ranking + ΔΔG) crashed at XLA compile time on every modern
GPU (Blackwell, H100, A100, L40S) under jaxlib 0.10.2, but not on older Turing hardware. A version
bisection pinned the regression precisely:

| jax/jaxlib version | Gradient-path result |
| :-- | :-- |
| 0.10.2 (project pin) | FAIL |
| 0.9.2 | PASS |
| 0.9.0 | PASS |
| 0.8.3 | PASS |
| 0.8.0 | PASS |

Fix: a scoped `uv run --with jax==0.9.2 --with jaxlib==0.9.2` override applied only to the two
affected benchmark scripts — not a project-wide pin. An upstream bug report is drafted (not yet
filed) at `.praxia/docs/research/260712_jax-xla-scf-if-gradient-regression-bug-report.md`.

## 2. Numerical port parity (E3.5) — re-confirmed today with fresh evidence

The original E3.5 gate (`scripts/ebm/checkpoint_parity_check.py`) checked one fixed synthetic
structure (n=12 residues, seed=0) against the real PyTorch reference model, using the real
checkpoint. For this report, the same real checkpoint and real reference model were re-exercised
across a **grid of 20 (size × seed) synthetic trials** — sizes 8/16/32/64 residues × 5 seeds each —
via a new script, `scripts/ebm/collect_synthetic_parity_evidence.py`, tracked through bathos
(run `b12dace4-bfb1-4474-aad6-db82d6a65d28`, git SHA `69678405`).

**Result, from today's run:**

- Per-residue energy: mean absolute error between 6.1×10⁻⁶ and 2.7×10⁻⁵ across all 20 cases (float32-precision-level agreement, far tighter than the original gate's `atol=1e-3` tolerance).
- Per-residue conservative-score (`-∇E`) direction: cosine similarity between the reference and JAX score vectors ranged from 0.999999998 to 1.0 across all 560 (residue × case) points — essentially exact directional agreement.
- Pearson correlation on per-residue energy: ≥ 0.9999999999781 in every case.

This validates that **the port is numerically faithful to the real reference implementation** — it
is a fresh, independent re-confirmation at much finer sample density than the original single-case
gate, not a restatement of it. It does **not** validate that the model reproduces the paper's
published accuracy figures (§5).

**Provenance for this section:** every input is **synthetic** —
`numpy.random.default_rng(seed).normal(...)` coordinates and `randint(0,21)` amino-acid types (see
`checkpoint_parity_check.py::_build_fixed_input`) — no real protein sequence or structure is
involved. This is deliberate: it isolates numerical port fidelity from biological realism.

## 3. Real-world validation (E7) — real experimental data

The one application in this epic validated against real biological data rather than a synthetic
proxy. Real point mutants of *E. coli* lipoate-protein ligase A (LplA) were scored by the ported
model's per-state energy gap `ΔE = E(closed, mutant_seq) − E(open, mutant_seq)` between two real
crystal conformations, and correlated against the real experimentally-measured "Promiscuous
Activity" readout.

Run via `scripts/ebm/lpla_biasing_check.py`, tracked through bathos (run
`b3a68dd9-080c-4ce6-a2ae-ea1c2061db06`, outcome **pass**, git SHA `69678405`).

**Result: Spearman r = 0.402 (p = 3.1×10⁻⁵), n = 101 real mutants.** A statistically significant,
moderate positive correlation — consistent with the design spec's own (qualitative, weaker-than-the
paper's-headline-numbers) expectation for this application ("sign/rank of ΔE … positive corr. w/
activity").

**Provenance for this section (real data, not synthetic):**

| Field | Value |
| :-- | :-- |
| Protein | LplA (lipoate-protein ligase A), *E. coli* |
| Closed state | PDB 1X2G (331/337 residues resolved) |
| Open state | PDB 3A7R (337/337 residues resolved) |
| Shared/scored residues | 331 (intersection mask; a 6-residue disordered loop in 1X2G excluded) |
| Mutants scored | 101 (all rows in `eval_data/lpla.csv` with a non-null real `Promiscuous Activity` value) |
| Activity readout | Real experimental measurement (Cavanaugh et al., via the `ConformationalBiasing` GitHub repo and the ProteinEBM reference repo's own `eval_data/lpla.csv`) |
| Diffusion time | t = 0.05 (ProteinEBM-x MVP target) |

## 4. Synthetic / invariant test coverage

`tests/ebm/` — 16 test files across backlog nodes E0–E10, all synthetic/fixture-based (no real
weights, no torch, no real datasets):

| Node | Covers | Validates against |
| :-- | :-- | :-- |
| E0 | Foundations & invariants | VP-SDE closed-form score, `E=0` on zero-`r`, autograd vs. analytic Gaussian score |
| E1/E2 | Trunk + diffuser | `FourierEmbedding` bounds, determinism given key, jit/vmap compatibility |
| E3 | Energy/Score/Aux readouts | `score = -∇E` on toy + real model, masked residues contribute zero, 2nd-order grad finite |
| E3.5 | Checkpoint weight-port | Synthetic state-dict remap/skip/shape logic (fast CI-safe complement to §2's real check) |
| E4 | Dispatch (Vmap/SafeMap) | Tiled dispatch matches a plain Python loop, all three application entry points |
| E4.5 | Bucket-boundary gate | Padding-waste bound on real + synthetic length distributions |
| E5/E6/E7 | Decoy/ΔΔG/biasing wiring | Fusion-primitive correctness on toy energies (accuracy vs. the paper: §5) |
| E8 | Training path | Loss composition, JAX-safe coin-flip, 2nd-order finite-difference gate on the real model |
| E9/E10 | Langevin + structure prediction | Toy-model convergence, detailed-balance cross-check, resampling statistics |

Suite status as of the epic's 2026-07-10 completion commit: 261 passed, 4 pre-existing skips (the
skips gate on a locally-present real-weight checkpoint, absent by default). **Freshly re-run for
this report** (2026-07-16, this exact worktree commit): **261 passed, 4 skipped, 0 failed** — an
exact match to the original completion count, confirming no regressions since the epic landed. (An
earlier attempt in this same session hit a sandbox limitation — this background job has no D-Bus
user session, so the local memory-capping hook's `systemd-run` wrapping failed outright on the
first try; a retry succeeded once the hook's own bus-probe fail-open path took over.)

## 5. Explicitly out of scope

**Decoy-ranking Spearman 0.838** and **ΔΔG Spearman 0.686** — the paper's own headline accuracy
numbers — were **not attempted** in this report, per explicit direction. Reproducing them requires:

- The real Rosetta decoy set: `https://files.ipd.uw.edu/pub/decoyset/decoys.zip` (5.18 GB, confirmed reachable) plus `rmsd.txt`/`rosettascore.txt`/`tmscore.txt` from `huggingface.co/jproney/ProteinEBM` (confirmed reachable, small).
- The ProteinGym/Tsuboyama-cDNA megascale stability subset: `DMS_substitutions.csv` filtered to `coarse_selection_type == 'Stability'` and filename containing `'Tsuboyama'`, plus matching AF2 structures — hosted at `marks.hms.harvard.edu` (unreliable/slow to reach from this environment during recon) and not otherwise mirrored on HuggingFace at the same completeness as the legacy ProteinGym v0.1 dataset (which predates the Tsuboyama stability assays).
- A from-scratch implementation of the reference's "32-sample unfolded-ensemble" Monte Carlo correction (`generate_random_backbone_coords`), which the existing `ddg_stability.py` module deliberately substitutes with a synthetic random-walk proxy, documented as such.

This is a real, scoped engineering task (data acquisition + a new per-protein scoring script
matching the reference `ddg_prediction.ipynb`/`rank_decoys.ipynb` notebooks), not attempted here.

## 6. Real accuracy-vs-paper data + preliminary measurements (added later 2026-07-16)

Per direct follow-up request, the real datasets §5 said were out of scope were acquired and real
(if small-sample, preliminary) correlations were measured — genuinely new evidence, not a
restatement of anything above.

### Data acquired

| Dataset | Source | Size | Verified |
| :-- | :-- | :-- | :-- |
| Rosetta decoy set | `files.ipd.uw.edu/pub/decoyset/decoys.zip` | 5.18 GB compressed, 23 GB unzipped | 133 native structures (matches the design spec's own count), ~957 decoys each |
| Real TM-score/RMSD/Rosetta-score labels | `huggingface.co/jproney/ProteinEBM` | ~26 MB | Joins cleanly to decoy filenames |
| ProteinGym v1.3 Tsuboyama-2023 stability subset | `marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/` | ~62 MB compressed | 64 real Tsuboyama assay CSVs + matching AF2 structures confirmed present |

**A real, fixable server issue was found and worked around, not silently bypassed**:
`marks.hms.harvard.edu` presents an incomplete TLS certificate chain (`unable to get local issuer
certificate` — a legitimate `*.hms.harvard.edu` InCommon-issued leaf certificate, valid through
Nov 2026, but a broken intermediate chain, confirmed via `openssl s_client`). This is a common
academic-server misconfiguration, not a suspicious substitution. The data was fetched with
`curl -k` (skip verification) — a deliberate, considered choice for a public-dataset download, not
an oversight; flagged here for anyone auditing this later.

### New scripts + a new real-data generator

- `scripts/ebm/real_decoy_ranking_benchmark.py` (+ `.bth.toml`) — real decoy-ranking Spearman, on
  top of the already-built (never-before-exercised) `aminx.ebm.decoy_ranking.rank_decoys_over_noise_time`.
- `scripts/ebm/real_ddg_stability_benchmark.py` (+ `.bth.toml`) — real ΔΔG Spearman, on top of
  `aminx.ebm.dispatch.score_mutant_ensemble` + `aminx.ebm.ddg_stability.unfolded_state_correction`.
- `aminx.ebm.ddg_stability.generate_real_unfolded_ensemble` — a new, faithful NumPy port of the
  reference's real NeRF-based backbone generator (`generate_random_backbone_coords`'s no-file,
  `uniform_sampling=True` fallback path — the only branch the reference's own
  `ddg_prediction.ipynb` actually exercises), added **alongside** (not replacing) the existing
  documented synthetic random-walk substitute, which stays as the CI-safe test fixture.
  Sanity-checked before trusting any downstream number (BATHOS discipline): no NaNs, plausible
  CA-CA spacing (2.96–3.67 Å), distinct ensemble members, deterministic given a seed, different
  given different seeds. `compute_ddg_stability` gained one new optional parameter
  (`unfolded_coords_ensemble`, default `None` = unchanged prior behavior) so the real generator's
  output can be substituted in without forking the ddG formula.

### Preliminary results — small samples, NOT a confirmatory measurement

Both scripts run end-to-end on real data and were tracked through bathos (`bth run`, sidecars,
outcomes recorded). At small sample sizes (3 natives × 20 decoys; 3 assays × 20 mutants — a fraction
of the ~957 decoys/native and full per-protein mutant sets available), both are **well below** the
paper's targets:

| Check | Sample | Result (\|Spearman\|) | Paper target |
| :-- | :-- | :-- | :-- |
| Decoy ranking, t=0.05 (pre-registered) | 3 natives × 20 decoys | 0.142 | 0.838 |
| Decoy ranking, best-of-sweep | 3 natives × 20 decoys | 0.376 | 0.838 |
| ΔΔG stability | 3 assays × 20 mutants | 0.434 | 0.686 |

**A real sign-convention finding, in both checks**: the raw (signed) Spearman correlations came out
*negative* for decoy-ranking (e.g. −0.696 on one native at n=20) — physically expected, since lower
energy = more native-like but higher TM-score also = more native-like, so the "correct" raw sign is
negative; the paper's own 0.838 figure is reported positive, consistent with an implicit
absolute-value convention. Both scripts now report signed *and* absolute means explicitly, and
document this in their own output JSON, so this is never silently conflated later.

**Read this table for what it is — a pipeline validation, not a verdict.** n=20 decoys out of ~957,
or n=20 mutants out of up to ~1300 per protein, non-randomly selected (first N by file/row order),
is far too small a sample to draw any conclusion about whether the port reproduces the paper's
numbers. What *is* established: both scripts run correctly end-to-end on real data, produce
non-degenerate real correlations (not zero, not NaN, not degenerate), and are ready to scale to the
full dataset. **Scaling to the full 133 natives / all 64 assays' complete mutant sets is real
remaining work**, not yet done — likely cluster-scale given the JAX JIT-recompilation cost of 133
distinct native lengths on CPU (a single 3-native/20-decoy smoke run already took ~85s locally).

## 7. Throughput depth: wall-clock, batching, heterogeneous systems, outer Langevin/E10 (same day)

Per direct follow-up request, the throughput comparison (§1) was extended well beyond the original
single-fixed-batch-size measurement. Reading the reference repo's own production scripts
(`score_decoys.py`, `run_dynamics.py`) showed the *real* usage scale is much larger than what §1's
numbers tested (batch 256/400 vs. the original 4–16) — this section closes that gap.

**All four E11a-d benchmark scripts extended** (`scripts/ebm/benchmarks/{decoy,ddg,biasing,langevin}_benchmark.py`):
raw wall-clock mean/std now persisted per row (previously computed then discarded); a `--batch-sizes`
sweep replaces the single fixed batch value, anchored on the reference's own real defaults; and a
3-way PyTorch comparison on the gradient-path metrics (decoy/ddg `score_grad_ms` only — the only
metrics where `create_graph` genuinely varies) — **shipped** (the reference's own `compute_score()`
wrapper verbatim, `create_graph=True`, "the public code" as literally distributed), **eager**
(the existing optimized bypass, `create_graph=False`), and **compiled** (`torch.compile` wrapping the
eager path). A real, precisely-characterized `torch.compile` finding: compiling in the same process
as aminx's own imports can trip `aminx.training`'s deliberate `NotImplementedError` stub via
Dynamo/Inductor's module introspection — confirmed via an isolated repro to be a process-level
interaction, **not** an incompatibility with the reference architecture itself. Recorded honestly
per-row (`compile_error`), not silently mislabeled as a working "compiled" number.

**Two genuinely new benchmarks**, closing real gaps no prior E11x script covered:

- `heterogeneous_batch_benchmark.py` — no existing benchmark ever tested a batch of *mixed* protein
  lengths (every one uses one fixed length per call, at exact bucket boundaries — zero padding waste
  by construction). Compares JAX's real bucket+pad+tile strategy against a naive PyTorch
  pad-to-batch-max approach and a zero-padding per-structure loop, on an identical realistic mixed-
  length batch (lengths drawn from E4.5's own `build_proxy_distribution`). **Smoke-scale local
  result**: JAX bucket+pad+tile beat naive PyTorch pad-to-max by ~4.6× while paying far less padding
  overhead (1.29× vs 2.39×) — real GPU numbers at production scale pending the cluster run.
- `langevin_annealing_benchmark.py` — `langevin_benchmark.py` (E11d) explicitly scopes itself to the
  *inner* fixed-`t` sampler only; the *outer* noise-schedule/model-swap loop
  (`run_annealing_schedule`) and the full E10 multi-round structure-prediction pipeline were never
  throughput-benchmarked, only correctness-tested. First measurement for both (model-swap uses the
  same real checkpoint on both branches as a stand-in, since only one real checkpoint exists in this
  environment — matches the epic's own already-documented E9 test-stand-in convention, stated
  honestly, not a new limitation).

All six scripts (four extended + two new) pass `--dry-run` (L1) and `--smoke` (L2) gates.

**Cluster confirmatory run**: submitted to Engaging's `pi_so3` partition (Blackwell), SLURM job
`18059808`. Uses the already-proven `XLA_FLAGS=--xla_gpu_shard_autotuning=false` workaround and the
`jax/jaxlib==0.9.2` grad-path pin (decoy/ddg only, per the epic's already-documented jaxlib 0.10.x
regression). Batch-size sweep capped per-length to avoid the already-confirmed L=512 memory ceiling
(256/400 only where memory-plausible, shrinking to 4/16 at L=512).

**Result: `decoy_benchmark`, `ddg_benchmark`, `biasing_benchmark` completed cleanly** across every
length × batch-size combination (52 rows each for decoy/ddg, 8 for biasing) — real 3-way PyTorch
data pulled from engaging and committed. Two findings worth noting in the data itself: decoy's
`torch.compile` variant fails 100% of the time (13/13 compiled rows hit the already-documented
`aminx.training` stub error), while ddg's compiled variant has zero failures — an asymmetry not yet
explained (both scripts wrap the same eager path in `torch.compile`; worth a follow-up look at what
decoy's compile graph touches that ddg's doesn't). These three scripts' batching charts are now live
in the artifact (§01b).

**Resolved 2026-07-19 (PR #118, not yet merged):** root-caused via a Dynamo-verbose repro
(`TORCHDYNAMO_VERBOSE=1`, then calling `decoy_benchmark.py`'s compiled path directly outside its
try/except). The traceback bottoms out in `self.dump(obj)` — Inductor's `FxGraphCache` pickling a
compiled artifact — which calls `pickle.whichmodule`, and that scans every entry in `sys.modules`
via `getattr(module, name, None)` to resolve where an object is defined. `aminx.training`'s stub
`__getattr__` (`src/aminx/training/__init__.py`) raised `NotImplementedError` for *any* name, which
breaks the `getattr(obj, name, default)`/`hasattr()` contract that `whichmodule` relies on to
silently skip modules it can't resolve against — the stub's exception propagates instead, surfacing
as `BackendCompilerFailed`, unrelated to anything the compiled function itself does. This was never
about what decoy's compile graph touches that ddg's doesn't — it's that `aminx.training` merely
needs to be in `sys.modules` (true for any script that does `import aminx`) *and* the specific
compiled graph needs to hit pickle's slow `whichmodule` scan path (a trivial `torch.compile(lambda
x: x + 1)` does not; the real `ProteinEBM.compute_energy` graph does). Fix: raise `AttributeError`
instead — verified via `decoy_benchmark.py --smoke`, all 4 combos now report `compile_error: None`.

**`langevin_benchmark` hit a real crash, not the known autotuning bug**: at length=128, batch=400 —
the reference's own `run_dynamics.py` default batch size, never tested by decoy/ddg (max batch 256)
— JAX's warmup compile call triggered `CUDA_ERROR_ILLEGAL_ADDRESS` on node4008 (Blackwell), with the
SM120 autotuning-hang workaround flag already active, so this is a distinct failure mode from the
already-documented one. Because `run_cluster_benchmarks.sh` had no per-step failure isolation
(`set -euo pipefail`, one linear pipeline), this single crash aborted every step queued after it,
including the two brand-new scripts (`heterogeneous_batch_benchmark`, `langevin_annealing_benchmark`)
— neither got to run at all on the first attempt.

Fixed both scripts before resubmitting: `langevin_benchmark.py` now writes its JSON payload
incrementally after every `(length, batch_size)` cell and wraps each cell in `try/except`, so a crash
loses only the in-flight cell instead of every already-timed one (an `impl="error"` row records the
exception and the sweep continues); `run_cluster_benchmarks.sh` gained a `run_step`/`should_run`
wrapper so one step's failure no longer kills the rest of the job, plus a `BENCHMARK_SKIP_STEPS` env
var to resume without re-running steps that already produced good data. The two never-run scripts
were also reordered ahead of `langevin_benchmark` so they get priority if it crashes again.

Resubmitted directly via `ssh`+`sbatch` (myxcel's `submit_job` MCP tool hit its 30s timeout twice
against a slow login node — a known myxcel-reliability pattern, worked around with direct
`scp`+`sbatch` rather than retried indefinitely) as SLURM job **`18069513`**, skipping
decoy/ddg/biasing via `BENCHMARK_SKIP_STEPS=decoy,ddg,biasing`, running heterogeneous-batch and
outer-annealing first.

**Job `18069513` completed (exit 0, 1h02m, node4008) with a mixed outcome that upgrades the crash
diagnosis.** Per-step result:

- **`heterogeneous_batch_benchmark` failed outright (exit 1, zero rows written)** — and it failed
  in the first ~30s of the job, before any other script ran, during XLA autotuning of the very
  first fused op (a triple-`vmap`'d `LayerNorm`+`Linear` score pass over the n=256 mixed-length
  batch). Same signature as the langevin crash: `CUDA_ERROR_ILLEGAL_ADDRESS` inside
  `cuda_executor`/`config_assigner` during `Failed to enqueue async memset operation`, with
  `hlo_rematerialization` warnings just before it reporting a ~40GiB compile-time memory footprint.
  This script wasn't part of the incremental-write fix (only `langevin_benchmark.py` was hardened),
  so the crash lost the entire run — the smoke-scale local result (JAX bucket+pad+tile ~4.6× over
  naive PyTorch pad-to-max, §7 above) remains the only heterogeneous-batching evidence; no
  production-scale GPU numbers exist yet.
- **`langevin_annealing_benchmark` completed cleanly at all four lengths**, zero errors — the
  first real throughput measurement for the outer noise-schedule/model-swap loop and the full E10
  pipeline. Outer-loop JAX-vs-PyTorch speedup: 26.9× (L=64), 35.8× (L=128), 41.8× (L=256), 34.1×
  (L=512). E10 multi-round pipeline: 11.1–15.2ms scaling mildly with length, 3 rounds each.
- **`langevin_benchmark` completed with the incremental-write fix doing exactly its designed job**:
  23 of 26 attempted (length, batch) cells succeeded; 3 crashed and were caught as `impl="error"`
  rows without losing the other cells in the same file — L=128/batch=400 (**the identical cell that
  crashed job `18059808`, now confirmed reproducible**, not a one-off), L=256/batch=64, and
  L=512/batch=16.

**Revised diagnosis**: this is not a large-batch-specific issue (L=512/batch=16 is a small batch by
any measure) — four independent crashes now, across two different scripts, all the same
`CUDA_ERROR_ILLEGAL_ADDRESS` during XLA autotuning of a large fused kernel, all on node4008. The
common thread looks like compile-time memory pressure (the heterogeneous crash's own log shows
`hlo_rematerialization` failing to get below ~38GiB just before the fault) rather than any one
script's batch-size choice.

**A follow-up retry (SLURM job `18149833`) tested that hypothesis directly and ruled it out.**
Two changes went into the retry: (1) `heterogeneous_batch_benchmark.py` got the same
incremental-write + per-strategy `try/except` hardening as `langevin_benchmark.py` (verified locally
by injecting a synthetic failure into `_jax_bucket_pad_tile` and confirming the other two strategies'
results still land in the output file), and (2) `--n-structures` was cut **4× (256 → 64)** via a new
`BENCHMARK_HETEROGENEOUS_N` override, to test whether a smaller heterogeneous batch avoids the crash.
It didn't: the **identical** fault recurred — same op
(`jit(_score_group)/vmap(vmap(eqx.nn.Linear))/dot_general`, `__triton_gemm` backend), same
`CUDA_ERROR_ILLEGAL_ADDRESS` inside `Failed to enqueue async memset operation`, this time in a
`gemm_fusion_dot` over an `f32[512,8704]` shape — with the L=512 bucket holding only ~7 structures
(vs. ~26+ at n=256). **A batch-size reduction of 4× made no difference**, which rules out total
vmap-batch-size as the trigger; the fault looks tied to the fused-kernel *shape* XLA:Triton generates
for this LayerNorm+Linear pattern at L=256/L=512 on this Blackwell (SM120) node, not to how many
structures are packed into that shape.

The incremental-write fix did work exactly as designed — the crash was caught, logged, and the
error row was persisted to `heterogeneous_batch_benchmark_full.json` — but the corrupted CUDA driver
state left behind by the illegal-address fault (the same "`failed to unload module ...; leaking`"
messages seen in every occurrence of this bug) apparently bloated the process's host memory
footprint enough that SLURM's own accounting killed the whole job as **`OUT_OF_MEMORY`** (`oom_kill`
event, exit 137) a few seconds later, before the two PyTorch-only strategies could run. So this
specific run recovered one honest data point (the crash itself, precisely characterized) but not the
PyTorch comparison numbers.

**Conclusion, treated as closed for this report**: this reads as a genuine XLA:Triton/Blackwell
(SM120) autotuning compiler limitation specific to this fusion pattern at L≥256, not a bug in either
benchmark script and not something `--n-structures` tuning fixes. A real resolution would need either
a targeted XLA flag beyond the already-active `--xla_gpu_shard_autotuning=false` workaround (to
disable Triton GEMM fusion for this op pattern) or running on non-Blackwell hardware — both out of
scope here. Recorded as an **open infra/compiler limitation**, not chased further with more cluster
time: no production-scale heterogeneous-batching number exists, and the smoke-scale local result
(JAX bucket+pad+tile ~4.6× over naive PyTorch pad-to-max, §7 above) remains the only evidence for
that comparison.

**New side effect, 2026-07-18 (after the device-placement fix, jobs `18284162`/`18293715`)**: now
that PyTorch actually shares the same GPU as JAX (see the rewritten §1 above), the corrupted CUDA
driver state left behind by the JAX crash also poisons both PyTorch strategies in the same process —
`pytorch_pad_to_batch_max` and `pytorch_per_structure_loop` now both fail with `AcceleratorError: CUDA
error: an illegal memory access was encountered` immediately after the JAX crash, whereas previously
they were unaffected (silently running on CPU, isolated from JAX's GPU-side corruption). This is not
a new bug and not a regression in the device-placement fix — it's the same already-diagnosed crash's
context-corruption reach extending to whatever else shares the process's GPU context. Net effect:
`heterogeneous_batch_benchmark_full.json` now has **zero** successful rows for either framework
(previously it had zero successful *comparison* rows too, since PyTorch's old CPU numbers were never
a valid GPU-vs-GPU comparison anyway — so no real evidence is lost, but the failure is now total
rather than partial). Process-isolating the PyTorch strategies from JAX's call within this script
(e.g., running each strategy in its own subprocess) would recover the PyTorch numbers even when JAX
crashes, but is out of scope for this report.

**jax/jaxlib 0.11.0 retest (SLURM job `18161115`, 2026-07-17) confirms the crash is version-independent,
not a regression a newer release fixes.** jax/jaxlib 0.11.0 (released 2026-07-16) had already been
confirmed to fix an unrelated, previously-pinned decoy/ddg grad-path regression (a `scf.if`
control-flow gradient compile crash present in 0.10.2 but not 0.8.0/0.8.3/0.9.0/0.9.2/0.11.0), so it
was retested directly against both the heterogeneous and langevin autotuning-crash cells — same
`XLA_FLAGS=--xla_gpu_shard_autotuning=false` workaround, no other flags, `UV_CACHE_DIR` pointed at
scratch storage to route around an unrelated pool-quota issue (see Open Issues below). Result:

- **`heterogeneous_batch_benchmark` (n=256) crashed identically** — same op
  (`jit(_score_group)/vmap(vmap(eqx.nn.Linear))/dot_general`, `__triton_gemm` backend), same
  `CUDA_ERROR_ILLEGAL_ADDRESS` inside `Failed to enqueue async memset operation`, same "55 out of X
  instructions" autotuning-config failure shape as jobs `18069513`/`18149833`.
- **`langevin_benchmark` crashed at the *exact same three cells* as job `18069513`**: L=128/batch=400,
  L=256/batch=64, L=512/batch=16 — nothing above or below those thresholds crashed, and every cell
  below each threshold produced clean, real throughput numbers (the incremental-write fix again did
  its job). Reproducing the identical crash boundary, cell for cell, under a different jax/jaxlib
  version is the strongest evidence yet that this is a hardware/compiler-pattern limitation (specific
  fused-kernel shape on Blackwell SM120) rather than anything version-, batch-size-, or
  script-dependent.

**This confirms, rather than reverses, the "closed as open infra limitation" conclusion above** — it
now holds across three separate jax/jaxlib versions (`0.9.2`, `0.10.2`, `0.11.0`). No jax-version
upgrade is expected to resolve this; a real fix still requires either a targeted XLA flag to disable
Triton GEMM fusion for this exact op pattern, or non-Blackwell hardware. The `0.9.2` grad-path pin in
`run_cluster_benchmarks.sh` remains separately justifiable to retire now that 0.11.0's fix for that
*specific* regression has a real (if still smoke-scale) confirmation — a full-scale confirmatory run
of that fix specifically is still pending and out of scope for this report.

One secondary observation: after every intended output file had already been fully and successfully
written to disk, SLURM OOM-killed the whole job (`exit 0:125`, "1 oom_kill event in
StepId=18161115.batch") a few seconds after the final script's completion echo. Consistent with the
`18149833` postmortem above, this looks like leaked CUDA driver state ("`failed to unload module
...; leaking`", logged after every crash) accumulating host-memory footprint across five sequential
`uv run` invocations sharing one job's 128G allocation, not a sizing problem with any individual
script call. No data was lost — every result file was already complete on disk before the kill.

**Mitigation added for future runs: `langevin_benchmark.py` now proactively dispatches the
`n_trajectories` axis through an xtrax `AxisSpec`/`BatchPlanner` decision instead of crashing and
retrying.** An initial version of this fix (commit `e437535`) caught the crash and retried at half the
batch size — it worked, but retrying at a smaller size means the reported `batch_size` no longer
matches what was requested, and repeated in-process crash/retry cycles risk confounding the timing
with residual state from the failed attempts. Commit `3841a66` replaces it with the same pattern
aminx's own production EBM dispatch already uses (`aminx.ebm.dispatch`/`aminx.ebm.plan`): a per-length
safe batch size (`SAFE_TRAJECTORY_BATCH_BY_LENGTH`, derived directly from the crash thresholds
established above — 64→400, 128→64, 256→16, 512→4) feeds an `AxisSpec`/`BatchPlanner` decision;
at or below the threshold the axis runs as a single `Vmap`, above it as `SafeMap` (`jax.lax.map` in
chunks of the safe size) — proactively, before any crash occurs, so there is nothing to retry. The
*requested* batch size is always what gets measured; each `jax` row now records `dispatch_strategy`
(`vmap`/`safe_map`) and, when chunked, `safe_map_chunk_size`. This is only applied to
`langevin_benchmark.py` — wiring the same mechanism into `heterogeneous_batch_benchmark.py` would be
pointless as a *fix* since total structure count is already known not to be that crash's trigger
(the `18149833` finding above), though it remains an open, untested idea whether a fully
non-vmapped (`SafeMap(batch_size=1)`) dispatch of the L≥256 buckets specifically — never tried, since
every heterogeneous attempt so far has still vmapped multiple structures per bucket — might avoid it
for a different reason (vmap changing the fused kernel's shape/rank, not total element count).
Verified locally: `ty`/`ruff` clean, L1/L2 gates unchanged for the below-threshold path, an injected
correctness check confirming `SafeMap`-chunked and `Vmap` dispatch produce bit-identical per-trajectory
results (trajectories are independent, so chunking cannot change values, only execution order), and an
end-to-end smoke run forcing a batch size above the L=128 threshold confirming the full pipeline
selects `safe_map`, records the chunk size, and produces a plausible number.

**Cluster-confirmed at real scale, 2026-07-17/18 — and a real bug found in the process.** An isolated
cluster test of the L=128/batch=400 cell (job `18182309`) initially *hung* rather than crashed —
concerning, since it was the exact cell that used to fail in under a second. Root-caused via a
zero-cluster-cost `jax.make_jaxpr` inspection: `xtrax.tiling.SafeMap`'s execution lowers to
`jax.lax.map`, which JAX's own docs describe as `scan`-based ("like scan... compiled once"). But
`run_langevin_equilibration` already contains an internal `jax.lax.while_loop`, deliberately chosen
(this module's own docstring) because `while_loop` compiles ~300-400x faster than `scan` on SM120 for
this kind of loop — wrapping it in `SafeMap`'s outer `scan` reintroduced exactly that penalty. Fixed
(commit `0faa1d0`) by dispatching `SafeMap`'s chunking through a plain Python-level `for` loop over
static-size slices instead (`_chunked_vmap`), each independently `jax.vmap`'d — zero `scan` nodes,
verified bit-identical to a single `Vmap` call. Re-tested in isolation (job `18192077`): the full
L=128/batch=400 cell (model load + compile + 5 timed repeats + single-step timing) completed in
**104 seconds**, on the identical cell and identical GPU that crashed every single time before this
session's fixes. Logged as tech debt in `xtrax` (`#795`) and `jaxlint`/`using-jax` (`#796`).

While isolating that fix, a second, much larger bug was found (see the rewritten §1 above): the
PyTorch reference model was never actually moved to GPU, in any of the 6 benchmark scripts, ever.
The full corrected re-run (jobs `18284162` + `18293715`, after fixing that bug too) confirms the
`BatchPlanner`/chunked-vmap fix holds at full production scale: `langevin_benchmark_full_L{64-128,256,
512}.json` show **zero** `CUDA_ERROR_ILLEGAL_ADDRESS` occurrences anywhere, and every cell up to
each length's known crash threshold (L=128/batch≤64, L=256/batch≤16, L=512/batch≤4) succeeds cleanly
via plain `Vmap`. The crash cells themselves (L=128/batch=400, L=256/batch=64, L=512/batch=16) now
fail with a *different*, expected error — a genuine PyTorch-side `CUDA out of memory` — since JAX and
PyTorch now compete for the same GPU memory instead of PyTorch running invisibly on CPU. This is a
new, real resource constraint, not evidence the `BatchPlanner` fix regressed: the previously-crashing
JAX side of these exact cells no longer crashes at all.

## Reproducibility

| Item | Value |
| :-- | :-- |
| Checkpoint | `model_6_expert_frozen_1m_md.pt` ("ProteinEBM-x"), `huggingface.co/jproney/ProteinEBM` |
| Reference repo | `~/repos/ProteinEBM` (Roney/Ou/Ovchinnikov, `jproney/ProteinEBM`) |
| §2 run | bathos `b12dace4-bfb1-4474-aad6-db82d6a65d28`, git SHA `69678405` |
| §3 run | bathos `b3a68dd9-080c-4ce6-a2ae-ea1c2061db06` (outcome: pass), git SHA `69678405` |
| New tracked scripts | `scripts/ebm/collect_synthetic_parity_evidence.py` (+ `.bth.toml`), `scripts/ebm/render_parity_report_data.py` |
| New tracked data | `outputs/ebm_benchmarks/synthetic_parity/*`, `outputs/ebm_benchmarks/lpla_biasing_real.json` |
| Reproduce §2 | `uv run python scripts/ebm/collect_synthetic_parity_evidence.py --checkpoint <ckpt.pt> --reference-repo ~/repos/ProteinEBM --out-dir outputs/ebm_benchmarks/synthetic_parity --sizes 8 16 32 64 --seeds 0 1 2 3 4` |
| Reproduce §3 | `uv run python scripts/ebm/lpla_biasing_check.py --orbax-model <ported_model_dir> --out outputs/ebm_benchmarks/lpla_biasing_real.json` |
| §6 scripts | `scripts/ebm/real_decoy_ranking_benchmark.py`, `scripts/ebm/real_ddg_stability_benchmark.py` (+ `.bth.toml` each) |
| §6 real data | Rosetta decoys (`files.ipd.uw.edu/pub/decoyset/decoys.zip`, `~/repos/ProteinEBM/eval_data/decoys/`), ProteinGym v1.3 Tsuboyama subset (`marks.hms.harvard.edu` — broken TLS chain, fetch with `curl -k`) |
| §7 scripts | `scripts/ebm/benchmarks/{decoy,ddg,biasing,langevin}_benchmark.py` (extended), `scripts/ebm/benchmarks/heterogeneous_batch_benchmark.py`, `scripts/ebm/benchmarks/langevin_annealing_benchmark.py` (both new) |
| §7 cluster jobs | SLURM `18059808` (decoy/ddg/biasing completed; langevin crashed at L=128/batch=400), SLURM `18069513` (resume: heterogeneous FAILED exit 1, annealing OK, langevin OK w/ 3 caught crashes), SLURM `18149833` (heterogeneous retry @ n=64, hardened script: same crash recurred, then OOM-killed), SLURM `18161115` (jax/jaxlib 0.11.0 retest: heterogeneous + langevin crashed at the identical cells as `18069513`, then OOM-killed after all files were written), SLURM `18182309`/`18190343`/`18192077`/`18192388` (BatchPlanner/SafeMap compile-time investigation and device-placement bug discovery, 2026-07-17/18 — see below), **SLURM `18284162` (2026-07-18, first full re-run with device-placement fix: biasing/heterogeneous/annealing/langevin succeeded, decoy/ddg produced zero files due to the run_per_length_group isolation bug), SLURM `18293715` (2026-07-18, decoy+ddg re-run with both fixes: all 6 length-group invocations OK, one expected PyTorch OOM per script at the largest cell)**, Engaging `pi_so3` (Blackwell, node4008) |
| §7 cluster wrapper | `scripts/ebm/benchmarks/run_cluster_benchmarks.sh` — `BENCHMARK_SKIP_STEPS=<comma list> bash scripts/ebm/benchmarks/run_cluster_benchmarks.sh` from the project root on the remote |
| §7 real data | **All corrected 2026-07-18 (device-placement fix, jobs `18284162`+`18293715`)**: `outputs/ebm_benchmarks/{decoy,ddg}_benchmark_full_L{64-128,256,512}.json` (real GPU-vs-GPU rows + 1 expected OOM `error` row each at the largest cell), `biasing_benchmark_full.json`, `langevin_benchmark_full_L{64-128,256,512}.json` (1 expected OOM `error` row per file at the largest cell), `langevin_annealing_benchmark_full.json` — all real production-scale, real-GPU data for both frameworks. `heterogeneous_batch_benchmark_full.json` remains diagnostic-only: the pre-existing JAX autotuning crash (§7 below, unrelated to this fix) now also corrupts the CUDA context for both PyTorch strategies in the same run (they were previously immune, running silently on CPU) — zero successful rows for either framework. |
| §7 report data | `scripts/ebm/render_parity_report_data.py --benchmarks-dir outputs/ebm_benchmarks` → `outputs/ebm_benchmarks/ebm_parity_report_data.json` (`throughput_depth` key; 6/6 scripts have a file, 5/6 have real GPU-vs-GPU measurements as of 2026-07-18, `heterogeneous`'s file is still diagnostic-only). The separate **hardcoded** `THROUGHPUT` dict in the same script (titanix/Turing + engaging/Blackwell-pinned numbers, §1's original 11–92× source) predates the device-placement fix and has not been re-verified against it — left in place as a historical record, not cited as current evidence. |
| §7 jax 0.11.0 retest data | `outputs/ebm_benchmarks/jax011_{heterogeneous_batch_benchmark_full,langevin_benchmark_full_L64\,128,langevin_benchmark_full_L256,langevin_benchmark_full_L512}.json` — version-comparison evidence only, not fed into `render_parity_report_data.py` (the production depth numbers already come from the default-jax `18069513` files above; these confirm the same crash boundary under 0.11.0, they don't add new comparison data) |
