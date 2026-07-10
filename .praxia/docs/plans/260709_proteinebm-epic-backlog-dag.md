# EPIC (proposal): ProteinEBM composable energy/score path in aminx — backlog DAG

- **task_id**: `260709_aminxtension`
- **status**: APPROVED 2026-07-09 — filed into praxia. User decisions: (1) orbax weight-port first (retrain deferred); (2) buckets `(64,128,256,512)`; (3) **E9/E10/E11d (Langevin + structure prediction) → FOLLOW-ON epic**, not this one; (4) PBCNet2.0/E12 → later epic. In-epic nodes: E0–E8 + gates E3.5/E4.5 + benchmarks E11a–c.
- **date**: 2026-07-09
- **branch**: `worktree-proteinebm-decomposition`
- **design spec**: [`specs/260709_proteinebm-aminx-decomposition.md`](../specs/260709_proteinebm-aminx-decomposition.md) (this EPIC supersedes its §7–§9 planning content with the resolved versions below)

## Provenance & review outcome

This EPIC is the output of the praxia pipeline: **compose → brainstorm → adversarial spec critique**. A 3-agent panel reviewed the design spec (audits in `.praxia/audits.jsonl`: `..._spec-challenge`, `..._spec-defense`):

- **Brainstorm** (design forks 1–11) — resolved every open decision with codebase evidence.
- **Challenger** — verdict `not_ready`: 2 BLOCKERs, 5 MAJORs, 5 MINORs.
- **Defender** — verdict `needs_revision`: core thesis strongly supported; 2 load-bearing miscitations.

**Consensus:** the central thesis is correct and well-evidenced — ProteinEBM is a genuinely new *forward energy/score* axis that must **not** be bolted onto the inverse-folding logit `StageSet`; it composes as a peer. The two BLOCKERs and the MAJORs are all about the *mechanism* details, and all are resolved below. After the resolutions, this is a defensible basis to file. The **spec has been corrected** for the confirmed-false claims (see spec §10 "Adversarial review resolutions").

---

## 1. Resolved design decisions (forks 1–11)

| # | Decision | Resolution | Why |
| :-- | :-- | :-- | :-- |
| **1** | Readout abstraction | **Additive peer slots** `energy_readout`/`score_readout` on the *same* `StageSet` + a new `EnergyScoreDecode` mode class in `host/plan.py`, resolved **before** the `sample_step`/`decode_step` checks (mirrors the STEDecode validation at `host/plan.py:448-464`). | Matches the RS-6/RS-7 precedent (how `encoding_fusion`/`decoding_fusion` were added as optional occupancy-checked slots); logit path stays byte-untouched; smallest reviewable diff. Rejects §3.2 option A (Protocol pollution) and B (field duplication). |
| **2** | Score-matching 2nd-order AD | **Nested `jax.grad`** (`score = -jax.grad(E)`; train loss differentiates through it → reverse-over-reverse) + **`jax.checkpoint`** on the 85M trunk. **New 4th synthetic invariant**: finite-difference check of the *outer* training gradient on an analytically-known toy, gating E8; `jax_debug_nans` once before any confirmatory run. | JAX has no `create_graph` flag — grad-of-grad is native but compile/memory-heavy on SM120; the aux non-conservative head avoids it *at simulation time* only. aminx has **zero** existing 2nd-order AD code — genuinely novel territory, must be gated. |
| **3** | Checkpoint strategy | **Orbax weight-port for E3–E7** + a **validation-only** JAX retrain (confirm the loop reduces DSM loss / doesn't NaN, no data-pipeline investment) to decouple "loop works" from "we have published weights". Full from-scratch retrain deferred indefinitely. | Weight-port is strictly cheaper for the near-term parity gate; full retrain is an independently-sized sub-project that must not block E3–E7. **New gate E3.5** (below) de-risks the remap. |
| **4** | MVP parity scope | **ProteinEBM-x @ t=0.05: decoy + ΔΔG + biasing** (E5/E6/E7). | Exercises all three fusion primitives the EPIC introduces (readout, mean-Fuse, difference-Fuse) at the smallest checkpoint footprint. Langevin (E9) buys nothing toward the 3 headline Spearman numbers. |
| **5** | All-atom aux head | **Drop for E3–E7; include for E8.** | `E_θ` depends only on `r⁽ⁱ⁾`, never `r_atom` (provably inert for scoring); but `L_atom` is a literal term in the published loss (needed for training parity). |
| **6** | Bucket boundaries | **`(64,128,256,512)`** — aligned exactly to the §8.3 benchmark sweep (drop the un-benchmarked `384`), so every throughput point lands at zero padding waste. **Mandatory xtrax HiTL gate E4.5** (run `xtrax.eda.analyze_bucket`/`explain_plan` on real CATH/AFDB/TED length stats before locking). | Resolves the challenger's §8.3-vs-§9.4 inconsistency (L=64 had no bucket ≤128 → 4× pad waste). |
| **7** | Langevin composition | **Outer** = xtrax `CarrySpec`+`Scan` (`carry={coords,key}`, `xs=noise_schedule`); **inner** Langevin = plain `lax.while_loop` (inference-only → ~300–400× faster compile than `scan` on SM120, per `using-jax`). Noise-level model-swap = **net-new `lax.cond` dispatcher** over pre-loaded trunk/readout instances. | **BLOCKER-1 fix.** `schedule_selector.py` is unrelated (chromatic decode-*order*); `AxisBoundary` **cannot** hold swappable weights (must flatten to zero JAX leaves). Real ref mechanism is `get_dynamics_model(t)`. Variable inner trip-count → pad-to-max+mask; MH accept → `lax.cond`/`jnp.where`; between-round resampling (multinomial + scipy clustering) → host-side `Sink`/`Tap`. |
| **8** | Training harness | **Adopt `xtrax.engine.Engine` for E8's new trainer only**; defer backfilling the existing (tested) sequence-diffusion `trainer.py`. | Matches spec §3.4 design intent without risking the working `training/test_diffusion_loop.py` path; E8 is the live prototype for whether Engine fits aminx's accumulation-microbatch + NaN-guard needs. |
| **9** | DAG sequencing | See §2. Key fixes: **E9 depends on the checkpoint gate, not E8**; **E4 pre-stubs 3 *separate* sibling dispatch functions** so E5/E6/E7 `[P]` land in disjoint code (else they conflict in the single `_sample_batch`); **E11 splits per-application**; gate nodes get explicit IDs. | Challenger MAJOR-7 + brainstorm Fork 9. |
| **10** (new) | PRNG discipline | Cross-cutting: outer `CarrySpec` carries a splitting `PRNGKeyArray` leaf; self-cond/seq-drop coin-flips get dedicated subkeys; per-element decoy/mutant keys via **`jax.random.fold_in(base_key, idx)`** — reuse the precedent already at `kernel_dispatch.py:244/332/428/512`. | `using-jax` PRNG rules; avoid key reuse across the whole path. |
| **11** (new) | Stale-skill note | The `using-xtrax` "⚠ GAP: `make_inference_plan` validator does not exist" note is **stale** — `validate_plan_topology` is wired at `host/plan.py:440-446` on xtrax 0.4.0a5. **Do not** re-file the ordered-Sink+Vmap conflict as an open risk; it's caught at plan-construction time for free. | Resolves defender A5 / challenger MINOR-1 partially. |

---

## 2. Corrected backlog DAG

Nodes are the filing units. `[P]` = parallelizable within tier. Gate nodes (`*.5`) are first-class (explicit `depends_on`), not asides. Every implementation node carries a **jaxlint gate** (`uv run jaxlint check` on touched files) and **ty/ruff/pytest** green.

```
E0  Foundations & invariants ────────────────────────────────────────────────┐
     • parity harness; synthetic invariants incl. NEW 2nd-order finite-diff    │
       (Fork 2); jaxtyping contracts (coords nm, atom37, E scalar, score N×3)  │
     • GATE: all invariants green on toy                                        │
        ├─> E1  Model trunk port (DiffusionTransformer/AdaLN/conditioners/      │ [P]
        │        FourierEmbedding/RelPos → eqx.Module; no custom CUDA)          │
        └─> E2  R3Diffuser (VP-SDE over CA) — reuse SinusoidalEmbedding/SwiGLU  │ [P]
                 from **model/diffusion_mpnn.py** (NOT training/diffusion.py);  │
                 distinct from the sequence-space NoiseSchedule                 │
E1,E2 ─> E3  EnergyReadout + ScoreReadout(-jax.grad E) + AuxScoreReadout        │
              • non-equivariance: sampled rotation constant w.r.t. grad;        │
                rotate=False for deterministic decoy/ΔE ranking                 │
              • coordinate_scaling applied ONCE (pin diffuser vs ebm.forward    │
                vs precondition — Fork/MINOR-3)                                 │
E3 ──> E3.5 (GATE) Weight-port parity — orbax flat-state_dict→PyTree remap +    │
              INDEPENDENT criterion (per-tensor allclose vs ref forward on a    │
              fixed input; E=0 on zero-r; -grad E == analytic Gaussian score).  │
              De-circularizes MAJOR-3: validates the port itself, not via       │
              the downstream Spearman gate.                                     │
E3.5 ─> E4  Composition wiring (serial; 3-file coupled contract) ──────────────┤
              • add energy_readout/score_readout slots to StageSet (Fork 1)     │
              • EnergyScoreDecode mode class in host/plan.py                    │
              • NEW EnergyFusionFn / generic xtrax Fuse[S,O]  (NOT the          │
                logit-typed DecodingFusionFn — BLOCKER-2)                       │
              • pre-stub 3 SEPARATE sibling dispatch fns (avoid E5/E6/E7 merge  │
                conflict in the single _sample_batch)                          │
E4 ──> E4.5 (GATE, xtrax HiTL) Bucket boundaries (64,128,256,512) — analyze_    │
              bucket on real length histogram before lock (Fork 6)             │
E4.5 ─┬─> E5  Decoy ranking — Vmap/SafeMap over decoys + noise-t sweep     [P]  │  parity ≥ 0.838
      ├─> E6  ΔΔG stability — Vmap mutants + mean-Fuse unfolded ensemble  [P]   │  parity ≥ 0.686
      └─> E7  Conformational biasing/multistate — difference-Fuse         [P]   │  (generic Fuse)
E3 ──> E8  Training path — L=3·L_DSM+0.75·L_aux+0.1·L_atom; nested jax.grad +   │
              jax.checkpoint (Fork 2 gate); xtrax Engine (new trainer, Fork 8); │
              optax OneCycle + clip@10; orbax; all-atom head; validation-only   │
              retrain (Fork 3)                                                  │
E3.5 ─> E9  Langevin sampler — outer CarrySpec+Scan, inner lax.while_loop, MH   │  (depends on
              via lax.cond, net-new model-swap dispatcher, aux score for sim    │   CHECKPOINT,
E9 ──> E10 Structure-prediction pipeline() of scans + host-side resampling      │   NOT E8)
              (multinomial + scipy clustering) as Sink/Tap                      │
E5─>E11a  E6─>E11b  E7─>E11c  E9─>E11d  Benchmarks (per-app, split): accuracy    │
              parity + throughput/latency vs PyTorch. Apples-to-apples:         │
              baseline drops create_graph at inference; SM120 XLA flag;         │
              exclude JIT warmup; lengths = bucket points {64,128,256,512}      │
E12 (future) differentiable multistate design (MPNN×EBM). PBCNet2.0 = separate  │
              lower-priority epic (dependency/domain argument; NOT topology-    │
              reuse — see spec §6 correction)                                   ┘
```

**Confirmed-correct sequencing:** E0 precedes all; E1∥E2 (disjoint new files); E3 is the first shared-file touch (serialization point); E4 serial (couples `stages.py`+`plan.py`+`kernel_dispatch.py`); E8 ∥ E4–E7 (disjoint `training/*` surface); E9→E10 serial.

---

## 3. Risk register (post-review)

| ID | Severity | Risk | Resolution / residual |
| :-- | :-- | :-- | :-- |
| BLOCKER-1 | resolved | Langevin model-swap grounded in wrong file + un-holdable in `AxisBoundary` | Net-new `lax.cond` dispatcher (Fork 7). **Residual scoped risk:** N×85M device memory if all checkpoints resident; mitigation = host-side per-t-range scan segmentation. Budget a memory spike for E9. |
| BLOCKER-2 | resolved | Scalar energy/coord-score is not a `DecodeOutput` → can't use `DecodingFusionFn` | New `EnergyFusionFn` / generic xtrax `Fuse[S,O]` (Fork 1); do **not** extend the JIT-boundary `DecodeOutput`/`InferenceBundle`. |
| MAJOR-3 | resolved | Every numeric gate blocked on an under-committed weight port | Independent parity gate **E3.5**; validation-only retrain fallback (Fork 3). |
| MAJOR-4 | resolved | Langevin ≠ clean nested scan (variable trip-count, MH, host resampling) | Explicitly modeled in E9/E10 (pad-to-max+mask; `lax.cond`; host `Sink`). |
| MAJOR-5 | resolved | Union Gate inert (metric rows, not hypotheses; targets conflated) | Real claim in §4 below (≥2 hypotheses incl. null; clause map; disaggregated targets). |
| MAJOR-6 | scoped | 2nd-order training grad compile/memory/NaN | `jax.checkpoint` + new 2nd-order invariant gate + `jax_debug_nans` (Fork 2). Self-conditioning: **stop-gradient on the recycled estimate** + 2× forward cost noted. |
| MAJOR-7 | resolved | DAG mis-sequencing (E9→E8), illusory E5/E6/E7 parallelism | E9→E3.5; 3 separate dispatch fns; E11 split (§2). |
| MINOR-1..5 | resolved | driver.decode deprecated; benchmark not apples-to-apples; coordinate_scaling double-apply; augmentation; PBCNet2.0 overstatement | Folded into E-node acceptance criteria + spec §10 corrections. |

---

## 4. bathos instruments (pre-registered before confirmatory runs)

**4.1 Literature-parity** (`parity.bth.toml`, Mode B / text-first) — gates E3/E3.5. 5-phase blind-reconstruction → adversarial-refutation → graded verdict; orchestrator re-derivation lock via the E0 synthetic invariants (incl. the new 2nd-order one). Headline scoring metrics run at *inference from ported checkpoints* — **not blocked** by the non-runnable reference `train.py` (defender's decisive point).

**4.2 Accuracy claim** (`claim.bth.toml`, real Union Gate — MAJOR-5 fix):

```toml
[claim]
headline = "The aminx JAX energy path reproduces ProteinEBM-x structure-QA and stability rankings within tolerance at t=0.05."
kill_condition = "any pinned parity target misses its equivalence bound, OR masked-sequence null does not degrade."

[[hypotheses]]  # main
id="H_energy_faithful" label="ported energy reproduces published rankings" predicted_signature="Spearman within bound on the pinned set"
[[hypotheses]]  # null / misspecified — REQUIRED
id="H_backbone_only" label="energy is structural-realism only (sequence inert / port misspecified)" predicted_signature="masked-seq ranking ≈ unmasked; or Spearman collapses"

[[confounds.reference_parity]] parity_run_id="<E3.5 run>"   # decouples remap fidelity
```

Pinned, **disaggregated** targets (challenger MAJOR-5): decoy Spearman **0.838** on the *Rosetta decoy test split*, per-target then averaged; ΔΔG Spearman **0.686** on the *ProteinGym Tsuboyama-cDNA* subset (state exact split + aggregation in the sidecar). Probes: scaled-divergence over noise `t` (main clause); **null-injection = masked-sequence must degrade** (Fig 2b control → eval-validity clause); information-ablation (biasing).

**4.3 Throughput/latency** (`benchmark.bth.toml`, split E11a–d) — vs the **original PyTorch** impl + paper baselines. Apples-to-apples (challenger MINOR-2): PyTorch baseline **drops `create_graph` at inference**; lengths = bucket points `{64,128,256,512}`; **exclude JIT warmup**; SM120 `--xla_gpu_shard_autotuning=false` mandatory (and noted as an XLA de-tune caveat in the writeup). Metrics: `energy_evals_per_sec`, `score_grad_ms` (1st-order), `langevin_steps_per_sec`, plus a compute-matched full structure-prediction wall-clock (paper App. H protocol).

---

## 5. Filed into praxia (2026-07-09) ✅

Filed via `workspace_handshake` → `scope set aminx` → `backlog add` (with `depends_on` edges), workspace `aminx`. Graph verified: all 14 dependency edges match §2. Langevin (E9/E10) and PBCNet2.0 (E12) intentionally **not** filed (follow-on/later epics per user decisions).

| Node | Backlog ID | depends_on |
| :-- | :-- | :-- |
| **EPIC** | **#3294** | — |
| E0 Foundations & invariants | #3295 | — |
| E1 Model trunk port | #3296 | E0 |
| E2 R3Diffuser (VP-SDE/CA) | #3297 | E0 |
| E3 Energy/Score/Aux readouts | #3298 | E1, E2 |
| E3.5 GATE weight-port parity | #3299 | E3 |
| E4 Composition wiring | #3300 | E3.5 |
| E4.5 GATE bucket HiTL | #3302 | E4 |
| E5 Decoy ranking (≥0.838) | #3303 | E4.5 |
| E6 ΔΔG stability (≥0.686) | #3304 | E4.5 |
| E7 Conformational biasing | #3305 | E4.5 |
| E8 Training path (2nd-order) | #3306 | E3 |
| E11a Benchmark decoy | #3307 | E5 |
| E11b Benchmark ΔΔG | #3308 | E6 |
| E11c Benchmark biasing | #3309 | E7 |

Follow-on epic (deferred): E9 Langevin sampler, E10 structure-prediction pipeline, E11d Langevin benchmark. Later epic (deferred): E12 differentiable multistate design; PBCNet2.0 pairwise scoring.

## 6. Still-open decisions for you

1. **Checkpoint**: confirm orbax weight-port first (recommended) vs. insisting on retrain for provenance.
2. **Bucket boundaries**: confirm `(64,128,256,512)` (the xtrax HiTL gate) or supply corpus length stats to derive them.
3. **Scope of E9/E10 (Langevin + structure prediction)** in *this* epic vs. a follow-on — they carry the heaviest residual risk (2nd-order + model-swap memory) and don't affect the 3 headline parity numbers.
4. **PBCNet2.0**: confirm deferral to a separate epic.
