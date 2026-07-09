# ProteinEBM → aminx: Decomposition into a Composable Energy/Score Inference + Training Path

- **task_id**: `260709_aminxtension`
- **status**: DRAFT design + assessment — no runtime code (per session scope decision)
- **author**: orchestrator session "aminxtension"
- **date**: 2026-07-09
- **branch**: `worktree-proteinebm-decomposition`

## Scope & sources

This document determines **how** to decompose ProteinEBM as a new inference/training path
in aminx, composed on xtrax, and assesses PBCNet2.0 for a similar purpose. It is a design
spec, not an implementation. The next step (see §7) is to file a praxia EPIC, decompose it
into a backlog DAG, brainstorm + adversarially critique the spec, and gate on user review
**before** the DAG is filed.

Primary sources (all read for this spec):

| Source | Location | Role |
| :-- | :-- | :-- |
| ProteinEBM paper — *"Protein Diffusion Models as Statistical Potentials"* (Roney, Ou, **Ovchinnikov**), bioRxiv `2025.12.09.693073v3` | `gdrive:00sources/mpnn_multistate/` | Method / architecture / benchmarks (parity targets) |
| ProteinEBM reference code (PyTorch 2.6 + Lightning) | `~/repos/ProteinEBM` | Port surface |
| PBCNet2.0 paper — *"Atomic-level protein–ligand recognition with PBCNet2.0"*, Nat. Chem. Biol. `s41589-026-02241-x` | `gdrive:00sources/cosolvent/` | Assessment target |
| PBCNet2.0 reference code (PyTorch + DGL 1.0.2) | `~/repos/PBCNet2.0` | Assessment port surface |
| aminx composition internals | `docs/COMPOSITION_GUIDE.md`, `src/aminx/{inference,host,training,model,run}/` | Target architecture |
| xtrax composition layer | `using-xtrax` skill, xtrax v0.3.0 + main | Composition primitives |
| bathos parity + claim tooling | `using-bathos` skill | Experiment / parity / benchmark discipline |

> **Note on the gdrive/`mpnn_multistate` filing.** The paper filed under `mpnn_multistate/`
> *is* the ProteinEBM preprint. The "multistate" relevance is real but indirect: ProteinEBM
> supplies a **direct per-state energy** `E_θ(x_state, s)` that complements ProteinMPNN's
> Bayes-ratio conformational biasing (see §5). That is the integration payoff, not a separate
> paper.

---

## 0. Executive summary / verdicts

**ProteinEBM → aminx: GREEN (decompose it).** ProteinEBM is architecturally the *inverse* of
today's aminx (a forward, generative **energy** `E_θ(x,s,t)` over CA coordinates vs. aminx's
inverse-folding decode-to-logits), so it does **not** fit the existing `StageSet` logit-fusion
topology. But it decomposes cleanly onto xtrax's **tiling + composition** primitives: the five
paper applications each map to a known xtrax strategy (Vmap/SafeMap scoring, difference-`Fuse`
conformational biasing, `Scan`/`CarrySpec` Langevin annealing, a `pipeline()` of scan rounds for
structure prediction), and training reuses xtrax `Trainer`/`Engine`/`ResumableState` + optax + orbax.
The numerically load-bearing core (energy head, VP-SDE diffuser, Boltz-1-style transformer trunk)
is dependency-light PyTorch with **no custom CUDA kernels** — a mechanical Equinox port. The real
cost is (a) JAX fixed-shape discipline vs. the reference's dynamic crop/pad + per-batch coin-flip
branching, (b) replacing PyTorch-Lightning wholesale, and (c) the fact that the reference
`train.py`/`dataset.py` **do not run as shipped** (three undefined-name bugs) and must be treated
as design intent, not a verified parity target.

**PBCNet2.0 → aminx: YELLOW (defer; reuse the pattern, not the model).** Only the *scoring
topology* — encode → **fuse-by-difference** → decode-to-scalar — overlaps aminx (it is literally
what the RS-6/RS-7 `EncodingFusionFn`/`DecodingFusionFn` machinery already expresses). The encoder
(TensorNet O(3)-equivariant rank-2 Cartesian tensors), RDKit/BioPython/docking-pose featurization,
DGL→jraph message passing, 8.6M-pair data pipeline, and pickled-`nn.Module` checkpoint are ~90%
net-new work in a different scientific domain (small-molecule affinity). There is **no** encoder or
weight reuse with LigandMPNN. Recommendation: **not** a near-term aminx path; if pursued later,
frame it as a *separate* xtrax-composed scoring model that reuses the composition scaffolding, at
lower priority than ProteinEBM. Details in §6.

**Benchmarking is first-class (per user).** Every parity target and a throughput/latency sweep vs.
the original PyTorch implementations (and the paper's own baselines — Rosetta, ESM-IF/ESM3,
ProteinMPNN, AF2/AF3) is pre-registered as a bathos claim + benchmark sidecar. See §8.

---

## 1. What ProteinEBM is (grounded)

**Learned object.** A scalar energy `E_θ(x, s, t)` defining a Boltzmann density
`p(x|s) ∝ exp(−β E_θ(x,s))`, interpreted as a coarse-grained (CA-level) free energy. Trained by
**energy-parameterized denoising score matching**: the score is the *negative gradient of the
energy*, `s_θ(x,t) = −∇_x E_θ(x,t)` (Du 2023 / Thornton 2025 style), which is what makes it a
genuine potential rather than a bare denoiser.

**Energy parameterization (key detail).** The final transformer layer emits a per-residue
3-vector `r⁽ⁱ⁾(x,s,t) ∈ ℝ³`, and `E_θ = Σᵢ ‖r⁽ⁱ⁾‖²` (masked sum). This guarantees `E_θ ≥ 0` and
well-behaved gradients. **The conservative score is `−∇_x E_θ` via autograd** (`torch.autograd.grad`
in the reference; a direct `jax.grad` analog in the port).

**Diffusion process.** VP-SDE over **CA translations only** (`data_dimension = 3`;
`diffuse_sidechain = False` in both shipped configs). Coordinates are scaled by
`coordinate_scaling = 0.1` (Å→~nm) to keep the process near unit variance. Schedule:
`b_t = min_b + t(max_b − min_b)` with `min_b=0.1, max_b=20.0`; integrated
`marginal_b_t = t·min_b + ½t²(max_b−min_b)`; `conditional_var = 1 − exp(−marginal_b_t)`;
closed-form DSM target `score = −(x_t − exp(−½·marginal_b_t)·x_0)/conditional_var`. **This is VP,
not VE** — the port must not import a VE convention.

**Backbone.** AlphaFold3 / **Boltz-1** diffusion-module design (`DiffusionTransformer`: AdaLN-
conditioned pair-bias attention, `SingleConditioning`/`PairwiseConditioning`, Fourier time
embedding, relative-position encoder). **Non-equivariant** + random SO(3) augmentation
(`center_random_augmentation`); IPA was *explicitly rejected* because optimizing second-order
derivatives (energy→gradient→score) through IPA is unstable. **85M parameters** (downscaled Boltz-1).

**Auxiliary heads (both matter).**
- **Non-conservative score head** `r_aux⁽ⁱ⁾` (separate linear projection from the same final
  layer): a cheap score surrogate used *during simulation* to avoid backprop-through-gradient at
  every Langevin step. Loss `L_aux`.
- **All-atom head** `r_atom⁽ⁱ⁾ ∈ ℝ³⁷ˣ³` (CA-centered heavy-atom prediction, **not** diffused):
  auxiliary supervision only. Loss `L_atom` (known to mean-collapse).

**Total training loss:** `L = 3·L_DSM + 0.75·L_aux + 0.1·L_atom`.

**Conditioning.** Sequence `s` enters directly (MSA-free); masked 10% of the time in training.
Self-conditioning 50% of the time (recycled denoised estimate). A per-residue "external-contact
flag" marks cropped interaction partners (zeroed at inference). Diffusion time `t` is a **tunable
inference knob**, not fixed at 0 (see §4).

**Representation contracts (for jaxtyping).**
- Diffused variable: CA coords `x ∈ ℝ^{N×3}` (nm after scaling).
- All-atom aux target: `atom37` `∈ ℝ^{N×37×3}`, CA at atom index 1, CA-centered.
- Energy head: per-residue `r⁽ⁱ⁾ ∈ ℝ³` → scalar `E`.
- `aatype ∈ [0,20]` (21-way embedding incl. mask token 20); residue mask `(N,)`; atom-presence
  mask `(N,37)`; contact flag `(N,)`; self-cond channel same shape as `x_t`.
- Time `t ∈ [0,1]` scalar (per batch element).

---

## 2. Why this is a NEW axis for aminx (the core finding)

aminx today is **inverse folding**: structure → sequence, an autoregressive decode producing
per-position logits over the 21-AA vocabulary. Its whole composition machinery is built around
that shape:

- `StageSet` slots (`logit_transform: (S,L,V)→(L,V)`, `ar_logit_transform: (S,V)→(V)`,
  `decode_step`, `sample_step`, `tie_group_fuse`) all operate on **logits `(…,V)`**.
- `driver.infer_topology` returns exactly three topologies — `AR`, `CONDITIONAL_SCORE`,
  `UNCONDITIONAL` — all producing `Logits` or a `SampleResult` of tokens.

ProteinEBM's outputs are a **scalar energy** and a **coordinate score/force `∇_x E ∈ ℝ^{N×3}`**.
There is no vocabulary, no per-position categorical, no token sampling. It cannot be expressed by
the *current* `StageSet`, whose slots are all logit-shaped. **This is the argument *for* the xtrax
composition design, not a limitation:** the logit `StageSet` is one stage-bundle; the energy path is
*another* composable stage-bundle, and xtrax composition (readout-agnostic `Fuse`/`Tap`/`Sink`/
`AxisBoundary` + tiling strategies) is exactly the mechanism that lets both coexist and interoperate
without either perturbing the other. We generalize the readout abstraction (see §3.2) so energy is a
peer of logits, not a special case bolted onto the logit path. Its "sampling" is coordinate Langevin
dynamics (a scan over noise/MCMC steps updating coordinates), not autoregressive token decode.

The existing `model/diffusion_mpnn.py` + `training/diffusion.py` is **not** the same thing: that is
**sequence-space** diffusion (diffusing the AA one-hot, denoising to logits). Its `NoiseSchedule`,
`SinusoidalEmbedding`, and `SwiGLU` are reusable *building blocks*, but the learned object is
different (coordinate energy vs. sequence logits). See §3.4.

**Conclusion:** ProteinEBM is a genuinely new, orthogonal capability for aminx — a *forward /
generative energy model* alongside the existing *inverse-folding decoder*. The decomposition below
adds it as a first-class path without perturbing the logit topology.

---

## 3. The decomposition onto xtrax (the heart)

The design principle mirrors the COMPOSITION_GUIDE: **assemble a new path from stages and xtrax
tiling strategies without touching the existing kernel math.** ProteinEBM adds (a) a new readout,
(b) a new topology, (c) a set of inference paths that are each a known xtrax strategy, and (d) a
training path on the xtrax run/training layer.

### 3.1 New readouts (the atom of the design)

Two composable `eqx.Module` readouts over the transformer trunk output `a`:

```
EnergyReadout(a, mask)   -> E: Float[Array, ""]            # E = sum_i mask_i * ||r_proj(a_i)||^2
ScoreReadout             -> s = -jax.grad(energy_wrt_coords)  # conservative score  (Float[N 3])
AuxScoreReadout(a)       -> r_aux: Float[N 3]              # cheap non-conservative score for sim
```

`ScoreReadout` is *derived*: `score = -jax.grad(lambda x: EnergyReadout(trunk(x,s,t), mask))(x)`.
This is the exact JAX analog of the reference's `torch.autograd.grad(energy.sum(), r_noisy,
create_graph=True)` and is the entire "EBM mechanism." No new math is invented; the readout is a
scalar head plus autograd.

### 3.2 New topology — a peer StageSet, not a bolt-on

The intent (per the composition-first framing above) is to make the readout **kind** a first-class,
readout-agnostic axis rather than special-casing logits. Two equivalent implementations, to be
chosen at spec-critique time:

- **(a) Generalize `StageSet`** so its decode/readout slot is typed over a `Readout` union
  (`LogitReadout | EnergyReadout | ScoreReadout`), and `infer_topology` dispatches on the readout
  kind — logits stay exactly as-is; energy is a peer variant.
- **(b) A parallel `EnergyStageSet`** bundle resolved by the same `make_inference_plan` factory.

Either way, extend `inference/driver.py::infer_topology` (and the mode-class resolution that
superseded it in `host/plan.py`) with an **energy/score topology**:

```
TOPOLOGY_ENERGY   — readout is EnergyReadout/ScoreReadout; output is (E, per_residue_E, score)
```

Selection is by readout kind, exactly as the current code selects by `decode_step`/`sample_step`
occupancy. This is **additive**; the three existing logit topologies are untouched. The unification
seam between the energy bundle and the logit bundle is a plain xtrax `Fuse` over a state axis (§5) —
which is why composing *other* StageSets under xtrax is the whole point. `host/kernel_dispatch.py`
`resolve_kernel_fn` gains one case (per the COMPOSITION_GUIDE "new host dispatch path" row).

### 3.3 The five inference paths → xtrax primitives

Each ProteinEBM application maps to a specific xtrax tiling strategy. This is where the
composition pays off — no bespoke loops, just declared axes.

| Paper application | Operation | xtrax primitive | Notes |
| :-- | :-- | :-- | :-- |
| **Decoy ranking / structure QA** (Spearman 0.838) | Score N decoys at fixed `t` | `AxisSpec(decoys)` → **Vmap** (small N) / **SafeMap** (large N) | Each decoy has distinct coords → no `encode-once`; vmap `EnergyReadout` over the decoy axis. Noise-time sweep = a second **Vmap** axis over `t`, or a `Fuse` that reduces the `t` axis. |
| **Mutation ΔΔG / stability** (Spearman 0.686, SOTA) | `E(x,s) − E(x,s′)` minus MC unfolded correction over `p_UF` samples | **Vmap** over mutants × a **Fuse** (mean) over the unfolded ensemble axis | The unfolded-ensemble mean is a textbook `Fuse[S]→[1]`. `dedup_eligible` if many mutants share a WT reference structure (**DedupGather**). |
| **Conformational biasing / multistate** | `E(x_state1,s) − E(x_state2,s)` per fixed `s` | **difference-`Fuse`** via a **new `EnergyFusionFn` / generic xtrax `Fuse[S,O]`** over a 2-element state axis (⚠ **NOT** the logit-typed `DecodingFusionFn` — a scalar energy is not a `DecodeOutput`; see §10 BLOCKER-2) | This is the multistate mechanism (§5). |
| **Langevin annealing / folding simulation** | reverse-SDE + local Langevin equilibration per noise level | Outer = **`Scan` + `CarrySpec`** (carry = `(coords, key)`; xs = noise-level schedule); inner Langevin = **`lax.while_loop`** (inference-only, fast compile) | ⚠ Multi-checkpoint handoff at `t=0.1` is a **net-new `lax.cond` dispatcher** over pre-loaded models — **NOT** `schedule_selector.py` (unrelated) and **NOT** an `AxisBoundary` (cannot hold weights); see §10 BLOCKER-1. Variable inner trip-count → pad-to-max+mask; MH accept → `lax.cond`; between-round resampling → host-side `Sink`. Aux score inside sim; conservative energy for rescoring. |
| **Structure prediction (MSA-free)** | initial-sample → resample (Boltzmann) → 3× refine → optional AF2Rank | **`pipeline()`** of `Scan` stages, with an importance-resampling **`Fuse`** between rounds | Clustering + AF2Rank are host-side post-processing (numpy/scipy, outside jit) — a `Sink`/`Tap`, not a traced stage. |

Design rules honored (from `using-xtrax`):
- Variable-length proteins → **`Bucket`** on the residue axis (`bucket_boundaries`) to bound XLA
  recompilation, replacing the reference's dynamic crop/pad. This directly resolves the "biggest
  structural change" flagged in the port surface (§4).
- The Langevin `CarrySpec.init` must be static-shape (bucketed max length) — the `🔬 HiTL` static-
  shape stop in the skill applies; the residue axis is bucketed, the carry is fixed per bucket.
- Ordered `Sink` (trajectory dump) + `Vmap` over trajectories is the one topology conflict the
  skill flags (`make_inference_plan` validator gap) — trajectories that need ordered per-step I/O
  must use a `Scan`/ordered path, not vmap; parallel trajectories that only dump at the end can vmap.

### 3.4 Training path → xtrax training layer

The reference training is PyTorch-Lightning and **must be replaced wholesale** (there is no
drop-in). Target the xtrax training layer already vendored in aminx:

- **State**: `xtrax.training.ResumableState` (step, key, model, opt_state) + orbax checkpointing
  (`using-orbax`). Replaces Lightning's `ModelCheckpoint`.
- **Loop**: `xtrax.training.Trainer.step` (`@eqx.filter_jit`) + `Engine.fit_sync` with callbacks.
  Replaces `LightningModule`/`Trainer`.
- **Optimizer**: optax Adam + OneCycle (`using-optax`), `clip_by_global_norm(10.0)` — matches the
  reference's hardcoded grad-clip and `OneCycleLR(pct_start=0.3, div_factor=25, final_div_factor=1e4)`.
- **Loss**: a new `score_matching_loss` implementing `L = 3·L_DSM + 0.75·L_aux + 0.1·L_atom`, all
  masked MSE. Lives beside the existing `training/losses.py`.
- **Diffuser**: a new `R3Diffuser`-equivalent (VP-SDE over CA coords, PRNG-threaded) — **distinct**
  from the existing sequence-space `training/diffusion.py::NoiseSchedule`, but reusing its
  `SinusoidalEmbedding` and `SwiGLU`. The reference's numpy/torch dual RNG must unify under
  `jax.random`.
- **Reuse**: `DiffusionAminx`'s time-embedding injection pattern is the template for wiring the
  Fourier/sinusoidal `t` embedding into the trunk.

**Coin-flip branches** (per-batch `random.random()>0.5` self-conditioning; 10% sequence-drop) must
become **always-compute-both-branches + mask select** or `lax.cond` on a traced boolean — the
single subtle correctness item in the training port.

### 3.5 Model port (net-new module, mechanical)

A new `eqx.Module` trunk (`DiffusionTransformer`, `AdaLN`, `AttentionPairBias`, conditioners,
`FourierEmbedding`, `RelativePositionEncoder`). It does **not** reuse the ProteinMPNN encoder/decoder
(different architecture: full transformer trunk vs. MPNN message passing). Port is mechanical:
`nn.Module`→`eqx.Module`, `torch.einsum`→`jnp.einsum`, einops works over JAX arrays unchanged,
in-place inits → Equinox init-time construction, `torch.autocast` → explicit float32 upcast. **No
custom CUDA kernels exist** in the core path.

---

## 4. Port surface, risks, and what NOT to trust

From the reference-code recon (`~/repos/ProteinEBM`):

**Clean / mechanical:** `ebm.py` energy head, `layers.py` transformer blocks, `boltz_utils.py`
rotation/augmentation utils, `r3_diffuser.py` VP-SDE math. Conservative score = `jax.grad` (direct).
Clustering/AF2Rank stay host-side numpy/scipy. `af2rank_cluster.py` already uses JAX/haiku via an
external AlphaFold checkout at a hardcoded path — **out of scope** for the core port.

**Hard / architectural:**
1. **Fixed-shape discipline.** Reference uses dynamic crop/pad + per-batch dynamic padding.
   → resolve via xtrax **`Bucket`** on the residue axis + full masking. *Biggest structural change.*
2. **Coin-flip control flow.** Self-conditioning and sequence-drop are per-batch Python `if
   random()>0.5`. → mask-select / `lax.cond`.
3. **Lightning replacement.** Entire training/checkpoint/DDP machinery → xtrax `Trainer`/`Engine` +
   optax + orbax.
4. **Checkpoint reuse (optional).** Reference `.pt` is a Lightning `state_dict` with a `"model."`
   prefix. Cross-loading old weights needs a one-time flat-key→Equinox-PyTree remap script (orbax);
   otherwise retrain in JAX. Decide during epic scoping.

**Do NOT port verbatim (reference-only, confirmed non-runnable as shipped):**
- `train.py`: `cluster_sizes` never defined (`NameError`); imports `protein_ebm.model.ema` which
  **does not exist** (`ImportError` at load).
- `dataset.py`: `ProteinNoisedDataset.__init__` references undefined `target_ids` (`NameError`).
- `ebm.py::compute_score`: `atom_mask` referenced but never extracted when `diffuse_sidechain=True`
  (dead path in shipped configs, but a `NameError` if enabled).
- Dead config flag `use_attention_mask` (time-dependent attention cutoff described in YAML comments,
  **not implemented**).

**Discipline (from `~/.claude/rules/BATHOS.md`): verify the measurement pipeline on synthetic
ground truth before trusting any parity conclusion.** Concretely, before any parity claim: feed the
VP-SDE `score` closed form a known `(x_0, t)` and assert it matches `−(x_t−√ᾱ x_0)/(1−ᾱ)`; feed
`EnergyReadout` a zero-`r` trunk and assert `E=0`; assert `−jax.grad(E)` equals the analytic score on
a Gaussian toy. These 30-second invariants gate the reimplementation (bathos literature-parity
Phase 5 invariant-test spec, §8).

### 4.1 aminx-side hygiene found while writing this spec

The EBM/score work builds on aminx's existing (sequence-space) diffusion path
(`model/diffusion_mpnn.py`, `training/{diffusion,train_diffusion,trainer}.py`). A `ruff --select F`
+ `ty` sweep confirms aminx does **not** carry ProteinEBM's crash bugs (`cluster_sizes`,
`target_ids`, missing `model.ema` — grep-clean), but surfaced the following in the diffusion path,
which E0/E1 should inherit clean:

| Finding | Location | Class | Disposition |
| :-- | :-- | :-- | :-- |
| Undefined name `Any` in `**kwargs: Any` | `model/diffusion_mpnn.py:170` | F821 (undefined name; latent at runtime only because of `from __future__ import annotations`, but fails CI `ruff`/`ty`) | **FIXED** in this branch (`from typing import TYPE_CHECKING, Any`) |
| `inference = True` set when `key is None` then **never used** | `model/decoder.py:629` | F841 dead store | **INVESTIGATED → not a bug, then RESOLVED.** Trace showed dropout was *already* correctly disabled at inference (two independent guards: `DecoderLayer.__call__:318-319` re-derives `inference=True` on `key is None`; `Dropout.__call__:52-54` returns `x` on `key is None`). Fixed anyway for correctness/symmetry: added `inference: bool = False` to the unconditional `Decoder.__call__` (mirrors `call_conditional`), threaded it into the layer call (F841 resolved), added determinism regression tests (`tests/model/test_decoder_unconditional_inference.py`, 3/3 green; 26/26 decode-path tests still pass). Behavior-preserving for the one live caller. |
| `UnconditionalDecode` fallback does not forward `config.inference` (the `decode_step` branch above it does) | `inference/decode/unconditional.py:100` | consistency / possible dropout-during-unconditional-scoring | **FLAGGED (new, gated on parity).** Now that `model.decoder` accepts `inference`, the fallback *could* forward `config.inference` to match the branch above. Doing so changes unconditional-scoring dropout when `config.inference=True` with a non-None key — a scoring-numerics change → gated on a parity check before wiring (comment updated in place; behavior unchanged for now). |
| `PRNGKeyArray` imported from jaxtyping then redefined as `jax.Array` | `model/encoder.py:30` | F811 redefinition | **FLAGGED** — cosmetic, zero behavioral risk |
| `super().__call__(...)` passes 7 positional args; `Aminx.__call__` expects 5 | `model/diffusion_mpnn.py:184` | `ty` `too-many-positional-arguments` (pre-existing) | **FLAGGED** — possible real signature bug in the diffusion path, or a `ty` false-positive; verify against `Aminx.__call__`'s actual signature before the port relies on it |

The `Any` (F821) and `decoder.py:629` (F841 + symmetry) items are **fixed** on this branch with a
regression test; the `decoder.py:629` investigation confirmed dropout was never actually running at
inference (two independent guards), so the fix is correctness/symmetry, not a behavior change. The
new `unconditional.py:100` fallback inconsistency and the `encoder.py:30` (cosmetic) and
`diffusion_mpnn.py:184` (`ty` positional-arg mismatch — verify vs. `Aminx.__call__`'s real signature)
items remain for epic triage; the fallback one is a scoring-numerics change and must go through a
bathos parity check (§8) before wiring.

---

## 5. The multistate integration payoff (MPNN × EBM)

The reason this belongs next to aminx's ProteinMPNN, not in a vacuum:

- ProteinMPNN conformational biasing needs a **Bayes-ratio** `log p(s′|x_open)/p(s′|x_closed)` and is
  exact only up to a sequence-independent constant (fine for ranking mutants, not absolute occupancy).
- ProteinEBM gives a **direct per-state energy** `E_θ(x_state, s)` for a *fixed* sequence, so a
  multistate objective is a plain **difference-`Fuse`** over a state axis: `ΔE = E(x_1,s) − E(x_2,s)`
  in units of `k_BT`.

This yields a native aminx **multistate scoring path** and, longer term, the differentiable
**multistate design** loop the paper flags as future work: design a sequence with ProteinMPNN, score
its per-state energy landscape with the EBM, and (eventually) backprop an energy-gap objective into
sequence logits. The EBM path and the existing decode path meet at exactly one composition seam — a
`Fuse` over states — which is why the decomposition is worth doing inside aminx rather than as a
separate repo.

**In-scope now:** the EBM per-state energy scoring path (a `Fuse`). **Out of scope / future epic:**
differentiable joint multistate sequence design (needs sequence-space gradients through the EBM).

---

## 6. PBCNet2.0 assessment (YELLOW — defer)

**What it is.** A TensorNet-style **O(3)-equivariant Siamese GNN** predicting **relative** binding
affinity `Δŷ = pAct_ref − pAct_query` between two protein–ligand complexes sharing the same pocket
(congeneric series or WT/mutant). Rank-2 Cartesian tensors per atom (scalar+skew+symmetric-traceless
decomposition), scatter-sum message passing (DGL), ligand-atom-pooled embeddings fused **by
difference** into a 3-layer FFN, with a sign-flip antisymmetry augmentation. Trained on 8.6M pairs;
needs a docked pose per ligand.

**Overlap with aminx (the only green part).** The outer topology encode(ref) → encode(query) →
**fuse-by-difference** → decode-to-scalar is *exactly* aminx's `EncodingFusionFn`/`DecodingFusionFn`
composition (RS-6/RS-7). The DGL scatter-sum maps trivially to jraph/`segment_sum`; the tensor
algebra is plain `jnp.einsum`/`matmul` (no e3nn needed). The sign-flip is a training trick, not
architecture.

**Why it's ~90% net-new (the yellow/red part).**
- The **entire encoder** is a from-scratch small-molecule+pocket equivariant GNN — no LigandMPNN
  encoder or weight reuse.
- Hard external dependency chain aminx doesn't carry: **RDKit** atom/bond featurization, BioPython
  pocket extraction, and an implicit **docking-pose** prerequisite (pose quality is load-bearing).
- Checkpoint is a pickled full `nn.Module` requiring a legacy PyTorch + DGL 1.0.2 / Python 3.8
  environment to even extract weights.
- Data/loss/pipeline (8.6M BindingDB-derived pairs, Tanimoto clustering, MCS pose selection) is a
  wholesale new data-engineering surface in a different scientific domain.

**Recommendation.** Do **not** fold PBCNet2.0 into the ProteinEBM epic. If protein–ligand affinity
becomes a goal, file it as a *separate, lower-priority* epic: "a new xtrax-composed pairwise scoring
model reusing aminx's Fuse/DecodingFusionFn/InferencePlan scaffolding," explicitly decoupled from
LigandMPNN. The reusable asset is the **composition pattern**, not the model. Its literature-parity
validation (equivariant-mixing numerical parity, exact graph-construction cutoffs) would be its own
`parity.bth.toml`.

---

## 7. From spec to filed EPIC (the praxia pipeline — gated on user review)

Per user direction, after this doc the flow is:

1. **File a praxia EPIC** (`epic_compose`) — "ProteinEBM composable energy/score path in aminx."
2. **Decompose into a backlog DAG** — the phase graph in §7.1 becomes backlog nodes with explicit
   `depends_on` edges (`backlog` / `dw_compose_sprint`).
3. **Autonomous brainstorm** (`contemplex` brainstorm session) on the open design forks (§9).
4. **Adversarial spec critique** — `spec-challenger` vs. `spec-defender` + `oracle` synthesis
   (the `spec-driven-dev` dynamic workflow), plus `plan-auditor`.
5. **→ USER REVIEW GATE ←** — present the critiqued EPIC + DAG for final review.
6. **Only then** file the entire EPIC backlog DAG in praxia.

This document is the input to steps 1–4. Nothing is filed in praxia yet.

### 7.1 Proposed phase graph (→ backlog DAG nodes)

Dependencies are the DAG edges. `[P]` = parallelizable within its tier.

```
E0  Foundations & invariants (parity harness, VP-SDE toy tests, jaxtyping contracts)
      └─> E1  Model trunk port (DiffusionTransformer/AdaLN/conditioners as eqx.Module)   [P]
      └─> E2  R3Diffuser (VP-SDE over CA) + NoiseSchedule reuse                            [P]
E1,E2 └─> E3  EnergyReadout + ScoreReadout (jax.grad) + AuxScoreReadout  ── literature-parity gate
E3    └─> E4  TOPOLOGY_ENERGY in driver/plan/kernel_dispatch + InferencePlan wiring
E4    └─> E5  Inference path: decoy ranking (Vmap/SafeMap)   ── parity: Spearman ≥ 0.838   [P]
E4    └─> E6  Inference path: ΔΔG stability (Vmap + unfolded Fuse) ── parity: Spearman ≥ 0.686 [P]
E4    └─> E7  Inference path: conformational biasing / multistate (difference-Fuse)        [P]
E3    └─> E8  Training path: score-matching loss + xtrax Trainer/Engine + optax/orbax
E8    └─> E9  Langevin annealing sampler (Scan+CarrySpec, schedule_selector handoff)
E9    └─> E10 Structure prediction pipeline (pipeline() of scans + resampling Fuse)
E5..E7,E9 └─> E11 Benchmark campaign: accuracy parity + throughput/latency vs PyTorch & baselines
E11   └─> E12 (future) Differentiable multistate design (MPNN×EBM); PBCNet2.0 = separate epic
```

**Checkpoint-reuse decision node** (attach to E3): port old weights via orbax remap, or retrain in
JAX. Retraining needs the CATH/AFDB/TED + BioEmu-MD data pipeline (a sub-track); weight-porting is
cheaper for a parity gate but couples to the legacy `.pt` format.

---

## 8. bathos experiment / parity / benchmark pre-registration

Three bathos instruments, all pre-registered **before** confirmatory runs.

### 8.1 Literature-parity validation (`parity.bth.toml`) — gates E3

ProteinEBM is a paper reimplementation whose reference code is partly non-runnable (§4), so it is a
**Mode B (text-first) literature-parity** target. Run the 5-phase protocol:

```toml
# parity.bth.toml  (alongside the energy-readout module)
[parity]
paper_pdf         = ".../2025.12.09.693073v3.full.pdf"
impl_paths        = ["src/aminx/model/ebm_trunk.py",
                     "src/aminx/inference/energy_readout.py",
                     "src/aminx/training/r3_diffuser.py"]
reference_code    = "~/repos/ProteinEBM/protein_ebm"
citation_note     = "Energy = sum_i ||r_i||^2 (App. A); score = -grad_x E; VP-SDE, coords in nm."
recon_lenses      = ["math", "algo", "protocol"]
attack_lenses     = ["stats", "hyper", "struct"]
hypotheses        = ["conservative score = -grad E is implemented (not the aux head)",
                     "VP (not VE) schedule; coordinate_scaling applied consistently",
                     "energy sum-of-squares parameterization reproduced"]
equivalence_bound = 0.02
N = 3
M = 3
```

**Orchestrator re-derivation lock:** after the agent phases, independently re-derive the decisive
findings with runnable invariant tests (§4 synthetic checks). The graded verdict
(PARITY/PARTIAL/FAIL) feeds `[confounds.reference_parity]` and gates the accuracy campaign.

### 8.2 Accuracy claim (`claim.bth.toml`) — confirmatory campaign, Union Gate

Headline: *"The JAX/Equinox aminx energy path reproduces ProteinEBM's published structure-QA and
stability rankings within tolerance."* One row per parity target below; `kill_condition` = any
target misses its equivalence bound against the paper value.

| Clause | Metric | Paper value (parity target) | Probe |
| :-- | :-- | :-- | :-- |
| decoy ranking | Spearman(E, TMScore) | **0.838** (Rosetta 0.757) | scaled-divergence over noise `t` |
| stability | Spearman ΔΔG (ProteinGym/Tsuboyama) | **0.686** (SOTA) | null-injection: masked-sequence must degrade |
| sequence-use | masked-seq decoy ranking drops | must fall (Fig 2b) | information-ablation probe |
| conf. biasing | sign/rank of `ΔE` on LplA open/closed | positive corr. w/ activity | information-ablation |

`[confounds.reference_parity]` cites the §8.1 parity run; a masked-sequence **null-injection** probe
proves the eval can falsify (energy must use sequence, not just backbone realism — the paper's own
Fig 2b control).

### 8.3 Throughput / latency benchmark (`benchmark.bth.toml`) — per user, first-class

Benchmark the JAX path against (a) the **original PyTorch** ProteinEBM and (b) the paper's baselines,
across protein lengths (the paper reports a length-vs-runtime curve, Fig S7 — a direct parity target).

```toml
[benchmark]
baseline_ref        = "<pytorch-proteinebm-run-id>"   # original impl, same hardware
metric              = "energy_evals_per_sec"          # also: langevin_steps_per_sec, score_grad_ms
regression_threshold = 0.05
target              = "JAX >= PyTorch throughput on pi_so3 (SM120) at L in {64,128,256,512}"

[result_schema]
energy_evals_per_sec   = "float"
langevin_steps_per_sec = "float"
score_grad_ms          = "float"
protein_length         = "int"
device                 = "str"
impl                   = "str"   # "jax" | "pytorch"
```

Dimensions to sweep: protein length {64,128,256,512}; batch size; energy-only vs. conservative-score
(`jax.grad`, the 2nd-derivative cost) vs. aux-score; single-structure scoring vs. vmapped decoy
batch vs. Langevin scan. Compare accuracy-matched wall-clock for the full structure-prediction
budget (the paper's compute-matching methodology, App. H, is the reference protocol). Cluster runs
via `bth submit` on `pi_so3` — **the SM120 `XLA_FLAGS=--xla_gpu_shard_autotuning=false` workaround is
mandatory** (`~/.claude/rules/CLUSTER.md`) or throughput numbers are ~1000× wrong.

**Measurement-pipeline sanity (BATHOS rule):** before trusting any throughput number, confirm the
timer excludes JIT warmup (discard first call) and that both impls score identical inputs — a
synthetic fixed structure with a known energy, asserted equal (within tol) across impls, before the
sweep.

---

## 9. Open questions / decisions (for brainstorm + user review)

1. **Checkpoint strategy:** port ProteinEBM `.pt` weights (orbax key-remap; couples to legacy
   format, fastest to a parity gate) **vs.** retrain in JAX (needs the full data pipeline; cleaner).
   Recommendation: port weights first for the E3–E7 parity gates; schedule retraining behind E8.
2. **Which checkpoints:** the paper uses up to 6 checkpoints (base, expert-x, MD-finetuned variants)
   with noise-level handoff. Minimum viable parity = **ProteinEBM-x @ t=0.05** (covers decoy +
   stability + biasing, the strongest results). Langevin/structure-pred (E9/E10) need the base +
   MD-finetuned set — defer.
3. **All-atom aux head:** port it (matches loss weights, needed for exact training parity) or drop it
   for scoring-only parity? Recommendation: drop for E3–E7, add for E8 training parity.
4. **Residue-axis bucketing boundaries:** choose `bucket_boundaries` (compile-count vs. padding-waste
   tradeoff — the skill's `🔬 HiTL` stop). Proposal: `(128, 256, 384, 512)`.
5. **Scope of multistate design (E12):** scoring-only now; differentiable design is a separate epic.
6. **PBCNet2.0:** confirm it is deferred to a separate lower-priority epic (§6).

---

## References

- Roney, Ou, Ovchinnikov. *Protein Diffusion Models as Statistical Potentials.* bioRxiv
  10.64898/2025.12.09.693073v3 (2026). Code: github.com/jproney/ProteinEBM.
- Yu, Sheng, et al. *Atomic-level protein–ligand recognition with PBCNet2.0 for probe discovery.*
  Nat. Chem. Biol. 10.1038/s41589-026-02241-x (2026). Code: `~/repos/PBCNet2.0`.
- Simeon & De Fabritiis. *TensorNet.* NeurIPS 2023 (PBCNet2.0 backbone).
- aminx `docs/COMPOSITION_GUIDE.md`; `using-xtrax`, `using-bathos`, `using-optax`, `using-orbax`,
  `developing-composable-jax` skills.
- Cluster: `~/.claude/rules/CLUSTER.md` (SM120 XLA flag); measurement discipline:
  `~/.claude/rules/BATHOS.md`.

---

## 10. Adversarial review resolutions (2026-07-09)

A 3-agent panel (brainstorm + spec-challenger + spec-defender; audits in `.praxia/audits.jsonl`)
reviewed this spec. The **core thesis held** (energy is a new peer axis, not a logit-`StageSet`
bolt-on), but the following claims were **wrong or underspecified** and are corrected here. The
forward-looking resolutions (design forks, corrected DAG, risk register, bathos claim) live in
[`plans/260709_proteinebm-epic-backlog-dag.md`](../plans/260709_proteinebm-epic-backlog-dag.md),
which supersedes §7–§9 of this spec.

| ID | Finding | Correction |
| :-- | :-- | :-- |
| **BLOCKER-1** | §3.3 claimed the Langevin noise-level model-swap reuses `inference/schedule_selector.py` wrapped as an `AxisBoundary`. **False**: `schedule_selector.py` is chromatic decode-*order* (W0.3) tooling; `AxisBoundary` must flatten to zero JAX leaves so it **cannot** hold swappable 85M-param models. | The model-swap is **net-new** (`lax.cond` over pre-loaded models, mirroring the reference `get_dynamics_model(t)`). Residual: N×85M memory if all-resident; else host-side per-t-range segmentation. (§3.3 corrected.) |
| **BLOCKER-2** | §3.3 called the multistate difference-fuse a `DecodingFusionFn`. **False + self-contradictory** with §3.2: `DecodeOutput` is `{sequences, logits}` only; a scalar energy / `N×3` score is not a `DecodeOutput`. | The seam is a **new `EnergyFusionFn`** implementing the **generic** `FuseFn`/xtrax `Fuse[S,O]` protocol (`types/stages.py:64`), a peer of the logit-typed fusion — not the logit-typed class. Do **not** extend the JIT-boundary `DecodeOutput`. (§3.3 corrected.) |
| **MAJOR-3** | Weight-port filed as "optional", yet every published-number gate depends on it (near-circular: port validated by the Spearman gate). | New **independent** gate node **E3.5** (per-tensor allclose vs ref forward; `E=0` on zero-`r`; `−∇E`==analytic score) validates the port itself. + validation-only retrain fallback. (EPIC §1 Fork 3, §2.) |
| **MAJOR-4** | §3.3 modeled Langevin as a clean nested `Scan`. Reference has variable inner trip-count, Metropolis-Hastings, and host-side importance resampling. | Corrected in §3.3 + EPIC E9/E10 (pad-to-max+mask; `lax.cond`; host `Sink`). |
| **MAJOR-5** | §8.2 accuracy "claim" was metric rows, not hypotheses (no null/misspec; no clause map) → Union Gate inert (bathos Signal 12); `0.838`/`0.686` conflated benchmarks. | Real claim with ≥2 hypotheses incl. a null (masked-sequence must degrade) + clause mapping + **disaggregated pinned targets** — EPIC §4.2. |
| **MAJOR-6** | §3.1 "no new math… scalar head plus autograd" understates the **second-order** training grad (∂²E/∂x∂θ); §4/§8.1 invariants test first-order only; self-conditioning omitted stop-gradient. | `jax.checkpoint` + a **new 2nd-order finite-diff invariant** gating E8; `jax_debug_nans` once; self-cond recycled estimate is **stop-gradient** (2× forward cost noted). EPIC §1 Fork 2. |
| **MAJOR-7** | §7.1 DAG edge `E9→E8` wrong; `E5/E6/E7 [P]` share the single `_sample_batch`. | EPIC §2: E9 depends on the **checkpoint gate E3.5**, not E8; E4 pre-stubs **3 separate dispatch fns**; E11 split per-application. |
| **MINOR** | `resolve_kernel_fn` (§3.2, from `COMPOSITION_GUIDE.md:283`) does not exist; `driver.decode` is deprecated (`infer_topology` inert — real dispatch is `host/plan.py` mode resolution + `kernel_dispatch.py`). | Wiring lands in `host/plan.py` (`EnergyScoreDecode` mode class) + `kernel_dispatch.py`, not `driver.decode`. (EPIC §1 Fork 1.) |
| **MINOR** | §3.4/E2 attributed `SinusoidalEmbedding`/`SwiGLU` to `training/diffusion.py`. | They live in **`model/diffusion_mpnn.py`**; `training/diffusion.py::NoiseSchedule` is the (distinct) sequence-space cosine schedule. |
| **MINOR** | §3.1 omitted non-equivariance/augmentation. | Energy is not SO(3)-invariant by construction; deterministic decoy/`ΔE` ranking needs `rotate=False`; any augmentation rotation inside the differentiated energy must be **constant w.r.t. the grad**. |
| **MINOR** | §8.3 lengths `{64,128,256,512}` vs §9.4 buckets `(128,256,384,512)` — L=64 → 4× pad waste; benchmark not apples-to-apples. | Buckets `(64,128,256,512)` aligned to the sweep (EPIC Fork 6); PyTorch baseline **drops `create_graph`** at inference; SM120 XLA-detune noted; exclude JIT warmup. |
| **MINOR** | `coordinate_scaling` applied in the diffuser **and** `ebm.forward` **and** under `precondition` — a ~10× score-error footgun. | §8.1 invariant must pin single-application explicitly. |
| **MINOR** | §6 PBCNet2.0 leaned on a topology-reuse argument that shares BLOCKER-2's flaw (scalar affinity ≠ `DecodeOutput`). | Deferral verdict stands, but the rationale now **leads with dependency/domain** (DGL 1.0.2/py3.8 pickled model, RDKit/pose, 8.6M-pair pipeline), not composition-topology reuse. |
| **Stale-skill** | The `using-xtrax` "⚠ GAP: `make_inference_plan` validator does not exist" note is **false** on xtrax 0.4.0a5 — `validate_plan_topology` is wired at `host/plan.py:440-446`. | The ordered-`Sink`+`Vmap` trajectory conflict (§3.3) is caught at plan-construction time for free; do not re-file as an open risk. |
