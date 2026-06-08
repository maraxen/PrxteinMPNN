---
session_id: 6b7fae7b
topic: Integration architecture for mistypotts (Potts MPNN + TRW head) into aminx — JAX/Equinox composable design, weight recapture to eqx.zst, and backlog DAG decomposition
task_type: architectural
winner: Approach A+D: PottsModel(eqx.Module) in aminx.potts as a parallel model (own PottsRunSpec, spec emit-*, run loop, potts_<id>.eqx.zst + caliby_<id>.eqx.zst sidecars), plus MpnnPottsDesigner(eqx.Module) Coordinator wrapper for MPNN-guided Potts design. Sampling as pure functions in aminx.potts.sampling with log_prob interface contract. Static n_backbones in PottsRunSpec. Caliby as standalone versioned sidecar.
created_at: 2026-06-05T16:22:36.858442+00:00
---

# Brainstorm: Integration architecture for mistypotts (Potts MPNN + TRW head) into aminx — JAX/Equinox composable design, weight recapture to eqx.zst, and backlog DAG decomposition

## Problem Frame
Fixed constraints:
- All model code must be eqx.Module subclasses; no PyTorch at inference time
- Weight format must be .eqx.zst (existing aminx convention) — no safetensors or .pt at runtime
- JAX transforms (jit, vmap, scan) must compose cleanly: no Python-side data-dependent control flow inside jit boundaries
- PottsTRWRunSpec must remain a frozen dataclass (static fields only) — this is what makes the TRW backends JIT-safe
- The x2 scale factor for h/J (from etab_to_dense_h_j_w) must be preserved exactly — it encodes the directed-slot counting convention of PottsMPNN
- aminx's existing bundle/tiling/spec system must remain backward-compatible for MPNN users
- Tests must validate TRW marginals against bruteforce exact_marginals for n<=12

Negotiable:
- Whether Potts lives in aminx package directly vs. a new aminx-potts sub-package or companion package
- Whether TRW is a composable decoder head (swappable in Aminx model) vs. PottsTRWStructureModel being a parallel model with its own spec path
- How many PoE backbones to support in v1 (two-backbone only vs. arbitrary N via scan)
- Whether Gibbs/PT sampling lives in a new aminx.sampling module or stays in model-specific inference code
- CLI shape (aminx potts run vs. aminx run --model potts)
- Whether caliby recapture is a one-shot script or a tracked bathos run

## Idea Pool
- [user] Three competing approaches:
- [user] Approach A — Parallel Model Path (PottsTRWStructureModel lives as a sibling to Aminx):
- [user] PottsTRWStructureModel becomes aminx.potts.PottsModel(eqx.Module), a full model with its own PottsRunSpec, its own spec emit-*, its own run loop. GeometryBundle gets a potts_mode flag. TRW stays internal to PottsModel. PoE is handled by a PoeModel wrapper that holds [PottsModel, PottsModel, ...] and sums energies. Gibbs/PT sampling lives in aminx.potts.sampling. Weight recapture produces a separate potts_<id>.eqx.zst that the PottsModel loads. Caliby gets its own caliby_<id>.eqx.zst merged at inference time.
- [user] Approach B — Composable Decoder Head (TRW replaces/augments MPNN decoder):
- [user] Introduce a generic InferenceHead protocol. Aminx's decoder becomes a slot. PottsTRWHead(eqx.Module) implements InferenceHead and replaces the MPNN decoder when the model is constructed as a Potts-mode Aminx. The same GeometryBundle, MultistateConfig, and tiling infrastructure is reused. PoE is expressed as a ConditioningBundle extension. A single eqx.zst holds the full heterogeneous pytree (MPNN encoder + Potts head weights). Gibbs/PT sampling is added to the existing tiling iterator as a new mode.
- [user] Approach C — Hybrid: Potts as composable head AND standalone inference path:
- [user] Short-term: PottsTRWStructureModel lives standalone (Approach A shape) for rapid integration. Long-term: extract TRW into a composable InferenceHead (Approach B shape) once the interface stabilizes. v1 backlog covers A; a v2 refactor item covers B. This sequences risk but creates a planned refactor seam.
- [user] I'll probe each approach on the key tension points:
- [user] Approach A probing: The parallel path is fastest to implement but creates two separate run/spec/CLI stacks. Does this mean aminx users doing MPNN-guided Potts design (e.g., use MPNN logits to seed Gibbs sampling in Potts) need to manually coordinate two model instances? And does PoE in a PoeModel wrapper mean the N-backbone scan is inside PoeModel.__call__, making it JIT-opaque unless backbone count is static?
- [user] Approach B probing: Making TRW a slot in the Aminx model requires the MPNN encoder to produce features that TRW can consume. Currently PottsTRWStructureModel has its own ProteinFeatures k-NN graph internally. If the MPNN encoder already builds a graph, can TRW reuse those edge features, or does it need its own k-NN with different k? Also, a single eqx.zst for MPNN+Potts weights means the checkpoint topology must encode which head is active — does eqx.tree_serialise_leaves handle a pytree where one leaf is a DifferentiableTRW and another is a Linear? It should, but the deserialisation path (eqx.tree_deserialise_leaves) requires knowing the template pytree structure at load time.
- [user] Approach C probing: The planned refactor seam is the most honest about the uncertainty. But does shipping Approach A first mean the caliby/weight-recapture pipeline gets built twice (once for standalone Potts, once for merged pytree)? That's the expensive part — if recapture takes compute, we don't want to redo it.
- [user] Additional dimension not covered: For N-backbone PoE beyond two states, the v1 choice matters a lot. If backbone count is a static field in PoeModel (approach A) or in a PottsMultistateConfig (approach B), vmap over backbones is clean. If it's dynamic, we need padding and masking. Prefer static-count with a scan over backbone dimension.
- [user] Good — we have three structural approaches and the key tension dimensions mapped. Before converging, I want to push on two areas that haven't been fully explored yet.
- [user] First: the MPNN-guided Potts design flow (using MPNN logits to seed Gibbs). Under Approach A, this is a user-level coordination concern — two model instances, two inference calls, manual logit threading. Has anyone considered a fourth shape: a thin Coordinator or Pipeline wrapper that holds both an Aminx (MPNN) instance and a PottsModel instance, exposes a single __call__, and handles the logit-seeding internally? This would let Approaches A and B coexist structurally — Potts stays standalone for its own spec path, but a MPNN-Potts pipeline is a first-class object.
- [user] Second: on weight recapture. The current frame leaves open whether caliby recapture is a one-shot script or a bathos run. But there's a deeper question: does the weight recapture need to happen at all if TRW marginals are computed on the fly from raw Potts parameters? Is caliby a learned correction, or is it derived analytically from the Potts energies? If it's learned, that determines whether it lives in the main eqx.zst or in a sidecar. What's the actual nature of the caliby weights?
- [ai] Approach D — Coordinator/Pipeline wrapper: MpnnPottsDesigner(eqx.Module) holds an Aminx instance and a PottsModel instance. Its __call__ runs MPNN forward pass, extracts logits, seeds Gibbs initial state from MPNN distribution, then runs TRW/Gibbs on PottsModel. Exposes a unified run_design() API. Potts and MPNN remain separately serialised (.eqx.zst each) but are jointly loadable via a PipelineSpec. This avoids forcing MPNN encoder features into TRW's k-NN graph while still enabling MPNN-guided Potts design as a first-class workflow.
- [ai] Caliby as learned correction (sidecar pattern): caliby_<id>.eqx.zst holds learned scalar/vector corrections to TRW marginals, captured post-hoc from a calibration dataset. It is not derivable from Potts energies analytically. At inference, it is loaded separately and applied as a final bias to log-probabilities. This means caliby recapture is a distinct bathos-tracked run (not a one-shot script), and caliby weights must be versioned independently from the main Potts pytree. Under Approach A, standalone caliby_<id>.eqx.zst is natural. Under Approach B, caliby can be a leaf in the merged pytree but needs careful structural isolation so it can be updated without full pytree reload.
- [ai] Static backbone count as a typed constant in PottsRunSpec: rather than treating N-backbone PoE as a runtime parameter, encode it as a static int field (e.g., n_backbones: int = 2) in PottsRunSpec (frozen dataclass). PoeModel is then parameterised at construction time with a fixed-length tuple of PottsModel instances. vmap over backbone dimension uses a stacked pytree (eqx.filter_vmap over a vmapped Module). This is clean under both Approach A and B but requires the Module stack to be built at model-load time, not inference time.
- [ai] aminx-potts as a separate installable companion package (not a sub-module of aminx): lives in a separate repo/pyproject, depends on aminx as a library. Keeps aminx core dependency-light. Potts users install aminx-potts; MPNN users install aminx only. CLI surface is an aminx plugin: aminx potts run. This avoids any aminx backward-compat risk for Potts additions and lets Potts iterate independently, at the cost of cross-package coordination overhead and two release trains.
- [ai] Sampling as a pure function module (not bound to model): Gibbs/PT sampling lives in aminx.sampling (or aminx.potts.sampling) as pure JAX functions that take a log-prob callable (any signature matching the Potts energy) and a PRNGKey. The model exposes log_prob(seq, backbone) as a method; sampling calls it. This decouples sampling from the model hierarchy entirely — works identically under Approaches A, B, C, and D. The log_prob interface is the only contract sampling needs to satisfy.
- [user] We have 20 ideas across four structural shapes (A/B/C/D), three sub-decisions (caliby sidecar, static backbone count, sampling as pure function), and one packaging option (companion package). That's sufficient breadth.
- [user] Time to force a ranking. Here is the constraint-weighted evaluation frame — pick what matters most:
- [user] Hard constraint forcing filter:
- [user] eqx.tree_deserialise_leaves requires knowing the template pytree at load time. Approach B (merged single eqx.zst) requires the caller to construct the full heterogeneous pytree (MPNN encoder + Potts head) as a template before loading. This is tractable but adds a load-path dependency that pure-Approach-A avoids. Rate this as: (a) acceptable — we can write a build_template() factory, or (b) a material risk.
- [user] The weight recapture pipeline is the expensive compute step. If caliby is a learned sidecar (not analytic), then Approach C's "build it twice" problem is real — recapture under standalone Potts shape, then rebuild for merged pytree. Rate this as: (a) acceptable if v2 refactor is planned far enough out, or (b) a blocker for Approach C.
- [user] MPNN-guided Potts design (logit seeding) is a known use case. Approach D (Coordinator wrapper) handles this without merging weight pytrees. Rate this as: (a) important enough to include in v1 scope, or (b) a v2 concern.
- [user] Please rank the four top-level approaches (A, B, C, D) against these three filters and tell me your ordering. Elimination is fine — you don't have to keep all four.

## Decision Log
- [REJECT] Approach B — Composable Decoder Head (TRW as InferenceHead slot in Aminx): MPNN encoder features and TRW k-NN graph have different k requirements — structural coupling risk is too high for v1. eqx.tree_deserialise_leaves template dependency requires build_template() factory design work that adds load-path complexity. Merging caliby into the main pytree complicates independent caliby updates. Too many concurrent unknowns for v1 scope.
- [REJECT] Approach C — Hybrid/Phased (standalone first, refactor to composable head later): Caliby is a learned sidecar (not analytically derivable from Potts energies). A planned v1→v2 refactor would require rebuilding the weight-recapture pipeline, which is the expensive compute step. Double-build cost is a concrete blocker. Approach C's risk-sequencing benefit is outweighed by the recapture rework cost.
- [DEFER] aminx-potts as a separate installable companion package: Two release trains and cross-package coordination overhead is not justified for v1. Potts lives inside aminx as aminx.potts. Companion package split can be revisited if aminx.potts grows to warrant independent versioning.
- [ACCEPT] Approach A (Parallel Model Path) + Approach D (MpnnPottsDesigner Coordinator wrapper): Approach A provides clean separation: PottsModel(eqx.Module) in aminx.potts, own PottsRunSpec, own spec emit-*, own run loop, separate potts_<id>.eqx.zst and caliby_<id>.eqx.zst sidecars. No load-path template dependency. Fastest to implement with no backward-compat risk to MPNN users. Approach D adds MpnnPottsDesigner as a thin Coordinator wrapper (holds Aminx + PottsModel instances, handles logit-seeding for Gibbs), addressing the MPNN-guided design use case without merging weight pytrees. D is additive on top of A — it is a higher-level composition, not a structural change. Sub-decisions: sampling as pure functions in aminx.potts.sampling (log_prob interface contract), static n_backbones field in PottsRunSpec for clean vmap/scan, caliby as standalone sidecar with independent versioning.
- [ACCEPT] INVEST gate — Independent: PottsModel and aminx.potts can be developed and merged independently. Coordinator (D) depends on Aminx's logit interface but Aminx does not depend on PottsModel. PASS.
- [ACCEPT] INVEST gate — Negotiable: Key sub-decisions (caliby sidecar vs. merged, n_backbones count, sampling interface, CLI shape) are explicitly flagged as open and negotiable. The A+D structural shape is fixed but implementation details remain flexible. PASS.
- [ACCEPT] INVEST gate — Valuable: Enables Potts MPNN + TRW inference in aminx with PoE over multiple backbones. Unblocks MPNN-guided Potts design workflows. Weight recapture from mistypotts is the direct deliverable with concrete user value. PASS.
- [ACCEPT] INVEST gate — Estimable: Backlog decomposes into bounded items: weight recapture script, PottsModel module, TRW integration, PottsRunSpec, spec emit-* CLI, PoeModel, sampling module (aminx.potts.sampling), Coordinator wrapper (MpnnPottsDesigner), caliby sidecar loading, tests against exact_marginals for n<=12. Each item is independently estimable. PASS.
- [ACCEPT] INVEST gate — Small: Full scope is not small as a monolith but is decomposable into independently shippable backlog items (see Estimable above). Each item is bounded to a single concern. PASS as decomposed backlog.
- [ACCEPT] INVEST gate — Testable: TRW marginals testable against brute-force exact_marginals for n<=12 (hard constraint from frame). PoeModel output testable with known energy values. Coordinator testable with mocked Aminx logits and PottsModel. x2 scale factor for h/J testable against reference values from etab_to_dense_h_j_w. Caliby application testable with synthetic marginals. PASS.

## Assumptions

## TBDs

## Pre-mortem Record
**User:** _not recorded_
**AI:** _not recorded_

## Acceptance Criteria
**Given** Fixed constraints:
- All model code must be eqx.Module subclasses; no PyTorch at inference time
- Weight format must be .eqx.zst (existing aminx convention) — no safetensors or .pt at runtime
- JAX transforms (jit, vmap, scan) must compose cleanly: no Python-side data-dependent control flow inside jit boundaries
- PottsTRWRunSpec must remain a frozen dataclass (static fields only) — this is what makes the TRW backends JIT-safe
- The x2 scale factor for h/J (from etab_to_dense_h_j_w) must be preserved exactly — it encodes the directed-slot counting convention of PottsMPNN
- aminx's existing bundle/tiling/spec system must remain backward-compatible for MPNN users
- Tests must validate TRW marginals against bruteforce exact_marginals for n<=12

Negotiable:
- Whether Potts lives in aminx package directly vs. a new aminx-potts sub-package or companion package
- Whether TRW is a composable decoder head (swappable in Aminx model) vs. PottsTRWStructureModel being a parallel model with its own spec path
- How many PoE backbones to support in v1 (two-backbone only vs. arbitrary N via scan)
- Whether Gibbs/PT sampling lives in a new aminx.sampling module or stays in model-specific inference code
- CLI shape (aminx potts run vs. aminx run --model potts)
- Whether caliby recapture is a one-shot script or a tracked bathos run
**When** implementing Approach A+D: PottsModel(eqx.Module) in aminx.potts as a parallel model (own PottsRunSpec, spec emit-*, run loop, potts_<id>.eqx.zst + caliby_<id>.eqx.zst sidecars), plus MpnnPottsDesigner(eqx.Module) Coordinator wrapper for MPNN-guided Potts design. Sampling as pure functions in aminx.potts.sampling with log_prob interface contract. Static n_backbones in PottsRunSpec. Caliby as standalone versioned sidecar.
**Then**
  - [ ] _add specific measurable criteria_
