---
title: Potts parallel model path, not StageSet consumer
decision_id: 260605_potts-parallel-not-stageset
date: 260605
status: Accepted 2026-06-05
decision_type: architectural
relates_to: 260605_integration-architecture-for-mistypotts
---

# Status: Accepted 2026-06-05

# Context: integration of PottsTRWStructureModel into aminx.potts.PottsModel

PottsTRWStructureModel (from mistypotts) is being integrated into aminx to enable Potts-based energy scoring and MPNN-guided design workflows. Three structural approaches were evaluated:

- **Approach A:** PottsTRWStructureModel becomes `aminx.potts.PottsModel`, a parallel model with its own spec path
- **Approach B:** TRW becomes a composable decoder head (replaces MPNN inference stage)
- **Approach C:** Hybrid — standalone first (Approach A), refactor to composable head later (Approach B)

Additionally, a fourth shape emerged:

- **Approach D:** MpnnPottsDesigner Coordinator wrapper (thin orchestrator holding Aminx + PottsModel instances for MPNN-guided design)

# Decision: parallel model family, NOT StageSet consumer

**Accepted:** Approach A + Approach D

PottsModel is a **parallel, independent model** — not a variant of Aminx's StageSet pipeline. It lives in `aminx.potts.model` with its own `PottsRunSpec`, its own `spec emit-*` CLI surface, and its own run loop. Weight recapture produces two sidecars: `potts_<id>.eqx.zst` (model weights) and `caliby_<id>.eqx.zst` (correction factors). Sampling is expressed as pure JAX functions in `aminx.potts.sampling`, consuming only a `log_prob(seq, backbone_idx) → float` interface. Static `n_backbones` field in `PottsRunSpec` enables clean vmap/scan over the PoE backbone dimension. MpnnPottsDesigner (Approach D) provides a thin Coordinator wrapper for MPNN-guided Potts design workflows, holding both model instances without merging weight pytrees.

# Rejected Alternatives

## Option II: decode_step adapter

**Idea:** Extract PottsTRWStructureModel as a pluggable `decode_step` function compatible with Aminx's stageset inference loop. The function receives `(node_f, edge_f, nei, mask) → logits(L, V)`.

**Problems:**
- **Currency mismatch:** Potts requires raw coordinates → `(marginals, h, J, rho)` (pairwise energy tensor). decode_step outputs logits currency `(L, V)` (per-position vocabularies). These currencies are incompatible; no lossless conversion exists.
- **Phase mismatch:** TRW is one-shot global inference (all nodes simultaneous, marginals computed from full energy). Aminx's decoder is autoregressive per-position. Forcing Potts into the autoregressive frame requires either (a) running full TRW at each position (prohibitively expensive), or (b) truncating TRW (approximate, breaks contract).
- **StageSet inflexibility:** Aminx's stageset is designed for position-conditional scoring. Potts energy is pair-wise and global; wrapping it as a position-wise step creates artificial boundaries and breaks the pairwise structure.

## Option III: EncoderOutput widening

**Idea:** Extend Aminx's EncoderOutput bundle to carry Potts-specific fields: `node_h`, `edge_J`, `rho_scale`. At inference time, the decoder checks an `inference_mode` flag and dispatches to either MPNN or TRW.

**Problems:**
- **Blast radius — load-bearing definition:** EncoderOutput (defined at aminx/types/bundles.py:359) is a fundamental contract throughout the inference stack. Every call site that constructs, reads, or serialises EncoderOutput must be updated to handle optional Potts fields. The change cascades through: `infer.tiling`, `host.plan`, `types.stages`, `inference.logits`, and all downstream consumers. This violates the principle of structural separation — Potts-specific fields leak into the MPNN-only bundle contract, forcing all MPNN paths to carry Potts-aware logic even when unused.
- **Backward compatibility risk:** Existing checkpoints (MPNN weights) serialized without Potts fields become opaque when loaded with the widened bundle. Deserialisation requires a migration path or graceful nulling. Any mistake here breaks weight recapture or inference reproducibility.
- **Semantic muddiness:** The bundle becomes neither pure-MPNN nor pure-Potts; it is a union type. Downstream code must perform mode dispatch at every site where EncoderOutput is used, multiplying the test surface and creating fragile conditional branches.

## Option IV: ProductOfProbabilities (PoE) at logits level

**Idea:** Express multi-backbone Potts PoE (mixture of energies from independent Potts models) as a ConditioningBundle extension. Represent PoE as a stack of logits tensors `(S, L, V)` (S backbones, L positions, V vocab), sum them, and apply the result as a factorization inside the tiling iterator.

**Problems:**
- **Currency incompatibility:** PoE for Potts combines pairwise energies from multiple backbones into a joint energy landscape. This operation is defined in the space of *pair interaction tensors* — the J tensor is `(N, N, 20, 20)` (node indices × node indices × amino acid indices × amino acid indices). The logits currency is `(L, V)` — per-position marginals. These two currencies are orthogonal. Potts J tensor is `(N, N, 20, 20)` for the standard amino acid alphabet (V=20), so the logits currency `(L, 20)` has no room for pairwise terms and no mechanism to recover them from positional logits alone.
- **Information loss:** Reducing pairwise interactions to per-position logits loses the interaction structure. You cannot reconstruct interactions from marginals (in general, this is NP-hard). Attempting to express PoE at logits level would require either dropping the pairwise structure entirely or implementing an approximation that breaks the Potts energy contract.

# Enforcement

**Hard boundaries:**
- `aminx.potts.{model, poe, sampling}` are **forbidden from importing** `aminx.inference.decode`, `aminx.inference.host.plan`, `aminx.types.stages`, `aminx.inference.logits`.
- Only `aminx.potts.designer` is exempt (Coordinator wrapper holds Aminx instance and must wire logits to Gibbs initialization).

**Verification:**
- Backlog item #1304 enforces import boundary via static AST checks in CI.
- Both `model.py` and `designer.py` must cite this ADR in module docstrings.

# Rationale

1. **Clean separation:** Potts is structurally disjoint from MPNN. A parallel model path avoids Frankenstein pytrees and stageset mode dispatch.
2. **No backward-compat risk:** MPNN users see zero changes. Potts is additive; Aminx core remains unchanged.
3. **Shipping velocity:** Parallel approach is fastest to implement. Option A → D iteration is incremental; no rework of weight recapture or checkpoint structure.
4. **Testability:** Each model family (MPNN, Potts) has isolated test surfaces. No cross-coupling in inference logic.
5. **MPNN-guided design enabled by Coordinator:** Approach D (MpnnPottsDesigner) provides first-class orchestration for the logit-seeding workflow without forcing MPNN+Potts into a single eqx.zst.

---

**Decision made:** 2026-06-05
**Reviewed by:** code-reviewer subagent
**Snapshot:** brainstorm spec at `.praxia/docs/specs/260605_integration-architecture-for-mistypotts.md`
