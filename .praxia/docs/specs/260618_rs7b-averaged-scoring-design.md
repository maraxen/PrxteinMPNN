---
task_id: 260618_autonomous-loop
backlog: RS-7b (child of RS-7 #1626)
parent_spec: 260611_runspec-unification.md  (AC-RS-7)
date: 260618
status: design-complete
author: staff (design pass, autonomous loop iteration 2)
---

# RS-7b Design — Averaged-Topology Scoring (AC-RS-7 primary clause)

Route `score()` through the InferencePlan averaging topology when
`average_node_features=True`, sharing the sampling encode→fuse→decode path.
RS-7a (temperature-N/A doc+test, HDF5 deferral test) is independent — land it first.

## 1. Feasibility — HIGH

`score_sequence` (scoring/score.py:56-125) is a monolith at the public boundary, but the
encode/decode seam already exists one level down in `inference/score_conditional.py:32-44`:
`encode_fn(bundle,k,config) -> EncoderOutput`, then `ConditionalDecode(...)` consumes only
`enc.{node_features,edge_features,neighbor_indices,mask}`. That is the SAME `EncoderOutput`
type `ArithmeticMeanEncodingFusion` (host/averaging.py:48-57) consumes and produces. Fusion
is type-preserving → a pre-fused encoding can be injected with NO decoder change. Structurally
identical to sampling unified path B (host/kernel_dispatch.py:309-328).

## 2. Recommended approach — refactor kernel to expose encode→(fuse)→score seam

Rejected: inlining the D-encode/fuse loop in the runner (duplicates plan/encoding_fusion,
drifts from the sampling topology AC-RS-7 says to share).

Seams / signatures:
- inference/score_conditional.py (SHARED KERNEL — serialize edits):
  - NEW `encode(model,key,bundle,config) -> EncoderOutput` (extract from kernel)
  - NEW `score_from_encoding(model,key,enc,bundle,config,stage_set) -> Logits` (decode half)
  - NEW `score_averaged(model,key,bundles_per_noise,config,stage_set,encoding_fusion)`:
    encode D×, `_stack_encoder_outputs` (leading-D axis like kernel_dispatch:321-325),
    `fused = encoding_fusion(stacked)` (D→1), `score_from_encoding(... fused, bundles[0] ...)`.
  - `kernel(...)` now composes encode+score_from_encoding — BYTE-IDENTICAL behaviour for
    existing non-averaged callers.
- scoring/score.py: extract `_nll_from_logits(logits,seq_oh,mask)` from score_sequence:116-125;
  both paths call it (single source of NLL formula).
- host/runner.py:316-322: branch — if `average_node_features`: `plan=make_inference_plan(model,spec)`
  (wires ArithmeticMeanEncodingFusion, plan.py:652-662) + `_make_averaged_score_fn(plan,spec)`
  (NEW closure builds D bundles from spec.backbone_noise [inherited, specs.py:197, default (0.0,)],
  calls score_averaged, NLL via _nll_from_logits). Else: `make_score_fn(model)` unchanged.
  Per-(structure,sequence) loop (runner.py:355-407) unchanged.

## 3. Parity / measurement oracle (BATHOS — verify before trusting)

AC-RS-7 means INTERPRETATION (A): mean node/edge features THEN score (one decode, one NLL).
NOT (B) mean of D independent scalar NLLs (differs: log_softmax is nonlinear). Justification:
sampling topology averages features before decode (averaging.py:52-53); `average_node_features`
names features; legacy averaged features not scores. Tests must assert (A) and guard vs (B).

Invariants:
1. D=1 degenerate identity (HARD GATE, bit-equal): backbone_noise=(0.0,) → fusion mean over
   length-1 axis is identity → averaged-path logits AND nll MUST equal current non-averaged
   score() on same (structure,sequence,key). `jnp.array_equal` (relax to atol=0,rtol<=1e-7 only
   if XLA layout forces it — else treat as real bug).
2. A vs B (D=2 distinct noise): nll_A == hand-computed (encode both, mean features, decode, NLL);
   nll_A != nll_B (mean of two independent scores) on a non-degenerate fixture.
3. Fusion math sanity (synthetic 30s): stacked node_features=stack([ones*1, ones*3]) →
   output==2.0; neighbor_indices/mask == stack[0].
4. Golden: small real structure + seq, D=3, pin scalar NLL.

## 4. Task DAG (RS-7b)

| ID | Size | Title | Files | Deps | Worktree |
|----|------|-------|-------|------|----------|
| RS7b-1 | quick | Split kernel → encode + score_from_encoding (no behaviour change) | score_conditional.py | — | SHARED KERNEL: serialize |
| RS7b-1b | quick | Extract _nll_from_logits from score_sequence:116-125 | scoring/score.py | — | pairs w/ RS7b-1 |
| RS7b-2 | standard | Add score_averaged + _stack_encoder_outputs | score_conditional.py | RS7b-1 | serialize after 1 |
| RS7b-3 | standard | Runner branch: make_inference_plan + _make_averaged_score_fn | host/runner.py | RS7b-2, RS7b-1b | safe |
| RS7b-4 | standard | Parity oracle suite (Invariants 1-3) | tests/scoring/test_averaged_parity.py (new) | RS7b-3 | safe |
| RS7b-5 | quick | Invariant-4 golden + docstring (close AC-RS-7) | test + specs.py:406 docstring | RS7b-4 | safe |

Critical path: RS7b-1 → -2 edit the shared kernel (not worktree-parallel); 1b parallel with 1;
3→5 on distinct files.

## 5. Risks + off-ramp

- R2 (medium): static_argnames drift → recompile-per-D. Build D bundles in Python (host),
  concrete arrays; do not make D a static jit arg.
- R3 (medium): reusing bundles[0] for decode conditioning assumes conditioning is noise-invariant
  (only backbone_noise varies) — assert D bundles' conditioning fields identical.
- R1/R4 (low): extra JIT compile (opt-in, acceptable); D=1 bit-equal may need atol=0,rtol<=1e-7.
- OFF-RAMP if parity unreachable: land RS7b-1/-1b extractions only; in score() raise
  NotImplementedError("averaged-feature scoring not yet supported; omit average_node_features")
  (mirrors HDF5 guard runner.py:312-314) + docstring. Satisfies AC "explicitly documented" as a
  degraded close (vs current silent ignore = the bug).

## 6. RS-7a independence — CONFIRMED

- Temperature N/A: score_sequence accepts multi_state_temperature, never reads it (score.py:71).
  RS-7a = docstring note + test asserting score() NLL identical across temperatures. No seam dep.
- HDF5/combine: already NotImplementedError (runner.py:312-314). RS-7a = test asserting it.
Land RS-7a first (worktree-safe, 2 tests + docstring); sequence its score.py docstring edit
before RS7b-1b to avoid a trivial merge.
