// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/260616_rs6-fuse-refactor.toml
// Regenerate: praxia dw emit-sprint 260616_rs6-fuse-refactor.toml
// task_id: 260616_rs6-fuse-refactor   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (A,B,C,D) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.

export const meta = {
  name: "260616_rs6-fuse-refactor",
  description: "R6-5/6/7: remove dead AggregationFn/AveragingMode/DecodeFn; add Fuse-shaped EncodingFusionFn + DecodingFusionFn; wire both through RunSpecification and kernel_dispatch",
  phases: [
    { title: "Track A — R6-5: Remove dead protocols; add DecodeOutput + DecodingFusionFn (#2003)" },
    { title: "Track B — R6-6: Add encoding_fusion + decoding_fusion to RunSpecification; wire encoding_fusion (#2004)" },
    { title: "Track C — R6-7: Wire decoding_fusion in kernel_dispatch.py (#2005)" },
    { title: "Track D — Auditor closeout" },
  ],
};

const TASK_ID = "260616_rs6-fuse-refactor";
const MAX_FIX_RETRIES = 1;

const VERDICT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["item_id", "verdict", "summary"],
  properties: {
    item_id: { type: "string" },
    verdict: { type: "string", enum: ["PASS", "NEEDS_WORK", "FAIL"] },
    summary: { type: "string" },
    issues: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["where", "problem", "fix"],
        properties: {
          where: { type: "string" },
          problem: { type: "string" },
          fix: { type: "string" },
        },
      },
    },
  },
};

// Shared context for the writing tracks (from recon, task 260616_rs6-fuse-refactor).
const EMITTER_CTX = `Research gate findings (from sprint 260620 post-mortem and encoding-aggregation research):\n\n1. AggregationFn, AveragingMode, DecodeFn in specs.py are dead — wrong binary signatures,\n   never called from dispatch. The real call site is kernel_dispatch.py:328:\n     fused_enc = plan.stage_set.encoding_fusion(stacked_enc)\n   which takes a single stacked EncoderOutput (unary — Fuse-shaped).\n\n2. The correct protocols already exist:\n   - EncodingFusionFn in aminx/types/stages.py:114 — Fuse[EncoderOutput, EncoderOutput]\n   - StageSet.encoding_fusion: EncodingFusionFn | None at stages.py:346 — already wired in dispatch\n   - ArithmeticMeanEncodingFusion in averaging.py — reference implementation\n\n3. build_run_spec() (spec.py:311-317) has dead AveragingConfig.average_encoding_mode path;\n   spec.encoding_fusion is never read or wired into StageSet.\n\n4. Decode fusion is a NEW step: after _safe_map(_call_decode_one_enc, fused_enc) at\n   kernel_dispatch.py:347 and :514, K decode outputs go directly to the transposed sink.\n   Inserting a fusion step here enables logit ensembling, best-of-K, etc.\n\n5. DecodeOutput bundle does not exist yet. DecodingFusionFn does not exist yet.\n\nScope: src/aminx/ only. xtrax unchanged.\n`;

// ---- per-track stage helpers ---------------------------------------------
const fixer = (prompt, label, phaseName, isolation = null, context = null) => {
  const opts = { agentType: "fixer", label, phase: phaseName };
  if (isolation) opts.isolation = isolation;
  if (context) Object.assign(opts, context);
  return agent(`${prompt}\n\nWhen done, end your message with 'verdict: done' on its own line.`, opts);
};

const reviewer = (itemId, prompt, label, phaseName, isolation = null, context = null) => {
  const opts = { agentType: "reviewer", label, phase: phaseName, schema: VERDICT_SCHEMA };
  if (isolation) opts.isolation = isolation;
  if (context) Object.assign(opts, context);
  return agent(prompt, opts);
};

// Sequential implement->review with bounded NEEDS_WORK repair cycles.
async function track(itemId, phaseName, fixerPrompt, reviewerPrompt, isolation = null, context = null) {
  log(`[${itemId}] implement`);
  await fixer(fixerPrompt, `fix:${itemId}`, phaseName, isolation, context);
  let verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}`, phaseName, isolation, context);
  for (let retry = 0; retry < MAX_FIX_RETRIES && verdict && verdict.verdict === "NEEDS_WORK"; retry++) {
    log(`[${itemId}] NEEDS_WORK — repair cycle ${retry + 1}/${MAX_FIX_RETRIES}`);
    const issues = (verdict.issues || [])
      .map((i) => `- ${i.where}: ${i.problem} -> ${i.fix}`)
      .join("\n");
    await fixer(
      `${fixerPrompt}\n\nA reviewer found issues — fix exactly these, nothing else:\n${issues}`,
      `fix:${itemId}:repair:${retry}`,
      phaseName,
      isolation,
      context
    );
    verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}:re:${retry}`, phaseName, isolation, context);
  }
  return verdict;
}

// ===== TRACK A — Track A — R6-5: Remove dead protocols; add DecodeOutput + DecodingFusionFn (#2003) =========================
const trackA = () =>
  track(
    "2003",
    "Track A — R6-5: Remove dead protocols; add DecodeOutput + DecodingFusionFn (#2003)",
    `task_id: ${TASK_ID}. Fix #2003: Remove dead code from specs.py and add new Fuse-shaped types.\n\nWORKING DIRECTORY: /home/marielle/projects/aminx\n\n--- STEP 1: Remove dead code from src/aminx/run/specs.py ---\n\nRemove these classes entirely:\n  - class AggregationFn(Protocol) — wrong binary signature, never called\n  - class AveragingMode(Enum) — produces lambdas with wrong signature, never called\n  - class DecodeFn(Protocol) — wrong unary untyped signature, never called\n\nRemove these fields from their dataclasses:\n  - RunSpecification.encoding_aggregation_fn (field with type AggregationFn | None)\n  - SamplingSpecification.decode_fn (field with type DecodeFn | None)\n\nRemove imports that become dead after deletion:\n  - "from enum import Enum" (if only used by AveragingMode)\n  - "import jax.numpy as jnp" (only used in AveragingMode.to_fn() lambdas)\n\nKeep:\n  - The register_spec decorator's average_encoding_mode backward-compat handling\n    (it strips the deprecated kwarg and emits a DeprecationWarning — this is still\n    needed for users migrating from the old API)\n  - All other fields, classes, and imports\n\n--- STEP 2: Add DecodeOutput to src/aminx/types/bundles.py ---\n\nRead the file first to understand its existing style, then add:\n\n  @dataclass(frozen=True)\n  class DecodeOutput:\n    sequences: Int[Array, "K L"]\n    logits: Float[Array, "K L V"]\n\nUse the jaxtyping imports already in the file (Int, Float, Array). If they are not\npresent, add: from jaxtyping import Array, Float, Int\n\n--- STEP 3: Add DecodingFusionFn to src/aminx/types/stages.py ---\n\nRead the file. After EncodingFusionFn (around line 114), add:\n\n  class DecodingFusionFn(Protocol):\n    def __call__(self, stacked: DecodeOutput) -> DecodeOutput: ...\n\nAdd the import for DecodeOutput at the top of stages.py:\n  from aminx.types.bundles import DecodeOutput\n\nUpdate __all__ in stages.py to include "DecodingFusionFn".\n\n--- STEP 4: Export from src/aminx/types/__init__.py ---\n\nAdd DecodeOutput and DecodingFusionFn to the exports in aminx/types/__init__.py.\nRead the file first to match the existing export pattern.\n\n--- STEP 5: Verify ---\n\n  cd /home/marielle/projects/aminx && uv run pytest tests/ -x -q 2>&1 | tail -20\n\nFix any import errors. The test suite should pass (or fail only on pre-existing failures).\n\n--- STEP 6: Commit ---\n\n  git add src/aminx/run/specs.py src/aminx/types/bundles.py src/aminx/types/stages.py src/aminx/types/__init__.py\n  git commit -m "refactor(R6-5): remove dead AggregationFn/AveragingMode/DecodeFn; add DecodeOutput + DecodingFusionFn (#2003)"\n\nWhen done, end with: verdict: done\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Review R6-5: dead protocol cleanup + new Fuse-shaped types.\n\nVERIFY 1: src/aminx/run/specs.py — AggregationFn class is GONE\nVERIFY 2: src/aminx/run/specs.py — AveragingMode class is GONE\nVERIFY 3: src/aminx/run/specs.py — DecodeFn class is GONE (the one defined IN specs.py; not any type alias elsewhere)\nVERIFY 4: src/aminx/run/specs.py — encoding_aggregation_fn field is GONE from RunSpecification\nVERIFY 5: src/aminx/run/specs.py — decode_fn field is GONE from SamplingSpecification\nVERIFY 6: src/aminx/run/specs.py — register_spec decorator still handles average_encoding_mode kwarg stripping (backward-compat preserved)\nVERIFY 7: src/aminx/types/bundles.py — DecodeOutput frozen dataclass exists with sequences and logits fields\nVERIFY 8: src/aminx/types/stages.py — DecodingFusionFn(Protocol) exists with __call__(self, stacked: DecodeOutput) -> DecodeOutput\nVERIFY 9: DecodingFusionFn is exported from aminx/types/__init__.py\nVERIFY 10: cd /home/marielle/projects/aminx && uv run pytest tests/ -q 2>&1 | tail -5 — passes (or same pre-existing failures as before)\n\nPASS if all 10 satisfied. FAIL if any dead class survives or new types are missing.\n`,
    null,
    null
  );

// ===== TRACK B — Track B — R6-6: Add encoding_fusion + decoding_fusion to RunSpecification; wire encoding_fusion (#2004) =========================
const trackB = () =>
  track(
    "2004",
    "Track B — R6-6: Add encoding_fusion + decoding_fusion to RunSpecification; wire encoding_fusion (#2004)",
    `task_id: ${TASK_ID}. Fix #2004: Add Fuse-shaped fusion fields to RunSpecification and wire encoding_fusion through build_run_spec().\n\nWORKING DIRECTORY: /home/marielle/projects/aminx\n\nPrerequisites: Track A (#2003) is complete — DecodeOutput and DecodingFusionFn now exist in aminx.types.\n\n--- STEP 1: Add fields to RunSpecification in src/aminx/run/specs.py ---\n\nAfter the existing noise: list[FeatureNoiseBundle] field, add:\n\n  encoding_fusion: EncodingFusionFn | None = None\n  decoding_fusion: DecodingFusionFn | None = None\n\nAdd imports at the top of specs.py:\n  from aminx.types.stages import EncodingFusionFn, DecodingFusionFn\n\n(EncodingFusionFn may already be imported — check first and only add what is missing.)\n\n--- STEP 2: Wire encoding_fusion in src/aminx/run/spec.py ---\n\nRead build_run_spec() (around line 311). There is a dead AveragingConfig construction\nthat references average_encoding_mode. Find it.\n\nThe AveragingConfig is used to build the RunSpec's averaging field. Replace the dead\naverage_encoding_mode argument with nothing (remove it from the AveragingConfig call).\nThen, find where StageSet is constructed in build_run_spec() and add:\n\n  encoding_fusion=spec.encoding_fusion\n\nto the StageSet(...) constructor call. StageSet already has\nencoding_fusion: EncodingFusionFn | None = None (at stages.py:346), so this is just\npassing through the spec field.\n\nAlso add to the RunSpec (or whichever structure carries plan-level fields):\n  decoding_fusion=spec.decoding_fusion  (if RunSpec has a decoding_fusion field, else skip — Track C will add it)\n\nIf RunSpec does not yet have decoding_fusion, add:\n  decoding_fusion: DecodingFusionFn | None = eqx.field(static=True, default=None)\nto the RunSpec class (src/aminx/run/spec.py or wherever RunSpec is defined — check\nwhich file defines the aminx RunSpec eqx.Module).\n\n--- STEP 3: Verify ---\n\n  cd /home/marielle/projects/aminx && uv run pytest tests/ -x -q 2>&1 | tail -20\n\n--- STEP 4: Commit ---\n\n  git add src/aminx/run/specs.py src/aminx/run/spec.py\n  git commit -m "feat(R6-6): add encoding_fusion + decoding_fusion to RunSpecification; wire encoding_fusion -> StageSet (#2004)"\n\nWhen done, end with: verdict: done\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Review R6-6: RunSpecification fusion fields + build_run_spec() wiring.\n\nVERIFY 1: RunSpecification has encoding_fusion: EncodingFusionFn | None = None\nVERIFY 2: RunSpecification has decoding_fusion: DecodingFusionFn | None = None\nVERIFY 3: build_run_spec() passes spec.encoding_fusion into StageSet(encoding_fusion=...) — grep -n "encoding_fusion" src/aminx/run/spec.py should show this wiring\nVERIFY 4: Dead average_encoding_mode argument is removed from AveragingConfig construction in build_run_spec()\nVERIFY 5: RunSpec (eqx.Module) has decoding_fusion field (static=True, for passing to kernel_dispatch)\nVERIFY 6: cd /home/marielle/projects/aminx && uv run pytest tests/ -q 2>&1 | tail -5 — passes\n\nPASS if all 6 satisfied. FAIL if encoding_fusion is not wired into StageSet or fields are missing.\n`,
    null,
    null
  );

// ===== TRACK C — Track C — R6-7: Wire decoding_fusion in kernel_dispatch.py (#2005) =========================
const trackC = () =>
  track(
    "2005",
    "Track C — R6-7: Wire decoding_fusion in kernel_dispatch.py (#2005)",
    `task_id: ${TASK_ID}. Fix #2005: Insert decode fusion step in kernel_dispatch.py after K-decode collection.\n\nWORKING DIRECTORY: /home/marielle/projects/aminx\n\nPrerequisites: Tracks A + B are complete. RunSpec now has decoding_fusion: DecodingFusionFn | None field.\n\n--- THE INSERTION POINT ---\n\nIn src/aminx/run/kernel_dispatch.py there are two decode dispatch paths:\n  1. Protein path: around line 347 — _safe_map(_call_decode_one_enc, fused_enc, batch_size=None)\n  2. Ligand path: around line 514 — similar _safe_map over decode\n\nAfter each _safe_map call that collects K stacked decode outputs (sequences + logits),\nand BEFORE the transpose step (around line 523), insert:\n\n  # Apply decode fusion if specified (e.g. logit ensembling, best-of-K)\n  if plan.decoding_fusion is not None:\n      from aminx.types.bundles import DecodeOutput\n      stacked_out = DecodeOutput(sequences=stacked_sequences, logits=stacked_logits)\n      fused_out = plan.decoding_fusion(stacked_out)\n      stacked_sequences, stacked_logits = fused_out.sequences, fused_out.logits\n\nWhere stacked_sequences and stacked_logits are the variable names used at that point.\nRead the actual file around those lines to get the exact variable names before editing.\n\nNOTE: "plan" here is the RunSpec (or InferencePlan) object passed into the dispatch\nfunction. Check how it's referenced — it may be \`spec\`, \`run_spec\`, or \`plan\`. Use\nwhatever variable name holds the RunSpec.\n\nIf decoding_fusion is None (default), no change — K outputs pass through as before.\n\n--- STEP 2: Verify ---\n\n  cd /home/marielle/projects/aminx && uv run pytest tests/ -x -q 2>&1 | tail -20\n\nRun a quick smoke test to confirm the None path (current behaviour) still works:\n  uv run python -c "\nfrom aminx.run.specs import RunSpecification\nspec = RunSpecification(inputs=['tests/data/1bc8.pdb'])\nassert spec.decoding_fusion is None\nprint('decoding_fusion=None default OK')\n"\n\n--- STEP 3: Commit ---\n\n  git add src/aminx/run/kernel_dispatch.py\n  git commit -m "feat(R6-7): wire decoding_fusion into kernel_dispatch — fusion step after K-decode (#2005)"\n\nWhen done, end with: verdict: done\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Review R6-7: decoding_fusion wiring in kernel_dispatch.py.\n\nVERIFY 1: kernel_dispatch.py has a fusion guard block (check for decoding_fusion is not None) after the K-decode _safe_map in BOTH protein and ligand paths\nVERIFY 2: The guard wraps sequences+logits into DecodeOutput, calls plan.decoding_fusion(stacked_out), and unpacks\nVERIFY 3: When decoding_fusion is None (default), the code path is identical to pre-patch — no change in output shape\nVERIFY 4: cd /home/marielle/projects/aminx && uv run pytest tests/ -q 2>&1 | tail -5 — passes\n\nPASS if all 4 satisfied. FAIL if fusion guard is missing in either dispatch path or if the None path changes behaviour.\n`,
    null,
    null
  );

// ===== TRACK D — Track D — Auditor closeout =========================
const trackD = () =>
  track(
    "2005",
    "Track D — Auditor closeout",
    `task_id: ${TASK_ID}. This is an auditor closeout track — no implementation needed.\n\nRun the full audit of the R6-5/6/7 sprint changes.\n\nScope: src/aminx/run/specs.py, src/aminx/run/spec.py, src/aminx/run/kernel_dispatch.py,\nsrc/aminx/types/bundles.py, src/aminx/types/stages.py, src/aminx/types/__init__.py\n\nWhen done, end with: verdict: done\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. You are the AUDITOR for sprint 260616_rs6-fuse-refactor. Review ALL changes together.\n\nAUDIT DIMENSIONS:\n\nA. CORRECTNESS\n  - DecodingFusionFn protocol matches Fuse[DecodeOutput, DecodeOutput] shape (unary, not binary)\n  - encoding_fusion wired into StageSet (not just stored on spec)\n  - decoding_fusion guard in kernel_dispatch handles None correctly (no-op)\n  - DecodeOutput field names and types match what kernel_dispatch actually produces\n  - AveragingMode backward-compat kwarg stripping still works in register_spec\n\nB. COMPLETENESS\n  - All three dead classes removed: AggregationFn, AveragingMode, DecodeFn\n  - Both dead fields removed: encoding_aggregation_fn, decode_fn\n  - Both new fields present on RunSpecification: encoding_fusion, decoding_fusion\n  - DecodeOutput exported from aminx.types\n  - DecodingFusionFn exported from aminx.types\n\nC. ARCHITECTURE\n  - EncodingFusionFn and DecodingFusionFn both follow Fuse[X, X] unary contract\n  - DecodeOutput bundle type is appropriate (frozen dataclass or eqx.Module matching pipeline output)\n  - build_run_spec() passes encoding_fusion to StageSet directly (not via AveragingConfig)\n\nD. TEST COVERAGE\n  - cd /home/marielle/projects/aminx && uv run pytest tests/ -q — passes\n  - Are there tests for the new encoding_fusion / decoding_fusion spec fields?\n  - Is the decoding_fusion=None default tested?\n\nE. REGRESSIONS\n  - grep -En "AggregationFn|AveragingMode|DecodeFn|encoding_aggregation_fn|decode_fn" src/aminx/run/specs.py should return 0 hits for class/field definitions\n\nRender PASS / NEEDS_WORK / FAIL with specific issues.\n`,
    null,
    null
  );

// ---- orchestrate: writing chain (A -> B -> C -> D, sequential) ----
log("260616_rs6-fuse-refactor: writing chain (A -> B -> C -> D, sequential)");
const a = await trackA();
const b = await trackB();
const c = await trackC();
const d = await trackD();

return {
  task_id: TASK_ID,
  sprint_id: 1,
  verdicts: {
    "2003": a,
    "2004": b,
    "2005": c,
    "2005_audit": d
  },
};
