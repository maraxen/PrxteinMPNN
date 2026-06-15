// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/260618_runspec-hygiene-p1.toml
// Regenerate: praxia dw emit-sprint 260618_runspec-hygiene-p1.toml
// task_id: 260618_runspec-hygiene-p1   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (A,B,C,D) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.

export const meta = {
  name: "260618_runspec-hygiene-p1",
  description: "RunSpec hygiene P1: conditioning fields to base, chain_mask fix, Fuse/Tap/Sink/AxisBoundary to xtrax",
  phases: [
    { title: "Track A — H1: Move sidechain_conditioning + fixed_mask to RunSpecification base (#1920)" },
    { title: "Track B — H2: Fix chain_mask hardcode in _sampling_helper.py:346 (#1921)" },
    { title: "Track C — Promote Fuse/Tap/Sink/AxisBoundary to xtrax/stages/boundaries.py (#1907)" },
    { title: "Track D — Convert aminx/types/boundaries.py to shim re-export from xtrax.stages (#1908)" },
  ],
};

const TASK_ID = "260618_runspec-hygiene-p1";
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

// Shared context for the writing tracks (from recon, task 260618_runspec-hygiene-p1).
const EMITTER_CTX = `[MANUAL: paste recon findings here before running emit-sprint]`;

// ---- per-track stage helpers ---------------------------------------------
const fixer = (prompt, label, phaseName, isolation = null) => {
  const opts = { agentType: "fixer", label, phase: phaseName };
  if (isolation) opts.isolation = isolation;
  return agent(`${prompt}\n\nWhen done, end your message with 'verdict: done' on its own line.`, opts);
};

const reviewer = (itemId, prompt, label, phaseName, isolation = null) => {
  const opts = { agentType: "reviewer", label, phase: phaseName, schema: VERDICT_SCHEMA };
  if (isolation) opts.isolation = isolation;
  return agent(prompt, opts);
};

// Sequential implement->review with bounded NEEDS_WORK repair cycles.
async function track(itemId, phaseName, fixerPrompt, reviewerPrompt, isolation = null) {
  log(`[${itemId}] implement`);
  await fixer(fixerPrompt, `fix:${itemId}`, phaseName, isolation);
  let verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}`, phaseName, isolation);
  for (let retry = 0; retry < MAX_FIX_RETRIES && verdict && verdict.verdict === "NEEDS_WORK"; retry++) {
    log(`[${itemId}] NEEDS_WORK — repair cycle ${retry + 1}/${MAX_FIX_RETRIES}`);
    const issues = (verdict.issues || [])
      .map((i) => `- ${i.where}: ${i.problem} -> ${i.fix}`)
      .join("\n");
    await fixer(
      `${fixerPrompt}\n\nA reviewer found issues — fix exactly these, nothing else:\n${issues}`,
      `fix:${itemId}:repair:${retry}`,
      phaseName,
      isolation
    );
    verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}:re:${retry}`, phaseName, isolation);
  }
  return verdict;
}

// ===== TRACK A — Track A — H1: Move sidechain_conditioning + fixed_mask to RunSpecification base (#1920) =========================
const trackA = () =>
  track(
    "1920",
    "Track A — H1: Move sidechain_conditioning + fixed_mask to RunSpecification base (#1920)",
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nRemove sidechain_conditioning (line 336) and fixed_mask (line 316) from SamplingSpecification. Add both to RunSpecification defaulted-field block (145-225) before run_spec = field(init=False) at line 226. Safe: build_run_spec reads via getattr MRO-agnostic; spec_json.py:186 already handles fixed_mask. Supersedes #1904 (which adds to InspectionSpec — wrong fix). Gate: pytest passes; round-trip test for ScoringSpec with fixed_mask; test asserts fields are on base not only SamplingSpec. LOC ~8.\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nRemove sidechain_conditioning (line 336) and fixed_mask (line 316) from SamplingSpecification. Add both to RunSpecification defaulted-field block (145-225) before run_spec = field(init=False) at line 226. Safe: build_run_spec reads via getattr MRO-agnostic; spec_json.py:186 already handles fixed_mask. Supersedes #1904 (which adds to InspectionSpec — wrong fix). Gate: pytest passes; round-trip test for ScoringSpec with fixed_mask; test asserts fields are on base not only SamplingSpec. LOC ~8.\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ===== TRACK B — Track B — H2: Fix chain_mask hardcode in _sampling_helper.py:346 (#1921) =========================
const trackB = () =>
  track(
    "1921",
    "Track B — H2: Fix chain_mask hardcode in _sampling_helper.py:346 (#1921)",
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nReplace jnp.ones hardcode with: if spec.fixed_mask is not None: fixed_mask_np = _broadcast_per_structure(spec.fixed_mask, batch_size=batch_size, expected_len=seq_len, dtype=jnp.float32, name='fixed_mask'); chain_mask = 1.0 - fixed_mask_np. Use _broadcast_per_structure (exists at lines 448-457) — do NOT use raw spec.fixed_mask directly (1-D, unpadded). Convention: chain_mask 1=designable 0=fixed; fixed_mask 1=fixed, hence complement. Gate: fixed_mask=None → all-ones (batch_size, seq_len); fixed_mask set → complement correct shape; assert chain_mask.dtype == jnp.float32. LOC ~15.\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nReplace jnp.ones hardcode with: if spec.fixed_mask is not None: fixed_mask_np = _broadcast_per_structure(spec.fixed_mask, batch_size=batch_size, expected_len=seq_len, dtype=jnp.float32, name='fixed_mask'); chain_mask = 1.0 - fixed_mask_np. Use _broadcast_per_structure (exists at lines 448-457) — do NOT use raw spec.fixed_mask directly (1-D, unpadded). Convention: chain_mask 1=designable 0=fixed; fixed_mask 1=fixed, hence complement. Gate: fixed_mask=None → all-ones (batch_size, seq_len); fixed_mask set → complement correct shape; assert chain_mask.dtype == jnp.float32. LOC ~15.\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ===== TRACK C — Track C — Promote Fuse/Tap/Sink/AxisBoundary to xtrax/stages/boundaries.py (#1907) =========================
const trackC = () =>
  track(
    "1907",
    "Track C — Promote Fuse/Tap/Sink/AxisBoundary to xtrax/stages/boundaries.py (#1907)",
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nCreate xtrax/stages/boundaries.py exporting Fuse, Tap, Sink, AxisBoundary verbatim from aminx/types/boundaries.py. All four types remain @runtime_checkable. AxisBoundary fields remain eqx.field(static=True). Docstring cross-references added between FuseFn and Fuse (distinct protocols, different contexts).\n\nUpdate xtrax/stages/__init__.py to re-export: from xtrax.stages.boundaries import AxisBoundary, Fuse, Sink, Tap. FuseFn is NOT removed or merged.\n\nAcceptance criteria:\n- Given: from xtrax.stages import Fuse, Tap, Sink, AxisBoundary; Then: all four resolve without ImportError\n- Given: AxisBoundary() instance (all-None); When: eqx.tree_flatten(ab) called; Then: succeeds and round-trips\n- Given: a class with def __call__(self, stacked): ...; When: isinstance(obj, Fuse); Then: True\n- Given: from xtrax.stages import FuseFn; Then: still resolves (not broken)\n\nPre-condition: grep aminx for isinstance.*(Fuse|Tap|Sink|AxisBoundary) — confirm zero matches before creating file.\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nCreate xtrax/stages/boundaries.py exporting Fuse, Tap, Sink, AxisBoundary verbatim from aminx/types/boundaries.py. All four types remain @runtime_checkable. AxisBoundary fields remain eqx.field(static=True). Docstring cross-references added between FuseFn and Fuse (distinct protocols, different contexts).\n\nUpdate xtrax/stages/__init__.py to re-export: from xtrax.stages.boundaries import AxisBoundary, Fuse, Sink, Tap. FuseFn is NOT removed or merged.\n\nAcceptance criteria:\n- Given: from xtrax.stages import Fuse, Tap, Sink, AxisBoundary; Then: all four resolve without ImportError\n- Given: AxisBoundary() instance (all-None); When: eqx.tree_flatten(ab) called; Then: succeeds and round-trips\n- Given: a class with def __call__(self, stacked): ...; When: isinstance(obj, Fuse); Then: True\n- Given: from xtrax.stages import FuseFn; Then: still resolves (not broken)\n\nPre-condition: grep aminx for isinstance.*(Fuse|Tap|Sink|AxisBoundary) — confirm zero matches before creating file.\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ===== TRACK D — Track D — Convert aminx/types/boundaries.py to shim re-export from xtrax.stages (#1908) =========================
const trackD = () =>
  track(
    "1908",
    "Track D — Convert aminx/types/boundaries.py to shim re-export from xtrax.stages (#1908)",
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nAfter xtrax/stages/boundaries.py is created, replace aminx/types/boundaries.py contents with a one-line re-export shim:\n  from xtrax.stages.boundaries import AxisBoundary, Fuse, Sink, Tap  # noqa: F401\n  __all__ = ['AxisBoundary', 'Fuse', 'Sink', 'Tap']\n\nAcceptance criteria:\n- Given: from aminx.types.boundaries import Fuse; Then: aminx.types.boundaries.Fuse is xtrax.stages.Fuse (same object)\n- Given: from aminx.types import boundaries; boundaries.Fuse accessed; Then: resolves (module-level attribute access works)\n\nDepends on: xtrax/stages/boundaries.py created (item 1907)\n\n[AUTO: no recon anchors — paste file:line references here, then write the full fixer prompt]\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # DRAFT — requires human review before dw emit-sprint\n\nBACKLOG DESCRIPTION:\nAfter xtrax/stages/boundaries.py is created, replace aminx/types/boundaries.py contents with a one-line re-export shim:\n  from xtrax.stages.boundaries import AxisBoundary, Fuse, Sink, Tap  # noqa: F401\n  __all__ = ['AxisBoundary', 'Fuse', 'Sink', 'Tap']\n\nAcceptance criteria:\n- Given: from aminx.types.boundaries import Fuse; Then: aminx.types.boundaries.Fuse is xtrax.stages.Fuse (same object)\n- Given: from aminx.types import boundaries; boundaries.Fuse accessed; Then: resolves (module-level attribute access works)\n\nDepends on: xtrax/stages/boundaries.py created (item 1907)\n\nREVIEWER CHECKLIST (fill in before running dw emit-sprint):\nVERIFY: cargo command or test that must pass — e.g. \`cargo nextest run -p <crate>\`\nVERIFY: observable assertion — e.g. "new test exists and calls the handler directly"\nVERIFY: no regression — e.g. "existing tests still pass"\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.`,
  );

// ---- orchestrate: writing chain (A -> B -> C -> D, sequential) ----
log("260618_runspec-hygiene-p1: writing chain (A -> B -> C -> D, sequential)");
const a = await trackA();
const b = await trackB();
const c = await trackC();
const d = await trackD();

return {
  task_id: TASK_ID,
  sprint_id: 1,
  verdicts: {
    "1920": a,
    "1921": b,
    "1907": c,
    "1908": d
  },
};
