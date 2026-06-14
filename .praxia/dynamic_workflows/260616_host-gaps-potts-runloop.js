// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 260616_host-gaps-potts-runloop   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (A,B) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time. Only the read-only
//   research/concurrent tracks (C,D,E) run concurrently.

export const meta = {
  name: "260616_host-gaps-potts-runloop",
  description: "Close three host dispatch wiring gaps (#1856 chain_mask, #1857 fixed_mask double-apply, #1858 STE del-kill), implement P-08 end-to-end Potts run loop, and land T2.5 multi-phase BatchPlanner in xtrax.",
  phases: [
    { title: "Track A — [host-dispatch] chain_mask uncomputed + unrouted for sidechain conditioning (#1856)" },
    { title: "Track B — [host-dispatch] fixed_mask double-applied in _prepare_fixed_controls (#1857)" },
    { title: "Track C — [host-dispatch] STE not routed from spec in InferencePlan — decode_mode hardcoded ConditionalMode (#1858)" },
    { title: "Track D — P-08: complete aminx.potts runner — add Gibbs/PT sampling integration (#1299)" },
    { title: "Track E — [xtrax T2.5+2.5b] Multi-phase BatchPlanner + axis-injection seam (#1554)" },
  ],
};

const TASK_ID = "260616_host-gaps-potts-runloop";
const MAX_FIX_RETRIES = 2;

function extractVerdict(text) {
  const m = String(text ?? "").match(/verdict:\s*([a-z_]+)/i);
  return m ? m[1].toLowerCase() : "advance";
}

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

// Shared context for the writing tracks (from recon, task 260616_host-gaps-potts-runloop).
const EMITTER_CTX = `Prior sprint (260615) closed at 0.1.0a6 with P-05/P-07/P-09/T0.3/RS-1/T2.1/T2.4 complete.\nStage 3 bathos cells still require the Python API directly (build_inference_bundle) because\nhost dispatch wiring is incomplete: chain_mask is unrouted, fixed_mask has a double-apply\nbug, and STE is killed before dispatch. These three bugs block campaign-level sidechain /\nfixed-residue / STE workflows through the spec+CLI layer.\nP-08 is the next Potts milestone: end-to-end inference (weight load -> TRW -> sample -> output).\nT2.5 is unblocked after T2.4 landed (xtrax repo).\nWorktree isolation IS enabled for concurrent tracks. The prior failure (sprint 260615)\nused relative ../xtrax paths — all prompts below use absolute paths so isolation is safe.\nTrack E workspace is /home/marielle/projects/xtrax (absolute); fixer agents must NOT use\nrelative ../xtrax references.\n`;

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
    const issues = verdict.issues || [];
    if (issues.length === 0) {
      log(`[${itemId}] NEEDS_WORK with no issues — reviewer must supply at least one actionable issue; escalating`);
      break;
    }
    log(`[${itemId}] NEEDS_WORK — repair cycle ${retry + 1}/${MAX_FIX_RETRIES} (${issues.length} issue(s))`);
    const issueText = issues
      .map((i) => `- ${i.where}: ${i.problem} -> ${i.fix}`)
      .join("\n");
    await fixer(
      `${fixerPrompt}\n\nA reviewer found issues — fix exactly these, nothing else:\n${issueText}`,
      `fix:${itemId}:repair:${retry}`,
      phaseName,
      isolation
    );
    verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}:re:${retry}`, phaseName, isolation);
  }
  return verdict;
}

// ===== TRACK A — Track A — [host-dispatch] chain_mask uncomputed + unrouted for sidechain conditioning (#1856) =========================
const trackA = () =>
  track(
    "1856",
    "Track A — [host-dispatch] chain_mask uncomputed + unrouted for sidechain conditioning (#1856)",
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/aminx\n\n## Objective\nFix TWO connected gaps in the LigandMPNN sidechain conditioning path:\n  1. chain_mask is hardcoded \`jnp.zeros(...)\` in _prepare_ligand_context — a semantically\n     meaningless all-fixed placeholder that disables side-chain design.\n  2. chain_mask is never forwarded to build_inference_bundle (it's in the returned dict\n     but not passed at any of the four call sites in kernel_dispatch.py).\n\n## Key files (read before implementing)\n- src/aminx/host/_sampling_helper.py line 343  — \`chain_mask = jnp.zeros((batch_size, seq_len), ...)\`\n  This is the hardcoded placeholder; needs to be computed from real chain data.\n- src/aminx/model/ligand_mpnn.py lines 277, 312-313  — chain_mask semantic: 1=designable, 0=fixed\n- src/aminx/host/_sampling_helper.py lines 247-263  — _prepare_ligand_context non-ligandmpnn branch\n  (also returns chain_mask: None — confirm appropriate for non-ligandmpnn paths)\n- src/aminx/host/kernel_dispatch.py lines 145-160  — _prepare_ligand_context call + dict usage\n- src/aminx/host/kernel_dispatch.py lines 191-221, 257-295, 349-387, 421-457\n  (atom_37/atom_37_mask forwarded; chain_mask never passed — four call sites to fix)\n- src/aminx/inference/bundle_builder.py lines 36-63  — no chain_mask param currently\n\n## Part 1: Compute chain_mask correctly\nIn _prepare_ligand_context (line 325+, sidechain_conditioning=True branch):\n- Replace \`chain_mask = jnp.zeros(...)\` with a value derived from batched_ensemble.chain_index\n  or spec.chain_mask (if SamplingSpecification exposes one).\n- If spec has a chain_mask field: read it. If not: derive a default (e.g., all-ones meaning\n  fully designable) — but explicitly choose a semantically correct value, NOT all-zeros.\n- Verify against the LigandMPNN semantic (1=designable, 0=fixed).\n\n## Part 2: Route chain_mask to build_inference_bundle\n- Determine whether to add a \`chain_mask: jax.Array | None = None\` parameter to\n  build_inference_bundle (bundle_builder.py:36) and thread it through to the model, OR\n  handle it via an existing mechanism. Check LigandMPNN model usage before deciding.\n- Apply routing at ALL FOUR build_inference_bundle call sites in kernel_dispatch.py.\n\n## Tests\nAdd a test asserting:\n  (a) When sidechain_conditioning=True, chain_mask is NOT all-zeros (confirm semantically correct)\n  (b) chain_mask reaches the bundle at a build_inference_bundle call site\n\n## Constraints\n- Edit only. Do not touch averaging.py, _prepare_fixed_controls, or Potts code.\n- Track B is the fixed_mask track and also edits _sampling_helper.py; coordinate scope\n  to avoid conflicts (Track A touches line 343+; Track B touches lines 438-495).\n- Run \`uv run pytest tests/ -x -q 2>&1 | tail -30\` before declaring done.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/aminx\n\n## Checks\nVERIFY: \`uv run pytest tests/ -x -q 2>&1 | tail -30\` — all existing tests pass\nVERIFY: grep -En "chain_mask.*zeros|zeros.*chain_mask" src/aminx/host/_sampling_helper.py\n        — ZERO matches (hardcoded zeros placeholder removed)\nVERIFY: New test asserts chain_mask is NOT all-zeros when sidechain_conditioning=True\nVERIFY: New test asserts chain_mask reaches the bundle/model at a call site\nVERIFY: ALL FOUR build_inference_bundle call sites include chain_mask routing\nVERIFY: \`uv run ty check 2>&1 | tail -20\` — no new type errors\n\nPASS if all VERIFY items are satisfied.\nFAIL if chain_mask is still zeros, routing is absent at any call site, or any test fails.\n`,
    "worktree"
  );

// ===== TRACK B — Track B — [host-dispatch] fixed_mask double-applied in _prepare_fixed_controls (#1857) =========================
const trackB = () =>
  track(
    "1857",
    "Track B — [host-dispatch] fixed_mask double-applied in _prepare_fixed_controls (#1857)",
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/aminx\n\n## Objective\nFix the double-application of spec.fixed_mask in \`_prepare_fixed_controls\` so that a\nfloat-valued shell mask (shape (L,) or (batch, L)) is applied exactly once.\n\n## Key file\nsrc/aminx/host/_sampling_helper.py lines 438-495 (_prepare_fixed_controls)\n\n## Bug\nThere are TWO blocks handling spec.fixed_mask in this function:\n  Block 1 (lines ~448-456): sets fixed_mask_np from spec.fixed_mask via np.broadcast_to\n  Block 2 (lines ~485-493): reads spec.fixed_mask AGAIN via _broadcast_per_structure and\n    takes jnp.maximum with the already-set fixed_mask_np\n\nThis double-application is fragile and inconsistent with the single-pass pattern used for\nfixed_positions. The correct single-path logic:\n  1. Start with zeros.\n  2. If spec.fixed_mask set: apply once via _broadcast_per_structure.\n  3. If spec.fixed_positions set: union (jnp.maximum).\n  4. If spec.fixed_tokens set: validate against combined mask.\n\n## Implementation\n- Edit only. Do not Write to existing files.\n- Collapse the two blocks into one, using _broadcast_per_structure at the START of the\n  function (before the fixed_positions block), matching the pattern used for fixed_positions.\n- Remove Block 2 (lines ~485-493) entirely.\n- Add a test in tests/host/test_sampling_helper.py (or nearest existing test file):\n  (a) float mask via spec.fixed_mask → correctly broadcast\n  (b) fixed_mask + fixed_positions → union taken\n  (c) idempotency: applying spec with same mask twice produces same result\n- Run \`uv run pytest tests/ -x -q 2>&1 | tail -30\` before declaring done.\n\n## Constraints\n- One file focus: _sampling_helper.py.\n- Do not touch kernel_dispatch.py, averaging.py, or Potts code.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/aminx\n\n## Checks\nVERIFY: \`uv run pytest tests/ -x -q 2>&1 | tail -30\` — all existing tests pass\nVERIFY: spec.fixed_mask appears exactly ONCE in _prepare_fixed_controls body\n        (grep -c "spec.fixed_mask" _sampling_helper.py — should be 1 in the function, plus test)\nVERIFY: New test covers (a) float mask broadcast, (b) fixed_mask+fixed_positions union, (c) idempotency\nVERIFY: \`uv run ty check 2>&1 | tail -20\` — no new type errors\n\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item fails or is untestable as written.\n`,
    "worktree"
  );

// ===== TRACK C — Track C — [host-dispatch] STE not routed from spec in InferencePlan — decode_mode hardcoded ConditionalMode (#1858) =========================
const trackC = () =>
  track(
    "1858",
    "Track C — [host-dispatch] STE not routed from spec in InferencePlan — decode_mode hardcoded ConditionalMode (#1858)",
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/aminx\n\n## Objective\nWire spec.sampling_strategy="straight_through" through to InferencePlan so that\nplan.decode() in kernel_dispatch.py uses STEDecode instead of staying on ConditionalMode.\n\n## Key files (read before implementing)\n- src/aminx/host/plan.py lines 635-690  — InferencePlan construction; lines 682-685 are the gap:\n    # Routing mode from spec.sampling_strategy is planned future work. To use a\n    # different mode (AR, STE, Unconditional), construct decode_fn manually via ...\n    decode_mode = ConditionalMode()\n  This hardcodes ConditionalMode regardless of spec.sampling_strategy.\n- src/aminx/host/plan.py lines 304-313  — STEDecode import and constraint ("STE requires\n  ConditionalDecodeStep"); STEDecode IS already imported and constrained here.\n- src/aminx/host/plan.py lines 262, 366+  — ARDecodeFn, DecodeScoreFn, STEDecodeFn protocol\n  references; plan accepts STEDecode as a valid decode_fn type.\n- src/aminx/host/kernel_dispatch.py lines 33, 230, 332, 399, 493  — plan.decode() call sites;\n  the plan is passed through from the outer function; sampling_strategy comes from spec.\n\n## Diagnosis\nThe \`del sampling_strategy, decoding_order_fn\` in averaging.py:362 is DEAD CODE — the\n\`sample_fn\` returned from that function (averaging.py:408-426) has its own \`sampling_strategy\`\nparameter and already dispatches to _ste_optimize_sequence_legacy at line 453. That path works.\n\nThe ACTUAL bug is in host/plan.py lines 682-685: when constructing the InferencePlan used\nby kernel_dispatch.py, decode_mode is always set to ConditionalMode(). The comment at line 682\nconfirms this is a known planned gap ("planned future work"). This means plan.decode() in\nkernel_dispatch.py:230 (and lines 332, 399, 493) always runs ConditionalDecode, never STEDecode.\n\n## Implementation\n- Edit only. Do not Write to existing files.\n- In host/plan.py (function that constructs InferencePlan, around line 635+):\n  Accept or read spec.sampling_strategy (or a \`sampling_strategy\` parameter).\n  When sampling_strategy == "straight_through": set decode_mode = STEDecode (already imported).\n  Else: keep decode_mode = ConditionalMode() (current default).\n- Verify that STEDecode requires ConditionalDecodeStep (plan.py:262 constraint) — confirm\n  the ConditionalDecodeStep is already set when sampling_strategy="straight_through".\n- Add a test in tests/host/ asserting:\n  (a) InferencePlan constructed with sampling_strategy="straight_through" uses STEDecode\n  (b) InferencePlan constructed with sampling_strategy="temperature" uses ConditionalMode\n- Run \`uv run pytest tests/ -x -q 2>&1 | tail -30\` before declaring done.\n\n## What NOT to do\n- Do NOT remove \`del sampling_strategy\` from averaging.py:362 — that code is dead but harmless;\n  removing it would change the semantics of the outer function's closed-over scope.\n- Do NOT modify kernel_dispatch.py — the plan.decode() calls are correct; fix the plan itself.\n\n## Constraints\n- Focus on host/plan.py only.\n- Do not touch averaging.py, _sampling_helper.py, or Potts code.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/aminx\n\n## Checks\nVERIFY: \`uv run pytest tests/ -x -q 2>&1 | tail -30\` — all existing tests pass\nVERIFY: New test asserts that sampling_strategy="straight_through" produces an InferencePlan\n        that uses STEDecode (not ConditionalMode) for its decode step\nVERIFY: \`grep -n "decode_mode = ConditionalMode" src/aminx/host/plan.py\` — the hardcoded\n        decode_mode = ConditionalMode() line is GONE from the InferencePlan construction path\n        (replaced by a conditional: ConditionalMode for temperature, STEDecode for straight_through)\nVERIFY: averaging.py is UNCHANGED (grep sha or confirm no diff to averaging.py)\nVERIFY: \`uv run ty check 2>&1 | tail -20\` — no new type errors\n\nPASS if all VERIFY items are satisfied.\nFAIL if decode_mode is still unconditionally ConditionalMode, or if averaging.py was modified.\n`,
    "worktree"
  );

// ===== TRACK D — Track D — P-08: complete aminx.potts runner — add Gibbs/PT sampling integration (#1299) =========================
const trackD = () =>
  track(
    "1299",
    "Track D — P-08: complete aminx.potts runner — add Gibbs/PT sampling integration (#1299)",
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/aminx\n\n## Objective\nComplete P-08 by adding Gibbs/PT sampling integration to the existing Potts runner.\nsrc/aminx/potts/runner.py ALREADY EXISTS (225 lines) with run_potts() covering:\n  weight loading (zstd+eqx.tree_deserialise_leaves), calibration, infer_params, PoeModel.\nThe MISSING piece: spec.n_samples > 0 should trigger GibbsSampler / PTSampler sampling\nand include sequences in PottsResult.\n\n## Read first (understand existing code)\n- src/aminx/potts/runner.py lines 1-225  — FULL EXISTING implementation; do NOT recreate it\n- src/aminx/potts/sampling.py  — GibbsSampler, PTSampler (P-07/#1295, P-09/#1297)\n- src/aminx/potts/spec.py  — PottsRunSpec fields (check for n_samples, n_chains, sampler_type)\n- tests/potts/test_runner_smoke.py  — EXISTING tests (225 lines); extend, do NOT replace\n\n## What to add (the missing piece)\nIn run_potts() (runner.py:114-225), after the infer_params + calibration block:\n  if spec.n_samples > 0 (or equivalent field):\n    Instantiate GibbsSampler or PTSampler from aminx.potts.sampling per spec.sampler_type.\n    Run sampling; store sequences in PottsResult.samples.\n  Add the \`samples\` field to PottsResult if not already present.\n\n## Test requirement\nIn tests/potts/test_runner_smoke.py (extend, do not replace):\n  Add a test with n_samples > 0 that asserts PottsResult.samples has shape (n_samples, L)\n  and is dtype int. Create a fixture using \`eqx.tree_serialise_leaves\` + zstd compression\n  to a tempfile to exercise the real .eqx.zst load path (not just a mock).\n\n## Acceptance criteria\n- \`uv run pytest tests/potts/ -x -q 2>&1 | tail -30\` passes (all existing + new tests)\n- New sampling test covers shape and dtype of PottsResult.samples\n- Fixture uses real zstd+eqx.tree_serialise_leaves path (not a bypass)\n- \`uv run pytest tests/ -x -q 2>&1 | tail -30\` — no regressions\n- \`uv run ty check 2>&1 | tail -20\` — clean\n- \`uv run ruff check src/aminx/potts/ 2>&1 | tail -10\` — clean\n\n## Constraints\n- Edit only. runner.py exists — do NOT clobber it. Extend it.\n- Do not touch host/kernel_dispatch.py, averaging.py, or xtrax code.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/aminx\n\n## Checks\nVERIFY: \`uv run pytest tests/potts/ -x -q 2>&1 | tail -30\` — passes (existing + new)\nVERIFY: runner.py was extended, NOT recreated\n        (grep for original content, e.g. \`grep -c "infer_params" src/aminx/potts/runner.py\` — >=2)\nVERIFY: Sampling test covers PottsResult.samples shape (n_samples, L) and dtype int\nVERIFY: Fixture uses real zstd+eqx.tree_serialise_leaves path (grep for "zst" in test file)\nVERIFY: \`uv run pytest tests/ -x -q 2>&1 | tail -30\` — no regressions\nVERIFY: \`uv run ty check 2>&1 | tail -20\` — no new type errors\nVERIFY: \`uv run ruff check src/aminx/potts/ 2>&1 | tail -10\` — clean\n\nPASS if all VERIFY items are satisfied.\nFAIL if runner.py was clobbered, sampling test is missing, or any suite regresses.\n`,
    "worktree"
  );

// ===== TRACK E — Track E — [xtrax T2.5+2.5b] Multi-phase BatchPlanner + axis-injection seam (#1554) =========================
const trackE = () =>
  track(
    "1554",
    "Track E — [xtrax T2.5+2.5b] Multi-phase BatchPlanner + axis-injection seam (#1554)",
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/xtrax\n\n## Objective\nT2.5 + T2.5b: upgrade xtrax BatchPlanner to multi-phase budget planner and add axis-injection\nseam so protein axis identifiers (n_states, n_structures) remain in aminx and are passed in.\n\n## Spec reference (read first)\n/home/marielle/projects/aminx/.praxia/docs/specs/260611_aminx-xtrax-refactor.md\n(T2.5, T2.5b, AC-2, A5, B4 sections)\n\n## Key files\n- src/xtrax/tiling/planner.py — current BatchPlanner\n- src/xtrax/tiling/dispatch.py:77 — heterogeneous-axis guard (uses string 'state', must use param)\n- src/xtrax/tiling/carry.py — CarrySpec, CarryShape\n- src/xtrax/tiling/dedup.py — DedupSpec\n- /home/marielle/projects/aminx/src/aminx/host/plan.py lines 1-50 — aminx planner consumer\n  (line 18 mixes aminx-specific axes; B4 moves these to caller side)\n- /home/marielle/projects/aminx/src/aminx/tiling/axes.py — protein axis definitions\n- /home/marielle/projects/aminx/src/aminx/tiling/carry.py:21 — n_states/n_structures\n\n## Requirements\nAC-2: BatchPlanner supports phases 0, 0b, 1, 2, 3 with parametric budget allocation.\n\nB4 (axis-injection seam):\n  - BatchPlanner accepts axis_specs: list[AxisSpec] and heterogeneous_axes: set[str] as params.\n  - \`grep -rn "n_states\|n_structures" src/xtrax/\` must return 0 results after the fix.\n  - aminx/host/plan.py must pass axis_specs and heterogeneous_axes explicitly to BatchPlanner.\n\nA5: dispatch.py:77 heterogeneous guard must use the heterogeneous_axes param, not 'state'.\n\n## Constraints\n- Edit only on existing files; new files in src/xtrax/tiling/ are allowed.\n- Update aminx/host/plan.py to pass axis_specs and heterogeneous_axes (aminx owns these).\n- Run \`uv run pytest\` from BOTH repos before declaring done.\n- Do not touch aminx/host/kernel_dispatch.py, averaging.py, or _sampling_helper.py.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/xtrax\n\n## Checks\nVERIFY: \`cd /home/marielle/projects/xtrax && uv run pytest -x -q 2>&1 | tail -30\` — passes\nVERIFY: \`cd /home/marielle/projects/aminx && uv run pytest -x -q 2>&1 | tail -30\` — no regression\nVERIFY: B4 xtrax — \`grep -rn "n_states\|n_structures" /home/marielle/projects/xtrax/src/xtrax/ 2>&1\` — ZERO matches\nVERIFY: B4 aminx — \`grep -En "axis_specs=|heterogeneous_axes=" /home/marielle/projects/aminx/src/aminx/host/plan.py 2>&1\`\n        — must show these params passed to BatchPlanner (confirming aminx caller doesn't re-hardcode)\nVERIFY: BatchPlanner accepts axis_specs and heterogeneous_axes parameters\nVERIFY: dispatch.py:77 uses heterogeneous_axes param (not string 'state')\nVERIFY: \`cd /home/marielle/projects/xtrax && uv run ty check 2>&1 | tail -20\` — no new type errors\n\nPASS if all VERIFY items are satisfied.\nFAIL if protein axis names remain in xtrax, or aminx re-hardcodes them, or either suite regresses.\n`,
    "worktree"
  );

// ---- orchestrate: sequential writing chain || read-only research ----------
log("Sprint 260616: Host dispatch gaps + P-08 run loop + T2.5 BatchPlanner: writing chain (A -> B, sequential) || research (C, D, E, read-only)");
const [writing, resC, resD, resE] = await Promise.all([
  (async () => {
    const a = await trackA();
    const b = await trackB();
    return { a, b };
  })(),
  trackC(),
  trackD(),
  trackE(),
]);

// Phase-5: Integrate worktree branches
{
  phase("Integrate");
  const intManifestPath = `.praxia/worktree_manifests/${TASK_ID}.json`;
  // Step 1: write manifest via CLI
  const _intCli = await agent(
    `Run shell command: praxia worktree integrate --sprint-id ${TASK_ID} and report the manifest path written.`,
    { label: "worktree:integrate-cli" }
  );
  // Step 2: integrator analysis
  const intReport = await agent(
    `task_id: ${TASK_ID}. Analyze the worktree integration manifest and report any merge conflicts.`,
    { label: "integrator", phase: "Integrate" }
  );
  // Step 3: merge executor if conflicts
  if (intReport && typeof intReport === "string" && intReport.includes("conflict")) {
    await agent(
      `Resolve merge conflicts: read the manifest at ${intManifestPath}, then run git merge for each branch listed. Use git merge --no-ff for each branch in dependency order. Report merged SHAs.`,
      { label: "fixer:merge", phase: "Integrate" }
    );
  }
}

return {
  task_id: TASK_ID,
  sprint_id: 1,
  verdicts: {
    "1856": writing.a,
    "1857": writing.b
  },
  research_1858: resC,
  research_1299: resD,
  research_1554: resE
};
