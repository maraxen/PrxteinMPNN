// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 260616_host-gaps-potts-runloop   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain () runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time. Only the read-only
//   research/concurrent tracks (A,B,C,D,E) run concurrently.

export const meta = {
  name: "260616_host-gaps-potts-runloop",
  description: "Close three host dispatch wiring gaps (#1856 chain_mask, #1857 fixed_mask double-apply, #1858 STE del-kill), implement P-08 end-to-end Potts run loop, and land T2.5 multi-phase BatchPlanner in xtrax.",
  phases: [
    { title: "Track A — [host-dispatch] chain_mask not forwarded for sidechain conditioning (#1856)" },
    { title: "Track B — [host-dispatch] fixed_mask double-applied in _prepare_fixed_controls (#1857)" },
    { title: "Track C — [host-dispatch] STE killed by del sampling_strategy in averaging legacy path (#1858)" },
    { title: "Track D — P-08: aminx.potts end-to-end inference run loop (#1299)" },
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

// ===== TRACK A — Track A — [host-dispatch] chain_mask not forwarded for sidechain conditioning (#1856) =========================
const trackA = () =>
  track(
    "1856",
    "Track A — [host-dispatch] chain_mask not forwarded for sidechain conditioning (#1856)",
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/aminx\n\n## Objective\nFix the routing gap for \`chain_mask\` in the LigandMPNN sidechain conditioning path so that\nStage 3 bathos cells can use sidechain conditioning through the spec+CLI layer.\n\n## Key files\n- src/aminx/host/_sampling_helper.py lines 247-352  (_prepare_ligand_context — returns\n  chain_mask at line 343; all code paths return this key)\n- src/aminx/host/kernel_dispatch.py lines 145-160  (_prepare_ligand_context call)\n- src/aminx/host/kernel_dispatch.py lines 191-221  (first build_inference_bundle call — atom_37\n  and atom_37_mask forwarded at lines 211-216; chain_mask is NOT forwarded)\n- src/aminx/host/kernel_dispatch.py lines 257-295, 349-387, 421-457  (three other call sites,\n  same pattern — atom_37/atom_37_mask forwarded, chain_mask missing)\n- src/aminx/inference/bundle_builder.py lines 36-63  (build_inference_bundle signature —\n  no chain_mask param currently)\n\n## Diagnosis\natom_37 and atom_37_mask are already being forwarded at all four call sites (confirmed).\nchain_mask is populated by _prepare_ligand_context when sidechain_conditioning=True but is\nnever passed anywhere. build_inference_bundle has no chain_mask parameter.\nDetermine the correct route BEFORE implementing:\n  (a) Add chain_mask to build_inference_bundle and wire through to the model, OR\n  (b) Route chain_mask as a contribution to fixed_mask (mask all chain_mask==1 positions as\n      fixed), OR\n  (c) Another mechanism — check LigandMPNN usage to confirm intended semantic.\n\n## Implementation\n- Edit only. Do not Write to existing files.\n- Apply the routing consistently at ALL FOUR build_inference_bundle call sites.\n- If chain_mask needs a new param on build_inference_bundle, add it with type\n  \`jax.Array | None = None\` to bundle_builder.py line 63 area.\n- Add a test in tests/host/test_kernel_dispatch.py (or tests/inference/test_bundle_builder.py)\n  asserting that a non-None chain_mask reaches the bundle when sidechain_conditioning=True.\n- Run \`uv run pytest tests/ -x -q 2>&1 | tail -30\` before declaring done.\n\n## Constraints\n- Do not touch averaging.py, _prepare_fixed_controls, or Potts code.\n- One file per focused edit (kernel_dispatch.py changes first, then bundle_builder.py if needed).\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/aminx\n\n## Checks\nVERIFY: \`uv run pytest tests/ -x -q 2>&1 | tail -30\` — all existing tests pass\nVERIFY: New test asserts chain_mask reaches the bundle when sidechain_conditioning=True\nVERIFY: ALL FOUR build_inference_bundle call sites in kernel_dispatch.py include chain_mask\n        (grep -n "build_inference_bundle" kernel_dispatch.py — count call sites; each must have chain_mask)\nVERIFY: atom_37/atom_37_mask forwarding still present at same call sites (no regression)\nVERIFY: \`uv run ty check 2>&1 | tail -20\` — no new type errors\n\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item fails or is untestable as written.\n`,
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

// ===== TRACK C — Track C — [host-dispatch] STE killed by del sampling_strategy in averaging legacy path (#1858) =========================
const trackC = () =>
  track(
    "1858",
    "Track C — [host-dispatch] STE killed by del sampling_strategy in averaging legacy path (#1858)",
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/aminx\n\n## Objective\nFix the STE path in \`_make_encoding_sampling_split_fn_legacy\` so that\nsampling_strategy="straight_through" reaches \`_ste_optimize_sequence_legacy\`.\n\n## Key file\nsrc/aminx/host/averaging.py\n\n## Bug\nIn _make_encoding_sampling_split_fn_legacy (line ~315):\n  Line 362: \`del sampling_strategy, decoding_order_fn  # Currently only temperature sampling supported\`\n  This kills sampling_strategy before the returned sample_fn closure can use it.\n  Result: regardless of what the caller passes, the split-fn path always runs temperature sampling.\n\nThe WORKING STE dispatch is in _sample_using_averaged_features_legacy (line ~408):\n  Line 453: \`if sampling_strategy == "straight_through": return _ste_optimize_sequence_legacy(...)\`\n  But this is only reachable when calling _sample_using_averaged_features_legacy directly,\n  not via the split-function path.\n\n## Implementation\n- Edit only. Do not Write to existing files.\n- Remove \`del sampling_strategy\` from _make_encoding_sampling_split_fn_legacy (line 362).\n  Keep \`del decoding_order_fn\` only if decoding_order_fn truly is unused.\n- Wire sampling_strategy into the returned sample_fn closure: branch on\n  sampling_strategy == "straight_through" → call _ste_optimize_sequence_legacy(...),\n  else → temperature path.\n- The closure must capture sampling_strategy, iterations, and learning_rate from outer scope\n  (add these as parameters to the outer function if not already present).\n- Add a test in tests/host/test_averaging.py:\n  (a) split fn with sampling_strategy="straight_through" → sample_fn routes to STE path\n  (b) split fn with sampling_strategy="temperature" → sample_fn routes to temperature path\n- Run \`uv run pytest tests/ -x -q 2>&1 | tail -30\` before declaring done.\n\n## Constraints\n- Focus on averaging.py only.\n- Do not touch kernel_dispatch.py, _sampling_helper.py, or Potts code.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/aminx\n\n## Checks\nVERIFY: \`uv run pytest tests/ -x -q 2>&1 | tail -30\` — all existing tests pass\nVERIFY: \`grep -n "del sampling_strategy" src/aminx/host/averaging.py\` — ZERO matches in\n        _make_encoding_sampling_split_fn_legacy\nVERIFY: "straight_through" appears in BOTH the split-fn closure AND\n        _sample_using_averaged_features_legacy (grep "straight_through" averaging.py — 2+ hits)\nVERIFY: New test covers both STE and temperature dispatch paths through the split-fn\nVERIFY: \`uv run ty check 2>&1 | tail -20\` — no new type errors\n\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item fails or is untestable as written.\n`,
    "worktree"
  );

// ===== TRACK D — Track D — P-08: aminx.potts end-to-end inference run loop (#1299) =========================
const trackD = () =>
  track(
    "1299",
    "Track D — P-08: aminx.potts end-to-end inference run loop (#1299)",
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/aminx\n\n## Objective\nImplement P-08: end-to-end Potts inference run loop in aminx.potts. First callable Potts\ninference path; unlocks Stage 3 bathos experiments.\n\n## Prior work (read first)\n- src/aminx/potts/model.py — PottsModel, __call__, joint_energy (P-05/#1296, vectorised)\n- src/aminx/potts/sampling.py — GibbsSampler, PTSampler (P-07/#1295, P-09/#1297)\n- src/aminx/io/weights.py — weight loading (eqx.deserialise_filtered / .eqx.zst)\n- src/aminx/run/spec.py lines 104-240 — RunSpec, PlannerTopology, topology_hash (RS-2/#1621)\n- src/aminx/run/specs.py — SamplingSpecification (pattern reference)\n\n## What to implement\nCreate \`src/aminx/potts/runner.py\` with a \`PottsRunner\` class or \`run_potts\` function:\n  1. Accept PottsRunSpec or (PottsModel + config dict); load weights from potts_<id>.eqx.zst\n     via eqx.deserialise_filtered (existing weights.py pattern).\n  2. Optionally load caliby_<id>.eqx.zst; use identity caliby if absent.\n  3. Build Protein / GeometryBundle from a PDB input — reuse existing aminx.io.pdb loaders.\n  4. Run PottsModel forward: marginals = potts_model(geom_bundle).\n  5. Optionally sample: if n_samples > 0, call GibbsSampler or PTSampler.\n  6. Return dict {"marginals": ..., "log_probs": ..., "samples": ...} or a dataclass.\n\nDesign decisions (locked):\n- eqx.Module pattern; no dataclasses for model state.\n- Weight loading: eqx.deserialise_filtered only.\n- Structure input: accept Protein or path to PDB; reuse existing loaders.\n- Output: json-serialisable scalars/arrays matching aminx run output conventions.\n- Smoke test gate: n<=50 residues, identity caliby, no sampling.\n\n## Acceptance criteria\n- \`uv run pytest tests/potts/test_runner.py -x -q 2>&1 | tail -30\` passes with smoke test:\n    (a) weight load from fixture .eqx.zst (or mock)\n    (b) forward pass marginals shape == (L, 20)\n    (c) log_probs are finite (jnp.isfinite check)\n- \`uv run pytest tests/ -x -q 2>&1 | tail -30\` — no regressions\n- \`uv run ty check 2>&1 | tail -20\` — clean\n- \`uv run ruff check src/aminx/potts/runner.py 2>&1\` — clean\n\n## Constraints\n- New file (runner.py) allowed; Edit only on existing files.\n- Do not add sampling until the non-sampling smoke test passes.\n- Do not touch host/kernel_dispatch.py, averaging.py, or xtrax code.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/aminx\n\n## Checks\nVERIFY: \`uv run pytest tests/potts/test_runner.py -x -q 2>&1 | tail -30\` — passes\nVERIFY: Smoke test covers: (a) weight load, (b) marginals shape (L, 20), (c) log_probs finite\nVERIFY: \`uv run pytest tests/ -x -q 2>&1 | tail -30\` — no regressions\nVERIFY: \`uv run ty check 2>&1 | tail -20\` — no new type errors\nVERIFY: \`uv run ruff check src/aminx/potts/ 2>&1 | tail -10\` — clean\n\nPASS if all VERIFY items are satisfied.\nFAIL if smoke test is missing, trivially passing, or any suite regresses.\n`,
    "worktree"
  );

// ===== TRACK E — Track E — [xtrax T2.5+2.5b] Multi-phase BatchPlanner + axis-injection seam (#1554) =========================
const trackE = () =>
  track(
    "1554",
    "Track E — [xtrax T2.5+2.5b] Multi-phase BatchPlanner + axis-injection seam (#1554)",
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/xtrax\n\n## Objective\nT2.5 + T2.5b: upgrade xtrax BatchPlanner to multi-phase budget planner and add axis-injection\nseam so protein axis identifiers (n_states, n_structures) remain in aminx and are passed in.\n\n## Spec reference (read first)\n/home/marielle/projects/aminx/.praxia/docs/specs/260611_aminx-xtrax-refactor.md\n(T2.5, T2.5b, AC-2, A5, B4 sections)\n\n## Key files\n- src/xtrax/tiling/planner.py — current BatchPlanner\n- src/xtrax/tiling/dispatch.py:77 — heterogeneous-axis guard (uses string 'state', must use param)\n- src/xtrax/tiling/carry.py — CarrySpec, CarryShape\n- src/xtrax/tiling/dedup.py — DedupSpec\n- /home/marielle/projects/aminx/src/aminx/host/plan.py lines 1-50 — aminx planner consumer\n  (line 18 mixes aminx-specific axes; B4 moves these to caller side)\n- /home/marielle/projects/aminx/src/aminx/tiling/axes.py — protein axis definitions\n- /home/marielle/projects/aminx/src/aminx/tiling/carry.py:21 — n_states/n_structures\n\n## Requirements\nAC-2: BatchPlanner supports phases 0, 0b, 1, 2, 3 with parametric budget allocation.\n\nB4 (axis-injection seam):\n  - BatchPlanner accepts axis_specs: list[AxisSpec] and heterogeneous_axes: set[str] as params.\n  - \`grep -rn "n_states\|n_structures" src/xtrax/\` must return 0 results after the fix.\n  - aminx/host/plan.py must pass axis_specs and heterogeneous_axes explicitly to BatchPlanner.\n\nA5: dispatch.py:77 heterogeneous guard must use the heterogeneous_axes param, not 'state'.\n\n## Constraints\n- Edit only on existing files; new files in src/xtrax/tiling/ are allowed.\n- Update aminx/host/plan.py to pass axis_specs and heterogeneous_axes (aminx owns these).\n- Run \`uv run pytest\` from BOTH repos before declaring done.\n- Do not touch aminx/host/kernel_dispatch.py, averaging.py, or _sampling_helper.py.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260616_host-gaps-potts-runloop\nworkspace: /home/marielle/projects/xtrax\n\n## Checks\nVERIFY: \`cd /home/marielle/projects/xtrax && uv run pytest -x -q 2>&1 | tail -30\` — passes\nVERIFY: \`cd /home/marielle/projects/aminx && uv run pytest -x -q 2>&1 | tail -30\` — no regression\nVERIFY: B4 seam — \`grep -rn "n_states\|n_structures" /home/marielle/projects/xtrax/src/xtrax/ 2>&1\` — ZERO matches\nVERIFY: BatchPlanner accepts axis_specs and heterogeneous_axes parameters\nVERIFY: aminx/host/plan.py passes these to BatchPlanner explicitly (not hard-coded)\nVERIFY: dispatch.py:77 uses heterogeneous_axes param (not string 'state')\nVERIFY: \`cd /home/marielle/projects/xtrax && uv run ty check 2>&1 | tail -20\` — no new type errors\n\nPASS if all VERIFY items are satisfied.\nFAIL if protein axis names remain in xtrax or either test suite regresses.\n`,
    "worktree"
  );

// ---- orchestrate: sequential writing chain || read-only research ----------
log("Sprint 260616: Host dispatch gaps + P-08 run loop + T2.5 BatchPlanner: writing chain (, sequential) || research (A, B, C, D, E, read-only)");
const [writing, resA, resB, resC, resD, resE] = await Promise.all([
  (async () => {
    return {  };
  })(),
  trackA(),
  trackB(),
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

  },
  research_1856: resA,
  research_1857: resB,
  research_1858: resC,
  research_1299: resD,
  research_1554: resE
};
