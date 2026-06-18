// Sprint 2 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/260618_rs7b-averaged-scoring.toml
// Regenerate: praxia dw emit-sprint 260618_rs7b-averaged-scoring.toml
// task_id: 260618_rs7b-averaged-scoring   sprint_id: 2
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (A) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.

export const meta = {
  name: "260618_rs7b-averaged-scoring",
  description: "Route score() through the InferencePlan averaging topology when average_node_features=True, sharing the sampling encode->fuse->decode path. Interpretation (A): mean node/edge features THEN score (one decode, one NLL). Backlog #2252 (child of RS-7 #1626 / EPIC #1541). Single sequential track: shared-kernel edits serialize, no fan-out.",
  phases: [
    { title: "RS-7b: encode/score_from_encoding split + score_averaged + runner branch + parity oracle" },
  ],
};

const TASK_ID = "260618_rs7b-averaged-scoring";
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

// Shared context for the writing tracks (from recon, task 260618_rs7b-averaged-scoring).
const EMITTER_CTX = `RS-7b is the second half of RS-7 (#1626), the primary AC-RS-7 clause. RS-7a (#2251) is DONE/merged.\nDesign doc (feasibility HIGH, READ IT FIRST): .praxia/docs/specs/260618_rs7b-averaged-scoring-design.md\nParent spec: .praxia/docs/specs/260611_runspec-unification.md (AC-RS-7).\n\nVERIFIED ANCHORS (re-confirmed against live code this iteration):\n- src/aminx/inference/score_conditional.py:15-44 = kernel(): encode_fn(bundle,k_enc,config)->enc\n  at :32-33, then ConditionalDecode(...)(key,enc,bundle,config,stage_set) at :37-44. Clean seam.\n- src/aminx/scoring/score.py:114 kernel call; :117-123 NLL formula\n  (log_softmax[...,:20] -> -(seq[...,:20]*logp).sum(-1) -> masked sum / mask_sum) = _nll_from_logits.\n- src/aminx/host/runner.py:312-314 HDF5 NotImplementedError guard (off-ramp pattern to mirror);\n  :316-322 make_score_fn import + score_fn=make_score_fn(model) = the branch point;\n  :355-407 per-(structure,sequence) loop — UNCHANGED.\n- host/averaging.py:48-57 ArithmeticMeanEncodingFusion consumes/produces the SAME EncoderOutput type.\n- host/kernel_dispatch.py:309-328 sampling unified path B (leading-D stack exemplar at :321-325).\n- run/plan.py:652-662 make_inference_plan wires ArithmeticMeanEncodingFusion.\n- run/specs.py:197 backbone_noise default (0.0,); :406 score docstring to update.\n\nINTERPRETATION (HARD): (A) mean features then score. NOT (B) mean of D scalar NLLs (log_softmax is\nnonlinear, they differ). Tests MUST assert (A) and guard against (B).\n\nOFF-RAMP (if D=1 bit-equal parity Invariant-1 is unreachable): land only the RS7b-1/-1b extractions\n(behaviour-preserving), and in score()/runner raise NotImplementedError("averaged-feature scoring not\nyet supported; omit average_node_features") mirroring the HDF5 guard at runner.py:312-314, plus the\ndocstring note. This satisfies AC-RS-7 'explicitly documented' as a degraded-but-clean close.\n`;

// ---- per-track stage helpers ---------------------------------------------
const fixer = (prompt, label, phaseName, isolation = null, context = null) => {
  const opts = { agentType: "fixer", label, phase: phaseName };
  if (isolation) opts.isolation = isolation;
  if (context) Object.assign(opts, context);
  const boilerplate = `

COMMIT SCOPE: stage ONLY the files you edited with explicit paths (git add <path> ...).
NEVER use \`git add -A\` or \`git add .\` — under concurrency they sweep other work into your commit. Commit your own changes yourself (including any NEW files you create).

When done, end your message with 'verdict: done' on its own line.`;
  return agent(`${prompt}${boilerplate}`, opts);
};

const reviewer = (itemId, prompt, label, phaseName, isolation = null, context = null) => {
  const opts = { agentType: "reviewer", label, phase: phaseName, schema: VERDICT_SCHEMA };
  if (isolation) opts.isolation = isolation;
  if (context) Object.assign(opts, context);
  return agent(prompt, opts);
};

// Sequential implement->review with bounded NEEDS_WORK repair cycles.
async function track(itemId, phaseName, fixerPrompt, reviewerPrompt, isolation = null, context = null, reconExecutorKey = 'recon') {
  log(`[${itemId}] implement`);
  const _recon = await agent(
    `task_id: ${TASK_ID}. Run recon for: ${phaseName}. Task: ${fixerPrompt.slice(0, 500)}`,
    { label: 'recon:' + itemId, phase: phaseName, agentType: reconExecutorKey }
  );
  const _fixerPromptWithRecon = 'RECON FINDINGS:\n' + (_recon || '(no findings)') + '\n\n---\n\n' + fixerPrompt;
  const _preFixerHead = (await agent(
    'Run: git rev-parse HEAD. Return only the 40-character SHA, nothing else.',
    { label: 'head:' + itemId, phase: phaseName }
  )).trim().match(/[0-9a-f]{40}/)?.[0] || 'unknown';
  await fixer(_fixerPromptWithRecon, `fix:${itemId}`, phaseName, isolation, context);
  let _effRp = reviewerPrompt;
  if (isolation === 'worktree') {
    const _branch = (await agent(
      'Run: git branch --show-current. Return only the branch name, nothing else.',
      { label: 'branch:' + itemId, phase: phaseName }
    )).trim().match(/[a-zA-Z0-9_./-]+/)?.[0] || '';
    _effRp = reviewerPrompt + '\n\nIMPORTANT -- WORKTREE BRANCH: The fixer committed to branch ' + _branch + '. Before evaluating, run: git log main...' + _branch + ' --oneline && git diff main...' + _branch + '. Do NOT evaluate main HEAD. Review ONLY the commits on branch ' + _branch + '.';
  }
  _effRp = _effRp + '\n\nIMPORTANT — NO-COMMIT DETECTION: Before evaluating anything, run: ' +
    'git log ' + _preFixerHead + '..HEAD --oneline' +
    '\n' +
    'If the output is EMPTY (no commits since ' + _preFixerHead + '), return FAIL immediately with issue: ' +
    '"fixer made no commit — git log shows no new commits since pre-fixer HEAD ' + _preFixerHead + '". ' +
    'Do NOT evaluate file content if no commit was made.';
  let verdict = await reviewer(itemId, _effRp, `review:${itemId}`, phaseName, isolation, context);
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
    verdict = await reviewer(itemId, _effRp, `review:${itemId}:re:${retry}`, phaseName, isolation, context);
  }
  return verdict;
}

// ===== TRACK A — RS-7b: encode/score_from_encoding split + score_averaged + runner branch + parity oracle =========================
const trackA = () =>
  track(
    "2252",
    "RS-7b: encode/score_from_encoding split + score_averaged + runner branch + parity oracle",
    `task_id: ${TASK_ID}. task_id: 260618_autonomous-loop\nREAD FIRST: .praxia/docs/specs/260618_rs7b-averaged-scoring-design.md (it has all signatures + invariants).\nEdit only on existing production files. Write ONLY for the new test file. Run the gates between steps.\nKeep behaviour byte-identical for existing non-averaged callers until the final wiring.\n\nSTEP 1 (RS7b-1, score_conditional.py) — Split the kernel into encode + decode halves, NO behaviour change.\n  In src/aminx/inference/score_conditional.py extract two helpers from kernel() (currently :15-44):\n    def encode(model, key, bundle, config) -> EncoderOutput:\n        # body = current :32-33 (make_encode_fn(model, use_rolling_state=False); enc = encode_fn(bundle,key,config))\n    def score_from_encoding(model, key, enc, bundle, config, stage_set) -> Logits:\n        # body = current :35-44 (ConditionalDecode(model=..., state_iterator=VmapIterator())(key=...,enc=enc,...))\n  Rewrite kernel(...) to: k_enc,k_dec = jax.random.split(prng_key); enc = encode(model,k_enc,bundle,config);\n    return score_from_encoding(model,k_dec,enc,bundle,config,stage_set). Existing scoring tests must stay green.\n  GATE: uv run pytest tests/scoring -q  (and any test importing score_conditional) — all PASS unchanged.\n\nSTEP 2 (RS7b-1b, scoring/score.py) — Extract the single-source NLL helper.\n  In src/aminx/scoring/score.py extract lines 117-123 into a module-level function:\n    def _nll_from_logits(logits, seq_one_hot, mask) -> jax.Array:\n        log_probability = jax.nn.log_softmax(logits, axis=-1)[..., :20]\n        score = -(seq_one_hot[..., :20] * log_probability).sum(-1)\n        mask_flat = _residue_mask_for_scoring(mask)\n        return (score * mask_flat).sum(-1) / (mask_flat.sum() + SCORE_EPS)\n  Replace the inline body in score_sequence with a call to _nll_from_logits(logits, sequence, mask)\n  so masked_score_sum/mask_sum is computed in exactly one place. score_sequence's return tuple unchanged.\n  GATE: uv run pytest tests/scoring -q — all PASS (NLL numerics identical).\n\nSTEP 3 (RS7b-2, score_conditional.py) — Add the averaged path. After score_from_encoding add:\n    def _stack_encoder_outputs(encs) -> EncoderOutput:\n        # stack node_features/edge_features along a new leading D axis (mirror kernel_dispatch.py:321-325);\n        # neighbor_indices/mask taken from encs[0] (geometry is noise-invariant — assert identical if cheap).\n    def score_averaged(model, key, bundles_per_noise, config, stage_set, encoding_fusion) -> Logits:\n        # encode each bundle (split a key per D); stacked = _stack_encoder_outputs([...]);\n        # fused = encoding_fusion(stacked)  # D->1, type-preserving (host/averaging.py ArithmeticMeanEncodingFusion);\n        # return score_from_encoding(model, key, fused, bundles_per_noise[0], config, stage_set)\n  Do NOT make D a static jit arg (risk R2: recompile-per-D). Build the D bundles in host Python (concrete arrays).\n  GATE: uv run ty check && uv run ruff check . — clean (no runtime test yet).\n\nSTEP 4 (RS7b-3, host/runner.py) — Branch the runner. At the make_score_fn site (:316-322), add:\n    if getattr(spec, 'average_node_features', False):\n        plan = make_inference_plan(model, spec)          # wires ArithmeticMeanEncodingFusion (plan.py:652-662)\n        score_fn = _make_averaged_score_fn(plan, spec)    # NEW closure (below)\n    else:\n        score_fn = make_score_fn(model)                   # unchanged\n  Add _make_averaged_score_fn(plan, spec): builds D bundles from spec.backbone_noise (default (0.0,),\n  specs.py:197) — assert all D bundles' CONDITIONING fields are identical (only backbone_noise varies, risk R3),\n  calls score_averaged(...), then nll via _nll_from_logits. The per-(structure,sequence) loop (:355-407) is UNCHANGED;\n  it still calls score_fn(subkey, seq_one_hot, coords, mask, residue_index, chain_index, ...).\n  GATE: uv run pytest tests/scoring tests/host -q — existing (non-averaged) paths PASS.\n\nSTEP 5 (RS7b-4, NEW tests/scoring/test_averaged_parity.py) — BATHOS parity oracle. Write Invariants 1-3:\n  - Inv-3 (fusion math sanity, pure, ~30s): stacked node_features = stack([ones*1, ones*3]) -> fused == 2.0;\n    neighbor_indices/mask == stack[0]. (Write/run this FIRST — it needs no model.)\n  - Inv-1 (D=1 degenerate identity, HARD GATE, bit-equal): a spec with backbone_noise=(0.0,) and\n    average_node_features=True -> averaged-path logits AND nll MUST equal the current non-averaged score()\n    on the same (structure, sequence, key). Use jnp.array_equal; relax to atol=0, rtol<=1e-7 ONLY if XLA\n    layout forces it (else treat a mismatch as a real bug, not a tolerance issue).\n  - Inv-2 (A vs B, D=2 distinct noise): nll_A == hand-computed (encode both, mean features, decode, NLL);\n    nll_A != nll_B (mean of two independent scalar scores) on a non-degenerate fixture.\n  GATE: uv run pytest tests/scoring/test_averaged_parity.py -q — Inv-1/2/3 PASS.\n  *** If Inv-1 (D=1 bit-equal) CANNOT be made to pass after a genuine attempt: APPLY THE OFF-RAMP ***\n  (raise NotImplementedError in the averaged branch mirroring runner.py:312-314 + docstring), keep\n  Steps 1-2 (behaviour-preserving extractions), and report clearly that the off-ramp was taken and why.\n\nSTEP 6 (RS7b-5) — Invariant-4 golden + docstring. Add a golden test: a small real structure + seq, D=3,\n  pin the scalar NLL (record the produced value). Update the score docstring (run/specs.py:406 and/or\n  scoring/score.py) to document averaged-feature scoring (Interpretation A) closing AC-RS-7.\n  GATE: uv run pytest tests/scoring -q && uv run ty check && uv run ruff check . && uv run ruff format .\n\nSTEP 7 — Commit. git add -A && git commit -m "feat(RS-7b): averaged-topology scoring via InferencePlan encode/fuse/score (#2252)"\n  (If the off-ramp was taken, use: "feat(RS-7b): document averaged-scoring off-ramp (NotImplementedError) + kernel split (#2252)".)\n  Report the commit SHA, exact gate outputs, and whether the full path or the off-ramp landed.\n\n\n${EMITTER_CTX}

IMPORTANT — GITIGNORE IN WORKTREE: This worktree has a .gitignore that blocks the .claude/ directory. If any output files are under .claude/ (e.g., .claude/skills/, .claude/workflows/), you MUST use \`git add -f <path>\` to force-add them. A plain \`git add\` or \`git add .\` will silently skip these files. Always run \`git status\` after staging to confirm they appear as 'Changes to be committed'.\n\n`,
    `task_id: ${TASK_ID}. task_id: 260618_autonomous-loop\nVERIFY checklist for RS-7b (#2252). Run every command; report PASS/FAIL per item with evidence (paste output).\n1. uv run pytest tests/scoring -q  AND  uv run pytest tests/host -q — all PASS, no regression in existing scoring/host tests.\n2. INVARIANT-1 (HARD GATE): tests/scoring/test_averaged_parity.py contains a D=1 (backbone_noise=(0.0,))\n   identity test asserting averaged-path logits AND nll equal the non-averaged score() bit-for-bit\n   (jnp.array_equal, or atol=0/rtol<=1e-7 with a justification comment). It PASSES.\n   [If the OFF-RAMP was taken instead: verify score()/runner raises NotImplementedError for averaged mode\n    with a message mirroring runner.py:312-314, AND Steps 1-2 extractions still landed (kernel split +\n    _nll_from_logits) with all existing tests green. That is an ACCEPTABLE PASS.]\n3. INTERPRETATION (A) guard: the suite asserts nll_A == hand-computed mean-features-then-decode AND\n   nll_A != nll_B (mean-of-scores) on a non-degenerate D=2 fixture. Both assertions present and PASS.\n4. Inv-3 fusion sanity present (stack([ones*1,ones*3]) -> 2.0; neighbor_indices/mask from stack[0]) and PASSES.\n5. SINGLE-SOURCE NLL: scoring/score.py has exactly one _nll_from_logits and score_sequence calls it\n   (no duplicated log_softmax/masked-sum formula). score_from_encoding/encode exist in score_conditional.py\n   and kernel() composes them (existing non-averaged behaviour byte-identical).\n6. NO STATIC-D RECOMPILE: D is not a static jit arg; D bundles built in host Python (risk R2). Conditioning\n   fields asserted identical across the D bundles (risk R3).\n7. uv run ty check clean; uv run ruff check . clean; uv run ruff format . no diff.\n8. SCOPE: git show --stat HEAD touches only score_conditional.py, scoring/score.py, host/runner.py,\n   run/specs.py (docstring), and tests/scoring/test_averaged_parity.py (+ golden). No unrelated files.\nPASS only if all hold (item 2 satisfied by either the full path OR the documented off-ramp). Else FAIL with the specific item.\n\n\n${EMITTER_CTX}`,
    "worktree",
    { track_id: "a", sprint_id: TASK_ID }
    , "recon"
  );

// ---- orchestrate: writing chain (A, sequential) ----
const _gitBase = (await agent(
  'Run: git rev-parse HEAD. Return only the 40-character SHA, nothing else.',
  { label: 'git-base', phase: 'Execute' }
)).trim();
log("RS-7b: averaged-topology scoring (average_node_features -> InferencePlan encode/fuse/score): writing chain (A, sequential)");
const a = await trackA();
// Log track completion via transduction_log
await agent(
  `task_id: ${TASK_ID}. ` +
  `Call mcp__praxia__transduction_log(action='append_audit', payload={` +
  `audit_id: '${TASK_ID}-track-a', ` +
  `task_id: '${TASK_ID}', ` +
  `verdict: '${a?.verdict ?? 'unknown'}', ` +
  `issues: []` +
  `}). ` +
  `This is a telemetry call — execute it and return 'logged'.`,
  { label: 'transduction-log:track-a', phase: 'Telemetry' }
);


// Phase-5: Integrate worktree branches
{
  phase("Integrate");
  const intManifestPath = `.praxia/worktree_manifests/${TASK_ID}.json`;
  await agent(
    `task_id: ${TASK_ID}. Integrate Claude Code worktree branches:

Step 1: Discover worktree branches
  Run: git branch --list 'wt-*' | sort
  This lists all branches created by the CC worktree infrastructure for this sprint.

Step 2: For each discovered branch, check for new commits
  Run: git log HEAD..<branch> --oneline
  If the output is non-empty, the branch has commits to integrate.

Step 3: DO NOT MERGE. Per mission.md, merge-to-main is a MANUAL human action — never autonomous.
  For each branch with new commits, report the branch name and `git log main..<branch> --oneline`.
  Leave ALL branches unmerged on disk for the human to merge + push manually. Run NO git merge.

Step 4: Write manifest JSON
  Write to: ${intManifestPath}
  Schema:
  {
    "sprint_id": "${TASK_ID}",
    "generated_at": "<ISO8601 timestamp>",
    "branches_discovered": ["wt-..."],
    "branches_merged": [],
    "branches_pending_manual_merge": ["wt-..."]
  }
  If no wt-* branches exist, write the manifest with empty arrays (not an error).

Report: list of branches PENDING MANUAL MERGE and the manifest path written. Do not merge.`,
    { label: "worktree:integrate", phase: "Integrate" }
  );
}

phase("Audit");
const _auditVerdict = await agent(
  'task_id: ' + TASK_ID + '. The fixer committed to a worktree branch that is NOT merged to main (per mission.md, merge is manual). Discover it: run `git branch --list "wt-*"`. For the branch carrying new commits, run `git log main..<branch> --oneline` and `git diff main...<branch>`. Review all changed files against the RS-7b sprint requirements (design: .praxia/docs/specs/260618_rs7b-averaged-scoring-design.md; AC-RS-7). Verify: encode/score_from_encoding split is behaviour-preserving, single-source _nll_from_logits, the D=1 bit-equal parity test (Invariant 1) OR the documented off-ramp, and (A) mean-features-then-score. Return PASS or FAIL with findings. Do NOT merge anything.',
  { label: 'auditor', phase: 'Audit', agentType: 'auditor' }
);

// Log sprint completion via transduction_log
await agent(
  `task_id: ${TASK_ID}. ` +
  `Call mcp__praxia__transduction_log(action='append_audit', payload={` +
  `audit_id: '${TASK_ID}-final', ` +
  `task_id: '${TASK_ID}', ` +
  `verdict: '${_auditVerdict?.verdict ?? 'unknown'}', ` +
  `issues: []` +
  `}). ` +
  `This is a telemetry call — execute it and return 'logged'.`,
  { label: 'transduction-log:final', phase: 'Telemetry' }
);

return {
  task_id: TASK_ID,
  sprint_id: 2,
  verdicts: {
    "2252": a
  },
  audit: _auditVerdict,
};
