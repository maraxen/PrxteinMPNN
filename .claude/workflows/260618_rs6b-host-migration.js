// Sprint 3 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/260618_rs6b-host-migration.toml
// Regenerate: praxia dw emit-sprint 260618_rs6b-host-migration.toml
// task_id: 260618_rs6b-host-migration   sprint_id: 3
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (A) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.

export const meta = {
  name: "260618_rs6b-host-migration",
  description: "Second half of RS-6 (#1625). Migrate ~56 flat spec.<generic-field> reads across 7 host hot-path files to spec.run_spec.<subconfig>.<field> (the canonical post-RS-6a path), add an ast-grep rule banning flat reads in host/*, exempt prep.py. SEMANTICS-PRESERVING only. Backlog #2226 (child of RS-6 #1625 / EPIC #1541). RS-6a (#2225) DONE+merged; RS-7b (#2252) DONE+merged (480b1fa).",
  phases: [
    { title: "RS-6b: host caller run_spec.* migration + ast-grep flat-field gate" },
  ],
};

const TASK_ID = "260618_rs6b-host-migration";
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

// Shared context for the writing tracks (from recon, task 260618_rs6b-host-migration).
const EMITTER_CTX = `RS-6b migrates host CALLERS to read generic fields from spec.run_spec.<subconfig>.<field>\n(RS-6a already added the sub-configs + build_run_spec wiring). Exemplar (correct pattern,\nalready in place): src/aminx/host/streaming.py:310 -> spec.run_spec.multistate.n_states.\nMigration map: .praxia/docs/plans/260614_runspec-migration-map.md (section 4).\nRecon inventory: transduction recon_id 260618_rs6b_inventory.\n\nFIELD -> SUBCONFIG MAP (the ONLY fields to migrate):\n  run_spec.sampling: backbone_noise, temperature, num_samples, random_seed, bias,\n                     fixed_mask, fixed_positions, fixed_tokens, compute_pseudo_perplexity,\n                     return_logits, return_decoding_orders\n  run_spec.io:       output_h5_path, cache_path\n  run_spec.plan:     use_unified_driver   (kernel_dispatch ~:172; default True both flat and via run_spec -> SAFE/mechanical)\n\nTARGET FILES (migrate every flat read of the above fields; rg is ground truth — line numbers\nbelow are recon-time and may have shifted, esp runner.py post-RS-7b):\n  kernel_dispatch.py (~9), streaming.py (~8; :310 already correct), runner.py (~26),\n  plan.py (1: num_samples), _sampling_helper.py (9: fixed_*), _sampling_grid_lineage.py (7),\n  campaign.py (2), prep.py (1: cache_path ONLY).\n\nSEMANTICS-PRESERVING: run_spec.<sub>.<field> holds the value build_run_spec copied from the\nflat spec (with normalization: backbone_noise/temperature -> float tuple, random_seed -> int,\nreturn_* -> bool). Preserve the surrounding expression EXACTLY (e.g. \`len(...)\`,\n\`... or (0.0,)\`, \`... or 42\`, \`... is not None\`, \`np.asarray(...)\`). No behavior change.\n\nuse_unified_driver IS IN SCOPE: kernel_dispatch.py ~:172 reads getattr(spec,'use_unified_driver',True);\nrun_spec.plan.use_unified_driver also defaults True -> migrating is semantics-preserving. Migrate it to\nspec.run_spec.plan.use_unified_driver and ADD it to the ast-grep banned set.\n\nEXPLICITLY OUT OF SCOPE (deferred to follow-ups; do NOT touch):\n  - src/aminx/host/_sampling_averaged.py: DEAD (not imported). Do NOT delete (no-auto-delete)\n    and do NOT migrate. It is exempt from the gate (or untouched).\n  - prep.py: migrate cache_path ONLY; its 21 protein-only flat reads stay flat. prep.py is\n    EXEMPT from the ast-grep gate.\n`;

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

// ===== TRACK A — RS-6b: host caller run_spec.* migration + ast-grep flat-field gate =========================
const trackA = () =>
  track(
    "2226",
    "RS-6b: host caller run_spec.* migration + ast-grep flat-field gate",
    `task_id: ${TASK_ID}. task_id: 260618_autonomous-loop\nEdit only (existing host files + new ast-grep rule + new test). SEMANTICS-PRESERVING migration.\n\nSTEP 1 — Migrate flat reads in each host file. For EACH target file, use rg to find every flat\nread of a migrated field, then replace spec.<field> -> spec.run_spec.<subconfig>.<field> per the\nFIELD->SUBCONFIG MAP in the sprint context. Preserve the surrounding expression byte-for-byte\n(only the attribute path changes). Files (ground-truth via rg, not fixed counts):\n  src/aminx/host/kernel_dispatch.py  (backbone_noise,temperature,bias,compute_pseudo_perplexity,use_unified_driver->run_spec.plan)\n  src/aminx/host/streaming.py        (output_h5_path,return_logits,backbone_noise,temperature)\n  src/aminx/host/runner.py           (return_logits,return_decoding_orders,random_seed,output_h5_path,backbone_noise)\n  src/aminx/host/plan.py             (num_samples)\n  src/aminx/host/_sampling_helper.py (fixed_mask,fixed_positions,fixed_tokens)\n  src/aminx/host/_sampling_grid_lineage.py (num_samples,random_seed,temperature,backbone_noise)\n  src/aminx/host/campaign.py         (temperature,backbone_noise)\n  src/aminx/host/prep.py             (cache_path ONLY — leave the 21 protein-only flat reads)\nMigrate use_unified_driver (kernel_dispatch:172) -> spec.run_spec.plan.use_unified_driver. DO NOT touch _sampling_averaged.py.\nTo find sites: \`rg -n 'spec[.](backbone_noise|temperature|num_samples|random_seed|bias|fixed_mask|fixed_positions|fixed_tokens|compute_pseudo_perplexity|return_logits|return_decoding_orders|output_h5_path|cache_path|use_unified_driver)' src/aminx/host/\`\nAfter migrating, re-run that rg over each file (except prep.py protein lines / use_unified_driver) and confirm ZERO remaining flat reads of the migrated fields.\nGATE after STEP 1: \`uv run pytest tests/host tests/scoring -q\` — ALL pass (semantics preserved, no behavior change). If any test changes value, you broke a wrapped expression — fix it.\n\nSTEP 2 — Add the ast-grep gate. Inspect .ast-grep/rules/ (e.g. xtrax-protein-symbols.yml) for the\nYAML format + how sgconfig wires rules. Add .ast-grep/rules/rs6b-host-flat-field-ban.yml: a python\nrule, severity error, matching \`spec.$F\` for the migrated field set (NOT use_unified_driver),\nscoped to files src/aminx/host/**/*.py with prep.py EXCLUDED, message pointing to the run_spec.* path.\nVerify the rule wiring: \`ast-grep scan src/aminx/host/\` (or the repo's sg invocation) reports ZERO\nviolations on the migrated code.\n\nSTEP 3 — Rule-fires test. Add tests/lint/test_rs6b_flat_field_gate.py (or the repo's convention):\nplant a temp file / string with \`spec.temperature\` in a host-like path and assert ast-grep flags it;\nassert the real src/aminx/host/ tree (minus prep.py) is clean. If the repo has no python harness for\nast-grep, write a subprocess-based test invoking the sg binary on a fixture and asserting exit code.\n\nSTEP 4 — Full gates:\n  uv run pytest tests/host tests/scoring -q   (no regression / no value change)\n  ast-grep scan over src/aminx/host/  -> 0 violations (prep.py exempt)\n  uv run ty check && uv run ruff check . && uv run ruff format .\n\nSTEP 5 — Commit (explicit paths; the worktree gitignores .claude/ but rules are in .ast-grep/):\n  git add src/aminx/host/*.py .ast-grep/rules/rs6b-host-flat-field-ban.yml tests/...\n  git commit -m "refactor(RS-6b): migrate host callers to run_spec.* + ast-grep flat-field gate (#2226)"\nReport: commit SHA, the rg-confirms-zero-remaining output per file, the ast-grep scan result, the\nrule-fires test result, and the full gate outputs. State explicitly that no test value changed.\n\n\n${EMITTER_CTX}

IMPORTANT — GITIGNORE IN WORKTREE: This worktree has a .gitignore that blocks the .claude/ directory. If any output files are under .claude/ (e.g., .claude/skills/, .claude/workflows/), you MUST use \`git add -f <path>\` to force-add them. A plain \`git add\` or \`git add .\` will silently skip these files. Always run \`git status\` after staging to confirm they appear as 'Changes to be committed'.\n\n`,
    `task_id: ${TASK_ID}. task_id: 260618_autonomous-loop\nVERIFY checklist for RS-6b (#2226). Run every command; report PASS/FAIL per item with evidence.\n1. \`uv run pytest tests/host tests/scoring -q\` — ALL pass, NO regression. (RS-6b is semantics-preserving;\n   any changed test VALUE is a FAIL — a wrapped expression like \`len(...)\`/\`or (0.0,)\`/\`is not None\` got broken.)\n2. NO REMAINING FLAT READS: \`rg -n 'spec[.](backbone_noise|temperature|num_samples|random_seed|bias|fixed_mask|fixed_positions|fixed_tokens|compute_pseudo_perplexity|return_logits|return_decoding_orders|output_h5_path|cache_path|use_unified_driver)' src/aminx/host/\` returns ONLY: prep.py protein-only lines (none of these fields except cache_path which must be migrated) — i.e. zero hits for the migrated fields in kernel_dispatch/streaming/runner/plan/_sampling_helper/_sampling_grid_lineage/campaign, and prep.py cache_path migrated. Any stray flat read = FAIL.\n3. AST-GREP GATE: .ast-grep/rules/rs6b-host-flat-field-ban.yml exists, scoped to src/aminx/host/**/*.py with prep.py excluded, bans the migrated field set (NOT use_unified_driver). \`ast-grep scan src/aminx/host/\` reports 0 violations.\n4. RULE FIRES: the new test plants a \`spec.temperature\` host-path violation and asserts ast-grep flags it (the gate actually works, not a no-op). Test PASSES.\n5. OUT-OF-SCOPE UNTOUCHED: \`git show --stat HEAD\` shows NO changes to _sampling_averaged.py; prep.py's 21 protein-only reads unchanged. use_unified_driver IS migrated (kernel_dispatch -> run_spec.plan) and IS in the banned set.\n6. uv run ty check clean; uv run ruff check . clean; uv run ruff format . no diff.\n7. SCOPE: HEAD touches only src/aminx/host/*.py, .ast-grep/rules/rs6b-host-flat-field-ban.yml, and the new test. No unrelated files.\nPASS only if all 7 hold. Else FAIL with the specific item.\n\n\n${EMITTER_CTX}`,
    "worktree",
    { track_id: "a", sprint_id: TASK_ID }
    , "recon"
  );

// ---- orchestrate: writing chain (A, sequential) ----
const _gitBase = (await agent(
  'Run: git rev-parse HEAD. Return only the 40-character SHA, nothing else.',
  { label: 'git-base', phase: 'Execute' }
)).trim();
log("RS-6b: migrate host callers to run_spec.* + ast-grep flat-field lint gate: writing chain (A, sequential)");
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
  For each branch with new commits, report the branch name and the output of: git log main..<branch> --oneline
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
  'task_id: ' + TASK_ID + '. The fixer committed to a worktree branch NOT merged to main (mission.md: merge is manual). Discover it: git branch --list "wt-*". For the branch with new commits run git log main..<branch> --oneline and git diff main...<branch>. RS-6b is a SEMANTICS-PRESERVING migration of flat spec.<field> reads to spec.run_spec.<subconfig>.<field> in src/aminx/host/* + an ast-grep flat-field ban rule. VERIFY: (1) no migrated-field flat reads remain in host/ except prep.py protein-only (rg the field set); (2) wrapped expressions preserved (len/or-default/is-not-None) so NO behavior change; (3) the ast-grep rule exists, scoped to host/** excluding prep.py, and a test proves it fires on a planted violation; (4) _sampling_averaged.py untouched. Return PASS or FAIL with specific findings. Do NOT merge anything.',
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
  sprint_id: 3,
  verdicts: {
    "2226": a
  },
  audit: _auditVerdict,
};
