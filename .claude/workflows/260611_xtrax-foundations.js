// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 260611_xtrax-foundations   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (D,E,I) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time. Only the read-only
//   research/concurrent tracks (A,B,C,F,G,H) run concurrently.

export const meta = {
  name: "260611_xtrax-foundations",
  description: "First executable slice of EPIC #1541 + RunSpec foundations (RS-1/RS-4/RS-5). Lands boundary-lint,\ntraining adoption (ResumableState + optimizer + T1.4 clean-break checkpoint), and parallel RunSpec prep.\nCluster smoke (#1544) DEFERRED. G1 training-parity gate WAIVED (user: training not active).\nRE-BASELINED 2026-06-12: Tracks A (#1542) and B (#1543) already done in main (pyproject.toml at\n>=3.13, xtrax editable pin live). Tracks G (#1623) and H (#1624) implemented but uncommitted\n(verify+commit). Sequential spine starts at #1545. Stack: uv + ty + ruff + pytest (`uv run`).\n",
  phases: [
    { title: "Track D — [xtrax T0.4] Boundary-lint: ruff banned-api + xtrax protein-agnostic guard (#1545)" },
    { title: "Track E — [xtrax T1.1-1.2] Adopt xtrax ResumableState + optimizer in trainer.py (#1548)" },
    { title: "Track I — [xtrax T1.4] Checkpoint clean-break → xtrax single-PyTree ResumableState (#1549)" },
    { title: "Track A — [DONE] py3.13 floor already in pyproject.toml (#1542)" },
    { title: "Track B — [DONE] editable xtrax pin already in pyproject.toml (#1543)" },
    { title: "Track C — [DEFERRED] L3 cluster smoke py3.13+JAX on SM120 (#1544)" },
    { title: "Track F — [RunSpec RS-1] Host field inventory → run_spec migration map (#1620)" },
    { title: "Track G — [RunSpec RS-4] Verify + commit aminx.run exports fix (#1623)" },
    { title: "Track H — [RunSpec RS-5] Verify + commit unified driver default fix (#1624)" },
  ],
};

const TASK_ID = "260611_xtrax-foundations";
const MAX_FIX_RETRIES = 1;

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

const RESEARCH_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["surfaces", "recommendation"],
  properties: {
    surfaces: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["surface", "rules_location", "frontmatter", "settings_triggers", "distribution"],
        properties: {
          surface: { type: "string" },
          rules_location: { type: "string" },
          frontmatter: { type: "string" },
          settings_triggers: { type: "string" },
          distribution: { type: "string" },
          confidence: { type: "string" },
        },
      },
    },
    cross_surface_differences: { type: "array", items: { type: "string" } },
    recommendation: { type: "string" },
    open_questions: { type: "array", items: { type: "string" } },
  },
};

// Shared context for the writing tracks (from recon, task 260611_xtrax-foundations).
const EMITTER_CTX = `USER DECISIONS (2026-06-11): T-R1-X=5%/T-R2=clean-break/T-R3b=prolix-YES confirmed. T1.4 IN\nfoundations. G1 WAIVED. Cluster smoke DEFERRED. HDF5 scoring streaming (#1444) PERMANENTLY SHELVED.\n\nRE-BASELINE (2026-06-12):\n- Tracks A/B DONE: pyproject.toml already has requires-python=">=3.13" and\n  xtrax={path="../xtrax",editable=true}. Confirmed live.\n- Track G DONE (uncommitted): src/aminx/run/_exports.py + tests/run/test_exports.py exist in\n  working tree; fixer must verify ACs and commit.\n- Track H DONE (uncommitted): kernel_dispatch.py:171 default already True +\n  tests/host/test_use_unified_driver_default.py in working tree; fixer must verify ACs and commit.\n- P2 xtrax recon STALE: xtrax tiling already has BatchPlanner/DedupSpec/DedupGather/AxisSpec —\n  re-recon needed before T2.1-T2.5 can be scoped in sprint 2 (not a sprint-1 blocker).\n\nGROUND TRUTH: .praxia/docs/research/260611_aminx-xtrax-refactor-codebase-model.md\n- xtrax editable at ../xtrax. xtrax.training.ResumableState + adamw_with_schedule confirmed present.\n- aminx/training/trainer.py still uses create_optimizer() — Track E is the real sequential entry.\n- Track D boundary-lint: no xtrax ast-grep rule exists yet; .ast-grep/ dir exists for pattern.\n- Dependency order: 1545→1548→1549. RS tracks F/G/H independent (concurrent).\n`;

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

// ===== TRACK D — Track D — [xtrax T0.4] Boundary-lint: ruff banned-api + xtrax protein-agnostic guard (#1545) =========================
const trackD = () =>
  track(
    "1545",
    "Track D — [xtrax T0.4] Boundary-lint: ruff banned-api + xtrax protein-agnostic guard (#1545)",
    `task_id: ${TASK_ID}. task_id: 260611_xtrax-refactor-spec. Depends on B (#1543) already done. Stack: uv/Python + ruff/ast-grep. New lint file allowed.\n\nOBJECTIVE: Preserve ADR 260605 boundary discipline across the xtrax move. ruff TID251 bans\naminx.* paths but xtrax.* is inherently unbanned, so the first xtrax.* import would silently\nbypass the boundary. This guard MUST land atomically with or before any xtrax.* import.\n\nCURRENT STATE: \`rg -rn "import xtrax|from xtrax" src/\` → zero hits (no xtrax imports yet).\n.ast-grep/ dir exists with existing rules as pattern (see potts-import-boundary.yml was removed;\ncheck git log for structure). pyproject.toml [tool.ruff.lint.flake8-tidy-imports.banned-api]\ncurrently bans only aminx.* paths.\n\nDO (both halves required, must both be atomic):\n(a) aminx→xtrax-internals boundary: an import-boundary check ensuring aminx protein modules\n    do not reach into xtrax internals they should not (mirror the ADR's intent for surfaces\n    that will move to xtrax). Implement as a ruff banned-api entry or ast-grep rule.\n(b) xtrax protein-agnostic guard: assert no xtrax module references the symbols\n    \`atom_37\`, \`residue_index\`, \`tie_group_map\`. Implement as an ast-grep rule (preferred,\n    matches repo convention) or a pytest that greps ../xtrax/src; wire into CI.\n\nSUCCESS CRITERIA:\n- Both checks are GREEN on the current tree.\n- Injecting a violation (e.g. a temp xtrax module with \`atom_37 = None\`) makes part (b) FAIL;\n  removing it passes again. Demonstrate both states.\n- \`uv run ruff check .\` exits 0.\n- New check wired into CI (ci.yml) or pytest suite.\nReport: rule/file added, the inject-fail/clean-pass output.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260611_xtrax-refactor-spec. Reviewer (HAS Bash). Stack: uv/Python + ruff/ast-grep.\n\nVERIFY:\n- Both halves exist and are wired: (a) protein→xtrax-internals guard, (b) xtrax protein-agnostic guard.\n- Demonstrate FAIL on injected violation and PASS once removed (run both).\n- \`uv run ruff check .\` exits 0.\n- CI / pytest integration present.\nPASS iff both guards are real and enforced. FAIL otherwise.\n`,
  );

// ===== TRACK E — Track E — [xtrax T1.1-1.2] Adopt xtrax ResumableState + optimizer in trainer.py (#1548) =========================
const trackE = () =>
  track(
    "1548",
    "Track E — [xtrax T1.1-1.2] Adopt xtrax ResumableState + optimizer in trainer.py (#1548)",
    `task_id: ${TASK_ID}. task_id: 260611_xtrax-refactor-spec. Depends on #1545 (boundary-lint must land with the first\nxtrax import). Stack: uv/Python. Edit-only on named files; this is the first real xtrax.* import.\n\nOBJECTIVE: Replace aminx's inline training state + optimizer with xtrax primitives (idea-2\n"free win" — xtrax already ahead; no feature flag needed).\n\nANCHORS:\n- src/aminx/training/trainer.py: \`create_optimizer()\` (~lines 73-106, inline\n  optax.warmup_cosine_decay + clip_by_global_norm) and the (model, opt_state, start_step)\n  state threading (~lines 125-199).\n- xtrax targets (confirmed present in ../xtrax):\n    xtrax.training.ResumableState  (xtrax/src/xtrax/training/types.py:43 — step/key/model/opt_state/extras)\n    xtrax.training.optim.adamw_with_schedule + make_optimizer  (xtrax/src/xtrax/training/optim.py)\n\nDO:\n- Replace (model, opt_state, step) tuple threading with a single ResumableState.\n- Replace create_optimizer() internals with adamw_with_schedule(...) / make_optimizer(...),\n  mapping aminx's TrainingSpecification fields (learning_rate, warmup_steps, total_steps,\n  weight_decay, gradient_clip) onto xtrax builder args.\n- KEEP in aminx: domain losses (losses.py), metrics, diffusion, dataloading — do NOT move them.\n- Update imports and any affected tests under tests/training/.\n\nDO NOT: change checkpoint format (that is #1549). Do not touch tiling/host.\n\nSUCCESS CRITERIA:\n- \`uv run ty check\` and \`uv run ruff check .\` pass.\n- \`uv run pytest -q tests/training -m 'not slow'\` green.\n- Structural adoption verified: ResumableState threading + xtrax optimizer builder in place.\n  G1 loss-curve gate WAIVED (user: training not active).\nReport: diff per file and test tail.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260611_xtrax-refactor-spec. Reviewer (HAS Bash). Stack: uv/Python.\n\nVERIFY:\n- Training state is xtrax.training.ResumableState; optimizer built via xtrax\n  adamw_with_schedule/make_optimizer (inline optax.chain removed from trainer.py).\n- Domain losses/metrics/diffusion/dataloading remain in aminx:\n  \`rg -n 'cross_entropy_loss|NoiseSchedule' src/aminx/training\` resolves locally.\n- First xtrax import is covered by #1545 boundary-lint (green).\n- \`uv run ty check\` clean; \`uv run ruff check .\` clean; \`uv run pytest -q tests/training -m 'not slow'\` green.\n- G1 loss-curve gate WAIVED — structural adoption + tests suffice.\nPASS iff all hold. FAIL with command output otherwise.\n`,
  );

// ===== TRACK I — Track I — [xtrax T1.4] Checkpoint clean-break → xtrax single-PyTree ResumableState (#1549) =========================
const trackI = () =>
  track(
    "1549",
    "Track I — [xtrax T1.4] Checkpoint clean-break → xtrax single-PyTree ResumableState (#1549)",
    `task_id: ${TASK_ID}. task_id: 260611_xtrax-refactor-spec. Depends on #1548 (ResumableState must land first).\nStack: uv/Python. T-R2=NO clean-break (user-confirmed: no in-flight runs, no migration tool).\n\nOBJECTIVE: Replace Composite orbax checkpoint with xtrax single-PyTree over ResumableState.\nMetrics go in extras. No migration CLI — clean break documented in CHANGELOG.\n\nPRE-CUTOVER AUDIT (required first — if any exist, STOP and report to user):\n- Search for legacy Composite checkpoint dirs:\n  \`find . -name "*.orbax-checkpoint-tmp-*" -o -name "checkpoint_manager" 2>/dev/null | head -20\`\n  \`rg -rn "Composite|StandardSave|PyTreeSave" src/aminx/training/\`\n- If any Composite checkpoint dirs exist ON DISK (not just in code), do not merge format break —\n  report to user and stop.\n\nANCHORS:\n- src/aminx/training/checkpoint.py — Composite(model, opt_state, metrics JsonSave) today.\n- src/aminx/training/trainer.py — save_checkpoint / restore_checkpoint call sites.\n- xtrax: mirror xtrax.checkpoint / ResumableState PyTree handler from ../xtrax examples.\n\nDO:\n- Rewrite checkpoint save/load to persist full ResumableState (model, opt_state, step, key, extras w/ metrics).\n- Document Composite→PyTree break in CHANGELOG (one bullet under the relevant version).\n- Update tests/training checkpoint tests for fresh save→load round-trip.\n\nDO NOT: build migration tooling. Do not require G1 loss-curve parity.\n\nSUCCESS CRITERIA:\n- Fresh ResumableState save→load round-trip test green (jnp.array_equal on arrays; step/key/extras match).\n- \`uv run ty check\` + \`uv run ruff check .\` + \`uv run pytest -q tests/training\` pass.\n- CHANGELOG entry present.\nReport: audit output (no legacy ckpts found) + test names + CHANGELOG entry.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260611_xtrax-refactor-spec. Reviewer (HAS Bash). Stack: uv/Python.\n\nVERIFY:\n- Pre-cutover audit documented (no Composite ckpt dirs on disk).\n- checkpoint.py no longer uses ocp.args.Composite for the primary path.\n- Fresh save→load round-trip test passes on ResumableState fields.\n- CHANGELOG documents format break. No migration CLI added.\n- \`uv run ty check\` clean; \`uv run pytest -q tests/training\` green.\nPASS iff AC-1 T1.4 holds. FAIL with output.\n`,
  );

// ===== TRACK A — Track A — [DONE] py3.13 floor already in pyproject.toml (#1542) =========================
const trackA = () =>
  track(
    "1542",
    "Track A — [DONE] py3.13 floor already in pyproject.toml (#1542)",
    `task_id: ${TASK_ID}. SKIP — already done. pyproject.toml has requires-python=">=3.13" and both\n[tool.uv.environments] markers at python_version >= '3.13'. Nothing to change.\nVerify: \`rg -n 'requires-python|python_version' pyproject.toml\` → all 3.13.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. VERIFY only: \`rg -n 'requires-python|python_version' pyproject.toml\` → all 3.13. PASS immediately.\n`,
  );

// ===== TRACK B — Track B — [DONE] editable xtrax pin already in pyproject.toml (#1543) =========================
const trackB = () =>
  track(
    "1543",
    "Track B — [DONE] editable xtrax pin already in pyproject.toml (#1543)",
    `task_id: ${TASK_ID}. SKIP — already done. pyproject.toml [project].dependencies has xtrax>=0.2.0 and\n[tool.uv.sources] has xtrax = { path = "../xtrax", editable = true }.\nVerify: \`uv run python -c "import xtrax; print(xtrax.__version__)"\` → 0.2.0.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. VERIFY only:\n- \`rg -n 'xtrax' pyproject.toml\` shows the dep + editable source with literal ../xtrax path.\n- \`uv run python -c "import xtrax; print(xtrax.__version__)"\` → 0.2.0.\n- \`rg -n "import xtrax|from xtrax" src\` → NO source imports yet.\nPASS immediately if all hold.\n`,
  );

// ===== TRACK C — Track C — [DEFERRED] L3 cluster smoke py3.13+JAX on SM120 (#1544) =========================
const trackC = () =>
  track(
    "1544",
    "Track C — [DEFERRED] L3 cluster smoke py3.13+JAX on SM120 (#1544)",
    `task_id: ${TASK_ID}. DEFERRED — user decision 2026-06-11. Cluster smoke will run in a separate cluster sprint.\nDo not execute this track. Report SKIPPED.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. DEFERRED — skip. Report SKIPPED.\n`,
  );

// ===== TRACK F — Track F — [RunSpec RS-1] Host field inventory → run_spec migration map (#1620) =========================
const trackF = () =>
  agent(
    `task_id: ${TASK_ID}. task_id: 260611_xtrax-refactor-spec. Research-only; no product code changes. Stack: uv/Python.\nConcurrent with D/E — no dependency on sequential spine.\n\nOBJECTIVE: Produce the AC-RS-1a inventory table mapping every flat spec field read in the host\ninference hot path to either run_spec.<subconfig> or "protein-only façade" (per RunSpec\nunification spec .praxia/docs/specs/260611_runspec-unification.md).\n\nANCHORS (grep before writing):\n- src/aminx/host/{plan,prep,runner,streaming,kernel_dispatch}.py — every \`spec.\` / \`getattr(spec,\` read.\n- src/aminx/run/specs.py — build_run_spec / _sync_run_spec call sites.\n- src/aminx/run/spec.py — RunSpec sub-config fields.\n\nOUTPUT: Commit .praxia/docs/research/260611_runspec-host-field-inventory.md with columns:\n| file:line | read pattern | current source | target run_spec path | migration sprint | notes |\n\nSUCCESS CRITERIA:\n- Table covers all five host modules with zero TBD rows.\n- Explicitly notes the 2 existing run_spec read sites (streaming.py, training/trainer.py).\n- Flags the three currently-ignored spec fields: spec.fixed_mask, spec.iterations/learning_rate/\n  sampling_strategy (del'd in averaging.py), and the atom_37/chain_mask gap in kernel_dispatch.\n- Lists which fields must NEVER move to xtrax (protein-only: ligand, fixed_mask, tied_positions).\nReport row count and protein-only field list.\n`,
    { agentType: "librarian", label: "research:1620", phase: "Track F — [RunSpec RS-1] Host field inventory → run_spec migration map (#1620)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK G — Track G — [RunSpec RS-4] Verify + commit aminx.run exports fix (#1623) =========================
const trackG = () =>
  track(
    "1623",
    "Track G — [RunSpec RS-4] Verify + commit aminx.run exports fix (#1623)",
    `task_id: ${TASK_ID}. task_id: 260611_xtrax-refactor-spec. Stack: uv/Python. Concurrent with D/E.\n\nCONTEXT: Implementation of AC-RS-4 appears to already exist as uncommitted work:\n- src/aminx/run/_exports.py (untracked new file)\n- tests/run/test_exports.py (untracked new file)\nVerify these satisfy the AC, then commit. If they are incomplete, finish them.\n\nOBJECTIVE: AC-RS-4 — \`from aminx.run import sample, score\` must route to spec-driven\nhost.runner (returns dict), while tensor APIs remain importable with DeprecationWarning.\n\nVERIFY EXISTING FILES:\n1. Read src/aminx/run/_exports.py — does it re-export spec-driven sample/score from host.runner?\n2. Read tests/run/test_exports.py — does it test that spec call returns dict, that tensor path\n   emits DeprecationWarning, and that aminx.sampling.sample / aminx.scoring.score still work?\n3. Read src/aminx/run/__init__.py — is _exports.py integrated (imported/re-exported)?\n\nRUN CHECKS:\n- \`uv run pytest -q tests/run\` — must be green.\n- \`uv run python -c "from aminx.run import sample, score; print(sample, score)"\` — must resolve.\n- \`uv run ty check\` and \`uv run ruff check .\` — must pass.\n\nIF ALL PASS: \`git add src/aminx/run/_exports.py tests/run/test_exports.py\` + any other\nmodified files, then commit with message \`feat(run): spec-driven aminx.run exports + deprecation shim (RS-4 #1623)\`.\n\nIF GAPS FOUND: complete the implementation before committing.\n\nSUCCESS CRITERIA:\n- \`from aminx.run import sample, score\` routes to host.runner (dict result).\n- Tensor path through aminx.run emits DeprecationWarning for one minor version.\n- \`uv run pytest -q tests/run\` green; \`uv run ty check\` clean.\nReport: what you found in the existing files, what (if anything) you completed, and the commit hash.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260611_xtrax-refactor-spec. Reviewer (HAS Bash). Stack: uv/Python.\n\nVERIFY:\n- \`from aminx.run import sample, score\` resolves to spec-driven host.runner paths.\n- Tensor APIs still importable from aminx.sampling / aminx.scoring.\n- DeprecationWarning test exists for legacy tensor call via aminx.run.\n- \`uv run pytest -q tests/run\` green; \`uv run ty check\` clean.\n- Commit exists for this track (\`git log --oneline -5\`).\nPASS iff AC-RS-4 holds and work is committed. FAIL with output.\n`,
    "worktree"
  );

// ===== TRACK H — Track H — [RunSpec RS-5] Verify + commit unified driver default fix (#1624) =========================
const trackH = () =>
  track(
    "1624",
    "Track H — [RunSpec RS-5] Verify + commit unified driver default fix (#1624)",
    `task_id: ${TASK_ID}. task_id: 260611_xtrax-refactor-spec. Stack: uv/Python. Concurrent with D/E.\n\nCONTEXT: Implementation of AC-RS-5 appears to already exist as uncommitted work:\n- src/aminx/host/kernel_dispatch.py (modified) — getattr default likely already True\n- tests/host/test_use_unified_driver_default.py (untracked new file)\nVerify these satisfy the AC, then commit. If incomplete, finish them.\n\nOBJECTIVE: AC-RS-5 — align kernel_dispatch.py getattr default with\nSamplingSpecification.use_unified_driver=True (was False, causing default mismatch).\n\nVERIFY EXISTING CHANGES:\n1. \`rg -n "use_unified_driver" src/aminx/host/kernel_dispatch.py\` — confirm default is True.\n2. Read tests/host/test_use_unified_driver_default.py — does it prove default SamplingSpecification()\n   routes through the unified driver path?\n3. \`uv run pytest -q tests/host\` — must be green.\n4. \`uv run ty check\` and \`uv run ruff check .\` — must pass.\n\nIF ALL PASS: \`git add src/aminx/host/kernel_dispatch.py tests/host/test_use_unified_driver_default.py\`\n+ any other modified files, then commit with message:\n\`fix(host): align use_unified_driver default to True (RS-5 #1624)\`.\n\nIF GAPS FOUND: complete the implementation. The fix is ONE line:\n\`getattr(spec, 'use_unified_driver', False)\` → \`getattr(spec, 'use_unified_driver', True)\`.\n\nDO NOT: add PlannerTopology / run_spec.planner field (that is RS-2, sprint 2).\n\nSUCCESS CRITERIA:\n- Default SamplingSpecification() routes through unified driver path.\n- Regression test green.\n- \`uv run ty check\` and \`uv run ruff check .\` pass.\nReport: confirmed line change, test result, commit hash.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260611_xtrax-refactor-spec. Reviewer (HAS Bash). Stack: uv/Python.\n\nVERIFY:\n- \`rg -n "use_unified_driver" src/aminx/host/kernel_dispatch.py\` shows default True.\n- Regression test proves default SamplingSpecification uses unified path.\n- \`uv run pytest -q tests/host\` green.\n- Commit exists for this track.\nPASS iff AC-RS-5 holds and work is committed. FAIL otherwise.\n`,
    "worktree"
  );

// ---- orchestrate: sequential writing chain || read-only research ----------
log("260611_xtrax-foundations — aminx→xtrax P0 foundations + RunSpec prep: writing chain (D -> E -> I, sequential) || research (A, B, C, F, G, H, read-only)");
const [writing, resA, resB, resC, resF, resG, resH] = await Promise.all([
  (async () => {
    const d = await trackD();
    const e = await trackE();
    const i = await trackI();
    return { d, e, i };
  })(),
  trackA(),
  trackB(),
  trackC(),
  trackF(),
  trackG(),
  trackH(),
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
    "1545": writing.d,
    "1548": writing.e,
    "1549": writing.i
  },
  research_1542: resA,
  research_1543: resB,
  research_1544: resC,
  research_1620: resF,
  research_1623: resG,
  research_1624: resH
};
