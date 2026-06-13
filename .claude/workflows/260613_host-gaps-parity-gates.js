// Sprint 2 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 260613_host-gaps-parity-gates   sprint_id: 2
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (A,B,C) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time. Only the read-only
//   research/concurrent tracks (D,E,F) run concurrently.

export const meta = {
  name: "260613_host-gaps-parity-gates",
  description: "Fix three silent host-dispatch bugs (STE, fixed_mask, atom_37), validate checkpoint clean-break training parity, complete RS-1 inventory to unblock RS-2+, and run T0.3 cluster smoke to unblock T2 tiling.",
  phases: [
    { title: "Track A — Restore + wire STE dispatch in averaging.py (#1767)" },
    { title: "Track B — Read spec.fixed_mask in _prepare_fixed_controls (#1766)" },
    { title: "Track C — Wire atom_37/atom_37_mask/chain_mask into build_inference_bundle (#1765)" },
    { title: "Track D — Training parity gate: checkpoint round-trip + loss curve (G1 #1550)" },
    { title: "Track E — RS-1 inventory: host/* getattr(spec) → run_spec migration map (#1620)" },
    { title: "Track F — T0.3 L3 cluster smoke: py3.13 + JAX/jaxlib on SM120 + bathos sidecar (#1544)" },
  ],
};

const TASK_ID = "260613_host-gaps-parity-gates";
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

// Shared context for the writing tracks (from recon, task 260613_host-gaps-parity-gates).
const EMITTER_CTX = `Prior sprint (260611_xtrax-foundations) landed:\n- Track I: checkpoint.py rewritten as xtrax.checkpoint.orbax re-export; trainer.py uses save_checkpoint(manager, resumable_state). Composite format is DEAD — no backwards compat.\n- Track E: ResumableState adopted; train_step() no longer takes current_step.\n- Track D: ast-grep boundary guards in CI; protein-symbols rule scans ../xtrax/src/xtrax.\n- Track G: aminx.run.sample/score are now spec-driven with deprecation dispatch shim.\n- Track H: use_unified_driver defaults to True.\n\nThree host-dispatch bugs were identified in the original issue and are silent no-ops:\n1. STE body was deleted from src/aminx/inference/decode/ste.py — tied-position STE decoding falls through silently.\n2. spec.fixed_mask is never read in _prepare_fixed_controls — mask-based fixed positions are a no-op.\n3. atom_37/atom_37_mask/chain_mask are never forwarded from LigandContext into build_inference_bundle.\n\nKey file anchors:\n- src/aminx/inference/decode/ste.py — STE implementation (currently incomplete/deleted)\n- src/aminx/potts/__init__.py — has uncommitted changes (check before editing)\n- src/aminx/host/_sampling_helper.py:448 — _prepare_fixed_controls\n- src/aminx/host/kernel_dispatch.py:191-215 — build_inference_bundle call sites\n- src/aminx/training/checkpoint.py — thin re-export of xtrax.checkpoint.orbax\n- src/aminx/training/trainer.py:127-220 — _init_checkpoint_and_model (new load_checkpoint path)\n- tests/sampling/test_ste_tied_positions.py — exists; may be skipped or failing\n- tests/training/ — new; test_checkpoint.py + test_resumable_state.py (8 tests, all pass)\n`;

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

// ===== TRACK A — Track A — Restore + wire STE dispatch in averaging.py (#1767) =========================
const trackA = () =>
  track(
    "1767",
    "Track A — Restore + wire STE dispatch in averaging.py (#1767)",
    `task_id: ${TASK_ID}. # Track A — Restore STE dispatch (backlog #1767)\n\n## Context\nThe STE (straight-through estimator) decode path was removed from\nsrc/aminx/inference/decode/ste.py in a prior session. The file currently\nhas an incomplete or stub body. averaging.py does not dispatch to it.\ntests/sampling/test_ste_tied_positions.py exists but may be skipping or failing.\n\n## Step 1 — Recover the implementation\nRun:\n  git log --all --oneline -- src/aminx/inference/decode/ste.py\nto find the last commit that had a working body. Read that version via:\n  git show <sha>:src/aminx/inference/decode/ste.py\n\n## Step 2 — Restore ste.py\nRestore the STE implementation. It must:\n- Implement straight-through estimation for tied-position autoregressive decoding\n- Be a pure JAX function compatible with eqx.filter_jit / jax.vmap\n- Have no side effects; all randomness via explicit PRNG key\n\n## Step 3 — Wire dispatch in averaging.py\nIn src/aminx/inference/decode/averaging.py (or equivalent dispatch point),\nadd the STE branch so it is called when the run spec requests tied-position\nSTE decoding. Check how other decode modes are dispatched and follow the\nsame pattern.\n\n## Step 4 — Verify\n  uv run pytest tests/sampling/test_ste_tied_positions.py -v\nAll tests must pass without skips.\nAlso run: uv run pytest tests/sampling/ -v --tb=short to check no regression.\n\n## Acceptance criteria\n- src/aminx/inference/decode/ste.py contains a working non-stub implementation\n- averaging.py (or the dispatch module) routes to STE when appropriate\n- tests/sampling/test_ste_tied_positions.py: all tests PASS (no skips)\n- No regression in tests/sampling/\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track A Reviewer — STE dispatch (#1767)\n\nVERIFY: src/aminx/inference/decode/ste.py contains a real implementation\n  (not just pass, raise NotImplementedError, or an emptied body)\n  Check: grep -n "def\|return\|jax\." src/aminx/inference/decode/ste.py | head -20\n\nVERIFY: averaging.py (or the dispatch entry point) imports and calls the STE function\n  Check: grep -rn "ste\|STE\|straight_through" src/aminx/inference/ src/aminx/host/\n\nVERIFY: uv run pytest tests/sampling/test_ste_tied_positions.py -v\n  All tests PASS. Zero skips.\n\nVERIFY: uv run pytest tests/sampling/ -v --tb=short\n  No new failures compared to baseline.\n\nPASS if all VERIFY items are satisfied.\nFAIL if ste.py is still a stub, dispatch is missing, or tests skip/fail.\n`,
    "worktree"
  );

// ===== TRACK B — Track B — Read spec.fixed_mask in _prepare_fixed_controls (#1766) =========================
const trackB = () =>
  track(
    "1766",
    "Track B — Read spec.fixed_mask in _prepare_fixed_controls (#1766)",
    `task_id: ${TASK_ID}. # Track B — fixed_mask gap (_prepare_fixed_controls, backlog #1766)\n\n## Context\nsrc/aminx/host/_sampling_helper.py, function _prepare_fixed_controls (around line 448),\nreads spec.fixed_positions to build the fixed-residue tensor but never reads\nspec.fixed_mask. RunSpec exposes fixed_mask as a first-class field (a boolean\narray of shape [N_residues]) but it is silently ignored — mask-based position\nfixing is a complete no-op.\n\n## Task\n1. Read src/aminx/host/_sampling_helper.py in full around _prepare_fixed_controls.\n2. Read the RunSpec/SamplingSpecification definition to confirm the type and\n   semantics of fixed_mask.\n3. In _prepare_fixed_controls, after handling fixed_positions, also handle fixed_mask:\n   - When spec.fixed_mask is set (not None), treat True positions as fixed.\n   - Combine with fixed_positions if both are set (union).\n4. Add a test in tests/sampling/test_fixed_positions.py (or new\n   tests/sampling/test_fixed_mask.py) that:\n   - Creates a spec with fixed_mask set (no fixed_positions)\n   - Verifies the output fixed tensor reflects the mask\n   - Verifies it differs from an unfixed run\n\n## Acceptance criteria\n- _prepare_fixed_controls reads spec.fixed_mask\n- When fixed_mask is set, the resulting fixed tensor reflects it\n- New test covers the fixed_mask path and passes\n- uv run pytest tests/sampling/test_fixed_positions.py -v all pass\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track B Reviewer — fixed_mask gap (#1766)\n\nVERIFY: grep -n "fixed_mask" src/aminx/host/_sampling_helper.py\n  Must show at least one read of spec.fixed_mask inside _prepare_fixed_controls.\n\nVERIFY: A test exercises the fixed_mask path specifically\n  Check: grep -rn "fixed_mask" tests/sampling/\n\nVERIFY: uv run pytest tests/sampling/ -v --tb=short\n  No failures. New fixed_mask test(s) present and PASS.\n\nPASS if all VERIFY items are satisfied.\nFAIL if fixed_mask is still unread, no test was added, or sampling tests regress.\n`,
    "worktree"
  );

// ===== TRACK C — Track C — Wire atom_37/atom_37_mask/chain_mask into build_inference_bundle (#1765) =========================
const trackC = () =>
  track(
    "1765",
    "Track C — Wire atom_37/atom_37_mask/chain_mask into build_inference_bundle (#1765)",
    `task_id: ${TASK_ID}. # Track C — atom_37 wiring gap (backlog #1765)\n\n## Context\nsrc/aminx/host/kernel_dispatch.py lines 191-215 contain the call sites for\nbuild_inference_bundle. LigandContext carries atom_37, atom_37_mask, and\nchain_mask but these are never forwarded into the bundle — silently dropped\nat the host dispatch boundary.\n\n## Task\n1. Read src/aminx/host/kernel_dispatch.py fully around lines 191-215.\n2. Read the LigandContext definition to confirm which fields it carries.\n3. Read build_inference_bundle's signature to confirm it accepts these fields.\n4. Forward atom_37, atom_37_mask, chain_mask from ligand_context into the\n   build_inference_bundle call site(s).\n   - Handle the case where ligand_context is None (preserve existing behaviour).\n5. Add or update a test that passes a LigandContext with non-trivial atom_37\n   and asserts it reaches the bundle.\n\n## Acceptance criteria\n- atom_37, atom_37_mask, chain_mask from ligand_context are forwarded when present\n- When ligand_context is None, behaviour is unchanged\n- Test covers the forwarding path\n- uv run pytest tests/ -k "kernel_dispatch or ligand or dispatch" -v passes\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track C Reviewer — atom_37 wiring (#1765)\n\nVERIFY: grep -n "atom_37\|atom_37_mask\|chain_mask" src/aminx/host/kernel_dispatch.py\n  Must show these fields being read from ligand_context and passed to build_inference_bundle.\n\nVERIFY: The None-ligand_context path is guarded (no AttributeError when ligand_context=None)\n  Check: grep -n "ligand_context" src/aminx/host/kernel_dispatch.py | head -20\n\nVERIFY: uv run pytest tests/ -k "kernel_dispatch or ligand or dispatch" -v --tb=short\n  No failures. New test exercises the forwarding path.\n\nPASS if all VERIFY items are satisfied.\nFAIL if fields are still not forwarded, None guard is missing, or tests regress.\n`,
    "worktree"
  );

// ===== TRACK D — Track D — Training parity gate: checkpoint round-trip + loss curve (G1 #1550) =========================
const trackD = () =>
  agent(
    `task_id: ${TASK_ID}. # Track D — Training parity gate (G1, backlog #1550)\n\n## Context\nTrack I (prior sprint) rewrote checkpoint.py to use xtrax.checkpoint.orbax\n(PyTree format, breaking change from ocp.args.Composite). The new API:\n  save_checkpoint(manager, resumable_state)\n  load_checkpoint(manager, state_template, step=None)\n  get_checkpoint_manager(directory, max_to_keep=N)\n\n8 tests in tests/training/ pass locally. This gate confirms training correctness.\n\n## Task\n1. Run full training test suite:\n     uv run pytest tests/training/ -v\n   Report all results.\n\n2. Run checkpoint round-trip smoke:\n     uv run python -c "\n     import jax, jax.numpy as jnp, equinox as eqx, tempfile, pathlib\n     from xtrax.training.types import ResumableState\n     from aminx.training.checkpoint import get_checkpoint_manager, save_checkpoint, load_checkpoint\n     key = jax.random.PRNGKey(0)\n     model = eqx.nn.Linear(8, 8, key=key)\n     state = ResumableState(step=jnp.int32(42), key=key, model=model, opt_state=None, extras={})\n     with tempfile.TemporaryDirectory() as d:\n         mgr = get_checkpoint_manager(pathlib.Path(d), max_to_keep=1)\n         save_checkpoint(mgr, state); mgr.close()\n         mgr2 = get_checkpoint_manager(pathlib.Path(d), max_to_keep=1)\n         restored = load_checkpoint(mgr2, state); mgr2.close()\n         assert int(restored.step) == 42\n         print('PARITY GATE: PASS')\n     "\n\n3. Write .praxia/docs/research/260613_training-parity-gate.md with results.\n\n## Acceptance criteria\n- tests/training/ all pass\n- Checkpoint round-trip smoke prints PARITY GATE: PASS\n- Research doc created\n`,
    { agentType: "librarian", label: "research:1550", phase: "Track D — Training parity gate: checkpoint round-trip + loss curve (G1 #1550)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK E — Track E — RS-1 inventory: host/* getattr(spec) → run_spec migration map (#1620) =========================
const trackE = () =>
  agent(
    `task_id: ${TASK_ID}. # Track E — RS-1 host-field inventory (backlog #1620)\n\n## Context\nRS-2+ RunSpec unification items require knowing exactly which spec fields\nare read in host/* so the migration can be planned safely.\n\n## Task\nGrep every place in src/aminx/host/ that reads directly from the spec:\n  grep -rn "spec\.\|getattr(spec" src/aminx/host/ | grep -v "run_spec\."\n\nFocus on:\n  src/aminx/host/runner.py\n  src/aminx/host/kernel_dispatch.py\n  src/aminx/host/_sampling_helper.py\n  src/aminx/host/_scoring_helper.py (if exists)\n\nFor each hit record:\n  - file:line\n  - field name\n  - target RunSpec subconfig (sampling, plan, precision, etc.)\n    OR "protein-only facade" if it's a protein-domain field that stays flat\n  - Already reading run_spec? (yes/no)\n\n## Deliverable\nWrite .praxia/docs/research/260613_runspec-host-field-inventory.md with:\n1. Full field inventory table\n2. Fields already on run_spec (done)\n3. Fields needing migration + target subconfig\n4. Protein-only fields that stay flat\n5. Any fields in spec with no run_spec equivalent (gaps that need RS additions)\n\n## Acceptance criteria\n- Doc exists with populated table covering the three key host files\n- At least 10 entries catalogued\n`,
    { agentType: "librarian", label: "research:1620", phase: "Track E — RS-1 inventory: host/* getattr(spec) → run_spec migration map (#1620)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK F — Track F — T0.3 L3 cluster smoke: py3.13 + JAX/jaxlib on SM120 + bathos sidecar (#1544) =========================
const trackF = () =>
  track(
    "1544",
    "Track F — T0.3 L3 cluster smoke: py3.13 + JAX/jaxlib on SM120 + bathos sidecar (#1544)",
    `task_id: ${TASK_ID}. # Track F — T0.3 cluster smoke (backlog #1544)\n\n## Context\nSM120 (Blackwell) nodes node4007/node4008 (partition pi_so3) require:\n  XLA_FLAGS=--xla_gpu_shard_autotuning=false\nWithout this flag XLA autotuning hangs (1170× slowdown, SLURM job 15294138).\nsbatch/squeue/sacct are transparent SSH wrappers to engaging — run them locally.\n\n## Task\n1. Write scripts/cluster/smoke_sm120.py:\n   - import xtrax, jax, aminx\n   - print(jax.devices()) — assert at least one GPU\n   - run a tiny jax.jit'd jnp.dot(A, B) to confirm compilation\n   - print("SMOKE: PASS")\n\n2. Write scripts/cluster/smoke_sm120.sbatch:\n   - #SBATCH --partition=pi_so3\n   - #SBATCH --nodes=1 --ntasks=1 --gpus=1\n   - #SBATCH --time=00:10:00\n   - XLA_FLAGS SM120 workaround (key on SLURM_JOB_NODELIST for node4007/node4008)\n   - SLURM_SUBMIT_DIR path anchoring (not BASH_SOURCE)\n   - uv run python scripts/cluster/smoke_sm120.py\n\n3. Submit: use mcp__myxcel__submit_job or sbatch scripts/cluster/smoke_sm120.sbatch\n   Wait for completion with mcp__myxcel__job_wait.\n\n4. Retrieve logs; confirm "SMOKE: PASS" present.\n\n5. Write .praxia/docs/research/260613_t03-sm120-smoke.md with job ID, node, result.\n\n## Acceptance criteria\n- scripts/cluster/smoke_sm120.py and .sbatch committed\n- Job completes (COMPLETED, not TIMEOUT/FAILED)\n- "SMOKE: PASS" in job output\n- Research doc records job ID, node, result\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track F Reviewer — T0.3 cluster smoke (#1544)\n\nVERIFY: scripts/cluster/smoke_sm120.py and scripts/cluster/smoke_sm120.sbatch exist\nVERIFY: .sbatch includes XLA_FLAGS workaround for node4007/node4008\nVERIFY: .praxia/docs/research/260613_t03-sm120-smoke.md exists with job ID + result\nVERIFY: Research doc states "SMOKE: PASS"\n\nPASS if all items satisfied.\nFAIL if scripts missing, XLA workaround absent, job failed/timed out, or doc missing.\n`,
    "worktree"
  );

// ---- orchestrate: sequential writing chain || read-only research ----------
log("260613_host-gaps-parity-gates: writing chain (A -> B -> C, sequential) || research (D, E, F, read-only)");
const [writing, resD, resE, resF] = await Promise.all([
  (async () => {
    const a = await trackA();
    const b = await trackB();
    const c = await trackC();
    return { a, b, c };
  })(),
  trackD(),
  trackE(),
  trackF(),
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
  sprint_id: 2,
  verdicts: {
    "1767": writing.a,
    "1766": writing.b,
    "1765": writing.c
  },
  research_1550: resD,
  research_1620: resE,
  research_1544: resF
};
