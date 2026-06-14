// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 260617_poe-rs3-pt-stagebundle-tiling-val   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain () runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time. Only the read-only
//   research/concurrent tracks (A,B,C,D,E) run concurrently.

export const meta = {
  name: "260617_poe-rs3-pt-stagebundle-tiling-val",
  description: "PoE sanity baseline, RS-3 CarrySpec wiring, PT edge-case tests, StageBundle adapter, xtrax-prolix tiling validation",
  phases: [
    { title: "Track A -- P-00d: PoE energy sanity check -- run BATHOS validation script (#1305)" },
    { title: "Track B -- [RunSpec RS-3] Wire CarrySpec/DedupSpec from SamplingSpecification -> BatchPlanner (#1622)" },
    { title: "Track C -- [debt] _parallel_tempering_exchange edge-case tests for replica-count boundary conditions (#1861)" },
    { title: "Track D -- [xtrax T2.6] aminx StageBundle wrap-adapter preserving N-sink multiplicity (#1555)" },
    { title: "Track E -- [xtrax #1564] Validate xtrax tiling CORE API against prolix 6-axis planner (#1599)" },
  ],
};

const TASK_ID = "260617_poe-rs3-pt-stagebundle-tiling-val";
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

// Shared context for the writing tracks (from recon, task 260617_poe-rs3-pt-stagebundle-tiling-val).
const EMITTER_CTX = `Recon completed 2026-06-14. #1300 done (alphabet_map implemented). #1305 has ready script never run. #1622 is 2-file wiring. #1861 zero boundary tests. #1555 needs StageSet->StageBundle adapter. #1599 needs falsifiable prolix compat test. All 5 independent.`;

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

// ===== TRACK A — Track A -- P-00d: PoE energy sanity check -- run BATHOS validation script (#1305) =========================
const trackA = () =>
  track(
    "1305",
    "Track A -- P-00d: PoE energy sanity check -- run BATHOS validation script (#1305)",
    `task_id: ${TASK_ID}. task_id: 260617_poe-rs3-pt-stagebundle-tiling-val\nworkspace: /home/marielle/projects/aminx\n\n## Context\nThe PoE sanity script exists and is fully implemented but has never been executed.\nRun it. If it fails, fix the implementation. If it passes, the deliverable is the run output.\n\n## Key files\n- scripts/validate/poe_energy_sanity.py (282 lines) -- BATHOS validation script\n- scripts/validate/poe_energy_sanity.bth.toml -- sidecar with 2 pre-registered hypotheses\n- src/aminx/potts/poe.py:208 -- joint_energy; :186 -- eqx.filter_vmap over backbones\n- tests/potts/test_poe.py:690 -- test_poe_sanity_check_two_identical_backbones\n\n## Hypotheses\nH1: PoeModel([p,p]).joint_energy(seq) approx 2*PottsModel.log_prob(seq)  tol=1e-5\nH2: energy(2*h, 2*J) approx 2*energy(h,J)  tol=1e-6\n\n## Steps\n1. uv run python scripts/validate/poe_energy_sanity.py --dry-run\n   Must exit 0.\n2. uv run python scripts/validate/poe_energy_sanity.py --n-residues 6 --num-aa 20 --seed 0 2>&1 | tee /tmp/poe_sanity.log\n   Must exit 0 and print "ALL ASSERTIONS PASSED".\n3. uv run pytest tests/potts/test_poe.py -x -q --tb=short 2>&1 | tail -20\n4. If L2 fails: fix PoeModel or the script, then re-run. Do NOT widen tolerances.\n5. uv run pytest tests/potts/ -x -q --tb=short 2>&1 | tail -20 before declaring done.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260617_poe-rs3-pt-stagebundle-tiling-val\nworkspace: /home/marielle/projects/aminx\n\n## Checks\nVERIFY: uv run python scripts/validate/poe_energy_sanity.py --dry-run exits 0\nVERIFY: uv run python scripts/validate/poe_energy_sanity.py --n-residues 6 --num-aa 20 --seed 0 exits 0 AND output contains "ALL ASSERTIONS PASSED"\nVERIFY: uv run pytest tests/potts/test_poe.py -x -q passes (including test_poe_sanity_check_two_identical_backbones)\nVERIFY: uv run pytest tests/potts/ -x -q 2>&1 | tail -5 -- all existing tests pass\n\nPASS if all VERIFY items satisfied.\nFAIL if script exits non-zero, "ALL ASSERTIONS PASSED" absent from output, or any test fails.\n`,
    "worktree"
  );

// ===== TRACK B — Track B -- [RunSpec RS-3] Wire CarrySpec/DedupSpec from SamplingSpecification -> BatchPlanner (#1622) =========================
const trackB = () =>
  track(
    "1622",
    "Track B -- [RunSpec RS-3] Wire CarrySpec/DedupSpec from SamplingSpecification -> BatchPlanner (#1622)",
    `task_id: ${TASK_ID}. task_id: 260617_poe-rs3-pt-stagebundle-tiling-val\nworkspace: /home/marielle/projects/aminx\n\n## Context\nBatchPlanner (plan.py:114-115) already has carries/dedup_specs fields but they always\nget empty defaults because SamplingSpecification never declares or passes them. 2-file fix.\n\n## Key files\n- src/aminx/run/specs.py:305-409 -- SamplingSpecification (no carry_specs/dedup_specs)\n- src/aminx/host/plan.py:96-100 -- make_sampling_planner BatchPlanner(...) call\n- src/aminx/host/plan.py:104-115 -- BatchPlanner: carries at 114, dedup_specs at 115\n- src/aminx/tiling/carry.py:22-44 -- CarrySpec (local)\n- src/aminx/tiling/dedup.py:47-84 -- DedupSpec (local)\n\n## Part 1: specs.py\nAdd to SamplingSpecification:\n  carry_specs: list[CarrySpec] = field(default_factory=list)\n  dedup_specs: list[DedupSpec] = field(default_factory=list)\nAdd imports: from aminx.tiling.carry import CarrySpec; from aminx.tiling.dedup import DedupSpec\n\n## Part 2: plan.py\nIn make_sampling_planner, add to BatchPlanner(...):\n  carries=getattr(spec, "carry_specs", []),\n  dedup_specs=getattr(spec, "dedup_specs", []),\n\n## Part 3: Test\nNew test in tests/host/ or tests/run/:\n  (a) SamplingSpecification with carry_specs=[CarrySpec(...)] and dedup_specs=[DedupSpec(...)]\n  (b) Call make_sampling_planner and assert BatchPlanner.carries and .dedup_specs are non-empty\n\n## Constraints\n- Edit only, no new source files (test file OK).\n- uv run pytest tests/ -x -q --tb=short -k "not chex" 2>&1 | tail -20 before done.\n- uv run ty check 2>&1 | tail -20 -- no new type errors.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260617_poe-rs3-pt-stagebundle-tiling-val\nworkspace: /home/marielle/projects/aminx\n\n## Checks\nVERIFY: SamplingSpecification has carry_specs and dedup_specs fields with default_factory=list\nVERIFY: make_sampling_planner passes carries= and dedup_specs= to BatchPlanner constructor\nVERIFY: New test asserts non-empty carry_specs/dedup_specs reach BatchPlanner\nVERIFY: uv run pytest tests/ -x -q --tb=short -k "not chex" 2>&1 | tail -10 -- all pass\nVERIFY: uv run ty check -- no new type errors\n\nPASS if all satisfied. FAIL if any field/wiring/test missing, or tests fail.\n`,
    "worktree"
  );

// ===== TRACK C — Track C -- [debt] _parallel_tempering_exchange edge-case tests for replica-count boundary conditions (#1861) =========================
const trackC = () =>
  track(
    "1861",
    "Track C -- [debt] _parallel_tempering_exchange edge-case tests for replica-count boundary conditions (#1861)",
    `task_id: ${TASK_ID}. task_id: 260617_poe-rs3-pt-stagebundle-tiling-val\nworkspace: /home/marielle/projects/aminx\n\n## Context\nP-06 vectorised _parallel_tempering_exchange with vmap but zero tests cover replica-count\nboundary conditions where parity logic changes behavior.\n\n## Key files\n- src/aminx/potts/sampling.py:186 -- _parallel_tempering_exchange signature -> (seqs, accept, betas_out) accept shape (k-1,)\n- src/aminx/potts/sampling.py:207 -- process_parity: edges = jnp.arange(parity, k_rep-1, 2)\n  k=1: both parities empty (accept_edge shape (0,)); k=2: even=[0], odd=[] (odd is no-op)\n- tests/potts/test_sampling.py:170 -- only n_replicas=4, no accept_edge shape assertion\n- tests/potts/test_sampling.py:189 -- only n_replicas=3\n\n## Add tests (to test_sampling.py or new tests/potts/test_pt_edge_cases.py)\n\n1. @pytest.mark.parametrize("n_replicas", [1, 2, 3, 4, 5])\n   test: assert accept_edge.shape == (n_replicas - 1,)\n\n2. n_replicas=1: assert seqs_out equals seqs_in (no swap possible)\n\n3. n_replicas=2: function completes without error; odd parity is a no-op\n\nUse _parallel_tempering_exchange directly if importable, else via public\nparallel_tempering(..., n_replicas=N, n_sweeps=1). Use PRNGKey(0), N=4 residues, q=5.\n\n## Constraints\n- Test only. Do not modify sampling.py.\n- uv run pytest tests/potts/ -x -q --tb=short 2>&1 | tail -20 before done.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260617_poe-rs3-pt-stagebundle-tiling-val\nworkspace: /home/marielle/projects/aminx\n\n## Checks\nVERIFY: Tests cover n_replicas in [1, 2, 3, 4, 5] (parametrized or individual)\nVERIFY: accept_edge.shape == (n_replicas - 1,) asserted for each value\nVERIFY: n_replicas=1 test asserts sequences unchanged\nVERIFY: n_replicas=2 test asserts function completes without error\nVERIFY: uv run pytest tests/potts/ -x -q --tb=short 2>&1 | tail -10 -- all pass\n\nPASS if all satisfied. FAIL if n_replicas=1 or 2 absent, shape assertion missing, or tests fail.\n`,
    "worktree"
  );

// ===== TRACK D — Track D -- [xtrax T2.6] aminx StageBundle wrap-adapter preserving N-sink multiplicity (#1555) =========================
const trackD = () =>
  track(
    "1555",
    "Track D -- [xtrax T2.6] aminx StageBundle wrap-adapter preserving N-sink multiplicity (#1555)",
    `task_id: ${TASK_ID}. task_id: 260617_poe-rs3-pt-stagebundle-tiling-val\nworkspace: /home/marielle/projects/aminx\n\n## Context\nxtrax StageBundle (xtrax/stages/bundle.py:22) is a strict eqx.Module with Optional[Callable]\nper stage field. aminx StageSet (src/aminx/types/stages.py:266) has tuple[EncoderSinkFn, ...]\nat lines 344-345. Build an adapter in aminx that wraps StageSet and presents the xtrax\nStageBundle interface without collapsing N sinks into one.\n\n## Key files (read before implementing)\n- /home/marielle/projects/xtrax/src/xtrax/stages/bundle.py:22 -- StageBundle (read fully)\n- src/aminx/types/stages.py:266 -- StageSet\n- src/aminx/types/stages.py:344-345 -- encoder_sink: tuple[...]; decoder_sink: tuple[...]\n- src/aminx/host/streaming_host.py -- how sinks fire today (firing order context)\n\n## Design\nclass StageBundleAdapter(eqx.Module):\n  _stage_set: StageSet\n\n  def _chain(self, sinks: tuple) -> Callable | None:\n    if not sinks: return None\n    def chained(*args, **kwargs):\n      for fn in sinks: fn(*args, **kwargs)\n    return chained\n\n  @property\n  def encoder_sink(self) -> Callable | None:\n    return self._chain(self._stage_set.encoder_sink)\n  # similarly for decoder_sink, encode_fn, decode_fn\n  # active_stages(), has_stage(name) -- inspect which properties are non-None\n\nCreate: src/aminx/host/stage_adapter.py\n\n## Test (AC-2): tests/host/test_stage_bundle_adapter.py\n- StageSet with 2 encoder sinks (each appends to a shared list for side-effect capture)\n- Wrap in StageBundleAdapter\n- Call encoder_sink() callable\n- Assert BOTH sinks fired AND in order (list has 2 entries in correct sequence)\n- Assert active_stages() returns correct names\n\n## Constraints\n- Do NOT modify xtrax source. Do NOT modify StageSet.\n- uv run pytest tests/host/ -x -q --tb=short 2>&1 | tail -20 before done.\n- uv run ty check 2>&1 | tail -20 -- no new type errors.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260617_poe-rs3-pt-stagebundle-tiling-val\nworkspace: /home/marielle/projects/aminx\n\n## Checks\nVERIFY: StageBundleAdapter (or equivalent) exists in src/aminx/host/ and is an eqx.Module\nVERIFY: encoder_sink / decoder_sink expose Optional[Callable] (not tuples) to callers\nVERIFY: AC-2 test asserts BOTH sinks fire in order (2-sink side-effect capture)\nVERIFY: active_stages() and has_stage(name) work correctly on the adapter\nVERIFY: uv run pytest tests/host/ -x -q --tb=short 2>&1 | tail -10 -- all pass including AC-2\nVERIFY: uv run ty check 2>&1 | tail -10 -- no new type errors\n\nPASS if all satisfied. FAIL if N-sink multiplicity collapsed, protocol absent, or tests fail.\n`,
    "worktree"
  );

// ===== TRACK E — Track E -- [xtrax #1564] Validate xtrax tiling CORE API against prolix 6-axis planner (#1599) =========================
const trackE = () =>
  agent(
    `task_id: ${TASK_ID}. task_id: 260617_poe-rs3-pt-stagebundle-tiling-val\nworkspace: /home/marielle/projects/aminx\nxtrax: /home/marielle/projects/xtrax\nprolix: /home/marielle/projects/prolix\n\n## Context\nWrite and run a falsifiable test: can prolix's 6-axis planner work using only\nxtrax.tiling CORE symbols? Both PASS and FAIL are valid -- FAIL documents the gap.\n\n## Key files\n- /home/marielle/projects/prolix/src/prolix/tiling/axes.py -- 6 AxisSpec instances\n  N_CONFORMERS (heterogeneous=True) + N_MOLS/N_SYSTEMS (heterogeneous=True)\n- /home/marielle/projects/xtrax/src/xtrax/tiling/__init__.py -- xtrax.tiling exports\n- /home/marielle/projects/aminx/.praxia/docs/specs/260611_aminx-xtrax-refactor.md:105-132\n  -- CORE vs OPTIONAL split table\n- /home/marielle/projects/aminx/.praxia/docs/specs/260611_aminx-xtrax-refactor.md:125-126\n  -- exact D7 criterion\n\n## Test to write: /home/marielle/projects/xtrax/tests/tiling/test_prolix_compat.py\n  (fallback: /home/marielle/projects/aminx/tests/tiling/test_xtrax_prolix_compat.py)\n\n1. Import CORE only:\n   from xtrax.tiling import AxisSpec, BatchPlanner\n\n2. Replicate prolix's 6 axes with xtrax.AxisSpec (read prolix/tiling/axes.py for constructor args):\n   N_ATOMS, N_BONDS, N_ANGLES, N_TORSIONS = ... (non-heterogeneous)\n   N_CONFORMERS = AxisSpec(name="n_conformers", ..., heterogeneous=True)\n   N_MOLS = AxisSpec(name="n_mols", ..., heterogeneous=True)\n\n3. Build plan:\n   planner = BatchPlanner(axes=[N_ATOMS,N_BONDS,N_ANGLES,N_TORSIONS,N_CONFORMERS,N_MOLS],\n                          budget_bytes=8*1024**3, estimate_memory=lambda ax,bs: bs*4)\n   plan = planner.plan()\n\n4. Falsifiable assertions:\n   assert plan.decision_for("n_mols").batch_size > 0\n   assert plan.decision_for("n_conformers").batch_size > 0\n\n5. CORE non-leaky assertions:\n   import xtrax.tiling as m\n   assert not hasattr(m, "DedupGather")\n   assert not hasattr(m, "io_callback_sink")\n\nRun: cd /home/marielle/projects/xtrax && uv run pytest tests/tiling/test_prolix_compat.py -v --tb=short 2>&1 | tee /tmp/prolix_compat.log\n\n## Deliverable: .praxia/docs/research/260617_xtrax-tiling-prolix-compat.md\nPASS: "xtrax.tiling CORE sufficient for prolix 6-axis planner as of xtrax <sha>."\nFAIL: "xtrax.tiling CORE insufficient: <error>. Gap: <what xtrax is missing>."\n\n## Constraints\n- Do NOT modify xtrax or prolix source.\n- Run the test. Record the result. Write the summary doc.\n`,
    { agentType: "librarian", label: "research:1599", phase: "Track E -- [xtrax #1564] Validate xtrax tiling CORE API against prolix 6-axis planner (#1599)", schema: RESEARCH_SCHEMA }
  );

// ---- orchestrate: sequential writing chain || read-only research ----------
log("260617_poe-rs3-pt-stagebundle-tiling-val: writing chain (, sequential) || research (A, B, C, D, E, read-only)");
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
  research_1305: resA,
  research_1622: resB,
  research_1861: resC,
  research_1555: resD,
  research_1599: resE
};
