// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 260614_potts-runspec-xtrax-gates   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain () runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time. Only the read-only
//   research/concurrent tracks (A,B,C1,C2,D,E,F,G,H) run concurrently.

export const meta = {
  name: "260614_potts-runspec-xtrax-gates",
  description: "Potts sampling layer (P-05/P-06/P-07/P-09/P-00c), RunSpec foundation (RS-1), xtrax phase gates (G1, T0.3), and T2.1 kickoff — completing the Potts define+test+guard layer and unblocking the full RS and T2 fan-out.",
  phases: [
    { title: "Track A — P-06: aminx.potts.sampling — Gibbs + parallel tempering as pure JAX functions (#1295)" },
    { title: "Track B — P-09: Tests — TRW marginals vs brute-force exact_marginals (n<=12) (#1294)" },
    { title: "Track C (item 1) — P-05: aminx.potts.poe — PoeModel(eqx.Module) for N-backbone PoE (#1296)" },
    { title: "Track C (item 2) — P-00c: ast-grep import boundary guardrail for aminx.potts (#1304)" },
    { title: "Track D — P-07: spec emit-* CLI for PottsRunSpec (aminx spec emit-potts) (#1297)" },
    { title: "Track E — G1 Training parity gate: loss curve + checkpoint round-trip (#1550)" },
    { title: "Track F — RS-1: Inventory host/* getattr(spec) -> run_spec migration map (#1620)" },
    { title: "Track G — T0.3: L3 cluster smoke on SM120 + bathos sidecar (#1544)" },
    { title: "Track H — T2.1 (xtrax repo): Add Scan.init field to xtrax tiling Scan strategy (#1551)" },
  ],
};

const TASK_ID = "260614_potts-runspec-xtrax-gates";
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

// Shared context for the writing tracks (from recon, task 260614_potts-runspec-xtrax-gates).
const EMITTER_CTX = `Prior sprint (260613_host-gaps-parity-gates) closed three host-dispatch bugs:\n- Track A: STE dispatch restored in averaging.py (backlog #1767)\n- Track B: fixed_mask branch wired in _prepare_fixed_controls (backlog #1766)\n- Track C: atom_37/atom_37_mask/chain_mask forwarded in kernel_dispatch.py (backlog #1765)\nTracks D (G1 #1550), E (RS-1 #1620), F (T0.3 #1544) were planned but not executed.\n\nPrior sprint (260613_xtrax-t0-potts-core) completed:\n- T0.2: xtrax editable pin verified (#1543)\n- T0.4: xtrax boundary lint rules (#1545)\n- P-01: PottsMPNN .pt -> potts_<id>.eqx.zst recapture script (#1291)\n- P-03: PottsModel(eqx.Module) with DifferentiableTRW (#1292)\n- P-04: PottsRunSpec frozen dataclass + JSON round-trip (#1293)\n\nResolved risks (pre-sprint):\n- G1 pass criteria: pytest tests/training/ all pass + checkpoint round-trip allclose(atol=1e-7) + loss decrease >=10% over 50 steps\n- T0.3 cluster: node4007/4008 alloc but will queue (15min walltime, not a blocker)\n- T2.1 cross-repo: xtrax at /home/marielle/projects/xtrax, editable=true (pyproject.toml:274)\n- P-05 dep (#1293 P-04): confirmed closed\n- RS-1 output: locked to .praxia/docs/plans/260614_runspec-migration-map.md\n\nKey anchors:\n- src/aminx/potts/ — active Potts module (P-03/P-04 landed; P-05/P-06/P-07/P-09 are next)\n- src/aminx/potts/model.py — PottsModel (P-03, complete)\n- src/aminx/potts/spec.py — PottsRunSpec (P-04, complete)\n- src/aminx/training/ — checkpoint.py + trainer.py (xtrax ResumableState, G1 gate)\n- src/aminx/host/ — runner/kernel_dispatch/_sampling_helper (RS-1 scan target)\n- /home/marielle/projects/xtrax/ — xtrax source (editable pin at ../xtrax)\n- pyproject.toml:274 — xtrax = { path = "../xtrax", editable = true }\n- pi_so3 partition: node4007/node4008 (SM120 Blackwell); XLA_FLAGS workaround mandatory\n`;

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

// ===== TRACK A — Track A — P-06: aminx.potts.sampling — Gibbs + parallel tempering as pure JAX functions (#1295) =========================
const trackA = () =>
  track(
    "1295",
    "Track A — P-06: aminx.potts.sampling — Gibbs + parallel tempering as pure JAX functions (#1295)",
    `task_id: ${TASK_ID}. # P-06: aminx.potts.sampling (backlog #1295)\n\n## Context\nPottsModel (P-03) and PottsRunSpec (P-04) are complete. This track implements the sampling\nlayer: single-sweep Gibbs sampling and parallel tempering, both as pure JAX functions\n(no side effects, explicit PRNG keys, jit-compatible).\n\n## What to implement\n\n### src/aminx/potts/sampling.py\n\n**gibbs_sweep(key, seq, h, J, w, mask, temperature=1.0)**\n  - Single Gibbs sweep over all positions (or masked subset)\n  - For each position i: compute conditional p(a_i | a_{-i}) from (h_i, J_{ij}, w)\n  - Sample from categorical; return updated seq\n  - Pure JAX: no Python loops over residues; use jax.lax.scan or jax.vmap\n  - Types: seq: Int32[N], h: Float32[N, q], J: Float32[N, N, q, q], mask: Bool[N]\n  - temperature scales logits before softmax\n\n**parallel_tempering(key, seq, h, J, w, mask, n_replicas=4, temperatures=None, n_sweeps=100)**\n  - n_replicas chains at different temperatures (default: linspace(0.5, 2.0, n_replicas))\n  - Each sweep: gibbs_sweep all replicas, then propose adjacent swaps (Metropolis-Hastings)\n  - Return (final_seqs: Int32[n_replicas, N], energies: Float32[n_replicas])\n  - Pure JAX; use jax.lax.scan for outer sweep loop\n\n**log_energy(seq, h, J, w, mask)**\n  - Potts energy E(seq) = -sum_i h[i,seq[i]] - sum_{i<j} J[i,j,seq[i],seq[j]]\n  - Pure function; Float32 scalar output\n\nExport all three from src/aminx/potts/__init__.py.\n\n## Tests — tests/potts/test_sampling.py\n1. test_gibbs_sweep_changes_sequence: correct output shape Int32[N]\n2. test_gibbs_sweep_jit_compatible: eqx.filter_jit(gibbs_sweep)(...) does not raise\n3. test_log_energy_synthetic: known 2-residue system, allclose(atol=1e-5)\n4. test_parallel_tempering_shape: seqs.shape==(n_replicas,N), energies.shape==(n_replicas,)\n\n## Acceptance criteria\n- uv run pytest tests/potts/test_sampling.py -v — all 4 tests pass\n- uv run ty check src/aminx/potts/sampling.py — exits 0\n- gibbs_sweep, parallel_tempering, log_energy exported from aminx.potts\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track A Reviewer — P-06 Potts Sampling (#1295)\n\nVERIFY: src/aminx/potts/sampling.py exists with gibbs_sweep, parallel_tempering, log_energy\n  Check: grep -n "def gibbs_sweep\|def parallel_tempering\|def log_energy" src/aminx/potts/sampling.py\n\nVERIFY: Pure JAX — no Python loops over residues (lax.scan/vmap OK)\n  Check: grep -n "^    for\|^        for" src/aminx/potts/sampling.py | grep -v "lax"\n\nVERIFY: uv run pytest tests/potts/test_sampling.py -v — all 4 tests pass, zero skips\n\nVERIFY: uv run ty check src/aminx/potts/sampling.py — exits 0\n\nVERIFY: grep "gibbs_sweep\|parallel_tempering\|log_energy" src/aminx/potts/__init__.py\n\nPASS if all VERIFY items satisfied.\nFAIL if sampling.py missing, Python residue loops present, any test skips/fails, or ty errors.\n`,
    "worktree"
  );

// ===== TRACK B — Track B — P-09: Tests — TRW marginals vs brute-force exact_marginals (n<=12) (#1294) =========================
const trackB = () =>
  track(
    "1294",
    "Track B — P-09: Tests — TRW marginals vs brute-force exact_marginals (n<=12) (#1294)",
    `task_id: ${TASK_ID}. # P-09: TRW marginals vs brute-force exact_marginals (backlog #1294)\n\n## Context\nPottsModel with DifferentiableTRW is implemented (P-03). This track writes the correctness\ngate: for small systems (n<=12 residues), compare TRW marginals against exact brute-force\nmarginals computed by summing over all q^n configurations.\n\n## What to implement — tests/potts/test_trw_marginals.py\n\n**helper: exact_marginals(h, J, mask, q=4)**\n  - Enumerate all q^n valid sequences (n=mask.sum()); use q=4 toy alphabet for tractability\n  - Compute unnorm prob: exp(-E(seq)), E = -sum_i h[i,s[i]] - sum_{i<j} J[i,j,s[i],s[j]]\n  - Return site marginals p_i(a) = sum_{seq: seq[i]=a} p(seq); shape: Float32[N, q]\n  - q=4 gives 4^12=16M configs in ~1s (tractable)\n\n**test_trw_vs_exact_n6_q4**: 6 residues, q=4, random h/J, temperature=1.0\n  Call DifferentiableTRW directly (check import: aminx.potts.model or aminx.potts.trw)\n  Compare TRW site marginals vs exact_marginals: allclose(atol=0.05, rtol=0.05)\n\n**test_trw_vs_exact_n4_q4_ferromagnet**: 4-residue ferromagnet (J_{ij}=-1 all i!=j, h=0)\n  Ground truth: uniform marginals by symmetry. Assert max deviation from uniform < 0.02.\n\n**test_trw_marginals_sum_to_one**:\n  allclose(marginals.sum(axis=-1)[mask], 1.0, atol=1e-4)\n\n## Acceptance criteria\n- tests/potts/test_trw_marginals.py with 3 tests\n- uv run pytest tests/potts/test_trw_marginals.py -v — all 3 pass\n- Test runtime < 120s total\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track B Reviewer — P-09 TRW Marginals Tests (#1294)\n\nVERIFY: tests/potts/test_trw_marginals.py exists with >= 3 tests\n  Check: grep -n "def test_" tests/potts/test_trw_marginals.py\n\nVERIFY: exact_marginals uses q=4 (not q=20)\n  Check: grep -n "q=4\|q = 4" tests/potts/test_trw_marginals.py\n\nVERIFY: uv run pytest tests/potts/test_trw_marginals.py -v — all 3 pass, zero skips\n\nVERIFY: ferromagnet test asserts max deviation from uniform < 0.02\nVERIFY: sum_to_one test uses atol <= 1e-4\n\nPASS if all VERIFY items satisfied.\nFAIL if test file missing, runtime > 120s, or any test skips/fails.\n`,
    "worktree"
  );

// ===== TRACK C1 — Track C (item 1) — P-05: aminx.potts.poe — PoeModel(eqx.Module) for N-backbone PoE (#1296) =========================
const trackC1 = () =>
  track(
    "1296",
    "Track C (item 1) — P-05: aminx.potts.poe — PoeModel(eqx.Module) for N-backbone PoE (#1296)",
    `task_id: ${TASK_ID}. # P-05: aminx.potts.poe — PoeModel(eqx.Module) (backlog #1296)\n\n## Context\nPoE coordinator: given N backbone conformations, runs PottsModel on each and combines\nmarginals as a product (in log space). Multi-backbone aggregation layer above PottsModel.\n\n## Dependencies confirmed closed: P-03 (#1292), P-04 (#1293)\n\n## What to implement — src/aminx/potts/poe.py\n\n**class PoeModel(eqx.Module)**\n  potts_model: PottsModel\n  n_backbones: int = eqx.field(static=True)\n\n  __call__(key, backbone_coords: Float32[B,N,3], mask: Bool[N],\n            residue_index: Int32[N], chain_index: Int32[N]) -> PoeOutput:\n    - Apply potts_model to each backbone via eqx.filter_vmap over B axis\n    - Aggregate: log_poe(a) = sum_b log_marginals_b(a); normalize via softmax\n    - Return PoeOutput(marginals, log_poe_unnorm, per_backbone_marginals)\n\n**class PoeOutput (NamedTuple or frozen dataclass)**\n  marginals: Float32[N, q]\n  log_poe_unnorm: Float32[N, q]\n  per_backbone_marginals: Float32[B, N, q]\n\nExport PoeModel and PoeOutput from src/aminx/potts/__init__.py.\n\n## Tests — tests/potts/test_poe.py\n1. test_poe_single_backbone_matches_potts: n_backbones=1 marginals match PottsModel (atol=1e-5)\n2. test_poe_shape: shapes correct for B=3, N=10, q=20\n3. test_poe_marginals_sum_to_one: poe_marginals.sum(axis=-1) ~= 1.0 (atol=1e-5)\n\nUse synthetic protein inputs (random coords, all-True mask).\n\n## Acceptance criteria\n- uv run pytest tests/potts/test_poe.py -v — all 3 tests pass\n- uv run ty check src/aminx/potts/poe.py — exits 0\n- PoeModel and PoeOutput exported from aminx.potts\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track C (item 1) Reviewer — P-05 PoeModel (#1296)\n\nVERIFY: src/aminx/potts/poe.py has PoeModel(eqx.Module) and PoeOutput\n  Check: grep -n "class PoeModel\|class PoeOutput" src/aminx/potts/poe.py\n\nVERIFY: Uses eqx.filter_vmap (not Python loop) over backbones\n  Check: grep -n "filter_vmap\|vmap" src/aminx/potts/poe.py\n\nVERIFY: PoE aggregation in log space\n  Check: grep -n "log_\|jnp.log\|log_softmax\|softmax" src/aminx/potts/poe.py\n\nVERIFY: uv run pytest tests/potts/test_poe.py -v — all 3 pass, zero skips\n\nVERIFY: uv run ty check src/aminx/potts/poe.py — exits 0\n\nVERIFY: grep "PoeModel\|PoeOutput" src/aminx/potts/__init__.py — both exported\n\nPASS if all VERIFY items satisfied.\nFAIL if poe.py missing, Python loop over backbones, any test fails, or ty errors.\n`,
    "worktree"
  );

const trackC2 = () =>
  track(
    "1304",
    "Track C (item 2) — P-00c: ast-grep import boundary guardrail for aminx.potts (#1304)",
    `task_id: ${TASK_ID}. # P-00c: Import boundary guardrail for aminx.potts (backlog #1304)\n\n## Context\naminx.potts must NOT import from aminx.model, aminx.host, or aminx.run.\npotts/ is a pure Potts-math layer; cross-contamination couples it to protein-domain code.\n\n## What to implement — add TestPottsBoundary to tests/potts/test_import_boundaries.py\n\n**test_potts_no_protein_domain_imports**\n  Glob all *.py in src/aminx/potts/; extract imports via AST\n  Assert no import starts with:\n    POTTS_FORBIDDEN = {"aminx.model", "aminx.host", "aminx.run", "aminx.cli"}\n  aminx.potts.*, aminx.io, aminx.training are fine\n\n**test_potts_no_protein_field_names**\n  Scan all *.py in src/aminx/potts/ for substrings:\n    {"atom_37", "atom_37_mask", "residue_index", "chain_index"}\n  Exclude comment lines. Report file + line on failure.\n\nIf any current violations exist: fix them first, then add the test.\n\n## Acceptance criteria\n- TestPottsBoundary class in tests/potts/test_import_boundaries.py\n- uv run pytest tests/potts/test_import_boundaries.py -v — all tests pass\n- grep -rn "from aminx.model\|from aminx.host\|from aminx.run" src/aminx/potts/ returns empty\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track C (item 2) Reviewer — P-00c import guardrail (#1304)\n\nVERIFY: TestPottsBoundary class in tests/potts/test_import_boundaries.py\n  Check: grep -n "TestPottsBoundary" tests/potts/test_import_boundaries.py\n\nVERIFY: POTTS_FORBIDDEN includes aminx.model, aminx.host, aminx.run, aminx.cli\n\nVERIFY: uv run pytest tests/potts/test_import_boundaries.py -v — all tests pass\n\nVERIFY: grep -rn "from aminx.model\|from aminx.host\|from aminx.run" src/aminx/potts/ is empty\n\nPASS if all VERIFY items satisfied.\nFAIL if TestPottsBoundary missing, violations exist, or tests fail.\n`,
    "worktree"
  );

// ===== TRACK D — Track D — P-07: spec emit-* CLI for PottsRunSpec (aminx spec emit-potts) (#1297) =========================
const trackD = () =>
  track(
    "1297",
    "Track D — P-07: spec emit-* CLI for PottsRunSpec (aminx spec emit-potts) (#1297)",
    `task_id: ${TASK_ID}. # P-07: spec emit-* CLI for PottsRunSpec (backlog #1297)\n\n## Context\nPottsRunSpec (P-04) has to_json()/from_json(). Add \`aminx spec emit-potts\` CLI subcommand\nthat prints a JSON template to stdout — analogous to \`aminx spec emit-sample\`.\n\n## Recon first\nRead src/aminx/cli/ or src/aminx/run/ to find where \`spec emit-sample\` is implemented.\nFollow the same pattern exactly.\n\n## What to implement\n\n1. Add emit-potts subcommand:\n   - \`aminx spec emit-potts [--weights-path PATH] [--k-neighbors N] [--n-backbones N]\`\n   - Default: PottsRunSpec(k_neighbors=48, n_backbones=1, weights_path="")\n   - Print stdout as json.dumps(..., indent=2)\n\n2. Wire into CLI entry point if needed.\n\n3. Tests — tests/cli/test_spec_emit_potts.py:\n\n   test_emit_potts_default:\n     subprocess.run(["uv", "run", "aminx", "spec", "emit-potts"], capture_output=True)\n     exits 0; stdout is valid JSON; JSON["k_neighbors"]==48; JSON["n_backbones"]==1\n\n   test_emit_potts_custom_args:\n     [..., "--k-neighbors", "32", "--n-backbones", "4"]\n     JSON["k_neighbors"]==32; JSON["n_backbones"]==4\n\n   test_emit_potts_round_trips:\n     PottsRunSpec.from_json(output) does not raise\n\n## Acceptance criteria\n- \`aminx spec emit-potts\` exits 0 and prints valid JSON\n- uv run pytest tests/cli/test_spec_emit_potts.py -v — all 3 tests pass\n- uv run ty check on modified CLI file — exits 0\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track D Reviewer — P-07 PottsRunSpec CLI (#1297)\n\nVERIFY: uv run aminx spec emit-potts exits 0 and prints JSON\n  Check: uv run aminx spec emit-potts 2>&1 | python3 -m json.tool > /dev/null && echo OK\n\nVERIFY: uv run pytest tests/cli/test_spec_emit_potts.py -v — all 3 tests pass\n\nVERIFY: test_emit_potts_round_trips passes\n\nVERIFY: --k-neighbors 32 -> JSON k_neighbors == 32\n\nVERIFY: uv run ty check on modified CLI file — exits 0\n\nPASS if all VERIFY items satisfied.\nFAIL if command missing, any test fails, or ty errors.\n`,
    "worktree"
  );

// ===== TRACK E — Track E — G1 Training parity gate: loss curve + checkpoint round-trip (#1550) =========================
const trackE = () =>
  agent(
    `task_id: ${TASK_ID}. # G1 Training parity gate (backlog #1550)\n\n## Context\nT1.4 (sprint 260612): trainer.py uses xtrax ResumableState checkpoint (single-PyTree format).\n8 tests in tests/training/ pass. G1 is the formal gate confirming training + checkpoint integrity.\n\n## Pass criteria (explicitly defined for this sprint)\n1. pytest tests/training/ — all pass, zero fail\n2. Checkpoint round-trip: fresh save->close->reopen->load; step==42, all params allclose(atol=1e-7, rtol=0)\n3. Loss curve: if loss/convergence tests exist in tests/training/, they must pass.\n   If none: run 50-step overfit smoke, assert loss at step 50 < loss at step 0 (>=10% decrease).\n\n## Task\n\n1. uv run pytest tests/training/ -v --tb=short 2>&1 | tail -40\n\n2. Checkpoint round-trip smoke:\n   uv run python -c "\nimport jax, jax.numpy as jnp, equinox as eqx, tempfile, pathlib\nfrom xtrax.training.types import ResumableState\nfrom aminx.training.checkpoint import get_checkpoint_manager, save_checkpoint, load_checkpoint\nkey = jax.random.PRNGKey(0)\nmodel = eqx.nn.Linear(8, 8, key=key)\nstate = ResumableState(step=jnp.int32(42), key=key, model=model, opt_state=None, extras={})\nwith tempfile.TemporaryDirectory() as d:\n    mgr = get_checkpoint_manager(pathlib.Path(d), max_to_keep=1)\n    save_checkpoint(mgr, state); mgr.close()\n    mgr2 = get_checkpoint_manager(pathlib.Path(d), max_to_keep=1)\n    restored = load_checkpoint(mgr2, state); mgr2.close()\n    assert int(restored.step) == 42\n    orig = jax.tree.leaves(eqx.filter(state, eqx.is_array))\n    rest = jax.tree.leaves(eqx.filter(restored, eqx.is_array))\n    for o, r in zip(orig, rest):\n        assert jnp.allclose(o, r, atol=1e-7, rtol=0)\n    print('CHECKPOINT ROUND-TRIP: PASS')\n" 2>&1\n\n3. uv run pytest tests/training/ -k "loss or curve or overfit or convergence" -v 2>&1\n   If no matches: run 50-step overfit check, report step-0 and step-50 loss.\n\n4. Write .praxia/docs/research/260614_g1-training-parity-gate.md with:\n   - pytest results (N passed, N failed)\n   - Checkpoint round-trip result\n   - Loss curve result\n   - GATE VERDICT: PASS (all three criteria met) or FAIL\n\n## Acceptance criteria\n- pytest tests/training/ all pass\n- Checkpoint round-trip prints CHECKPOINT ROUND-TRIP: PASS\n- Loss curve criterion satisfied\n- .praxia/docs/research/260614_g1-training-parity-gate.md contains GATE VERDICT: PASS\n`,
    { agentType: "librarian", label: "research:1550", phase: "Track E — G1 Training parity gate: loss curve + checkpoint round-trip (#1550)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK F — Track F — RS-1: Inventory host/* getattr(spec) -> run_spec migration map (#1620) =========================
const trackF = () =>
  agent(
    `task_id: ${TASK_ID}. # RS-1: host-field inventory (backlog #1620)\n\n## Context\nRS-2+ RunSpec unification requires a complete map of every spec field read in src/aminx/host/.\nOutput path (fixed): .praxia/docs/plans/260614_runspec-migration-map.md\nRS-2 will consume this as its primary input.\n\n## Task\n\n### Step 1 — Grep for direct spec reads\ngrep -rn "spec\." src/aminx/host/ --include="*.py" \\n  | grep -v "run_spec\.\|#\|\"spec\.\|\'spec\." | sort\n\n### Step 2 — Read key files in full\nRead: runner.py, kernel_dispatch.py, _sampling_helper.py, averaging.py\nAlso check: _scoring_helper.py, streaming_host.py (if they exist)\n\n### Step 3 — Migration table\nFor each spec.<field>:\n| file | line | field_name | already_on_run_spec | target_subconfig | notes |\n\ntarget_subconfig options:\n  sampling / plan / precision / protein / ligand / model / new\n  protein = protein-domain fields that stay flat (do NOT migrate)\n  new = no clear home (flag as RS-gap)\n\n### Step 4 — Write .praxia/docs/plans/260614_runspec-migration-map.md\nSections:\n1. Summary (unique fields, already migrated count, needs-migration count)\n2. Full migration table\n3. Already migrated\n4. Needs migration (grouped by target_subconfig)\n5. RS-gaps (no clear home — candidate RS-2/RS-3 additions)\n6. Protein-only fields (stay flat, not migrate)\n\n## Acceptance criteria\n- .praxia/docs/plans/260614_runspec-migration-map.md exists\n- Covers kernel_dispatch.py, _sampling_helper.py, runner.py\n- Table has >= 10 entries\n- RS-gaps and protein-only sections present\n`,
    { agentType: "librarian", label: "research:1620", phase: "Track F — RS-1: Inventory host/* getattr(spec) -> run_spec migration map (#1620)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK G — Track G — T0.3: L3 cluster smoke on SM120 + bathos sidecar (#1544) =========================
const trackG = () =>
  track(
    "1544",
    "Track G — T0.3: L3 cluster smoke on SM120 + bathos sidecar (#1544)",
    `task_id: ${TASK_ID}. # T0.3: L3 cluster smoke on SM120 Blackwell (backlog #1544)\n\n## Context\npi_so3 nodes (node4007/4008) currently allocated — jobs will queue, this is expected.\nMandatory: XLA_FLAGS=--xla_gpu_shard_autotuning=false (without this: 1170x slowdown, job 15294138)\nsbatch/squeue are transparent SSH wrappers to engaging — run locally.\nxtrax editable at ../xtrax (pyproject.toml:274).\n\n## Task\n\n### Step 1 — scripts/cluster/smoke_sm120.py\nimport sys, jax, jax.numpy as jnp, xtrax, aminx\nprint(f"Python: {sys.version}")\nprint(f"JAX: {jax.__version__}, xtrax: {xtrax.__version__}")\nprint(f"devices: {jax.devices()}")\nassert len(jax.devices()) > 0\n\n@jax.jit\ndef dot(a, b): return jnp.dot(a, b)\nA = jax.random.normal(jax.random.PRNGKey(0), (64, 64))\nC = dot(A, A.T); C.block_until_ready()\nprint(f"dot(64,64): shape={C.shape}")\nprint("SMOKE: PASS")\n\n### Step 2 — scripts/cluster/smoke_sm120.sbatch\n#!/bin/bash\n#SBATCH --job-name=aminx-smoke-sm120\n#SBATCH --partition=pi_so3\n#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gpus=1\n#SBATCH --time=00:15:00\n#SBATCH --output=outputs/logs/slurm/smoke_sm120_%j.out\n\n# SM120 Blackwell XLA workaround (mandatory — without this: 1170x slowdown)\n_NODE="\${SLURM_JOB_NODELIST:-$(hostname -s)}"\nif [[ "\${_NODE}" == *node4007* ]] || [[ "\${_NODE}" == *node4008* ]]; then\n    export XLA_FLAGS="\${XLA_FLAGS:+\${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"\nfi\n_PROJ="\${SLURM_SUBMIT_DIR:-$(cd "$(dirname "\${BASH_SOURCE[0]}")/../.." && pwd)}"\ncd "\${_PROJ}"\necho "Node: $(hostname -s), XLA_FLAGS=\${XLA_FLAGS:-<unset>}"\nuv run python scripts/cluster/smoke_sm120.py\n\n### Step 3 — Submit\nmkdir -p outputs/logs/slurm\nsbatch scripts/cluster/smoke_sm120.sbatch\n(Record job ID from sbatch output)\n\n### Step 4 — Wait\nUse mcp__myxcel__job_wait with job ID, timeout 25min. Nodes may queue — expected.\n\n### Step 5 — Verify\nConfirm "SMOKE: PASS" in logs via mcp__myxcel__tail_job_log.\n\n### Step 6 — Gate doc\nWrite .praxia/docs/research/260614_t03-sm120-smoke.md:\n- Job ID, node, partition\n- XLA_FLAGS value\n- JAX/xtrax/Python versions\n- GATE VERDICT: PASS or FAIL\n\n## Acceptance criteria\n- scripts/cluster/smoke_sm120.py and .sbatch committed\n- Job COMPLETED (not TIMEOUT/FAILED)\n- "SMOKE: PASS" in output\n- .praxia/docs/research/260614_t03-sm120-smoke.md with GATE VERDICT: PASS\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track G Reviewer — T0.3 cluster smoke (#1544)\n\nVERIFY: scripts/cluster/smoke_sm120.py and .sbatch exist\n  Check: ls scripts/cluster/smoke_sm120.py scripts/cluster/smoke_sm120.sbatch\n\nVERIFY: .sbatch has XLA_FLAGS node4007/node4008 guard\n  Check: grep "XLA_FLAGS\|node4007\|node4008" scripts/cluster/smoke_sm120.sbatch\n\nVERIFY: .sbatch uses SLURM_SUBMIT_DIR path anchoring\n  Check: grep "SLURM_SUBMIT_DIR" scripts/cluster/smoke_sm120.sbatch\n\nVERIFY: .praxia/docs/research/260614_t03-sm120-smoke.md exists with GATE VERDICT: PASS\n\nPASS if all VERIFY items satisfied.\nFAIL if scripts missing, XLA workaround absent, job TIMEOUT/FAILED, or gate doc missing.\n`,
    "worktree"
  );

// ===== TRACK H — Track H — T2.1 (xtrax repo): Add Scan.init field to xtrax tiling Scan strategy (#1551) =========================
const trackH = () =>
  track(
    "1551",
    "Track H — T2.1 (xtrax repo): Add Scan.init field to xtrax tiling Scan strategy (#1551)",
    `task_id: ${TASK_ID}. # T2.1: Add Scan.init field to xtrax tiling Scan strategy (backlog #1551)\n\n## IMPORTANT: Cross-repo work in /home/marielle/projects/xtrax (NOT aminx)\nxtrax editable dep: pyproject.toml:274 — xtrax = { path = "../xtrax", editable = true }\nChanges to xtrax are immediately visible to aminx. Commit in xtrax repo.\nDo NOT modify aminx source files in this track.\n\n## Recon first\nRead /home/marielle/projects/xtrax/src/xtrax/tiling/strategy.py — find Scan class, fields, carry flow.\nRead /home/marielle/projects/xtrax/src/xtrax/tiling/dispatch.py — find where Scan is consumed.\n\n## What to implement\n\nIn strategy.py, add to Scan class:\n  init: PyTree | None = None\n  (backwards compatible: None = use existing default, no behavior change)\n\nIn dispatch.py: if jax.lax.scan is called with init, pass Scan.init when not None.\n\n## Tests (in xtrax repo)\nCheck for existing scan tests. Add test_scan_init_field:\n1. Scan(init={"counter": 0}) — field accessible\n2. Scan() — works without init (backwards compat)\n\nRun from /home/marielle/projects/xtrax:\n  uv run pytest tests/ -k "scan" -v\n\n## Commit in xtrax repo\ngit -C /home/marielle/projects/xtrax add -u\ngit -C /home/marielle/projects/xtrax commit -m "feat(tiling): add Scan.init field for configurable carry init (aminx T2.1)"\n\n## Verify from aminx\nuv run python -c "from xtrax.tiling import Scan; s = Scan(init={'x': 0}); print(s.init)"\nuv run pytest -x -q -m "not slow and not parity_heavy and not parity_audit" | tail -5\n\n## Acceptance criteria\n- Scan.init field in xtrax; committed with T2.1 message\n- Scan() without init works identically (backwards compat)\n- xtrax scan tests pass\n- aminx regression suite shows no new FAIL\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track H Reviewer — T2.1 xtrax Scan.init (#1551)\n\nVERIFY: Scan.init field exists in /home/marielle/projects/xtrax/src/xtrax/tiling/strategy.py\n  Check: grep -n "init" /home/marielle/projects/xtrax/src/xtrax/tiling/strategy.py | head -10\n\nVERIFY: Scan.init defaults to None\n  Check: grep "init.*=.*None" /home/marielle/projects/xtrax/src/xtrax/tiling/strategy.py\n\nVERIFY: T2.1 commit exists in xtrax\n  Check: git -C /home/marielle/projects/xtrax log --oneline -3\n\nVERIFY: uv run pytest tests/ -k "scan" -v (from xtrax root) — all pass\n\nVERIFY: Backwards compat: from xtrax.tiling import Scan; Scan() — no error\n\nVERIFY: aminx regression: uv run pytest -x -q -m "not slow and not parity_heavy and not parity_audit" | tail -5\n  No new FAIL\n\nPASS if all VERIFY items satisfied.\nFAIL if Scan.init absent, no T2.1 commit, xtrax tests fail, or aminx regresses.\n`,
    "worktree"
  );

// ---- orchestrate: sequential writing chain || read-only research ----------
log("260614_potts-runspec-xtrax-gates: writing chain (, sequential) || research (A, B, C1, C2, D, E, F, G, H, read-only)");
const [writing, resA, resB, resC, resD, resE, resF, resG, resH] = await Promise.all([
  (async () => {
    return {  };
  })(),
  trackA(),
  trackB(),
  trackC1(),
  trackC2(),
  trackD(),
  trackE(),
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

  },
  research_1295: resA,
  research_1294: resB,
  research_1296: resC,
  research_1304: resC,
  research_1297: resD,
  research_1550: resE,
  research_1620: resF,
  research_1544: resG,
  research_1551: resH
};
