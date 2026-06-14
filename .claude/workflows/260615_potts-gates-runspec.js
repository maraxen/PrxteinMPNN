// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 260615_potts-gates-runspec   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (E,F) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time. Only the read-only
//   research/concurrent tracks (A,B,C,D) run concurrently.

export const meta = {
  name: "260615_potts-gates-runspec",
  description: "Vectorise the two remaining Python loops in the Potts stack (P-06 sampling, P-05 PoE), close the G1 training gate, add PlannerTopology to RunSpec (RS-2), and port CarrySpec/DedupSpec + factory dispatch from aminx.tiling into xtrax.tiling (T2.2-2.3, T2.4).",
  phases: [
    { title: "Track E — T2.2-2.3: Port CarrySpec, CarryShape, DedupSpec from aminx.tiling → xtrax.tiling; enrich BatchPlanner with Phase 0/0b (#1552)" },
    { title: "Track F — T2.4: Refactor xtrax make_axis_dispatch to factory-style; port iterator types from aminx (#1553)" },
    { title: "Track A — P-06: Replace Python loops in _parallel_tempering_exchange with jax.vmap (#1295)" },
    { title: "Track B — P-05: Replace Python loops in PoeModel.__call__ and joint_energy with JAX-native ops (#1296)" },
    { title: "Track C — G1: Training parity gate (pytest + checkpoint round-trip + overfit smoke) (#1550)" },
    { title: "Track D — RS-2: Add PlannerTopology sub-config to RunSpec + topology_hash golden (#1621)" },
  ],
};

const TASK_ID = "260615_potts-gates-runspec";
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

// Shared context for the writing tracks (from recon, task 260615_potts-gates-runspec).
const EMITTER_CTX = `Prior sprint (260614_potts-runspec-xtrax-gates) completed:\n- P-07 (#1297): spec emit-potts CLI\n- P-09 (#1294): TRW marginals vs brute-force tests\n- T0.3 (#1544): SM120 cluster smoke gate PASS\n- RS-1 (#1620): 67-field host-field inventory → .praxia/docs/plans/260614_runspec-migration-map.md\n- T2.1 (#1551): Scan.init field added to xtrax (3 commits: f9847a9, eff90d4, a24dd06)\n  NOTE: T2.1 was believed failed but actually landed in xtrax — discovered at sprint compose time.\n\nOpen backlog items entering this sprint:\n- P-06 (#1295): sampling.py _parallel_tempering_exchange still has Python for/while loops (lines 206-224)\n- P-05 (#1296): poe.py __call__ (line 172) and joint_energy (line 234) still have Python loops\n- G1 (#1550): training parity gate not yet run (tests/training/ + checkpoint round-trip + overfit smoke)\n- RS-2 (#1621): PlannerTopology sub-config not yet added to RunSpec\n- T2.2-2.3 (#1552): CarrySpec, CarryShape, DedupSpec not yet in xtrax\n- T2.4 (#1553): xtrax make_axis_dispatch still eager (not factory-style)\n\nKey recon finding (260614 pre-sprint):\naminx.tiling is the MATURE library — CarrySpec, CarryShape, DedupSpec, multi-phase BatchPlanner,\nand factory-style make_axis_dispatch all already exist in aminx.tiling/. T2.2-2.3 and T2.4 are\nPORT tasks (aminx → xtrax), not new designs. Do NOT re-invent these APIs.\n\nPlannerTopology stays in aminx for RS-2. aminx BatchPlanner (multi-phase) ≠ xtrax BatchPlanner\n(single-phase). Once T2.5 ships, PlannerTopology will gain an xtrax.ExecutionProfile field.\n\nKey file anchors:\n- src/aminx/potts/sampling.py:202–225 — _parallel_tempering_exchange (P-06 target)\n- src/aminx/potts/poe.py:162–238 — PoeModel.__call__ + joint_energy (P-05 target)\n- src/aminx/run/spec.py:96–114 — RunSpec + sub-configs (RS-2 target)\n- src/aminx/run/spec.py:218 — build_run_spec (RS-2 target)\n- src/aminx/tiling/carry.py — CarrySpec reference implementation for T2.2-2.3\n- src/aminx/tiling/carry_shape.py — CarryShape reference implementation for T2.2-2.3\n- src/aminx/tiling/dedup.py — DedupSpec reference implementation for T2.2-2.3\n- src/aminx/tiling/planner.py:104–165 — Phase 0/0b reference for T2.2-2.3\n- src/aminx/tiling/dispatch.py — factory dispatch reference for T2.4\n- src/aminx/tiling/iterator.py — VmapIterator/SafeMapIterator/JaxScanIterator reference for T2.4\n- /home/marielle/projects/xtrax/ — xtrax repo (editable at ../xtrax per pyproject.toml:274)\n- /home/marielle/projects/xtrax/src/xtrax/tiling/strategy.py — Scan, DedupGather, etc.\n- /home/marielle/projects/xtrax/src/xtrax/tiling/plan.py — xtrax BatchPlanner (single-phase)\n- /home/marielle/projects/xtrax/src/xtrax/tiling/dispatch.py — eager make_axis_dispatch (T2.4 target)\n`;

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

// ===== TRACK E — Track E — T2.2-2.3: Port CarrySpec, CarryShape, DedupSpec from aminx.tiling → xtrax.tiling; enrich BatchPlanner with Phase 0/0b (#1552) =========================
const trackE = () =>
  track(
    "1552",
    "Track E — T2.2-2.3: Port CarrySpec, CarryShape, DedupSpec from aminx.tiling → xtrax.tiling; enrich BatchPlanner with Phase 0/0b (#1552)",
    `task_id: ${TASK_ID}. # T2.2-2.3: Port CarrySpec + DedupSpec to xtrax (backlog #1552)\n\n## IMPORTANT: Work in /home/marielle/projects/xtrax (NOT aminx)\nxtrax is an editable dep: pyproject.toml:274 — changes immediately visible to aminx.\nCommit in the xtrax repo.\n\n## This is a PORT task — not a new design\naminx.tiling is the reference. Copy-adapt these files, replacing aminx.* imports\nwith xtrax.* equivalents. Do NOT re-invent the API.\n\n## READ FIRST (reference implementations in aminx)\n- /home/marielle/projects/aminx/src/aminx/tiling/carry.py      → CarrySpec\n- /home/marielle/projects/aminx/src/aminx/tiling/carry_shape.py → CarryShape\n- /home/marielle/projects/aminx/src/aminx/tiling/dedup.py       → DedupSpec + get_k_bucket\n- /home/marielle/projects/aminx/src/aminx/tiling/planner.py:104-165 → Phase 0/0b logic\nRead each in full before writing anything.\n\nAlso read:\n- /home/marielle/projects/xtrax/src/xtrax/tiling/strategy.py — ScanTransition already exists\n- /home/marielle/projects/xtrax/src/xtrax/tiling/plan.py — current BatchPlanner (single-phase)\n- /home/marielle/projects/xtrax/src/xtrax/tiling/__init__.py — current exports\n\n## Files to create in /home/marielle/projects/xtrax/src/xtrax/tiling/\n\n### carry.py (new)\nPort from aminx/tiling/carry.py.\nReplace: \`from aminx.tiling.strategy import ScanTransition\`\nWith:    \`from xtrax.tiling.strategy import ScanTransition\`\nKeep the _HETEROGENEOUS_AXIS_NAMES guard and __post_init__ validation exactly.\nNo other aminx imports needed.\n\n### carry_shape.py (new)\nPort from aminx/tiling/carry_shape.py.\nNo aminx imports — only jax and jax.numpy. Exact copy-adapt.\n\n### dedup.py (new)\nPort from aminx/tiling/dedup.py.\nReplace: \`from aminx.tiling.strategy import DedupFn, DedupGather, GatherFn\`\nWith:    \`from xtrax.tiling.strategy import DedupFn, DedupGather, GatherFn\`\nKeep get_k_bucket, DedupSpec exactly. Note: the high-k TODO comment is valuable — keep it.\n\n## File to modify: /home/marielle/projects/xtrax/src/xtrax/tiling/plan.py\n\n### Enrich BatchPlanner with Phase 0 (CarrySpec) and Phase 0b (DedupSpec)\n\nAdd imports at top:\n  from xtrax.tiling.carry import CarrySpec\n  from xtrax.tiling.dedup import DedupSpec, get_k_bucket\n\nModify BatchPlanner.__init__ to accept:\n  carry_specs: list[CarrySpec] | None = None\n  dedup_specs: list[DedupSpec] | None = None\n\nModify BatchPlanner.plan() to add Phase 0 and 0b BEFORE the existing budget loop.\nReference: aminx/tiling/planner.py lines 104–165.\n\nPhase 0 — pre-demote axes with declared CarrySpec to Scan:\n  For each CarrySpec cs in self.carry_specs:\n    Find the AxisSpec with ax.name == cs.axis_name\n    Force its decision: strategy = Scan(transition=cs.transition, init=cs.init)\n    Remove it from the pending list (skip the budget loop for it)\n\nPhase 0b — pre-demote dedup_eligible axes with declared DedupSpec to DedupGather:\n  For each DedupSpec ds in self.dedup_specs:\n    Find the AxisSpec with ax.name == ds.axis_name\n    Force its decision: strategy = DedupGather(dedup_fn=ds.dedup_fn, gather_fn=ds.gather_fn, k_bucket=ds.k)\n    Remove it from the pending list\n\nRemaining axes go through the existing cardinality/budget rules unchanged.\n\n## File to modify: /home/marielle/projects/xtrax/src/xtrax/tiling/__init__.py\n\nAdd exports:\n  from xtrax.tiling.carry import CarrySpec\n  from xtrax.tiling.carry_shape import CarryShape\n  from xtrax.tiling.dedup import DedupSpec, get_k_bucket\n\nAdd to __all__: "CarrySpec", "CarryShape", "DedupSpec", "get_k_bucket"\n\n## Commit in xtrax repo\ncd /home/marielle/projects/xtrax\ngit add src/xtrax/tiling/carry.py src/xtrax/tiling/carry_shape.py src/xtrax/tiling/dedup.py\ngit add src/xtrax/tiling/plan.py src/xtrax/tiling/__init__.py\ngit commit -m "feat(tiling): port CarrySpec, CarryShape, DedupSpec from aminx; Phase 0/0b BatchPlanner (aminx T2.2-2.3)"\n\n## Verify\nFrom /home/marielle/projects/xtrax:\n  uv run pytest tests/ -v --tb=short 2>&1 | tail -20\n\nFrom /home/marielle/projects/aminx (regression check):\n  uv run pytest -x -q -m "not slow" 2>&1 | tail -10\n\n## Acceptance criteria\n- xtrax/tiling/carry.py, carry_shape.py, dedup.py created — API matches aminx originals\n- BatchPlanner.plan() accepts carry_specs and dedup_specs; Phase 0/0b runs before budget loop\n- CarrySpec, CarryShape, DedupSpec, get_k_bucket exported from xtrax.tiling\n- xtrax tests pass; aminx regression clean\n- T2.2-2.3 commit in xtrax repo\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track E Reviewer — T2.2-2.3 xtrax CarrySpec + DedupSpec (#1552)\n\nVERIFY: New files exist in xtrax\n  ls /home/marielle/projects/xtrax/src/xtrax/tiling/carry.py\n  ls /home/marielle/projects/xtrax/src/xtrax/tiling/carry_shape.py\n  ls /home/marielle/projects/xtrax/src/xtrax/tiling/dedup.py\n\nVERIFY: CarrySpec API matches aminx (axis_name, init, transition, ordered_sinks fields)\n  grep -n "axis_name\|init\|transition\|ordered_sinks" /home/marielle/projects/xtrax/src/xtrax/tiling/carry.py\n\nVERIFY: DedupSpec API matches aminx (unique_indices, index_map, k, dedup_fn, gather_fn)\n  grep -n "unique_indices\|index_map\|dedup_fn\|gather_fn" /home/marielle/projects/xtrax/src/xtrax/tiling/dedup.py\n\nVERIFY: BatchPlanner accepts carry_specs and dedup_specs\n  grep -n "carry_specs\|dedup_specs" /home/marielle/projects/xtrax/src/xtrax/tiling/plan.py\n\nVERIFY: CarrySpec, CarryShape, DedupSpec exported from xtrax.tiling.__init__\n  grep -n "CarrySpec\|CarryShape\|DedupSpec" /home/marielle/projects/xtrax/src/xtrax/tiling/__init__.py\n\nVERIFY: xtrax tests pass\n  cd /home/marielle/projects/xtrax && uv run pytest tests/ -v --tb=short | tail -10\n\nVERIFY: aminx regression clean\n  uv run pytest -x -q -m "not slow" 2>&1 | tail -5\n\nVERIFY: T2.2-2.3 commit in xtrax\n  git -C /home/marielle/projects/xtrax log --oneline -3\n\nPASS if all eight satisfied.\nFAIL if files missing, API mismatch, tests fail, aminx regresses, or commit absent.\n`,
  );

// ===== TRACK F — Track F — T2.4: Refactor xtrax make_axis_dispatch to factory-style; port iterator types from aminx (#1553) =========================
const trackF = () =>
  track(
    "1553",
    "Track F — T2.4: Refactor xtrax make_axis_dispatch to factory-style; port iterator types from aminx (#1553)",
    `task_id: ${TASK_ID}. # T2.4: xtrax make_axis_dispatch factory refactor (backlog #1553)\n\n## IMPORTANT: Work in /home/marielle/projects/xtrax (NOT aminx)\nCommit in the xtrax repo after Track E is committed (both touch __init__.py —\nthis track runs after E to avoid conflicts).\n\n## This is a PORT task — not a new design\naminx.tiling.dispatch and aminx.tiling.iterator are the references.\nCopy-adapt to xtrax; replace aminx.* imports with xtrax.* equivalents.\n\n## READ FIRST (reference implementations in aminx)\n- /home/marielle/projects/aminx/src/aminx/tiling/iterator.py   → VmapIterator, SafeMapIterator, JaxScanIterator, MapIterator\n- /home/marielle/projects/aminx/src/aminx/tiling/dispatch.py   → factory make_axis_dispatch\nRead both in full before writing anything.\n\nAlso read:\n- /home/marielle/projects/xtrax/src/xtrax/tiling/dispatch.py       — current eager dispatch\n- /home/marielle/projects/xtrax/src/xtrax/tiling/__init__.py        — current exports\n- /home/marielle/projects/xtrax/tests/                              — existing tests for make_axis_dispatch\n\n## Files to create/modify in /home/marielle/projects/xtrax/\n\n### src/xtrax/tiling/iterator.py (new)\nPort from aminx/tiling/iterator.py.\n- Replace aminx.* imports with xtrax.* equivalents\n- xtrax.transforms.map.safe_map already exists (used in dispatch.py)\n- xtrax.transforms.scan.safe_scan already exists (used in dispatch.py)\n- Keep VmapIterator, SafeMapIterator, JaxScanIterator, MapIterator API identical\n\n### src/xtrax/tiling/dispatch.py (modify)\nCurrent signature: make_axis_dispatch(strategy, fn, xs, init=None) → Any  [EAGER]\nNew signature:     make_axis_dispatch(strategy, *, axis="") → iterator     [FACTORY]\n\nKey changes:\n1. make_axis_dispatch(strategy, *, axis="") returns an iterator object, not a result\n   - Vmap      → VmapIterator()\n   - SafeMap   → SafeMapIterator(tile=strategy.batch_size)\n   - Scan      → JaxScanIterator()\n   - DedupGather → raise DispatchRejected (handled by BatchPlanner + _dispatch_axis)\n   - Bucket    → raise DispatchRejected or TypeError (host-side only)\n\n2. Create DispatchRejected exception class (port from aminx.tiling.dispatch):\n   class DispatchRejected(Exception): ...\n\n3. Add backward-compat shim (prevents breaking existing callers):\n   def axis_dispatch(strategy, fn, xs, init=None):\n       """Eager shim: make_axis_dispatch(strategy)(fn, xs, init)."""\n       it = make_axis_dispatch(strategy)\n       return it(fn, xs, init)\n   This preserves the old 4-arg eager API under a new name.\n\n4. If existing xtrax tests use make_axis_dispatch(strategy, fn, xs) — check tests/ first.\n   If they do: rename the old eager function to axis_dispatch in tests too (or add both).\n   Do NOT silently break tests.\n\n## Update __init__.py\nAdd: from xtrax.tiling.dispatch import DispatchRejected, axis_dispatch\nAdd to __all__: "DispatchRejected", "axis_dispatch"\nAlso add: from xtrax.tiling.iterator import VmapIterator, SafeMapIterator, JaxScanIterator, MapIterator\nAdd to __all__: "VmapIterator", "SafeMapIterator", "JaxScanIterator", "MapIterator"\n\n## 4 invariants to preserve\n1. VmapIterator(fn, xs) produces same result shape as jax.vmap(fn)(xs)\n2. SafeMapIterator is chunk-order-stable (same as safe_map)\n3. JaxScanIterator returns (final_carry, stacked_outputs) matching safe_scan convention\n4. DedupGather → DispatchRejected (not handled here; handled by BatchPlanner + _dispatch_axis)\n\n## Commit in xtrax repo\ncd /home/marielle/projects/xtrax\ngit add src/xtrax/tiling/iterator.py src/xtrax/tiling/dispatch.py src/xtrax/tiling/__init__.py\ngit commit -m "feat(tiling): factory-style make_axis_dispatch + iterator types; axis_dispatch compat shim (aminx T2.4)"\n\n## Verify\nFrom /home/marielle/projects/xtrax:\n  uv run pytest tests/ -v --tb=short 2>&1 | tail -20\n\nFrom /home/marielle/projects/aminx:\n  uv run pytest -x -q -m "not slow" 2>&1 | tail -10\n\n## Acceptance criteria\n- xtrax/tiling/iterator.py created with VmapIterator, SafeMapIterator, JaxScanIterator, MapIterator\n- make_axis_dispatch(strategy) returns iterator (factory pattern)\n- axis_dispatch(strategy, fn, xs, init) provides backward-compat eager call\n- DispatchRejected raised for DedupGather (not TypeError)\n- All 4 invariants hold\n- xtrax tests pass; aminx regression clean\n- T2.4 commit in xtrax repo\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track F Reviewer — T2.4 xtrax factory dispatch (#1553)\n\nVERIFY: iterator.py created\n  ls /home/marielle/projects/xtrax/src/xtrax/tiling/iterator.py\n\nVERIFY: VmapIterator, SafeMapIterator, JaxScanIterator present\n  grep -n "class VmapIterator\|class SafeMapIterator\|class JaxScanIterator" \\n    /home/marielle/projects/xtrax/src/xtrax/tiling/iterator.py\n\nVERIFY: make_axis_dispatch is factory (returns iterator, not result)\n  grep -n "def make_axis_dispatch" /home/marielle/projects/xtrax/src/xtrax/tiling/dispatch.py\n  Confirm signature does NOT include fn/xs params (factory = strategy only)\n\nVERIFY: axis_dispatch backward-compat shim exists\n  grep -n "def axis_dispatch" /home/marielle/projects/xtrax/src/xtrax/tiling/dispatch.py\n\nVERIFY: DispatchRejected raised for DedupGather\n  grep -n "DispatchRejected\|DedupGather" /home/marielle/projects/xtrax/src/xtrax/tiling/dispatch.py\n\nVERIFY: xtrax tests pass\n  cd /home/marielle/projects/xtrax && uv run pytest tests/ -v --tb=short | tail -10\n\nVERIFY: aminx regression clean\n  uv run pytest -x -q -m "not slow" 2>&1 | tail -5\n\nVERIFY: T2.4 commit in xtrax repo\n  git -C /home/marielle/projects/xtrax log --oneline -3\n\nPASS if all eight satisfied.\nFAIL if iterator.py missing, dispatch still eager, tests fail, aminx regresses, or commit absent.\n`,
  );

// ===== TRACK A — Track A — P-06: Replace Python loops in _parallel_tempering_exchange with jax.vmap (#1295) =========================
const trackA = () =>
  track(
    "1295",
    "Track A — P-06: Replace Python loops in _parallel_tempering_exchange with jax.vmap (#1295)",
    `task_id: ${TASK_ID}. # P-06: sampling.py Python loops → jax.vmap (backlog #1295)\n\n## What to change\n\nFile: src/aminx/potts/sampling.py\nFunction: _parallel_tempering_exchange (starts around line 192)\nTarget lines: 206–224 — the \`for parity in range(2)\` + \`while i + 1 < k_rep\` loops\n\n## READ FIRST\nRead src/aminx/potts/sampling.py lines 192–245 in full before writing anything.\nAlso read _attempt_adjacent_swap to understand its exact signature and return shape.\n\n## Current code to replace\n\n\`\`\`python\nfor parity in range(2):\n    start = parity\n    i = start\n    while i + 1 < k_rep:\n        key_cur, sk = jax.random.split(key_cur)\n        seq_cur, key_cur, acc = _attempt_adjacent_swap(\n            sk, seq_cur, jnp.int32(i), betas[i], betas[i + 1],\n            h, j, w, mask, swap_pair_energy_only=swap_pair_energy_only,\n        )\n        accept_edge = accept_edge.at[i].set(acc)\n        i += 2\n\`\`\`\n\n## Why this is safe to vmap\n\nk_rep = int(seqs.shape[0]) is statically known at trace time.\nWithin each parity group, swaps are non-overlapping:\n  even parity: pairs (0,1), (2,3), (4,5)... — no index shared\n  odd parity:  pairs (1,2), (3,4), (5,6)... — no index shared\nSo vmap over edges within each parity group is correct.\n\n## Replacement approach\n\nStep 1 — Read _attempt_adjacent_swap signature. It takes:\n  (key, seqs, i, beta_i, beta_next, h, j, w, mask, swap_pair_energy_only)\n  Returns: (updated_seqs, new_key, accept_scalar)\n  Note: it takes the FULL seqs array and edge index i (as jnp.int32).\n\nStep 2 — For each parity, build static edge-index arrays:\n  even_edges = jnp.arange(0, k_rep - 1, 2)   # shape (n_even,)\n  odd_edges  = jnp.arange(1, k_rep - 1, 2)   # shape (n_odd,)\n\nStep 3 — For each parity group, vmap over edges.\n  Because _attempt_adjacent_swap reads and writes seqs at positions i and i+1,\n  and edges within a parity group are non-overlapping, ALL calls in a group see\n  the same input seqs and produce disjoint position updates.\n  Strategy:\n    - Pre-split n_edges keys: subkeys = jax.random.split(key_cur, n_edges + 1)\n      key_cur, subkeys = subkeys[0], subkeys[1:]\n    - vmapped call: accepts, new_seqs_per_edge = jax.vmap(swap_fn)(subkeys, edges)\n      where swap_fn(sk, edge_i) = _attempt_adjacent_swap(sk, seq_cur, edge_i, ...)\n      BUT: _attempt_adjacent_swap returns full updated seqs — scatter only changed pairs.\n    - Alternatively: write a scalar helper that extracts just the new pair and accept:\n        def _swap_pair(sk, i):\n            _, _, acc = _attempt_adjacent_swap(sk, seq_cur, i, betas[i], betas[i+1], ...)\n            # Extract new seq values at positions i and i+1 from the returned seqs\n            new_seqs, _, acc = _attempt_adjacent_swap(sk, seq_cur, i, betas[i], betas[i+1], ...)\n            return new_seqs[i], new_seqs[i+1], acc\n      Then scatter: seq_cur = seq_cur.at[even_edges].set(new_i) ...\n\n  IMPORTANT: Check if betas[i] and betas[i+1] can be gathered inside vmap:\n    beta_i     = betas[even_edges]      # shape (n_even,)\n    beta_next  = betas[even_edges + 1]  # shape (n_even,)\n    These can be passed as in_axes=0 to vmap.\n\nStep 4 — Build accept_edge from vmap output directly (no .at[i].set loop).\n\nStep 5 — Keep the two-parity structure (even then odd) as two sequential vmap calls —\n  do NOT try to merge parities into one vmap (they must be sequential: odd parity\n  uses seqs updated by even parity).\n\n## Edit only — do not Write existing files. One file: sampling.py.\n\n## Verify\nuv run pytest tests/potts/ -v --tb=short 2>&1 | tail -20\nuv run ty check src/aminx/potts/sampling.py 2>&1\ngrep -n "for parity\|while i" src/aminx/potts/sampling.py  # must be empty\n\n## Acceptance criteria\n- No \`for parity\` or \`while i + 1\` loops in _parallel_tempering_exchange\n- All tests/potts/ pass\n- uv run ty check clean\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track A Reviewer — P-06 sampling.py vmap (#1295)\n\nVERIFY: Python loops removed from _parallel_tempering_exchange\n  grep -n "for parity\|while i + 1" src/aminx/potts/sampling.py\n  Expected: empty output\n\nVERIFY: jax.vmap or eqx.filter_vmap present in the function\n  grep -n "vmap" src/aminx/potts/sampling.py | head -10\n\nVERIFY: uv run pytest tests/potts/ -v --tb=short — zero failures\n\nVERIFY: uv run ty check src/aminx/potts/sampling.py — exits 0\n\nPASS if all four satisfied.\nFAIL if Python loops remain, tests fail, or ty errors.\n`,
    "worktree"
  );

// ===== TRACK B — Track B — P-05: Replace Python loops in PoeModel.__call__ and joint_energy with JAX-native ops (#1296) =========================
const trackB = () =>
  track(
    "1296",
    "Track B — P-05: Replace Python loops in PoeModel.__call__ and joint_energy with JAX-native ops (#1296)",
    `task_id: ${TASK_ID}. # P-05: poe.py loops → JAX-native (backlog #1296)\n\n## READ FIRST\nRead src/aminx/potts/poe.py lines 140–250 in full.\nPay special attention to:\n  - PoeModel.__call__ line ~172: \`for i in range(self.n_backbones):\`\n  - PoeModel.joint_energy line ~234: \`for h, j, w in params_list:\`\n  - PoeModel.infer_all_params line ~247: already uses eqx.filter_vmap — use as reference\n\n## File: src/aminx/potts/poe.py\n\n## Change 1: PoeModel.__call__ backbone loop (line ~172)\n\nCurrent:\n  all_marginals = []\n  for i in range(self.n_backbones):\n      backbone = self.backbones[i]\n      key_b = keys[i]\n      edge_knn_b = edge_knn_stack[i]\n      nei_b = nei_stack[i]\n      marginals, _, _, _ = backbone(edge_knn=edge_knn_b, nei=nei_b, mask=mask, key=key_b)\n      all_marginals.append(marginals)\n  per_backbone_marginals = jnp.stack(all_marginals, axis=0)\n\nReplace with eqx.filter_vmap following the pattern already in infer_all_params:\n  def _call_one(backbone, key_b, edge_knn_b, nei_b):\n      marginals, _, _, _ = backbone(edge_knn=edge_knn_b, nei=nei_b, mask=mask, key=key_b)\n      return marginals\n  per_backbone_marginals = eqx.filter_vmap(_call_one)(\n      self.backbones, keys, edge_knn_stack, nei_stack\n  )\n  # per_backbone_marginals shape: (B, N, q)\n\nNote: self.backbones is a list/pytree of identical-structure eqx.Modules.\neqx.filter_vmap handles Equinox modules correctly — it's the same pattern used in\ninfer_all_params. Follow that exactly.\n\n## Change 2: PoeModel.joint_energy params loop (line ~234)\n\nCurrent:\n  total_energy = 0.0\n  for h, j, w in params_list:\n      energy_b = PottsModel.log_prob(seq, h, j, w)\n      total_energy = total_energy + energy_b\n  return jnp.asarray(total_energy)\n\nReplace with:\n  # Stack params along batch axis (Python-level at trace time)\n  h_stack = jnp.stack([p[0] for p in params_list])  # (B, n, q)\n  j_stack = jnp.stack([p[1] for p in params_list])  # (B, n, n, q, q)\n  w_stack = jnp.stack([p[2] for p in params_list])  # (B, n, n)\n  # vmap log_prob over batch axis\n  energies = jax.vmap(PottsModel.log_prob, in_axes=(None, 0, 0, 0))(\n      seq, h_stack, j_stack, w_stack\n  )\n  return jnp.sum(energies)\n\nVerify PottsModel.log_prob signature before calling: it should be\n  log_prob(seq, h, j, w) -> scalar  (or a classmethod/staticmethod)\nRead src/aminx/potts/model.py to confirm.\n\n## Edit only — do not Write existing files. One file: poe.py.\n\n## Verify\nuv run pytest tests/potts/ -v --tb=short 2>&1 | tail -20\nuv run ty check src/aminx/potts/poe.py 2>&1\ngrep -n "^    for i in range\|^    for h, j, w" src/aminx/potts/poe.py  # must be empty\n\n## Acceptance criteria\n- No Python backbone loop in __call__; no Python params loop in joint_energy\n- eqx.filter_vmap used in __call__ following infer_all_params pattern\n- jax.vmap + jnp.sum used in joint_energy\n- All tests/potts/ pass\n- uv run ty check clean\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track B Reviewer — P-05 poe.py JAX-native (#1296)\n\nVERIFY: Python loops removed from __call__ and joint_energy\n  grep -n "for i in range(self.n_backbones)\|for h, j, w in params_list" src/aminx/potts/poe.py\n  Expected: empty output\n\nVERIFY: eqx.filter_vmap present in __call__\n  grep -n "filter_vmap" src/aminx/potts/poe.py | head -5\n\nVERIFY: jax.vmap present in joint_energy\n  grep -n "jax.vmap\|jnp.sum" src/aminx/potts/poe.py | head -5\n\nVERIFY: uv run pytest tests/potts/ -v --tb=short — zero failures\n\nVERIFY: uv run ty check src/aminx/potts/poe.py — exits 0\n\nPASS if all five satisfied.\nFAIL if Python loops remain, wrong vmap function used, tests fail, or ty errors.\n`,
    "worktree"
  );

// ===== TRACK C — Track C — G1: Training parity gate (pytest + checkpoint round-trip + overfit smoke) (#1550) =========================
// NOTE: G1 is a reviewer gate (runs pytest + bash smoke), NOT a librarian research task.
// is_research=true in TOML was a misnomer — the emitter routed this wrong; corrected here.
const trackC = () =>
  track(
    "1550",
    "Track C — G1: Training parity gate (pytest + checkpoint round-trip + overfit smoke) (#1550)",
    `task_id: ${TASK_ID}. # G1: Training parity gate (backlog #1550)\n\n## Context\nT1.4 (sprint 260612) landed: trainer.py uses xtrax ResumableState, single-PyTree checkpoint.\ntests/training/ has test_checkpoint.py and test_resumable_state.py (8+ tests total).\nG1 = formal verification that all three criteria pass.\n\n## Gate criteria (all three must pass)\n\n### G1.1 — pytest suite\nuv run pytest tests/training/ -v --tb=short 2>&1 | tee /tmp/g1_pytest.txt\n\n### G1.2 — Checkpoint round-trip smoke\nRead tests/training/test_checkpoint.py first to find the fixture shape.\nThen run:\n\nuv run python - <<'PY'\nimport jax, jax.numpy as jnp, equinox as eqx, tempfile, pathlib\nfrom xtrax.training.types import ResumableState\nfrom aminx.training.checkpoint import get_checkpoint_manager, save_checkpoint, load_checkpoint\nkey = jax.random.PRNGKey(0)\nmodel = eqx.nn.Linear(8, 8, key=key)\nopt_state = None\nstate = ResumableState(step=jnp.int32(42), key=key, model=model, opt_state=opt_state, extras={})\nwith tempfile.TemporaryDirectory() as d:\n    mgr = get_checkpoint_manager(pathlib.Path(d), max_to_keep=None)\n    save_checkpoint(mgr, state); mgr.close()\n    mgr2 = get_checkpoint_manager(pathlib.Path(d), max_to_keep=None)\n    restored = load_checkpoint(mgr2, state); mgr2.close()\n    assert int(restored.step) == 42, f"step mismatch: {restored.step}"\n    orig_leaves = jax.tree.leaves(eqx.filter(state, eqx.is_array))\n    rest_leaves = jax.tree.leaves(eqx.filter(restored, eqx.is_array))\n    for i, (o, r) in enumerate(zip(orig_leaves, rest_leaves)):\n        assert jnp.allclose(o, r, atol=1e-7, rtol=0), f"leaf {i} mismatch: max diff={jnp.max(jnp.abs(o-r))}"\n    print("G1.2 CHECKPOINT ROUND-TRIP: PASS")\nPY\n\nAdapt if ResumableState signature or checkpoint API differs — read the actual types first.\n\n### G1.3 — 50-step overfit smoke\nRead src/aminx/training/trainer.py to find the Trainer API.\nConstruct a minimal dummy batch (4-residue protein, eqx.nn.Linear + cross_entropy_loss).\nRun 50 steps; assert loss[49] < loss[0] * 0.9 (10% decrease).\nPrint step-0 and step-49 loss values.\n\n## Output\nWrite .praxia/docs/research/260614_g1-training-parity-gate.md with frontmatter gate: G1, date: 260615,\nthen sections for G1.1 pytest results, G1.2 checkpoint round-trip, G1.3 overfit smoke,\nand a final line: GATE VERDICT: PASS (only if all three criteria met; else GATE VERDICT: FAIL)\n\n## Acceptance criteria\n- All training tests pass (G1.1)\n- Checkpoint round-trip prints PASS (G1.2)\n- Loss decreases >=10% over 50 steps (G1.3)\n- .praxia/docs/research/260614_g1-training-parity-gate.md contains GATE VERDICT: PASS\n\nWhen done, end with: verdict: done\n`,
    `task_id: ${TASK_ID}. # Track C Reviewer — G1 training gate (#1550)\n\nVERIFY: uv run pytest tests/training/ -v — 0 FAIL\n  Run it; report exact counts (N passed, N failed, N error)\n\nVERIFY: Checkpoint round-trip — re-run the smoke or confirm gate doc shows PASS\n  Look for "G1.2 CHECKPOINT ROUND-TRIP: PASS" in gate doc\n\nVERIFY: .praxia/docs/research/260614_g1-training-parity-gate.md exists\n\nVERIFY: grep "GATE VERDICT: PASS" .praxia/docs/research/260614_g1-training-parity-gate.md\n\nPASS if all four satisfied.\nFAIL if any training test fails, round-trip errors, gate doc absent, or verdict is FAIL.\n`
  );

// ===== TRACK D — Track D — RS-2: Add PlannerTopology sub-config to RunSpec + topology_hash golden (#1621) =========================
const trackD = () =>
  track(
    "1621",
    "Track D — RS-2: Add PlannerTopology sub-config to RunSpec + topology_hash golden (#1621)",
    `task_id: ${TASK_ID}. # RS-2: PlannerTopology to RunSpec (backlog #1621)\n\n## READ FIRST\nRead src/aminx/run/spec.py in full — understand all existing sub-configs and build_run_spec.\nKey lines: 96–114 (sub-config classes + RunSpec), 218–295 (build_run_spec body).\n\n## Placement: aminx only, NOT xtrax\n\nPlannerTopology is an aminx-specific wrapper around aminx kernel dispatch topology.\n- use_unified_driver is a flag for kernel_dispatch.py (aminx-only concept)\n- aminx.tiling.BatchPlanner (multi-phase) ≠ xtrax.tiling.BatchPlanner (single-phase)\n- Once T2.5 ships (xtrax multi-phase BatchPlanner), PlannerTopology gains an\n  xtrax.ExecutionProfile field. For RS-2: one field only.\n\n## File: src/aminx/run/spec.py\n\n### Step 1 — Add PlannerTopology class (after PrecisionConfig, before RunSpec)\n\n\`\`\`python\nclass PlannerTopology(eqx.Module):\n    """aminx kernel dispatch topology config.\n\n    Wraps aminx.tiling.BatchPlanner config fields for RunSpec.\n    Will gain xtrax.ExecutionProfile once xtrax BatchPlanner reaches multi-phase parity (T2.5).\n    """\n    use_unified_driver: bool = eqx.field(static=True)\n\`\`\`\n\n### Step 2 — Add plan field to RunSpec\n\nAdd \`plan: PlannerTopology\` to the RunSpec class fields.\nOrder: keep existing fields; append plan after precision.\n\n### Step 3 — Add topology_hash (module-level function, NOT a method)\n\n\`\`\`python\ndef topology_hash(plan: PlannerTopology) -> str:\n    """Deterministic 16-char hex hash of PlannerTopology for cache-key derivation."""\n    import hashlib, json\n    payload = {"use_unified_driver": plan.use_unified_driver}\n    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]\n\`\`\`\n\n### Step 4 — Update build_run_spec to populate plan\n\nFind the final RunSpec(...) constructor call in build_run_spec. Before it, add:\n\n\`\`\`python\nplan = PlannerTopology(\n    use_unified_driver=bool(getattr(spec, "use_unified_driver", True)),\n    # Default is True — RS-5 fix in 26c9bb5 confirmed False was wrong\n)\n\`\`\`\n\nPass plan=plan to RunSpec(...).\n\n### Step 5 — Export topology_hash\n\nCheck src/aminx/run/__init__.py — add topology_hash to exports if it exports other\nspec-level functions (it exports build_run_spec; add topology_hash alongside it).\n\n### Step 6 — Tests\n\nFind or create tests/run/test_run_spec.py. Add:\n\n\`\`\`python\ndef test_planner_topology_default():\n    from aminx.run.spec import PlannerTopology, topology_hash\n    pt = PlannerTopology(use_unified_driver=True)\n    assert pt.use_unified_driver is True\n    h = topology_hash(pt)\n    assert len(h) == 16\n    assert h == topology_hash(pt)  # determinism\n\ndef test_planner_topology_false():\n    from aminx.run.spec import PlannerTopology, topology_hash\n    pt_t = PlannerTopology(use_unified_driver=True)\n    pt_f = PlannerTopology(use_unified_driver=False)\n    assert topology_hash(pt_t) != topology_hash(pt_f)  # different inputs → different hash\n\ndef test_build_run_spec_plan_field(minimal_spec):\n    from aminx.run.spec import build_run_spec\n    rs = build_run_spec(minimal_spec)\n    assert hasattr(rs, "plan")\n    assert isinstance(rs.plan.use_unified_driver, bool)\n\`\`\`\n\nUse existing fixtures from the file if present; create minimal_spec fixture if needed\n(a simple object with no attributes — build_run_spec uses getattr with defaults).\n\n## Edit only — no Write on existing files.\n\n## Verify\nuv run ty check src/aminx/run/spec.py 2>&1\nuv run pytest tests/run/ -v --tb=short 2>&1 | tail -20\n\n## Acceptance criteria\n- PlannerTopology(eqx.Module) added with use_unified_driver: bool = eqx.field(static=True)\n- RunSpec has plan: PlannerTopology field\n- build_run_spec populates plan with correct default (True)\n- topology_hash is a module-level function returning 16-char hex string\n- topology_hash is deterministic (same input → same output)\n- topology_hash(True) != topology_hash(False)\n- All tests/run/ pass\n- uv run ty check clean\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Track D Reviewer — RS-2 PlannerTopology (#1621)\n\nVERIFY: PlannerTopology class added to spec.py\n  grep -n "class PlannerTopology" src/aminx/run/spec.py\n\nVERIFY: use_unified_driver uses eqx.field(static=True)\n  grep -n "use_unified_driver\|eqx.field(static" src/aminx/run/spec.py | head -5\n\nVERIFY: RunSpec has plan field\n  grep -n "plan:" src/aminx/run/spec.py | head -3\n\nVERIFY: build_run_spec populates plan with default True\n  grep -n "use_unified_driver.*True\|getattr.*use_unified" src/aminx/run/spec.py\n\nVERIFY: topology_hash function exists at module level\n  grep -n "def topology_hash" src/aminx/run/spec.py\n\nVERIFY: uv run pytest tests/run/ -v — zero failures\n\nVERIFY: uv run ty check src/aminx/run/spec.py — exits 0\n\nPASS if all seven satisfied.\nFAIL if class missing, wrong default, no test, ty errors, or tests fail.\n`,
    "worktree"
  );

// ---- orchestrate: sequential writing chain || read-only research ----------
log("260615_potts-gates-runspec: writing chain (E -> F, sequential) || research (A, B, C, D, read-only)");
const [writing, resA, resB, resC, resD] = await Promise.all([
  (async () => {
    const e = await trackE();
    const f = await trackF();
    return { e, f };
  })(),
  trackA(),
  trackB(),
  trackC(),
  trackD(),
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
    "1552": writing.e,
    "1553": writing.f
  },
  research_1295: resA,
  research_1296: resB,
  research_1550: resC,
  research_1621: resD
};
