// Sprint 1 runner — aminx→xtrax P0 foundations + training adoption
// Source spec: .praxia/docs/specs/260611_aminx-xtrax-refactor.md  (EPIC backlog #1541)
// task_id: 260611_xtrax-refactor-spec   sprint_id: 1
//
// Polished from the `praxia dw emit-sprint` skeleton: helper/orchestration logic is
// verbatim; per-track fixer/reviewer prompts are the review-ready versions.
//
// RACE SAFETY (parallel fixers race on git-status scope checks in praxia):
//   the writing chain A->B->C->D->E runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.
// Dependency order: 1542 -> 1543 -> {1544, 1545}; 1548 needs 1543 AND 1545.
//   Linear A,B,C,D,E is a valid topological order. (1544 could run concurrently
//   with the 1545->1548 branch, but is kept sequential for working-tree safety.)

export const meta = {
  name: "260611_xtrax-foundations",
  description:
    "aminx->xtrax P0 foundations + idea-2 training adoption: py3.13 bump, editable xtrax pin, SM120 cluster smoke, boundary-lint, ResumableState+optimizer. Excludes user-decision TBDs (#1546 T-R2, #1547 T-R3b) and what they gate.",
  phases: [
    { title: "Track A — [xtrax T0.1] Bump aminx to Python >=3.13 (#1542)" },
    { title: "Track B — [xtrax T0.2] Add editable xtrax pin + dependency (#1543)" },
    { title: "Track C — [xtrax T0.3] L3 cluster smoke: py3.13 + JAX on SM120 + bathos sidecar (#1544)" },
    { title: "Track D — [xtrax T0.4] Boundary-lint + ruff banned-api for xtrax.* (#1545)" },
    { title: "Track E — [xtrax T1.1-1.2] Adopt xtrax ResumableState + optimizer (#1548)" },
  ],
};

const TASK_ID = "260611_xtrax-refactor-spec";
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

// Shared validated context (Wave-1+2 recon; full model in
// .praxia/docs/research/260611_aminx-xtrax-refactor-codebase-model.md).
const EMITTER_CTX = `VALIDATED GROUND TRUTH:
- aminx imports NOTHING from xtrax today (greenfield). xtrax v0.2.0 on PyPI, requires py>=3.13.
- A1 (live-validated): under requires-python >=3.12, adding editable ../xtrax fails the WHOLE uv lock
  (the 3.12 env split cannot satisfy xtrax>=3.13). Bumping the floor to >=3.13 AND both
  [tool.uv.environments] markers fixes it; import then yields 0.2.0. Active interpreter is 3.13.12;
  cluster uses 'uv run python' (no pin) so 3.13 is fine.
- A4: ruff TID251 bans only aminx.* paths; xtrax.* is unbanned, so the first xtrax.* import silently
  bypasses ADR 260605 unless an equivalent boundary guard lands in the SAME change (Track D before/with Track E).
- KEEP in aminx (do not move): protein bundles, domain losses/metrics/diffusion/dataloading.`;

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

// ===== TRACK A — [xtrax T0.1] Bump aminx to Python >=3.13 (#1542) ==========
const trackA = () =>
  track(
    "1542",
    "Track A — [xtrax T0.1] Bump aminx to Python >=3.13 (#1542)",
    `task_id: ${TASK_ID}. Edit-only; one concern. Stack: uv/Python (NOT cargo).

OBJECTIVE: Raise aminx's Python floor to 3.13 so it can depend on xtrax (which requires >=3.13).

ANCHORS (verify before editing):
- pyproject.toml:9 — requires-python = ">=3.12" -> change to ">=3.13".
- pyproject.toml [tool.uv.environments] — TWO env markers currently say python_version >= '3.12';
  change BOTH to python_version >= '3.13'. Grep: rg -n "tool.uv.environments|python_version" pyproject.toml

WHY (validated): under >=3.12 uv resolves a 3.12 env split that cannot satisfy xtrax>=3.13, failing the
whole lock (Wave-2 A1). Bumping the floor + both markers is the proven fix.

DO NOT add the xtrax dependency here (that is #1543/Track B). Touch only pyproject.toml.

SUCCESS CRITERIA:
- rg -n 'requires-python|python_version' pyproject.toml shows 3.13 everywhere.
- uv lock succeeds.
- uv run pytest -q -m 'not parity_heavy and not slow' is green on 3.13.
Report the exact pyproject diff and the uv lock + pytest tail.

${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Reviewer (HAS Bash). Stack: uv/Python. Emit the structured verdict.

VERIFY (run):
- rg -n 'requires-python|python_version' pyproject.toml -> all show 3.13 (floor + BOTH [tool.uv.environments] markers).
- uv lock exits 0 (no resolution error).
- uv run python --version -> 3.13.x.
- uv run pytest -q -m 'not parity_heavy and not slow' -> green (no NEW failures vs baseline).
- Only pyproject.toml / uv.lock changed (git diff --name-only).
PASS iff all hold; else NEEDS_WORK/FAIL with the failing command output.`,
  );

// ===== TRACK B — [xtrax T0.2] Add editable xtrax pin + dependency (#1543) ==
const trackB = () =>
  track(
    "1543",
    "Track B — [xtrax T0.2] Add editable xtrax pin + dependency (#1543)",
    `task_id: ${TASK_ID}. Edit-only; one concern. Depends on #1542 (py3.13 floor) merged first. Stack: uv/Python.

OBJECTIVE: Add xtrax as a dependency pinned to a LOCAL editable checkout for lockstep dev.

ANCHORS:
- pyproject.toml [project].dependencies (~lines 10-31) — add "xtrax>=0.2.0".
- pyproject.toml [tool.uv.sources] (~line 263; currently pins proxide, torch to pypi) — add
  xtrax = { path = "../xtrax", editable = true }. Pin the path EXPLICITLY as ../xtrax (do NOT run
  'uv add' and trust its cwd-derived relative path — Wave-2 A1b showed it emits a wrong climbing path).

DO NOT change requires-python (done in #1542). Do NOT import xtrax in any source file yet (the first
xtrax.* import is gated on the boundary-lint #1545 landing atomically — Track D / Track E).

SUCCESS CRITERIA:
- uv lock resolves with xtrax present.
- uv run python -c "import xtrax; print(xtrax.__version__)" prints 0.2.0.
Report the pyproject diff and the import output.

${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Reviewer (HAS Bash). Stack: uv/Python. Emit the structured verdict.

VERIFY:
- pyproject [project].dependencies contains xtrax; [tool.uv.sources] has
  xtrax = { path = "../xtrax", editable = true } with the literal ../xtrax path.
- uv lock exits 0.
- uv run python -c "import xtrax; print(xtrax.__version__)" -> 0.2.0.
- rg -n "import xtrax|from xtrax" src -> NO source imports yet.
PASS iff all hold; else NEEDS_WORK/FAIL with output.`,
  );

// ===== TRACK C — [xtrax T0.3] L3 cluster smoke (#1544) =====================
const trackC = () =>
  track(
    "1544",
    "Track C — [xtrax T0.3] L3 cluster smoke: py3.13 + JAX on SM120 + bathos sidecar (#1544)",
    `task_id: ${TASK_ID}. Depends on #1543. Stack: uv/Python + SLURM (myxcel). New sbatch + bathos sidecar allowed.

OBJECTIVE: Prove the py3.13 + JAX/jaxlib toolchain runs on the Blackwell SM120 cluster nodes BEFORE any
cluster perf gate trusts it (AS4 / R3a / pre-mortem PM-c).

DO:
- Create scripts/cluster/smoke_py313_jax.sbatch following scripts/cluster/recapture_pottsmpnn.sbatch
  (anchors: _PROJ; uv sync --quiet; uv run python; and the Blackwell guard that exports
  XLA_FLAGS=--xla_gpu_shard_autotuning=false when the node is node4007 or node4008). The job: a
  'uv run python -c' that imports jax, prints jax.__version__ and jax.devices(), and runs one tiny
  jit'd matmul on GPU.
- Add a bathos sidecar recording resolved jaxlib version, hostname, and confirmation XLA_FLAGS was
  applied (BATHOS verify-the-pipeline rule).
- Submit via sbatch --array=0-0 on the SM120 partition; wait for completion.

SUCCESS CRITERIA:
- L1: bash -n scripts/cluster/smoke_py313_jax.sbatch parses.
- L3: the array job COMPLETED; log shows a real GPU device and a non-hung matmul (NOT the 1170x
  autotuning-hung profile); sidecar captured jaxlib version + hostname + XLA_FLAGS-applied.
Report the job id, log tail, and sidecar contents.

${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Reviewer (HAS Bash + myxcel). Stack: uv/Python + SLURM. Emit the structured verdict.

VERIFY:
- bash -n scripts/cluster/smoke_py313_jax.sbatch exits 0; script anchors to its own dir (no bare
  relative paths) and contains the node4007/4008 XLA_FLAGS guard.
- The submitted array job reached COMPLETED (sacct); log shows a GPU device and a successful jit matmul.
- Bathos sidecar recorded jaxlib version, hostname, XLA_FLAGS-applied.
PASS iff the toolchain demonstrably runs on a real SM120 node; else FAIL (and BLOCK downstream perf gates).`,
  );

// ===== TRACK D — [xtrax T0.4] Boundary-lint for xtrax.* (#1545) ============
const trackD = () =>
  track(
    "1545",
    "Track D — [xtrax T0.4] Boundary-lint + ruff banned-api for xtrax.* (#1545)",
    `task_id: ${TASK_ID}. Depends on #1543. Stack: uv/Python + ruff/ast-grep. New lint rule/file allowed.

OBJECTIVE: Preserve ADR 260605 boundary discipline across the xtrax move (A4). ruff TID251 bans only
aminx.* paths; xtrax.* is unbanned, so the first xtrax.* import would silently bypass the boundary.
This guard MUST be mergeable atomically with the first xtrax.* import in any later slice.

ANCHORS:
- pyproject.toml [tool.ruff.lint.flake8-tidy-imports.banned-api] (~lines 137-141) bans
  aminx.inference.decode / aminx.host.plan / aminx.types.stages / aminx.inference.logits (ADR 260605).
- .ast-grep/ rules dir exists (see backlog #1304's ast-grep potts-isolation rule as a pattern).

DO (two halves, both required):
(a) An import-boundary check that aminx protein modules do not reach into xtrax internals they should not.
(b) A check asserting xtrax stays protein-AGNOSTIC: no xtrax module references the symbols atom_37,
    residue_index, tie_group_map. Prefer an ast-grep rule (repo convention) or a small pytest that
    greps xtrax source; wire it into CI / pre-commit.

SUCCESS CRITERIA:
- The check is green on the current tree.
- Injecting a violation (e.g. a temp xtrax module referencing atom_37) makes it FAIL; removing it passes.
- uv run ruff check . and the new check both pass.
Report the rule/file added and the inject-fail / clean-pass demonstration.

${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Reviewer (HAS Bash). Stack: uv/Python + ruff/ast-grep. Emit the structured verdict.

VERIFY:
- Both halves exist: (a) protein-module -> xtrax-internals guard; (b) xtrax-protein-agnostic guard
  (no atom_37/residue_index/tie_group_map).
- Demonstrate it FAILS on an injected violation and PASSES once removed (run both ways).
- uv run ruff check . exits 0; the new check is wired into CI / pre-commit.
PASS iff the guard is real and enforced; else NEEDS_WORK/FAIL.`,
  );

// ===== TRACK E — [xtrax T1.1-1.2] ResumableState + optimizer (#1548) =======
const trackE = () =>
  track(
    "1548",
    "Track E — [xtrax T1.1-1.2] Adopt xtrax ResumableState + optimizer (#1548)",
    `task_id: ${TASK_ID}. Depends on #1543 AND #1545 (boundary-lint must land with the first xtrax import).
Stack: uv/Python. Edit-only on the named files; this is the first real xtrax.* import.

OBJECTIVE: Rebuild aminx's training state + optimizer setup on xtrax primitives (the idea-2 free win
where xtrax is already at-or-ahead; no feature flag).

ANCHORS:
- aminx src/aminx/training/trainer.py: create_optimizer() (~lines 73-106, inline
  optax.warmup_cosine_decay + clip_by_global_norm) and the (model, opt_state, start_step) threading (~125-199).
- xtrax targets: xtrax.training.ResumableState (xtrax/src/xtrax/training/types.py:43 —
  step/key/model/opt_state/extras); xtrax.training.optim.adamw_with_schedule + make_optimizer.

DO:
- Replace the (model, opt_state, step) tuple threading with a single ResumableState.
- Replace create_optimizer() internals with adamw_with_schedule(...) / make_optimizer(...), mapping
  TrainingSpecification fields (learning_rate, warmup_steps, total_steps, weight_decay, gradient_clip).
- KEEP in aminx: domain losses (losses.py), metrics, diffusion (diffusion.py / train_diffusion.py),
  dataloading. Do NOT move them.
- Repoint imports and update affected tests under tests/training/.

DO NOT change checkpoint format (that is #1549, gated on the T-R2 decision). Do NOT touch tiling/host.

SUCCESS CRITERIA:
- uv run ty check and uv run ruff check . pass.
- uv run pytest -q tests/training -m 'not slow' green.
- A short smoke training run reproduces the legacy loss/accuracy curve within numerical tolerance.
Report the per-file diff, the test tail, and the loss-curve comparison.

${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Reviewer (HAS Bash). Stack: uv/Python. Emit the structured verdict.

VERIFY:
- Training state is xtrax.training.ResumableState; optimizer via xtrax adamw_with_schedule/make_optimizer
  (inline optax.chain removed).
- Domain losses/metrics/diffusion/dataloading remain in aminx
  (rg -n 'cross_entropy_loss|NoiseSchedule' src/aminx/training still resolves locally).
- The first xtrax import is covered by the #1545 boundary-lint (it is green).
- uv run ty check clean; uv run ruff check . clean; uv run pytest -q tests/training -m 'not slow' green.
- Smoke run loss curve within tolerance of legacy.
PASS iff all hold; else NEEDS_WORK/FAIL with command output.`,
  );

// ---- orchestrate: writing chain (A -> B -> C -> D -> E, sequential) ----
log("260611_xtrax-foundations: writing chain (A -> B -> C -> D -> E, sequential)");
const a = await trackA();
const b = await trackB();
const c = await trackC();
const d = await trackD();
const e = await trackE();

return {
  task_id: TASK_ID,
  sprint_id: 1,
  verdicts: {
    "1542": a,
    "1543": b,
    "1544": c,
    "1545": d,
    "1548": e,
  },
};
