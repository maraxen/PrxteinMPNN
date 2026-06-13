// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 260613_xtrax-t0-potts-core   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (A,B) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time. Only the read-only
//   research/concurrent tracks (C,D,E) run concurrently.

export const meta = {
  name: "260613_xtrax-t0-potts-core",
  description: "xtrax T0 pin + boundary lint, Potts recapture script + model + spec verification",
  phases: [
    { title: "Track A — [xtrax T0.2] Add editable xtrax pin + dependency (#1543)" },
    { title: "Track B — [xtrax T0.4] Boundary-lint (ADR 260605) + ruff banned-api for xtrax.* — atomic with first xtrax import (#1545)" },
    { title: "Track C — P-01: Weight recapture — PottsMPNN .pt → potts_<id>.eqx.zst (bathos-tracked) (#1291)" },
    { title: "Track D — P-03: aminx.potts.model — PottsModel(eqx.Module) with DifferentiableTRW internal (#1292)" },
    { title: "Track E — P-04: aminx.potts.spec — PottsRunSpec frozen dataclass (#1293)" },
  ],
};

const TASK_ID = "260613_xtrax-t0-potts-core";
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

// Shared context for the writing tracks (from recon, task 260613_xtrax-t0-potts-core).
const EMITTER_CTX = `[MANUAL: paste recon findings here before running emit-sprint]`;

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

// ===== TRACK A — Track A — [xtrax T0.2] Add editable xtrax pin + dependency (#1543) =========================
const trackA = () =>
  track(
    "1543",
    "Track A — [xtrax T0.2] Add editable xtrax pin + dependency (#1543)",
    `task_id: ${TASK_ID}. # xtrax editable dependency — verify pin completeness\n\n## Background\nxtrax is a JAX training/tiling library at ../xtrax (editable install). The editable pin was\nadded to pyproject.toml in commit 4f1b86a as part of the T0.1 Python 3.13 upgrade.\nThis task verifies the pin is correct and the gate commands pass — no new code is expected.\n\n## Recon anchors\n- pyproject.toml:26 — "xtrax>=0.2.0" in [project].dependencies\n- pyproject.toml:264 — xtrax = { path = "../xtrax", editable = true } in [tool.uv.sources]\n\n## What to check\n1. Open pyproject.toml and confirm both anchors are present.\n2. Run the acceptance gate from the project root (/home/marielle/projects/aminx):\n     uv lock --check\n     uv run python -c 'import xtrax; print(xtrax.__version__)'\n   Expected output: 0.2.0\n\n## If uv lock --check fails\nRun \`uv lock\` (no --check) to regenerate the lockfile, then re-run \`uv lock --check\`.\n\n## If import fails\nCheck that the xtrax source exists:\n  ls /home/marielle/projects/xtrax/src/xtrax/__init__.py\nIf missing, the xtrax repo is not checked out — report this as a blocker.\nDo NOT attempt to fix by changing the pin path.\n\n## If either anchor is absent from pyproject.toml\nAdd the missing line:\n  - To [project].dependencies array: "xtrax>=0.2.0",\n  - To [tool.uv.sources] section: xtrax = { path = "../xtrax", editable = true }\nThen run \`uv lock\` and re-run the gate.\n\n## Expected outcome\nBoth anchors present, \`uv lock --check\` exits 0, import prints 0.2.0.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Reviewer for T0.2 — xtrax editable dependency\n\n## Spec\npyproject.toml must contain BOTH:\n  [project].dependencies: "xtrax>=0.2.0"\n  [tool.uv.sources]: xtrax = { path = "../xtrax", editable = true }\nGate: \`uv lock\` resolves cleanly; import prints 0.2.0\n\n## VERIFY items\nVERIFY: \`grep 'xtrax' pyproject.toml\` shows both the version pin (>=0.2.0) and editable source entry\nVERIFY: \`uv lock --check\` exits 0 (run from /home/marielle/projects/aminx)\nVERIFY: \`uv run python -c 'import xtrax; print(xtrax.__version__)'\` prints exactly \`0.2.0\`\nVERIFY: uv.lock is committed or unchanged if it was already up to date\n\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or the import fails.\n`,
  );

// ===== TRACK B — Track B — [xtrax T0.4] Boundary-lint (ADR 260605) + ruff banned-api for xtrax.* — atomic with first xtrax import (#1545) =========================
const trackB = () =>
  track(
    "1545",
    "Track B — [xtrax T0.4] Boundary-lint (ADR 260605) + ruff banned-api for xtrax.* — atomic with first xtrax import (#1545)",
    `task_id: ${TASK_ID}. # xtrax import boundary enforcement (ADR 260605)\n\n## Background\nADR 260605_potts-parallel-not-stageset requires that aminx protein modules only import from\nxtrax's public __init__ re-exports, not from implementation submodules directly. Also, xtrax\nmust never reference aminx-specific field names (atom_37, residue_index, tie_group_map).\n\nThe current ruff TID251 config at pyproject.toml:136-140 bans specific aminx.* module paths\nfrom potts modules. It does NOT enforce xtrax.* boundary. This task adds that enforcement\natomically — before any xtrax.* imports land in aminx code.\n\n## Recon anchors\n- pyproject.toml:133-140 — existing [tool.ruff.lint.flake8-tidy-imports.banned-api] section\n- tests/potts/test_import_boundaries.py:18-23 — existing FORBIDDEN_IMPORTS set (aminx.* only)\n- /home/marielle/projects/xtrax/src/xtrax/ — xtrax public API structure:\n  Each subpackage has __init__.py that re-exports public API; implementation files\n  (bundle.py, protocols.py, dispatch.py, etc.) are internal to each subpackage.\n\n## Change 1 — pyproject.toml: extend banned-api\nAdd the following entries to [tool.ruff.lint.flake8-tidy-imports.banned-api]\n(insert after the existing aminx.inference.logits entry at line 140):\n\n"xtrax.stages.bundle" = {msg = "Use \`from xtrax.stages import ...\` — import from public __init__ only (ADR 260605)"}\n"xtrax.stages.protocols" = {msg = "Use \`from xtrax.stages import ...\` — import from public __init__ only (ADR 260605)"}\n"xtrax.tiling.bucket" = {msg = "Use \`from xtrax.tiling import ...\` — import from public __init__ only (ADR 260605)"}\n"xtrax.tiling.dedup" = {msg = "Use \`from xtrax.tiling import ...\` — import from public __init__ only (ADR 260605)"}\n"xtrax.tiling.dispatch" = {msg = "Use \`from xtrax.tiling import ...\` — import from public __init__ only (ADR 260605)"}\n"xtrax.tiling.plan" = {msg = "Use \`from xtrax.tiling import ...\` — import from public __init__ only (ADR 260605)"}\n"xtrax.tiling.strategy" = {msg = "Use \`from xtrax.tiling import ...\` — import from public __init__ only (ADR 260605)"}\n"xtrax.engine.engine" = {msg = "Use \`from xtrax.engine import ...\` — import from public __init__ only (ADR 260605)"}\n"xtrax.engine.io" = {msg = "Use \`from xtrax.engine import ...\` — import from public __init__ only (ADR 260605)"}\n"xtrax.safety.manager" = {msg = "Use \`from xtrax.safety import ...\` — import from public __init__ only (ADR 260605)"}\n\n## Change 2 — tests/potts/test_import_boundaries.py: add TestXtraxBoundaries class\nAdd a new test class TestXtraxBoundaries at the bottom of the file with two tests:\n\n(a) test_aminx_potts_no_xtrax_internals\n    Glob all *.py files in src/aminx/potts/. For each file, extract imports via AST using the\n    existing extract_imports() helper. Assert no import matches any of the following set:\n      XTRAX_BANNED_INTERNAL = {\n          "xtrax.stages.bundle", "xtrax.stages.protocols",\n          "xtrax.tiling.bucket", "xtrax.tiling.dedup", "xtrax.tiling.dispatch",\n          "xtrax.tiling.plan", "xtrax.tiling.strategy",\n          "xtrax.engine.engine", "xtrax.engine.io",\n          "xtrax.safety.manager",\n      }\n    Use the existing find_forbidden_imports() helper (prefix-match logic).\n\n(b) test_xtrax_no_aminx_field_names\n    Locate xtrax source root: /home/marielle/projects/xtrax/src/xtrax/\n    If that path does not exist: pytest.skip("xtrax source not found at expected path")\n    Scan all .py files under that root. For each file, read its text and assert it does NOT\n    contain any of these substrings: "atom_37", "residue_index", "tie_group_map".\n    On failure, report: which file, which forbidden string.\n\n## Gate: verify the new banned-api rules trigger a violation\nAfter writing both changes, run from /home/marielle/projects/aminx:\n  uv run ruff check . 2>&1 | head -5\n(should exit 0 — no production code violates the new rules)\n\nThen inject a temporary violation to verify the rule fires:\n  Add this comment to src/aminx/potts/spec.py after the last import:\n    from xtrax.stages.bundle import FuseFn  # noqa: F401\n  Run: uv run ruff check src/aminx/potts/spec.py 2>&1 | grep TID251\n  Expected: line reporting TID251 violation for xtrax.stages.bundle\n  Remove the injected line after confirming the rule fires.\n\nRun \`uv run pytest tests/potts/test_import_boundaries.py -v\` — all tests must pass.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Reviewer for T0.4 — xtrax import boundary enforcement\n\n## Spec\npyproject.toml [tool.ruff.lint.flake8-tidy-imports.banned-api] must include TID251 bans for:\n  xtrax.stages.bundle, xtrax.stages.protocols, xtrax.tiling.bucket, xtrax.tiling.dedup,\n  xtrax.tiling.dispatch, xtrax.tiling.plan, xtrax.tiling.strategy,\n  xtrax.engine.engine, xtrax.engine.io, xtrax.safety.manager\ntests/potts/test_import_boundaries.py must include TestXtraxBoundaries with:\n  (a) test that aminx.potts.* doesn't import xtrax internal paths\n  (b) test that xtrax source doesn't contain atom_37/residue_index/tie_group_map\n\n## VERIFY items\nVERIFY: \`uv run ruff check .\` exits 0 with no new TID251 violations in production code\nVERIFY: \`uv run pytest tests/potts/test_import_boundaries.py -v\` — all tests pass including TestXtraxBoundaries\nVERIFY: Injecting \`from xtrax.stages.bundle import FuseFn\` in any src/aminx/potts/*.py triggers TID251\nVERIFY: All 10 xtrax internal module paths appear in pyproject.toml banned-api section\n\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or the gate injection test does not trigger.\n`,
  );

// ===== TRACK C — Track C — P-01: Weight recapture — PottsMPNN .pt → potts_<id>.eqx.zst (bathos-tracked) (#1291) =========================
const trackC = () =>
  track(
    "1291",
    "Track C — P-01: Weight recapture — PottsMPNN .pt → potts_<id>.eqx.zst (bathos-tracked) (#1291)",
    `task_id: ${TASK_ID}. # P-01: PottsMPNN .pt -> potts_<id>.eqx.zst weight recapture script\n\n## Background\nThe recapture script ports mistypotts/pottsmpnn_ckpt_export.py into a bathos-tracked pipeline.\nIt loads a PyTorch .pt checkpoint, converts h/J via etab_to_dense_h_j_w (preserving x2 scale),\nserializes to .eqx.zst using the PottsCheckpointData pytree, and bakes in k_neighbors metadata.\n\n## Current state\nScript is largely complete at scripts/recapture/pottsmpnn_to_eqx.py (487 lines).\nBathos sidecar at scripts/recapture/pottsmpnn_to_eqx.bth.toml.\n\nKey functions expected by tests/potts/test_pottsmpnn_conversion.py:\n  - etab_to_dense_h_j_w at line 65: (etab, e_idx, mask) -> (h, J, W)\n    * etab shape: (1, L, K, q, q); e_idx shape: (1, L, K)\n    * self-loops (e_idx[i,k]==i) go to h diagonal\n    * off-diagonal pairs accumulate in J, symmetrize: j = 0.5*(j + j^T)\n    * x2 scale: h *= 2.0, j *= 2.0 (directed-slot PottsMPNN convention)\n  - extract_k_neighbors_from_config at line 137: reads payload['args']['k_neighbors']\n    then payload['hyper_params']['k_neighbors']; raises ValueError "k_neighbors not found" if absent\n  - save_checkpoint at line ~286: PottsCheckpointData pytree -> .eqx.zst\n    via eqx.tree_serialise_leaves + zstandard compression\n\nPottsCheckpointData (line 44):\n  class PottsCheckpointData(eqx.Module):\n    h, j, w, mask arrays; k_neighbors: int = eqx.field(static=True)\n\n## What to do\n1. Run the conversion tests from /home/marielle/projects/aminx:\n     uv run pytest tests/potts/test_pottsmpnn_conversion.py -v --no-header 2>&1 | tail -30\n\n2. If ALL tests PASS: task is complete. Report PASS.\n\n3. If any test FAILS:\n   a. Read the failing test to understand the expected contract.\n   b. Read scripts/recapture/pottsmpnn_to_eqx.py to find the gap.\n   c. Fix the gap in the script.\n   d. Re-run \`uv run pytest tests/potts/test_pottsmpnn_conversion.py -v\`.\n\n## Key test expectations\n- test_etab_to_dense_h_j_w_synthetic_2residue: 2-residue ground-truth, tolerance 1e-6\n- test_save_potts_checkpoint_format: output starts with zstd magic 0x28 0xb5 0x2f 0xfd\n- test_k_neighbors_from_model_config: ValueError raised with "k_neighbors not found" for empty payload\n\n## No external deps needed\nAll three tests use synthetic data only — no actual .pt checkpoint required.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Reviewer for P-01 — PottsMPNN weight recapture script\n\n## Spec\nscripts/recapture/pottsmpnn_to_eqx.py must expose:\n  - etab_to_dense_h_j_w: (etab, e_idx, mask) -> (h, J, W) with x2 scale factor\n  - save_checkpoint: serializes PottsCheckpointData to .eqx.zst (zstd-compressed pytree)\n  - extract_k_neighbors_from_config: reads k_neighbors from payload dict, raises ValueError if absent\nBathos sidecar at scripts/recapture/pottsmpnn_to_eqx.bth.toml must exist.\n2-residue synthetic ground-truth check must pass within 1e-6 tolerance.\n\n## VERIFY items\nVERIFY: \`uv run pytest tests/potts/test_pottsmpnn_conversion.py -v\` — all 3 tests pass\nVERIFY: scripts/recapture/pottsmpnn_to_eqx.bth.toml exists and is non-empty\nVERIFY: test_etab_to_dense_h_j_w_synthetic_2residue passes (h/J match reference within 1e-6)\nVERIFY: test_save_potts_checkpoint_format passes (output is valid zstd)\nVERIFY: test_k_neighbors_from_model_config passes (ValueError raised for empty payload)\n\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.\n`,
    "worktree"
  );

// ===== TRACK D — Track D — P-03: aminx.potts.model — PottsModel(eqx.Module) with DifferentiableTRW internal (#1292) =========================
const trackD = () =>
  track(
    "1292",
    "Track D — P-03: aminx.potts.model — PottsModel(eqx.Module) with DifferentiableTRW internal (#1292)",
    `task_id: ${TASK_ID}. # P-03: aminx.potts.model — PottsModel(eqx.Module) with DifferentiableTRW\n\n## Background\nPottsModel is a Potts MPNN with TRW inference on k-NN graphs built by ProteinFeatures.\nIt projects edge features to unary (h) and pairwise (J) potentials, then runs differentiable\ntree-reweighted message passing for marginals.\n\n## Current state\nPottsModel(eqx.Module) is fully implemented at src/aminx/potts/model.py:71\n\nKey elements:\n  - __call__(key, coords, mask, residue_index, chain_index) -> (marginals, h, J, rho) at line 241\n  - infer_params returning PottsParams namedtuple at line 353\n  - log_prob(seq, h, j, w, mask) @staticmethod at line 397 — pure function, no model state\n  - DifferentiableTRW: held as static field self.trw (eqx.field(static=True)) at line 97\n  - trw_spec: PottsTRWRunSpec static field at line 98\n  - k_neighbors: int = eqx.field(static=True) at line 89\n  - x2 scale factor preserved; documented in module docstring lines 5-8\n  - Training mode guard at lines 167-172: raises ValueError if trw_loop='fori' and training=True\n\n## What to do\n1. Run model tests from /home/marielle/projects/aminx:\n     uv run pytest tests/potts/test_potts_model.py tests/potts/test_potts_correctness.py -v --no-header 2>&1 | tail -40\n\n2. Run type check:\n     uv run ty check src/aminx/potts/model.py 2>&1 | tail -20\n\n3. If ALL tests PASS and ty is clean: task is complete. Report PASS.\n\n4. If any test FAILS or ty reports new errors:\n   a. Read model.py lines 71-180 (class definition) and 241-351 (__call__).\n   b. Fix the specific failing requirement.\n   c. Re-run tests and ty check.\n\n5. Verify PottsModel is exported from aminx.potts:\n     grep "PottsModel" src/aminx/potts/__init__.py\n   If absent: add \`from aminx.potts.model import PottsModel\` and "PottsModel" to __all__.\n\n## Weight loading (not inside PottsModel)\nCaller uses load_weights(local_path=..., skeleton=PottsModel(...)) from aminx.io.weights.\nNo fork of weights.py needed.\n\n## Batch-axis convention\n__call__ operates on single N-residue input (no batch dim).\nCaller applies eqx.filter_vmap for batching. If this is undocumented in the __call__\ndocstring (lines 245-266), add one line: "No batch axis. Caller applies filter_vmap."\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Reviewer for P-03 — PottsModel(eqx.Module)\n\n## Spec\nsrc/aminx/potts/model.py must have PottsModel(eqx.Module) with:\n  - __call__(key, coords, mask, residue_index, chain_index) -> (marginals, h, J, rho)\n  - log_prob @staticmethod (pure function: no self)\n  - DifferentiableTRW as eqx.field(static=True)\n  - k_neighbors as eqx.field(static=True)\n  - x2 scale factor preserved and documented\n  - Training guard: raises ValueError if trw_loop='fori' and training=True\n  - PottsModel exported from src/aminx/potts/__init__.py\n\n## VERIFY items\nVERIFY: \`uv run pytest tests/potts/test_potts_model.py tests/potts/test_potts_correctness.py -v\` — all tests pass\nVERIFY: \`uv run ty check src/aminx/potts/model.py\` — exits 0 or no new errors introduced\nVERIFY: PottsModel.log_prob is a @staticmethod (grep confirms decorator in model.py)\nVERIFY: PottsModel.trw and PottsModel.trw_spec have eqx.field(static=True) annotation\nVERIFY: \`grep "PottsModel" src/aminx/potts/__init__.py\` — PottsModel is exported\n\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.\n`,
    "worktree"
  );

// ===== TRACK E — Track E — P-04: aminx.potts.spec — PottsRunSpec frozen dataclass (#1293) =========================
const trackE = () =>
  track(
    "1293",
    "Track E — P-04: aminx.potts.spec — PottsRunSpec frozen dataclass (#1293)",
    `task_id: ${TASK_ID}. # P-04: aminx.potts.spec — PottsRunSpec frozen dataclass\n\n## Background\nPottsRunSpec couples model checkpoint paths (weights_path, caliby_path), TRW numerics config\n(trw_spec), and inference metadata (n_backbones, k_neighbors, training). It is a frozen\ndataclass (not eqx.Module) for avoid PyTree registration. Round-trips to/from JSON.\n\n## Current state\nPottsRunSpec(frozen=True) is fully implemented at src/aminx/potts/spec.py:36\n\nFields:\n  n_backbones: int = 1\n  weights_path: str = ""\n  caliby_path: str | None = None   (None = identity calibration, valid)\n  trw_spec: PottsTRWRunSpec | None = None\n  k_neighbors: int = 0\n  training: bool = False\n\n__post_init__ at line 62:\n  - fills trw_spec default if None\n  - raises ValueError if k_neighbors <= 0 (line 68-73)\n  - raises ValueError if training=True and trw_spec.trw_loop='fori' (line 75-80)\n  - raises ValueError if n_backbones < 1 (line 82-84)\n\nJSON round-trip:\n  - to_json() at line 86: asdict + trw_spec.to_json_dict()\n  - from_json() at line 93: json.loads + PottsTRWRunSpec.from_json_dict()\n\ncaliby_path load-time file-absent check: in runner.py:140 (FileNotFoundError) — NOT in __post_init__\n\n## What to do\n1. Run spec tests from /home/marielle/projects/aminx:\n     uv run pytest tests/potts/test_spec.py -v --no-header 2>&1 | tail -30\n\n2. Run type check:\n     uv run ty check src/aminx/potts/spec.py 2>&1 | tail -20\n\n3. Verify JSON round-trip:\n     uv run python -c "\nfrom aminx.potts.spec import PottsRunSpec\ns = PottsRunSpec(weights_path='/tmp/w', k_neighbors=48)\ns2 = PottsRunSpec.from_json(s.to_json())\nassert s == s2, f'round-trip mismatch'\nprint('round-trip OK')\n" 2>&1\n\n4. If ALL tests PASS, ty clean, round-trip works: task is complete. Report PASS.\n\n5. If any fails: read spec.py lines 35-161 and tests/potts/test_spec.py. Fix and re-run.\n\n## Not required\n- File existence check at __post_init__ for caliby_path (that is runner.py's job)\n- trw_spec as eqx.field (this is a plain frozen dataclass)\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. # Reviewer for P-04 — PottsRunSpec frozen dataclass\n\n## Spec\nsrc/aminx/potts/spec.py must have PottsRunSpec(frozen=True) with:\n  - Fields: n_backbones, weights_path, caliby_path (str|None), trw_spec, k_neighbors, training\n  - k_neighbors > 0 validation at construction (raises ValueError)\n  - n_backbones >= 1 validation at construction (raises ValueError)\n  - trw_loop='fori' + training=True guard at construction (raises ValueError naming OOM risk)\n  - caliby_path=None is valid (no error raised)\n  - JSON round-trip via to_json() / from_json() preserving all fields including nested trw_spec\n  - caliby_path file-absent check is load-time in runner.py (not __post_init__)\n\n## VERIFY items\nVERIFY: \`uv run pytest tests/potts/test_spec.py -v\` — all tests pass\nVERIFY: \`uv run ty check src/aminx/potts/spec.py\` — exits 0 or no new errors\nVERIFY: \`PottsRunSpec(weights_path='/tmp/w', k_neighbors=48).to_json()\` is parseable JSON\nVERIFY: JSON round-trip: from_json(to_json(s)) == s\nVERIFY: \`PottsRunSpec(weights_path='/tmp/w', k_neighbors=0)\` raises ValueError\nVERIFY: \`PottsRunSpec(weights_path='/tmp/w', k_neighbors=48, n_backbones=0)\` raises ValueError\n\nPASS if all VERIFY items are satisfied.\nFAIL if any VERIFY item is not met or untestable as written.\n`,
    "worktree"
  );

// ---- orchestrate: sequential writing chain || read-only research ----------
log("260613_xtrax-t0-potts-core: writing chain (A -> B, sequential) || research (C, D, E, read-only)");
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
    "1543": writing.a,
    "1545": writing.b
  },
  research_1291: resC,
  research_1292: resD,
  research_1293: resE
};
