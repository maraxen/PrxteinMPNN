// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/260621_r7-2-axisspec-rename.toml
// Regenerate: praxia dw emit-sprint 260621_r7-2-axisspec-rename.toml
// task_id: 260621_r7-2-axisspec-rename   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (A) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.

export const meta = {
  name: "260621_r7-2-axisspec-rename",
  description: "R7-2: rename xtrax AxisSpec.batch_size → default_batch_size and granularity → tile_granularity",
  phases: [
    { title: "Track A — R7-2: Rename AxisSpec fields to canonical names (#1964)" },
  ],
};

const TASK_ID = "260621_r7-2-axisspec-rename";
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

// Shared context for the writing tracks (from recon, task 260621_r7-2-axisspec-rename).
const EMITTER_CTX = `R7-1 gate (#1926) decided canonical AxisSpec field names:\n  - default_batch_size  (replaces xtrax's batch_size; avoids collision with AxisDecision.batch_size output)\n  - tile_granularity    (replaces xtrax's granularity; disambiguates from bucket/sequence granularity)\n\nSee: .praxia/docs/research/260616_axisspec-field-map.md\n\nScope (xtrax repo only — /home/marielle/projects/xtrax):\n  - xtrax/tiling/plan.py: 2 AxisSpec field declarations + 16 spec.batch_size read sites\n  - xtrax/tests/tiling/: 62 AxisSpec(batch_size=...) constructions + 7 granularity= usages\n  - Add __getattr__ shim on AxisSpec for deprecated names (read-only backward compat)\n\nOut of scope:\n  - AxisDecision.batch_size — this is the OUTPUT field (final resolved batch size), NOT the AxisSpec input field; must NOT be renamed\n  - SafeMap(batch_size=...) / SafeMapIterator / dispatch.py strategy.batch_size — these are strategy fields, not AxisSpec\n  - aminx AxisSpec — aminx re-exports xtrax.tiling.AxisSpec; no separate class to update\n  - prolix AxisSpec — already uses default_batch_size/tile_granularity; no changes needed\n  - RunSpecification.batch_size in aminx (trainer.py, prep.py) — completely different field\n`;

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

// ===== TRACK A — Track A — R7-2: Rename AxisSpec fields to canonical names (#1964) =========================
const trackA = () =>
  track(
    "1964",
    "Track A — R7-2: Rename AxisSpec fields to canonical names (#1964)",
    `task_id: ${TASK_ID}. Fix #1964: Rename xtrax AxisSpec field names to match R7-1 canonical decisions.\n\nWORKING DIRECTORY: /home/marielle/projects/xtrax\n\nCRITICAL OUT-OF-SCOPE (do NOT touch):\n  - AxisDecision.batch_size — output field, different concept, stays as batch_size\n  - SafeMap(batch_size=...) / SafeMapIterator / strategy.batch_size — strategy fields, not AxisSpec\n  - aminx/src/ files (RunSpecification.batch_size is a different field entirely)\n\n--- STEP 1: Rename the two fields in xtrax/src/xtrax/tiling/plan.py ---\n\nIn class AxisSpec (around line 26):\n\nChange field declarations:\n  batch_size: int\n  granularity: int = 1\n\nTo:\n  default_batch_size: int\n  tile_granularity: int = 1\n\nUpdate the docstring Attributes section:\n  batch_size: ... → default_batch_size: Default batch size hint for SafeMap and Vmap threshold.\n  granularity: ... → tile_granularity: Alignment granularity (default 1, no constraint).\n\n--- STEP 2: Add __getattr__ shim for backward compat (read-only) ---\n\nAdd this method to the AxisSpec class, AFTER __post_init__:\n\n  def __getattr__(self, name: str):\n      import warnings\n      if name == "batch_size":\n          warnings.warn(\n              "AxisSpec.batch_size is deprecated; use AxisSpec.default_batch_size",\n              DeprecationWarning,\n              stacklevel=2,\n          )\n          return object.__getattribute__(self, "default_batch_size")\n      if name == "granularity":\n          warnings.warn(\n              "AxisSpec.granularity is deprecated; use AxisSpec.tile_granularity",\n              DeprecationWarning,\n              stacklevel=2,\n          )\n          return object.__getattribute__(self, "tile_granularity")\n      raise AttributeError(f"type object 'AxisSpec' has no attribute {name!r}")\n\n--- STEP 3: Update all internal spec.batch_size reads in plan.py ---\n\nIn plan.py, all occurrences of spec.batch_size (there are ~16) refer to AxisSpec.batch_size.\nReplace ALL of them with spec.default_batch_size.\n\nIMPORTANT: do NOT change:\n  - AxisDecision.batch_size (the output dataclass around line 77)\n  - SafeMap(batch_size=...) construction sites (these are SafeMap strategy fields)\n  - Any string in error messages like "batch_size={spec.default_batch_size}" is fine — just keep consistent\n\nAlso replace any spec.granularity references with spec.tile_granularity (there may be 0 — verify with grep).\n\n--- STEP 4: Update all tests ---\n\nIn xtrax/tests/tiling/test_plan.py:\n\nUse sed or Edit to replace ALL occurrences:\n  AxisSpec(..., batch_size=   →   AxisSpec(..., default_batch_size=\n  granularity=               →   tile_granularity=\n\nThere are ~62 AxisSpec(batch_size=...) constructions and ~7 granularity= usages in the tests.\nUse: sed -i 's/AxisSpec(\(.*\)batch_size=/AxisSpec(\1default_batch_size=/g' tests/tiling/test_plan.py\n\nOr read the file and use Edit with replace_all=true for each pattern.\n\nAlso check for other test files:\n  grep -rn "AxisSpec(" tests/ | grep "batch_size=" to find any other test files\n\n--- STEP 5: Verify ---\n\nRun the full xtrax test suite:\n  cd /home/marielle/projects/xtrax && uv run pytest\n\nVerify the shim works:\n  uv run python -c "\n  import warnings\n  from xtrax.tiling import AxisSpec\n  spec = AxisSpec(name='batch', cardinality=32, default_batch_size=8)\n  assert spec.default_batch_size == 8\n  with warnings.catch_warnings(record=True) as w:\n      warnings.simplefilter('always')\n      val = spec.batch_size\n      assert len(w) == 1 and issubclass(w[0].category, DeprecationWarning)\n      assert val == 8\n  print('Shim OK — batch_size deprecated alias fires DeprecationWarning')\n  "\n\nVerify AxisDecision is untouched:\n  uv run python -c "\n  from xtrax.tiling.plan import AxisDecision, AxisSpec\n  spec = AxisSpec(name='b', cardinality=32, default_batch_size=8)\n  from xtrax.tiling.strategy import SafeMap\n  dec = AxisDecision(spec=spec, batch_size=8, reasoning='test', strategy=SafeMap(batch_size=8))\n  assert dec.batch_size == 8\n  print('AxisDecision.batch_size unchanged — OK')\n  "\n\n--- STEP 6: Check aminx tests still pass ---\n\n  cd /home/marielle/projects/aminx && uv run pytest\n\n--- STEP 7: Commit from xtrax directory ---\n\n  git add src/xtrax/tiling/plan.py tests/\n  git commit -m "refactor(R7-2): rename AxisSpec.batch_size→default_batch_size, granularity→tile_granularity; add deprecation shim (#1964)"\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Review R7-2: AxisSpec canonical field name migration.\n\nVERIFY 1: xtrax/src/xtrax/tiling/plan.py — AxisSpec has default_batch_size: int and tile_granularity: int = 1 (NOT batch_size or granularity)\nVERIFY 2: AxisSpec has __getattr__ method that returns default_batch_size for "batch_size" and tile_granularity for "granularity" with DeprecationWarning\nVERIFY 3: AxisDecision.batch_size is UNCHANGED — still named batch_size (it is the output field, not the input hint)\nVERIFY 4: All plan.py internal accesses use spec.default_batch_size (grep -n "spec\.batch_size" src/xtrax/tiling/plan.py should return 0 hits for AxisSpec accesses)\nVERIFY 5: All tests use AxisSpec(..., default_batch_size=...) — no remaining batch_size= kwargs in AxisSpec constructions in test files\nVERIFY 6: cd /home/marielle/projects/xtrax && uv run pytest — full suite passes\nVERIFY 7: cd /home/marielle/projects/aminx && uv run pytest — full suite passes\nVERIFY 8: Deprecation shim smoke-check passes:\n  cd /home/marielle/projects/xtrax && uv run python -c "\n  import warnings; from xtrax.tiling import AxisSpec\n  spec = AxisSpec(name='b', cardinality=32, default_batch_size=8)\n  with warnings.catch_warnings(record=True) as w:\n      warnings.simplefilter('always')\n      val = spec.batch_size\n  assert len(w) == 1 and issubclass(w[0].category, DeprecationWarning) and val == 8\n  print('OK')\n  "\n\nPASS if all 8 satisfied.\nFAIL if batch_size or granularity remain as first-class AxisSpec fields, or if AxisDecision.batch_size was renamed.\n`,
  );

// ---- orchestrate: writing chain (A, sequential) ----
log("260621_r7-2-axisspec-rename: writing chain (A, sequential)");
const a = await trackA();

return {
  task_id: TASK_ID,
  sprint_id: 1,
  verdicts: {
    "1964": a
  },
};
