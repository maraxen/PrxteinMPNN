// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/260618_runspec-xtrax-run.toml
// Regenerate: praxia dw emit-sprint 260618_runspec-xtrax-run.toml
// task_id: 260618_runspec-xtrax-run   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (A1,A2,B) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.

export const meta = {
  name: "260618_runspec-xtrax-run",
  description: "P1 sprint: wire fixed_mask through inspection path, add chain_mask_fixed, verify _sync_run_spec guard.",
  phases: [
    { title: "Track A-1 -- Wire fixed_mask into inspection runner path (#1880)" },
    { title: "Track A-2 -- Add chain_mask_fixed to SamplingSpecification (#1881)" },
    { title: "Track B -- Verify _sync_run_spec double-fire guard (e0791e4, verify-only)" },
  ],
};

const TASK_ID = "260618_runspec-xtrax-run";
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

// Shared context for the writing tracks (from recon, task 260618_runspec-xtrax-run).
const EMITTER_CTX = `Recon findings (2026-06-16):\n\n1. FIELD LOCATIONS -- \`fixed_mask: ArrayLike | None = None\` (line 268) and\n   \`sidechain_conditioning: bool = False\` (line 269) are declared on BASE class\n   \`RunSpecification\` (src/aminx/run/specs.py:147). ALL subclasses, including\n   \`InspectionSpecification\` (line 580), already inherit them -- the fields are\n   NOT missing. But \`runner.py\` inspect() path (lines 548-559) does NOT pass\n   \`spec.fixed_mask\` or \`spec.sidechain_conditioning\` into \`build_inference_bundle\`.\n   They are silently ignored in the inspection code path. #1904 = fix wiring, not add fields.\n\n2. CHAIN_MASK WIRING -- \`src/aminx/host/_sampling_helper.py:343-365\` computes\n   chain_mask from \`spec.fixed_mask\` (complement) or all-ones fallback.\n   \`build_inference_bundle\` at \`src/aminx/inference/bundle_builder.py:47\` accepts\n   \`chain_mask: jax.Array | None = None\` but runner.py does not pass it.\n   \`chain_mask_fixed\` (#1905) is a NEW field that bypasses the derivation entirely.\n\n3. DOUBLE-FIRE -- Already fixed. \`RunSpecification._sync_run_spec()\` at specs.py:272-278\n   checks \`self._run_spec_synced\` before firing. Guard committed in e0791e4 (on main).\n   Item #1906 is verify-only -- no implementation needed.\n`;

// ---- per-track stage helpers ---------------------------------------------
const fixer = (prompt, label, phaseName, isolation = null, context = null) => {
  const opts = { agentType: "fixer", label, phase: phaseName };
  if (isolation) opts.isolation = isolation;
  if (context) Object.assign(opts, context);
  return agent(`${prompt}\n\nWhen done, end your message with 'verdict: done' on its own line.`, opts);
};

const reviewer = (itemId, prompt, label, phaseName, isolation = null, context = null) => {
  const opts = { agentType: "reviewer", label, phase: phaseName, schema: VERDICT_SCHEMA };
  if (isolation) opts.isolation = isolation;
  if (context) Object.assign(opts, context);
  return agent(prompt, opts);
};

// Sequential implement->review with bounded NEEDS_WORK repair cycles.
async function track(itemId, phaseName, fixerPrompt, reviewerPrompt, isolation = null, context = null) {
  log(`[${itemId}] implement`);
  await fixer(fixerPrompt, `fix:${itemId}`, phaseName, isolation, context);
  let verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}`, phaseName, isolation, context);
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
    verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}:re:${retry}`, phaseName, isolation, context);
  }
  return verdict;
}

// ===== TRACK A1 — Track A-1 -- Wire fixed_mask into inspection runner path (#1880) =========================
const trackA1 = () =>
  track(
    "1904",
    "Track A-1 -- Wire fixed_mask into inspection runner path (#1880)",
    `task_id: ${TASK_ID}. Fix #1880: the fields \`fixed_mask\` and \`sidechain_conditioning\` are already on\n\`RunSpecification\` (inherited by \`InspectionSpecification\`), but the inspection\nrunner silently ignores them.\n\nWORKING DIRECTORY: /home/marielle/projects/aminx\n\nFILES TO CHANGE:\n- src/aminx/host/runner.py (inspection path, approx line 548)\n- src/aminx/inference/bundle_builder.py (check signature at line 47 before adding kwargs)\n\nWHAT TO DO:\nIn runner.py's inspect() function, find every \`build_inference_bundle(...)\` call\n(around line 548). Add:\n\n  fixed_mask=getattr(spec, "fixed_mask", None),\n\nCheck \`bundle_builder.py:47\` for whether \`sidechain_conditioning\` is also an accepted\nparam. If yes, also add:\n  sidechain_conditioning=getattr(spec, "sidechain_conditioning", False)\nIf not, leave a TODO comment for future wiring.\n\nDo NOT add new fields to any spec class -- they already exist on the base class.\n\nACCEPTANCE:\n1. \`InspectionSpecification(inputs=["x.pdb"], fixed_mask=None)\` round-trips without error.\n2. runner.py inspect() path passes \`fixed_mask\` from spec to \`build_inference_bundle\`.\n3. \`uv run pytest tests/ -q --ignore=tests/parity\` clean.\n\nCOMMIT:\n  git add src/aminx/host/runner.py\n  git commit -m "fix(#1880): thread fixed_mask into inspection runner path"\n\nWhen done, end with: verdict: done\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Review #1880 fix: inspect path fixed_mask wiring.\n\nVERIFY 1: grep -n "fixed_mask" src/aminx/host/runner.py -- should appear at the\n  build_inference_bundle call site in inspect(), not just in comments.\nVERIFY 2: Field types on the call site match \`build_inference_bundle\` signature\n  at src/aminx/inference/bundle_builder.py:47 (jax.Array | None).\nVERIFY 3: \`InspectionSpecification(inputs=["x.pdb"], fixed_mask=None)\` does not raise.\nVERIFY 4: \`uv run pytest tests/ -q --ignore=tests/parity\` -- no regressions.\n\nPASS if all 4 satisfied. FAIL if fixed_mask is still silently dropped in inspect().\n`,
    null,
    null
  );

const trackA2 = () =>
  track(
    "1905",
    "Track A-2 -- Add chain_mask_fixed to SamplingSpecification (#1881)",
    `task_id: ${TASK_ID}. Fix #1881: add \`chain_mask_fixed: ArrayLike | None = None\` to \`SamplingSpecification\`\nand wire it as a direct-override priority in \`_sampling_helper.py\`.\n\nWORKING DIRECTORY: /home/marielle/projects/aminx\n\nPrerequisites: Track A-1 (#1904) must be committed first.\n\nSTEP 1 -- Add field to SamplingSpecification:\n  File: src/aminx/run/specs.py, around line 476 (after \`dedup_specs\` field).\n  Add:\n    chain_mask_fixed: ArrayLike | None = None\n\n  \`ArrayLike\` is already imported in the file.\n\nSTEP 2 -- Wire in _sampling_helper.py:\n  File: src/aminx/host/_sampling_helper.py, lines 343-365.\n  Current chain_mask computation block:\n    if spec.fixed_mask is not None:\n        fixed_mask_np = _broadcast_per_structure(...)\n        chain_mask = 1.0 - fixed_mask_np\n        assert chain_mask.dtype == jnp.float32\n    else:\n        chain_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)\n\n  Prepend a new guard that fires FIRST:\n    if getattr(spec, "chain_mask_fixed", None) is not None:\n        chain_mask = _broadcast_per_structure(\n            spec.chain_mask_fixed, batch_size=batch_size, expected_len=seq_len,\n            dtype=jnp.float32, name="chain_mask_fixed",\n        )\n        assert chain_mask.dtype == jnp.float32\n    elif spec.fixed_mask is not None:\n        (existing logic unchanged)\n    else:\n        chain_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)\n\n  Priority order: chain_mask_fixed > fixed_mask complement > all-ones fallback.\n\nACCEPTANCE:\n1. \`SamplingSpecification(inputs=..., chain_mask_fixed=None)\` round-trips.\n2. When \`chain_mask_fixed\` is set, _sampling_helper uses it verbatim (not derived).\n3. When \`chain_mask_fixed=None\` and \`fixed_mask\` set, complement derivation unchanged.\n4. When both None, all-ones fallback unchanged.\n5. \`uv run pytest tests/ -q --ignore=tests/parity\` clean.\n\nCOMMIT:\n  git add src/aminx/run/specs.py src/aminx/host/_sampling_helper.py\n  git commit -m "feat(#1881): add chain_mask_fixed to SamplingSpecification; wire in _sampling_helper"\n\nWhen done, end with: verdict: done\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Review #1881: chain_mask_fixed field + _sampling_helper wiring.\n\nVERIFY 1: grep -n "chain_mask_fixed" src/aminx/run/specs.py -- appears in SamplingSpecification.\nVERIFY 2: grep -n "chain_mask_fixed" src/aminx/host/_sampling_helper.py -- appears before\n  the fixed_mask elif block (priority guard fires first).\nVERIFY 3: Fallback path (chain_mask_fixed=None, fixed_mask=None) produces all-ones float32.\nVERIFY 4: \`uv run pytest tests/ -q --ignore=tests/parity\` -- no regressions.\n\nPASS if all 4 satisfied. FAIL if chain_mask_fixed not wired or fallback broken.\n`,
    null,
    null
  );

// ===== TRACK B — Track B -- Verify _sync_run_spec double-fire guard (e0791e4, verify-only) =========================
const trackB = () =>
  track(
    "1906",
    "Track B -- Verify _sync_run_spec double-fire guard (e0791e4, verify-only)",
    `task_id: ${TASK_ID}. VERIFY ONLY -- this fix is already on main (commit e0791e4). No code changes needed.\n\nThe guard-flag \`_run_spec_synced: bool = field(init=False, default=False)\` is at\n\`src/aminx/run/specs.py:272\`. The guard check is in \`_sync_run_spec()\` at lines 274-278.\n\nACTION: Run the two verify commands. If both pass, the fix is confirmed.\n\n  grep -n "_run_spec_synced" src/aminx/run/specs.py\n  uv run pytest tests/run/ -q\n\nIf guard is present and tests pass: end with verdict: done\nIf double-fire is still observable: fix by ensuring the reset in each subclass\n__post_init__ happens BEFORE super().__post_init__() (not after).\n\nWhen done, end with: verdict: done\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Review #1906: _sync_run_spec double-fire guard -- verify only.\n\nVERIFY 1: specs.py line ~272 -- \`_run_spec_synced: bool = field(init=False, default=False)\` exists.\nVERIFY 2: \`_sync_run_spec()\` lines 274-278 -- first line is \`if self._run_spec_synced: return\`.\nVERIFY 3: Each subclass __post_init__ resets \`_run_spec_synced\` to False BEFORE super().__post_init__().\nVERIFY 4: \`uv run pytest tests/run/ -q\` -- all run-spec tests pass.\n\nPASS if guard is in place and tests are green.\n`,
    null,
    null
  );

// ---- orchestrate: writing chain (A1 -> A2 -> B, sequential) ----
log("260618_runspec-xtrax-run: writing chain (A1 -> A2 -> B, sequential)");
const a1 = await trackA1();
const a2 = await trackA2();
const b = await trackB();

return {
  task_id: TASK_ID,
  sprint_id: 1,
  verdicts: {
    "1904": a1,
    "1905": a2,
    "1906": b
  },
};
