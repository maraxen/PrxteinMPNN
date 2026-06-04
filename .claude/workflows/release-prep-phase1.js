/**
 * Release Preparedness Phase 1A + 1B Workflow
 *
 * Phase 1A: CI Modernization — CI-001..005 (serialized: 001→002→{003,004,005})
 * Phase 1B: Repo Cleanup — CLEAN-001..014 ({001-010,011,013}→014→012)
 *
 * Generated for task_id: 260604_release-prep-sprint-compose
 * Project: prxteinmpnn
 * Revised: oracle NEEDS_REVISION findings (B1 quote-mismatch, B2 missing CLEAN-012/014,
 *   R1 CI write-race, R2/R3 non-idempotent batch, R5 incomplete CLEAN-013 verification)
 */

export const meta = {
  name: 'release-prep-phase1',
  description: 'Release Prep Phase 1A (CI-001..005, serialized) + Phase 1B (CLEAN-001..014) — 19 spec items',
  phases: [
    { title: 'CI Modernization', detail: 'CI-001→002 sequential (shared files), then CI-003/004/005 parallel' },
    { title: 'Repo Cleanup', detail: 'CLEAN-001-010+011+013 parallel, then CLEAN-014 dep update, then CLEAN-012 INDEX' },
  ],
};

// ============================================================================
// HELPERS
// ============================================================================

function extractVerdict(agentOutput) {
  if (!agentOutput) return null;
  const lines = agentOutput.split('\n');
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (line.startsWith('verdict:')) {
      return line.substring('verdict:'.length).trim();
    }
  }
  return null;
}

const MAX_FIX_RETRIES = 2;

// ============================================================================
// CONSTANTS
// ============================================================================

const TASK_ID = '260604_release-prep-sprint-compose';
const PROJECT_ROOT = '/home/marielle/projects/prxteinmpnn';
const WF_DIR = `${PROJECT_ROOT}/.github/workflows`;

// ============================================================================
// PHASE 1A: CI MODERNIZATION
// CI-001 and CI-002 edit the same three files → must run sequentially.
// CI-003 (read-only audit), CI-004 (new ADR file), CI-005 (ci.yml only) run
// in parallel after 001+002 complete — no file conflicts.
// ============================================================================

async function phaseCI001() {
  const PHASE = 'CI Modernization';
  const item = 'CI-001: Bump Python 3.11 → 3.12 in GitHub Actions workflows';
  log(`[CI-001] Starting: ${item}`);

  const reconPrompt = `[role: recon] [phase: recon_phase] [task_id: ${TASK_ID}]

You are a recon agent for CI-001: Bump Python 3.11 → 3.12 in GitHub Actions workflow files.

Project root: ${PROJECT_ROOT}

Steps:
1. List all files in ${WF_DIR}/
2. Grep for python-version declarations (any value): grep -rn "python-version" ${WF_DIR}/
3. Report exact line numbers of '3.11' occurrences and note the quote style used per file (single or double quotes).
4. Check for any blockers.

End with 'verdict: advance' on its own line.`;

  const reconResult = await agent(reconPrompt, { label: 'recon:ci001', phase: PHASE, agentType: 'recon' });

  const reconVerdict = extractVerdict(reconResult);
  if (reconVerdict !== 'advance') {
    return { item, status: 'blocked', reason: `Recon: ${reconVerdict}` };
  }

  const fixerPrompt = `[role: fixer] [phase: fix_phase] [task_id: ${TASK_ID}]

Implement CI-001: Bump Python 3.11 → 3.12 in three workflow files.

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report before making any changes.

Files to edit:
  ${WF_DIR}/ci.yml
  ${WF_DIR}/parity-audit.yml
  ${WF_DIR}/parity.yml

IMPORTANT — quote style: workflow files may use single quotes ('3.11') or double quotes ("3.11") around the version string. Use the Edit tool to find and replace each occurrence by exact match; do not assume a single quote style. Replace every python-version field that declares version 3.11 with 3.12, preserving the surrounding quote style of each individual file.

After edits, verify:
  grep -rn "3\\.11" ${WF_DIR}/ — must return 0 matches (exit 1 is acceptable if no matches)

End with 'verdict: advance' on its own line.`;

  let fixResult = await agent(fixerPrompt, { label: 'fixer:ci001#0', phase: PHASE, agentType: 'fixer' });
  let fixVerdict = extractVerdict(fixResult);
  let fixRetries = 0;
  while (fixVerdict !== 'advance' && fixRetries < MAX_FIX_RETRIES) {
    fixRetries++;
    fixResult = await agent(fixerPrompt, { label: `fixer:ci001#${fixRetries}`, phase: PHASE, agentType: 'fixer' });
    fixVerdict = extractVerdict(fixResult);
  }

  if (fixVerdict !== 'advance') {
    return { item, status: 'failed', reason: `Fixer: ${fixVerdict}` };
  }

  return { item, status: 'passed' };
}

async function phaseCI002() {
  const PHASE = 'CI Modernization';
  const item = 'CI-002: Replace uv venv with uv run throughout workflows';
  log(`[CI-002] Starting: ${item} (runs after CI-001 — shared files)`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CI-002: Replace uv venv + source .venv pattern with uv run.

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report before making any changes.

Note: CI-001 has already been applied to these files (Python version bumped). The files are clean and committed.

Remove uv venv steps and source .venv/bin/activate lines. Add 'uv run' prefix to python/pytest invocations.
Files: ${WF_DIR}/ci.yml, ${WF_DIR}/parity-audit.yml, ${WF_DIR}/parity.yml

Verify: grep -c 'source .venv' ${WF_DIR}/*.yml — must return 0 for each file.

End with 'verdict: advance' on its own line.`;

  let implResult = await agent(implPrompt, { label: 'fixer:ci002#0', phase: PHASE, agentType: 'fixer' });
  let implVerdict = extractVerdict(implResult);
  let implRetries = 0;
  while (implVerdict !== 'advance' && implRetries < MAX_FIX_RETRIES) {
    implRetries++;
    implResult = await agent(implPrompt, { label: `fixer:ci002#${implRetries}`, phase: PHASE, agentType: 'fixer' });
    implVerdict = extractVerdict(implResult);
  }

  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

async function phaseCI003() {
  const PHASE = 'CI Modernization';
  const item = 'CI-003: Audit parity workflows for path correctness';
  log(`[CI-003] Starting: ${item} (read-only; runs after CI-001+002)`);

  const auditPrompt = `[role: auditor] [phase: review_phase] [task_id: ${TASK_ID}]

Audit CI-003: verify REFERENCE_PATH, checkout paths, inline Python scripts use absolute paths anchored to GITHUB_WORKSPACE.

Note: CI-001+002 have already been applied; files are committed. This is a read-only path audit.

Read ${WF_DIR}/parity.yml and ${WF_DIR}/parity-audit.yml in full.
Check every path reference and environment variable for correctness.
Report any paths that are relative or assume a specific cwd rather than anchoring to $GITHUB_WORKSPACE.

This item is defined as a manual walkthrough audit (Done When: manual walkthrough passes). If path issues are found, report them clearly as findings but do not attempt to fix them — they require human review.

End with 'verdict: advance' (no issues found) or 'verdict: blocked' (issues found, needs human review) on its own line.`;

  const auditResult = await agent(auditPrompt, { label: 'auditor:ci003', phase: PHASE, agentType: 'auditor' });
  const auditVerdict = extractVerdict(auditResult);
  return {
    item,
    status: (auditVerdict === 'advance' ? 'passed' : 'blocked'),
    reason: auditVerdict !== 'advance' ? `Audit found path issues: ${auditVerdict} — needs human review` : undefined,
  };
}

async function phaseCI004() {
  const PHASE = 'CI Modernization';
  const item = 'CI-004: Write ADR for deferring ty/ruff as CI gates';
  log(`[CI-004] Starting: ${item} (independent — writes new file only)`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CI-004: Write ADR deferring ty check and ruff check as CI gates.

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report before making any changes.

Steps:
1. Create directory if missing: ${PROJECT_ROOT}/.praxia/docs/adr/
2. Write ADR to: ${PROJECT_ROOT}/.praxia/docs/adr/260604_defer-ty-ruff-ci-gates.md
   Frontmatter: status: accepted, date: 260604
   Decision: ty check and ruff check are deferred from CI gates until code surface stabilizes post-release.
   Rationale: confidence in full code surface needed first; re-enable criteria: after README + CONTRIBUTING verified against src/, all public API examples execute.
   Consequences: quality checks run locally (uv run ty check, uv run ruff check) but not enforced in CI.
3. Update ${PROJECT_ROOT}/.praxia/docs/INDEX.md under ## ADRs: add entry for this ADR; remove any stub suffix for this item if present.

Verify: test -f ${PROJECT_ROOT}/.praxia/docs/adr/260604_defer-ty-ruff-ci-gates.md && echo ok

End with 'verdict: advance' on its own line.`;

  let implResult = await agent(implPrompt, { label: 'fixer:ci004#0', phase: PHASE, agentType: 'fixer' });
  let implVerdict = extractVerdict(implResult);
  let implRetries = 0;
  while (implVerdict !== 'advance' && implRetries < MAX_FIX_RETRIES) {
    implRetries++;
    implResult = await agent(implPrompt, { label: `fixer:ci004#${implRetries}`, phase: PHASE, agentType: 'fixer' });
    implVerdict = extractVerdict(implResult);
  }
  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

async function phaseCI005() {
  const PHASE = 'CI Modernization';
  const item = 'CI-005: Remove ty/ruff steps from quality-checks job';
  log(`[CI-005] Starting: ${item} (edits ci.yml only; runs after CI-001+002)`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CI-005: Remove ty check and ruff check steps from ci.yml quality-checks job.
This is a consequence of the CI-004 ADR (ty/ruff deferred from CI gates).

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report before making any changes.

Edit ${WF_DIR}/ci.yml: find the quality-checks job and remove the 'ty check' and 'ruff check' steps entirely (full step blocks, not just the run lines).

Verify:
  grep -n 'ty check' ${WF_DIR}/ci.yml — must return no output
  grep -n 'ruff check' ${WF_DIR}/ci.yml — must return no output

End with 'verdict: advance' on its own line.`;

  let implResult = await agent(implPrompt, { label: 'fixer:ci005#0', phase: PHASE, agentType: 'fixer' });
  let implVerdict = extractVerdict(implResult);
  let implRetries = 0;
  while (implVerdict !== 'advance' && implRetries < MAX_FIX_RETRIES) {
    implRetries++;
    implResult = await agent(implPrompt, { label: `fixer:ci005#${implRetries}`, phase: PHASE, agentType: 'fixer' });
    implVerdict = extractVerdict(implResult);
  }
  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

// ============================================================================
// PHASE 1B: REPO CLEANUP
// CLEAN-001..010, CLEAN-011, CLEAN-013 run in parallel (no shared file conflicts).
// CLEAN-014 runs after CLEAN-013 (both edit pyproject.toml).
// CLEAN-012 runs last (INDEX.md holistic update after all archives+ADRs exist).
// ============================================================================

async function phaseCLEAN001to010() {
  const PHASE = 'Repo Cleanup';
  const item = 'CLEAN-001..010: Batch delete and archive legacy files';
  log(`[CLEAN-001..010] Starting: ${item}`);

  const implPrompt = `[role: fixer] [phase: fix_phase] [task_id: ${TASK_ID}]

Implement CLEAN-001..010: Delete and archive 10 legacy items from the prxteinmpnn repo.

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report before making any changes.

Project root: ${PROJECT_ROOT}

IMPORTANT — all git rm commands must use --ignore-unmatch (idempotent; file may already be gone on retry).
IMPORTANT — all archive creation must guard with: test -e <source> before running tar.
Use uv run python for any Python scripts.

Execute each item in order:

CLEAN-001:
  git rm --ignore-unmatch ${PROJECT_ROOT}/AGENTS.md

CLEAN-002: Archive .agent/ directory
  test -e ${PROJECT_ROOT}/.agent && tar -C ${PROJECT_ROOT} -caf ${PROJECT_ROOT}/.praxia/docs/archive/260528_agent-scaffolding.tar.zst .agent/
  Write manifest: ${PROJECT_ROOT}/.praxia/docs/archive/260528_agent-scaffolding.md
    frontmatter: archive: 260528_agent-scaffolding.tar.zst, created: 260528, source: .agent/
    contents: list files from .agent/ if it exists, else mark as "archived in prior run"
  git rm --ignore-unmatch -r ${PROJECT_ROOT}/.agent/
  Add '.agent/' to ${PROJECT_ROOT}/.gitignore if not already present

CLEAN-003: Archive .agents/ directory
  test -e ${PROJECT_ROOT}/.agents && tar -C ${PROJECT_ROOT} -caf ${PROJECT_ROOT}/.praxia/docs/archive/260528_agents-sprint-artifacts.tar.zst .agents/
  Write manifest: ${PROJECT_ROOT}/.praxia/docs/archive/260528_agents-sprint-artifacts.md
    frontmatter: archive: 260528_agents-sprint-artifacts.tar.zst, created: 260528, source: .agents/
    contents: list files from .agents/ if it exists, else mark as "archived in prior run"
  git rm --ignore-unmatch -r ${PROJECT_ROOT}/.agents/

CLEAN-004: Archive PROXIDE_LIMITATIONS.md + PROXIDE_UPGRADE_SPEC.md
  test -e ${PROJECT_ROOT}/PROXIDE_LIMITATIONS.md && tar -C ${PROJECT_ROOT} -caf ${PROJECT_ROOT}/.praxia/docs/archive/260124_proxide-integration-docs.tar.zst PROXIDE_LIMITATIONS.md PROXIDE_UPGRADE_SPEC.md
  Write manifest: ${PROJECT_ROOT}/.praxia/docs/archive/260124_proxide-integration-docs.md
    frontmatter: archive: 260124_proxide-integration-docs.tar.zst, created: 260124, source: PROXIDE_LIMITATIONS.md + PROXIDE_UPGRADE_SPEC.md
  git rm --ignore-unmatch ${PROJECT_ROOT}/PROXIDE_LIMITATIONS.md ${PROJECT_ROOT}/PROXIDE_UPGRADE_SPEC.md

CLEAN-005: Archive legacy parity docs
  Source files (archive what exists; skip missing with a note in manifest):
    ${PROJECT_ROOT}/docs/FINAL_VALIDATION_RESULTS.md
    ${PROJECT_ROOT}/docs/parity_audit.md
    ${PROJECT_ROOT}/docs/parity/parity_report.md
    ${PROJECT_ROOT}/docs/archive/parity_audit_legacy.md
    ${PROJECT_ROOT}/docs/archive/parity_validation_results_legacy.md
  Create archive: ${PROJECT_ROOT}/.praxia/docs/archive/260410_parity-validation-legacy.tar.zst (add only files that exist)
  Write manifest: ${PROJECT_ROOT}/.praxia/docs/archive/260410_parity-validation-legacy.md
    frontmatter: archive: 260410_parity-validation-legacy.tar.zst, created: 260410, source: docs/ parity legacy files
    contents: list which files were archived
  git rm --ignore-unmatch ${PROJECT_ROOT}/docs/FINAL_VALIDATION_RESULTS.md ${PROJECT_ROOT}/docs/parity_audit.md ${PROJECT_ROOT}/docs/parity/parity_report.md ${PROJECT_ROOT}/docs/archive/parity_audit_legacy.md ${PROJECT_ROOT}/docs/archive/parity_validation_results_legacy.md

CLEAN-006: Archive docs/superpowers/
  test -e ${PROJECT_ROOT}/docs/superpowers && tar -C ${PROJECT_ROOT}/docs -caf ${PROJECT_ROOT}/.praxia/docs/archive/260512_docs-superpowers-status.tar.zst superpowers/
  Write manifest: ${PROJECT_ROOT}/.praxia/docs/archive/260512_docs-superpowers-status.md
    frontmatter: archive: 260512_docs-superpowers-status.tar.zst, created: 260512, source: docs/superpowers/
    contents: list files if directory exists, else "archived in prior run"
  git rm --ignore-unmatch -r ${PROJECT_ROOT}/docs/superpowers/

CLEAN-007:
  git rm --ignore-unmatch ${PROJECT_ROOT}/.readthedocs.yaml

CLEAN-008:
  Add 'docs/_build/' to ${PROJECT_ROOT}/.gitignore if not already present
  git rm --ignore-unmatch --cached -r ${PROJECT_ROOT}/docs/_build/

CLEAN-009:
  git rm --ignore-unmatch ${PROJECT_ROOT}/examples/test_nb.ipynb

CLEAN-010:
  git rm --ignore-unmatch ${PROJECT_ROOT}/colab_training_test.ipynb

After all operations, verify:
  test ! -f ${PROJECT_ROOT}/AGENTS.md && echo "CLEAN-001 ok"
  test ! -f ${PROJECT_ROOT}/PROXIDE_LIMITATIONS.md && echo "CLEAN-004 ok"
  test ! -f ${PROJECT_ROOT}/.readthedocs.yaml && echo "CLEAN-007 ok"
  test ! -f ${PROJECT_ROOT}/examples/test_nb.ipynb && echo "CLEAN-009 ok"
  test ! -f ${PROJECT_ROOT}/colab_training_test.ipynb && echo "CLEAN-010 ok"
  test -f ${PROJECT_ROOT}/.praxia/docs/archive/260124_proxide-integration-docs.md && echo "manifests ok"

End with 'verdict: advance' on its own line.`;

  let implResult = await agent(implPrompt, { label: 'fixer:clean001-010#0', phase: PHASE, agentType: 'fixer' });
  let implVerdict = extractVerdict(implResult);
  let implRetries = 0;
  while (implVerdict !== 'advance' && implRetries < MAX_FIX_RETRIES) {
    implRetries++;
    implResult = await agent(implPrompt, { label: `fixer:clean001-010#${implRetries}`, phase: PHASE, agentType: 'fixer' });
    implVerdict = extractVerdict(implResult);
  }

  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

async function phaseCLEAN011() {
  const PHASE = 'Repo Cleanup';
  const item = 'CLEAN-011: Write ADR for parity docs as CI-autogenerated';
  log(`[CLEAN-011] Starting: ${item}`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CLEAN-011: Write ADR for parity docs as CI-autogenerated artifacts.

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report before making any changes.

Steps:
1. Create directory if missing: ${PROJECT_ROOT}/.praxia/docs/adr/
2. Write ADR to: ${PROJECT_ROOT}/.praxia/docs/adr/260604_parity-docs-as-ci-artifacts.md
   Frontmatter: status: accepted, date: 260604
   Decision: parity validation docs are auto-generated by CI parity jobs, not hand-maintained.
   Rationale: manual docs drift from actual test results; CI-generated artifacts are authoritative. Hand-written versions were diverging from the parity CI output.
   Consequences: docs/parity/ manual files are archived; parity CI jobs write reports directly to artifacts.
3. Update ${PROJECT_ROOT}/.praxia/docs/INDEX.md under ## ADRs: add entry for this ADR; remove stub suffix if present.

Verify: test -f ${PROJECT_ROOT}/.praxia/docs/adr/260604_parity-docs-as-ci-artifacts.md && echo ok

End with 'verdict: advance' on its own line.`;

  let implResult = await agent(implPrompt, { label: 'fixer:clean011#0', phase: PHASE, agentType: 'fixer' });
  let implVerdict = extractVerdict(implResult);
  let implRetries = 0;
  while (implVerdict !== 'advance' && implRetries < MAX_FIX_RETRIES) {
    implRetries++;
    implResult = await agent(implPrompt, { label: `fixer:clean011#${implRetries}`, phase: PHASE, agentType: 'fixer' });
    implVerdict = extractVerdict(implResult);
  }
  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

async function phaseCLEAN013() {
  const PHASE = 'Repo Cleanup';
  const item = 'CLEAN-013: Dependency audit — migrate flax, pin deps, bump requires-python';
  log(`[CLEAN-013] Starting: ${item}`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CLEAN-013: Dependency audit, flax → eqx.Module migration, and pyproject.toml cleanup.

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report before making any changes.

Project root: ${PROJECT_ROOT}

Step 1 — Migrate flax.struct.dataclass → eqx.Module:

  FILE A: ${PROJECT_ROOT}/src/prxteinmpnn/training/metrics.py
  Current state (read to verify before editing):
    line 11: from flax.struct import dataclass
    line 14: @dataclass                     ← decorator on TrainingMetrics
    line 37: @dataclass                     ← decorator on EvaluationMetrics
  Changes:
    1. Remove line 11 (the flax import)
    2. Add 'import equinox as eqx' after the jax imports
    3. Remove the @dataclass decorator at line 14; change 'class TrainingMetrics:' → 'class TrainingMetrics(eqx.Module):'
    4. Remove the @dataclass decorator at line 37; change 'class EvaluationMetrics:' → 'class EvaluationMetrics(eqx.Module):'
    5. Preserve all field declarations and methods unchanged (including grad_norm default = None)

  FILE B: ${PROJECT_ROOT}/src/prxteinmpnn/utils/data_structures.py
  Current state (read to verify before editing):
    line 8:  from dataclasses import dataclass as dc   ← STDLIB — DO NOT TOUCH
    line 11: from flax.struct import dataclass          ← flax — remove this line only
    line 33: @dc                                        ← stdlib decorator on TrajectoryStaticFeatures — DO NOT TOUCH
    line 46: @dataclass                                 ← flax decorator on EstatInfo — change this class only
  Changes:
    1. Remove ONLY line 11 (the flax import). Keep line 8 (stdlib dc alias) intact.
    2. Add 'import equinox as eqx' after the 'import numpy as np' line
    3. Remove the @dataclass decorator at line 46; change 'class EstatInfo:' → 'class EstatInfo(eqx.Module):'
    4. Do NOT modify TrajectoryStaticFeatures or its @dc decorator in any way.
    5. Preserve all field declarations and methods unchanged.

Step 2 — Edit ${PROJECT_ROOT}/pyproject.toml:
  - Remove "flax" from [project.dependencies]
  - Remove pytest-asyncio, trio, anyio from [project.optional-dependencies.tests] and any dev group
    (zero async tests in the codebase — verify with: grep -rn 'async def test_' ${PROJECT_ROOT}/tests/ — should return 0)
  - Change requires-python to ">=3.12"
  - Add lower-bound pins to unversioned runtime deps:
      jax>=0.4.35, numpy>=1.26, optax>=0.2, jaxtyping>=0.2.30,
      joblib>=1.4, psutil>=5.9, h5py>=3.11, grain>=0.2

Step 3 — Run: cd ${PROJECT_ROOT} && uv lock

Verify (all must pass):
  grep -rn 'from flax' ${PROJECT_ROOT}/src/ — must return nothing
  grep -n 'requires-python' ${PROJECT_ROOT}/pyproject.toml — must show >=3.12
  uv run python -c 'from prxteinmpnn.training.metrics import TrainingMetrics; print("TrainingMetrics ok")'
  uv run python -c 'from prxteinmpnn.training.metrics import EvaluationMetrics; print("EvaluationMetrics ok")'
  uv run python -c 'from prxteinmpnn.utils.data_structures import EstatInfo; print("EstatInfo ok")'
  uv run python -c 'from prxteinmpnn.utils.data_structures import TrajectoryStaticFeatures; print("TrajectoryStaticFeatures ok")'
  uv run pytest -x --tb=short -q -k "metrics or data_struct" ${PROJECT_ROOT}/tests/ 2>&1 | tail -5

End with 'verdict: advance' on its own line.`;

  let implResult = await agent(implPrompt, { label: 'fixer:clean013#0', phase: PHASE, agentType: 'fixer' });
  let implVerdict = extractVerdict(implResult);
  let implRetries = 0;
  while (implVerdict !== 'advance' && implRetries < MAX_FIX_RETRIES) {
    implRetries++;
    implResult = await agent(implPrompt, { label: `fixer:clean013#${implRetries}`, phase: PHASE, agentType: 'fixer' });
    implVerdict = extractVerdict(implResult);
  }

  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

async function phaseCLEAN014() {
  const PHASE = 'Repo Cleanup';
  const item = 'CLEAN-014: Dependency update audit — resolve torch inconsistency, evaluate upgrades, regenerate uv.lock';
  log(`[CLEAN-014] Starting: ${item} (runs after CLEAN-013 — both edit pyproject.toml)`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CLEAN-014: Dependency update audit.

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report before making any changes.

Note: CLEAN-013 has already been applied. pyproject.toml now has requires-python>=3.12 and lower-bound pins. This step resolves version inconsistencies and evaluates upgrades.

Project root: ${PROJECT_ROOT}

Step 1 — Read ${PROJECT_ROOT}/pyproject.toml in full. Identify:
  a) The torch version inconsistency: benchmark extra uses torch>=2.12.0, but dev group uses torch>=2.9.1.
     Resolution: align both to torch>=2.12.0 (the more conservative floor).
  b) Any other version inconsistencies between [project.optional-dependencies] extras and [dependency-groups].

Step 2 — Edit pyproject.toml:
  - Set torch>=2.12.0 in BOTH locations (benchmark extra and dev group) — use the Edit tool to find each exact occurrence.
  - Fix any other inconsistencies found in Step 1.

Step 3 — Evaluate upgrades for focus packages:
  Run: cd ${PROJECT_ROOT} && uv pip index versions jax 2>&1 | tail -3
  Run: cd ${PROJECT_ROOT} && uv pip index versions equinox 2>&1 | tail -3
  Run: cd ${PROJECT_ROOT} && uv pip index versions optax 2>&1 | tail -3
  Run: cd ${PROJECT_ROOT} && uv pip index versions grain 2>&1 | tail -3
  Report the latest versions found. If a latest version is meaningfully newer than the lower-bound pin
  already in pyproject.toml (set by CLEAN-013), update the lower-bound pin to the newer version.
  Do not downgrade any existing pins. Do not change upper bounds.

Step 4 — Regenerate lock file:
  cd ${PROJECT_ROOT} && uv lock

Verify:
  grep -n 'torch' ${PROJECT_ROOT}/pyproject.toml — both occurrences must show >=2.12.0
  uv lock output exits 0 (no resolver errors)
  uv run python -c 'import jax; print(jax.__version__)' — must print a version

End with 'verdict: advance' on its own line.`;

  let implResult = await agent(implPrompt, { label: 'fixer:clean014#0', phase: PHASE, agentType: 'fixer' });
  let implVerdict = extractVerdict(implResult);
  let implRetries = 0;
  while (implVerdict !== 'advance' && implRetries < MAX_FIX_RETRIES) {
    implRetries++;
    implResult = await agent(implPrompt, { label: `fixer:clean014#${implRetries}`, phase: PHASE, agentType: 'fixer' });
    implVerdict = extractVerdict(implResult);
  }

  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

async function phaseCLEAN012() {
  const PHASE = 'Repo Cleanup';
  const item = 'CLEAN-012: Update .praxia/docs/INDEX.md — all archive entries and ADR section current';
  log(`[CLEAN-012] Starting: ${item} (runs last — after all archives and ADRs exist)`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CLEAN-012: Update .praxia/docs/INDEX.md holistically.

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report before making any changes.

This step runs after all archives and ADRs from Phase 1B have been created by earlier parallel tasks.

Steps:
1. Read ${PROJECT_ROOT}/.praxia/docs/INDEX.md in full.

2. Under ## Archive section: ensure entries exist for all six new archives created in CLEAN-001..010:
   - [260528_agent-scaffolding](archive/260528_agent-scaffolding.md) — .agent/ scaffolding, archived 260528
   - [260528_agents-sprint-artifacts](archive/260528_agents-sprint-artifacts.md) — .agents/ sprint outputs, archived 260528
   - [260124_proxide-integration-docs](archive/260124_proxide-integration-docs.md) — Proxide integration docs, archived 260124
   - [260410_parity-validation-legacy](archive/260410_parity-validation-legacy.md) — Legacy parity validation docs, archived 260410
   - [260512_docs-superpowers-status](archive/260512_docs-superpowers-status.md) — docs/superpowers/ skill outputs, archived 260512
   Add any that are missing; do not duplicate existing entries.

3. Under ## ADRs section: ensure all three Phase 1 ADRs appear:
   - [260604_defer-ty-ruff-ci-gates](adr/260604_defer-ty-ruff-ci-gates.md) — Defer ty/ruff as CI gates
   - [260604_parity-docs-as-ci-artifacts](adr/260604_parity-docs-as-ci-artifacts.md) — Parity docs as CI-autogenerated
   Remove any stub suffix (e.g., '(stub — to be written, CI-004)') from existing entries.
   Add any ADRs that are missing.

4. Verify the INDEX.md exists on disk:
   test -f ${PROJECT_ROOT}/.praxia/docs/INDEX.md && echo ok

End with 'verdict: advance' on its own line.`;

  let implResult = await agent(implPrompt, { label: 'fixer:clean012#0', phase: PHASE, agentType: 'fixer' });
  let implVerdict = extractVerdict(implResult);
  let implRetries = 0;
  while (implVerdict !== 'advance' && implRetries < MAX_FIX_RETRIES) {
    implRetries++;
    implResult = await agent(implPrompt, { label: `fixer:clean012#${implRetries}`, phase: PHASE, agentType: 'fixer' });
    implVerdict = extractVerdict(implResult);
  }

  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

log('Starting Release Prep Phase 1A + 1B');
log('CI track: CI-001 → CI-002 → parallel(CI-003, CI-004, CI-005)');
log('CLEAN track: parallel(CLEAN-001..010, CLEAN-011, CLEAN-013) → CLEAN-014 → CLEAN-012');

// ── Phase 1A: CI Modernization (serialized where needed) ──────────────────

phase('CI Modernization');

// CI-001 and CI-002 edit the same three workflow files — must run sequentially.
const ci001Result = await phaseCI001();
log(`[CI-001] ${ci001Result.status}${ci001Result.reason ? ': ' + ci001Result.reason : ''}`);

const ci002Result = await phaseCI002();
log(`[CI-002] ${ci002Result.status}${ci002Result.reason ? ': ' + ci002Result.reason : ''}`);

// CI-003 (read-only), CI-004 (new ADR file), CI-005 (ci.yml only) — no conflicts.
const [ci003Result, ci004Result, ci005Result] = await parallel([
  () => phaseCI003(),
  () => phaseCI004(),
  () => phaseCI005(),
]);

const ciResults = [ci001Result, ci002Result, ci003Result, ci004Result, ci005Result].filter(Boolean);

// ── Phase 1B: Repo Cleanup ────────────────────────────────────────────────

phase('Repo Cleanup');

// CLEAN-001..010 (archives + git rm), CLEAN-011 (ADR), CLEAN-013 (flax + dep pins) — no shared file conflicts.
const [clean01010Result, clean011Result, clean013Result] = await parallel([
  () => phaseCLEAN001to010(),
  () => phaseCLEAN011(),
  () => phaseCLEAN013(),
]);

// CLEAN-014 reads the pyproject.toml modified by CLEAN-013 — must run after.
const clean014Result = await phaseCLEAN014();

// CLEAN-012 updates INDEX.md after all archives and ADRs from this phase exist.
const clean012Result = await phaseCLEAN012();

const cleanResults = [clean01010Result, clean011Result, clean013Result, clean014Result, clean012Result].filter(Boolean);

// ── Summary ───────────────────────────────────────────────────────────────

const allResults = [...ciResults, ...cleanResults];
const passedCount = allResults.filter(r => r.status === 'passed').length;
const failedCount = allResults.filter(r => r.status === 'failed').length;
const blockedCount = allResults.filter(r => r.status === 'blocked').length;

return {
  task_id: TASK_ID,
  phase_1a_ci_modernization: ciResults,
  phase_1b_repo_cleanup: cleanResults,
  summary: {
    total: allResults.length,
    passed: passedCount,
    failed: failedCount,
    blocked: blockedCount,
    success: failedCount === 0 && blockedCount === 0,
    note: 'CI-003 blocked = parity path issues found; needs human review. CLEAN-014 dep upgrades are advisory.',
  },
};
