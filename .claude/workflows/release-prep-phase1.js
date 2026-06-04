/**
 * Release Preparedness Phase 1A + 1B Workflow
 *
 * Orchestrates 16 P1 items across two parallel tracks:
 * - Phase 1A: CI Modernization (CI-001..005)
 * - Phase 1B: Repo Cleanup + Audit (CLEAN-001..013)
 *
 * Generated for task_id: 260604_release-prep-sprint-compose
 * Project: prxteinmpnn
 */

export const meta = {
  name: 'release-prep-phase1',
  description: 'Release Preparedness Phase 1A (CI) + Phase 1B (Cleanup) — 16 P1 items',
  phases: [
    { title: 'CI Modernization', detail: 'CI-001..005: Python bump, uv run, parity audit, ty/ruff ADR, CI cleanup' },
    { title: 'Repo Cleanup', detail: 'CLEAN-001..013: archive 10 legacy items, parity ADR, dep audit + flax migration' },
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

// ============================================================================
// PHASE 1A: CI MODERNIZATION (Parallel)
// ============================================================================

async function phaseCI001() {
  const PHASE = 'CI Modernization';
  const item = 'CI-001: Bump Python 3.11 → 3.12 in GitHub Actions workflows';
  log(`[CI-001] Starting: ${item}`);

  const reconPrompt = `[role: recon] [phase: recon_phase] [task_id: ${TASK_ID}]

You are a recon agent for CI-001: Bump Python 3.11 → 3.12 in GitHub Actions workflow files.

Project root: ${PROJECT_ROOT}
Epic spec: ${PROJECT_ROOT}/.praxia/docs/specs/260604_release-preparedness-epic-for-prxteinmpn.md

Your goal is to locate every Python version declaration and confirm no blockers exist.

Steps:
1. Read .github/workflows/ci.yml, parity-audit.yml, parity.yml
2. Grep for python-version declarations
3. Report exact line numbers of '3.11' occurrences
4. Check for any blockers

End with 'verdict: advance' on its own line.`;

  const reconResult = await agent(reconPrompt, { label: 'recon:ci001', phase: PHASE, agentType: 'recon' });

  const reconVerdict = extractVerdict(reconResult);
  if (reconVerdict !== 'advance') {
    return { item, status: 'blocked', reason: `Recon: ${reconVerdict}` };
  }

  const fixerPrompt = `[role: fixer] [phase: fix_phase] [task_id: ${TASK_ID}]

Implement CI-001: Bump Python 3.11 → 3.12 in three workflow files.

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report before making any changes.

Edit each file and replace all 'python-version: '3.11'' with '3.12' (preserve quote style).
Files: ${PROJECT_ROOT}/.github/workflows/ci.yml, parity-audit.yml, parity.yml

After edits, verify: grep -rn "3\\.11" ${PROJECT_ROOT}/.github/workflows/ returns 0

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
  log(`[CI-002] Starting: ${item}`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CI-002: Replace uv venv + source .venv pattern with uv run.

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report before making any changes.

Remove uv venv steps and source .venv/bin/activate lines. Add 'uv run' prefix to python/pytest invocations.
Files: ${PROJECT_ROOT}/.github/workflows/ci.yml, parity-audit.yml, parity.yml

Verify: grep -c 'source .venv' ${PROJECT_ROOT}/.github/workflows/*.yml returns 0

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
  log(`[CI-003] Starting: ${item}`);

  const auditPrompt = `[role: auditor] [phase: review_phase] [task_id: ${TASK_ID}]

Audit CI-003: verify REFERENCE_PATH, checkout paths, inline scripts use absolute paths.

Read ${PROJECT_ROOT}/.github/workflows/parity.yml and parity-audit.yml in full.
Check every path reference and environment variable for correctness.
Report any paths that are relative or assume a specific cwd.

End with 'verdict: advance' on its own line.`;

  const auditResult = await agent(auditPrompt, { label: 'auditor:ci003', phase: PHASE, agentType: 'auditor' });
  const auditVerdict = extractVerdict(auditResult);
  return { item, status: (auditVerdict === 'advance' ? 'passed' : 'blocked'), reason: auditVerdict !== 'advance' ? `Audit: ${auditVerdict}` : undefined };
}

async function phaseCI004() {
  const PHASE = 'CI Modernization';
  const item = 'CI-004: Write ADR for deferring ty/ruff as CI gates';
  log(`[CI-004] Starting: ${item}`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CI-004: Write ADR deferring ty check and ruff check as CI gates.

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report before making any changes.

Steps:
1. Create directory if missing: ${PROJECT_ROOT}/.praxia/docs/adr/
2. Write ADR to: ${PROJECT_ROOT}/.praxia/docs/adr/260604_defer-ty-ruff-ci-gates.md
   Content: frontmatter (status: accepted, date: 260604), rationale (code surface not yet stable for strict type gating), decision (ty check and ruff check are deferred from CI until code surface stabilizes post-release).
3. Update ${PROJECT_ROOT}/.praxia/docs/INDEX.md: find the line for this ADR stub and remove the '(stub — to be written, CI-004)' suffix.

Verify: test -f ${PROJECT_ROOT}/.praxia/docs/adr/260604_defer-ty-ruff-ci-gates.md && echo ok

End with 'verdict: advance' on its own line.`;

  let implResult = await agent(implPrompt, { label: 'fixer:ci004#0', phase: PHASE, agentType: 'fixer' });
  let implVerdict = extractVerdict(implResult);
  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

async function phaseCI005() {
  const PHASE = 'CI Modernization';
  const item = 'CI-005: Remove ty/ruff steps from quality-checks job';
  log(`[CI-005] Starting: ${item}`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CI-005: Remove ty check and ruff check steps from ci.yml quality-checks job.
This is a consequence of the CI-004 ADR (ty/ruff deferred from CI gates).

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report before making any changes.

Edit ${PROJECT_ROOT}/.github/workflows/ci.yml: find the quality-checks job and remove the 'ty check' and 'ruff check' steps entirely.

Verify:
- grep -n 'ty check' ${PROJECT_ROOT}/.github/workflows/ci.yml returns no output (exit 1 is fine)
- grep -n 'ruff check' ${PROJECT_ROOT}/.github/workflows/ci.yml returns no output

End with 'verdict: advance' on its own line.`;

  let implResult = await agent(implPrompt, { label: 'fixer:ci005#0', phase: PHASE, agentType: 'fixer' });
  let implVerdict = extractVerdict(implResult);
  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

// ============================================================================
// PHASE 1B: REPO CLEANUP (Parallel)
// ============================================================================

async function phaseCLEANQuick() {
  const PHASE = 'Repo Cleanup';
  const item = 'CLEAN-001..010: Batch delete and archive legacy files';
  log(`[CLEAN-001..010] Starting: ${item}`);

  const implPrompt = `[role: fixer] [phase: fix_phase] [task_id: ${TASK_ID}]

Implement CLEAN-001..010: Delete and archive 10 legacy items from the prxteinmpnn repo.

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report before making any changes.

Project root: ${PROJECT_ROOT}

For each item below, execute the specified action. Use uv run python for any Python scripts.

CLEAN-001: git rm AGENTS.md
CLEAN-002: Create tar.zst archive of .agent/ → ${PROJECT_ROOT}/.praxia/docs/archive/260528_agent-scaffolding.tar.zst
  Write manifest: ${PROJECT_ROOT}/.praxia/docs/archive/260528_agent-scaffolding.md
  Then: git rm -r .agent/ && echo '.agent/' >> .gitignore
CLEAN-003: Create tar.zst archive of .agents/ → ${PROJECT_ROOT}/.praxia/docs/archive/260528_agents-sprint-artifacts.tar.zst
  Write manifest, then: git rm -r .agents/
CLEAN-004: Archive PROXIDE_LIMITATIONS.md + PROXIDE_UPGRADE_SPEC.md → ${PROJECT_ROOT}/.praxia/docs/archive/260528_proxide-docs.tar.zst
  Write manifest, then: git rm PROXIDE_LIMITATIONS.md PROXIDE_UPGRADE_SPEC.md
CLEAN-005: Archive hand-written parity docs (docs/parity_audit_legacy.md and similar) → ${PROJECT_ROOT}/.praxia/docs/archive/260528_parity-docs-handwritten.tar.zst
  Write manifest, then: git rm the archived files
CLEAN-006: Archive docs/superpowers/ → ${PROJECT_ROOT}/.praxia/docs/archive/260528_docs-superpowers.tar.zst
  Write manifest, then: git rm -r docs/superpowers/
CLEAN-007: git rm .readthedocs.yaml (the Python 3.11 stale one — keep .readthedocs.yml)
CLEAN-008: Add docs/_build/ to .gitignore if not present; git rm --cached -r docs/_build/ if tracked
CLEAN-009: git rm examples/test_nb.ipynb
CLEAN-010: git rm colab_training_test.ipynb

After all operations:
Update ${PROJECT_ROOT}/.praxia/docs/INDEX.md: add entries under ## Archive for the 6 new archives.

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
   Content: frontmatter (status: accepted, date: 260604), decision (parity validation docs are auto-generated by CI parity jobs, not hand-maintained), rationale (manual docs drift from actual test results; CI-generated artifacts are authoritative).
3. Update ${PROJECT_ROOT}/.praxia/docs/INDEX.md: find the line for this ADR stub and remove the stub suffix.

Verify: test -f ${PROJECT_ROOT}/.praxia/docs/adr/260604_parity-docs-as-ci-artifacts.md && echo ok

End with 'verdict: advance' on its own line.`;

  let implResult = await agent(implPrompt, { label: 'fixer:clean011#0', phase: PHASE, agentType: 'fixer' });
  let implVerdict = extractVerdict(implResult);
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
  Files confirmed to use flax.struct.dataclass:
  - ${PROJECT_ROOT}/src/prxteinmpnn/training/metrics.py (TrainingMetrics, EvaluationMetrics at lines 11, 14, 37)
  - ${PROJECT_ROOT}/src/prxteinmpnn/utils/data_structures.py (EstatInfo at lines 11, 46)

  For each class: replace @flax.struct.dataclass with class <Name>(eqx.Module):
  Convert field declarations to standard Python class variables with type annotations.
  Add import equinox as eqx; remove import flax.

Step 2 — Edit ${PROJECT_ROOT}/pyproject.toml:
  - Remove "flax" from [project.dependencies]
  - Remove pytest-asyncio, trio, anyio from [project.optional-dependencies.tests] and dev
    (confirmed zero async tests in the codebase)
  - Change requires-python to ">=3.12"
  - Add lower-bound pins: jax>=0.4.35, numpy>=1.26, optax>=0.2, jaxtyping>=0.2.30,
    joblib>=1.4, psutil>=5.9, h5py>=3.11, grain>=0.2

Step 3 — Run: cd ${PROJECT_ROOT} && uv lock

Verify:
  grep -rn 'from flax' ${PROJECT_ROOT}/src/ — should return nothing
  grep -n 'requires-python' ${PROJECT_ROOT}/pyproject.toml — should show >=3.12
  uv run python -c 'from prxteinmpnn.training.metrics import TrainingMetrics; print("ok")' — should print ok

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

// ============================================================================
// MAIN EXECUTION
// ============================================================================

log('Starting Release Prep Phase 1A + 1B — CI Modernization and Repo Cleanup running in parallel');

phase('CI Modernization');
const ciResults = await parallel([
  () => phaseCI001(),
  () => phaseCI002(),
  () => phaseCI003(),
  () => phaseCI004(),
  () => phaseCI005(),
]);

phase('Repo Cleanup');
const cleanResults = await parallel([
  () => phaseCLEANQuick(),
  () => phaseCLEAN011(),
  () => phaseCLEAN013(),
]);

const allResults = [...ciResults.filter(Boolean), ...cleanResults.filter(Boolean)];
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
    success: passedCount === allResults.length,
  },
};
