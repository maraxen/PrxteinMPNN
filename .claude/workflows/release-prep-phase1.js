/**
 * Release Preparedness Phase 1A + 1B Workflow
 *
 * Orchestrates 16 P1 items across two parallel tracks:
 * - Phase 1A: CI Modernization (CI-001..005)
 * - Phase 1B: Repo Cleanup + Audit (CLEAN-001..014)
 *
 * Generated for task_id: 260604_release-prep-sprint-compose
 * Project: prxteinmpnn
 */

export const meta = {
  name: 'release-prep-phase1',
  description: 'Release Preparedness Phase 1A (CI) + Phase 1B (Cleanup) — 16 P1 items',
  phases: {
    ci_modernization: 'CI-001 through CI-005: Python bump, uv run pattern, parity audit, ADR for ty/ruff deferral',
    repo_cleanup: 'CLEAN-001 through CLEAN-013: Archival, dependency audit, ADR for parity docs'
  }
};

// ============================================================================
// HELPERS
// ============================================================================

function extractVerdict(agentOutput) {
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
  const item = 'CI-001: Bump Python 3.11 → 3.12 in GitHub Actions workflows';
  console.log(`\n[CI-001] Starting: ${item}`);

  // Recon phase
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

  const reconResult = await agent({
    role: 'recon',
    phase: 'recon_phase',
    task_id: TASK_ID,
    prompt: reconPrompt
  });

  const reconVerdict = extractVerdict(reconResult);
  if (reconVerdict !== 'advance') {
    return { item, status: 'blocked', reason: `Recon: ${reconVerdict}` };
  }

  // Fixer phase
  const fixerPrompt = `[role: fixer] [phase: fix_phase] [task_id: ${TASK_ID}]

Implement CI-001: Bump Python 3.11 → 3.12 in three workflow files.

Edit each file and replace all 'python-version: '3.11'' with '3.12' (preserve quote style).

After edits, verify: grep -rn "3\\.11" ${PROJECT_ROOT}/.github/workflows/ returns 0

End with 'verdict: advance' on its own line.`;

  let fixResult = await agent({
    role: 'fixer',
    phase: 'fix_phase',
    task_id: TASK_ID,
    prompt: fixerPrompt
  });

  let fixVerdict = extractVerdict(fixResult);
  let fixRetries = 0;
  while (fixVerdict !== 'advance' && fixRetries < MAX_FIX_RETRIES) {
    fixRetries++;
    fixResult = await agent({
      role: 'fixer',
      phase: 'fix_phase',
      task_id: TASK_ID,
      attempt: fixRetries + 1,
      prompt: fixerPrompt
    });
    fixVerdict = extractVerdict(fixResult);
  }

  if (fixVerdict !== 'advance') {
    return { item, status: 'failed', reason: `Fixer: ${fixVerdict}` };
  }

  return { item, status: 'passed' };
}

async function phaseCI002() {
  const item = 'CI-002: Replace uv venv with uv run throughout workflows';
  console.log(`\n[CI-002] Starting: ${item}`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CI-002: Replace uv venv + source .venv pattern with uv run.

Remove uv venv steps and source .venv/bin/activate lines. Add 'uv run' prefix to python/pytest invocations.

Verify: grep -c 'source .venv' .github/workflows/*.yml returns 0

End with 'verdict: advance' on its own line.`;

  let implResult = await agent({
    role: 'fixer',
    phase: 'implementation',
    task_id: TASK_ID,
    prompt: implPrompt
  });

  let implVerdict = extractVerdict(implResult);
  let implRetries = 0;
  while (implVerdict !== 'advance' && implRetries < MAX_FIX_RETRIES) {
    implRetries++;
    implResult = await agent({
      role: 'fixer',
      phase: 'implementation',
      task_id: TASK_ID,
      attempt: implRetries + 1,
      prompt: implPrompt
    });
    implVerdict = extractVerdict(implResult);
  }

  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

async function phaseCI003() {
  const item = 'CI-003: Audit parity workflows for path correctness';
  console.log(`\n[CI-003] Starting: ${item}`);

  const auditPrompt = `[role: auditor] [phase: review_phase] [task_id: ${TASK_ID}]

Audit CI-003: verify REFERENCE_PATH, checkout paths, inline scripts use absolute paths.

Manual walkthrough of parity.yml and parity-audit.yml.

End with 'verdict: advance' on its own line.`;

  const auditResult = await agent({
    role: 'auditor',
    phase: 'review_phase',
    task_id: TASK_ID,
    prompt: auditPrompt
  });

  const auditVerdict = extractVerdict(auditResult);
  return { item, status: (auditVerdict === 'advance' ? 'passed' : 'blocked') };
}

async function phaseCI004() {
  const item = 'CI-004: Write ADR for deferring ty/ruff as CI gates';
  console.log(`\n[CI-004] Starting: ${item}`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CI-004: Write ADR deferring ty/ruff as CI gates.

Create directory: mkdir -p ${PROJECT_ROOT}/.praxia/docs/adr
Write ADR: ${PROJECT_ROOT}/.praxia/docs/adr/260604_defer-ty-ruff-ci-gates.md (status: accepted, rationale: code surface not stable)
Update ${PROJECT_ROOT}/.praxia/docs/INDEX.md: remove '(stub — to be written, CI-004)' suffix

Verify: test -f ${PROJECT_ROOT}/.praxia/docs/adr/260604_defer-ty-ruff-ci-gates.md

End with 'verdict: advance' on its own line.`;

  let implResult = await agent({
    role: 'fixer',
    phase: 'implementation',
    task_id: TASK_ID,
    prompt: implPrompt
  });

  let implVerdict = extractVerdict(implResult);
  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

async function phaseCI005() {
  const item = 'CI-005: Remove ty/ruff steps from quality-checks job';
  console.log(`\n[CI-005] Starting: ${item}`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CI-005: Remove ty/ruff steps from ci.yml quality-checks job (consequence of CI-004 ADR).

Edit ${PROJECT_ROOT}/.github/workflows/ci.yml: remove 'ty check' and 'ruff check' steps.

Verify: grep -n 'ty check' ${PROJECT_ROOT}/.github/workflows/ci.yml returns 0

End with 'verdict: advance' on its own line.`;

  let implResult = await agent({
    role: 'fixer',
    phase: 'implementation',
    task_id: TASK_ID,
    prompt: implPrompt
  });

  let implVerdict = extractVerdict(implResult);
  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

// ============================================================================
// PHASE 1B: REPO CLEANUP (Parallel)
// ============================================================================

async function phaseCLEANQuick() {
  const item = 'CLEAN-001..010: Batch delete and archive legacy files';
  console.log(`\n[CLEAN] Starting: ${item}`);

  const implPrompt = `[role: fixer] [phase: fix_phase] [task_id: ${TASK_ID}]

Implement CLEAN-001..010: Delete and archive 10 legacy items.

For each item:
- CLEAN-001: git rm AGENTS.md
- CLEAN-002: tar .agent/, create manifest, git rm -r, add to .gitignore
- CLEAN-003: tar .agents/, create manifest, git rm -r
- CLEAN-004: tar PROXIDE_*.md, create manifest, git rm
- CLEAN-005: tar 5 parity docs, create manifest, git rm
- CLEAN-006: tar docs/superpowers/, create manifest, git rm -r
- CLEAN-007: git rm .readthedocs.yaml
- CLEAN-008: add docs/_build/ to .gitignore, git rm --cached if tracked
- CLEAN-009: git rm examples/test_nb.ipynb
- CLEAN-010: git rm colab_training_test.ipynb

Update ${PROJECT_ROOT}/.praxia/docs/INDEX.md with 6 archive entries.

End with 'verdict: advance' on its own line.`;

  let implResult = await agent({
    role: 'fixer',
    phase: 'fix_phase',
    task_id: TASK_ID,
    prompt: implPrompt
  });

  let implVerdict = extractVerdict(implResult);
  let implRetries = 0;
  while (implVerdict !== 'advance' && implRetries < MAX_FIX_RETRIES) {
    implRetries++;
    implResult = await agent({
      role: 'fixer',
      phase: 'fix_phase',
      task_id: TASK_ID,
      attempt: implRetries + 1,
      prompt: implPrompt
    });
    implVerdict = extractVerdict(implResult);
  }

  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

async function phaseCLEAN011() {
  const item = 'CLEAN-011: Write ADR for parity docs as CI-autogenerated';
  console.log(`\n[CLEAN-011] Starting: ${item}`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CLEAN-011: Write ADR for parity docs as CI-autogenerated artifacts.

Create directory: mkdir -p ${PROJECT_ROOT}/.praxia/docs/adr
Write ADR: ${PROJECT_ROOT}/.praxia/docs/adr/260604_parity-docs-as-ci-artifacts.md (status: accepted, decision: docs are auto-generated by CI)
Update ${PROJECT_ROOT}/.praxia/docs/INDEX.md: remove stub suffix

End with 'verdict: advance' on its own line.`;

  let implResult = await agent({
    role: 'fixer',
    phase: 'implementation',
    task_id: TASK_ID,
    prompt: implPrompt
  });

  let implVerdict = extractVerdict(implResult);
  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

async function phaseCLEAN013() {
  const item = 'CLEAN-013: Dependency audit — verify, pin, bump requires-python';
  console.log(`\n[CLEAN-013] Starting: ${item}`);

  const implPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement CLEAN-013: Dependency audit and updates.

Changes:
1. Migrate flax.struct.dataclass → eqx.Module in src/:
   - src/prxteinmpnn/training/metrics.py
   - src/prxteinmpnn/utils/data_structures.py

2. Edit pyproject.toml:
   - Remove "flax" from [project.dependencies]
   - Remove pytest-asyncio, trio, anyio from tests and dev extras
   - Change requires-python to ">=3.12"
   - Add lower-bound pins to jax, numpy, optax, jaxtyping, joblib, psutil, h5py, grain

3. Run: uv lock

Verify:
- grep -rn 'from flax' src/ returns 0
- grep -n 'requires-python' pyproject.toml shows >=3.12
- uv run python -c 'from prxteinmpnn.training.metrics import TrainingMetrics' works

End with 'verdict: advance' on its own line.`;

  let implResult = await agent({
    role: 'fixer',
    phase: 'implementation',
    task_id: TASK_ID,
    prompt: implPrompt
  });

  let implVerdict = extractVerdict(implResult);
  let implRetries = 0;
  while (implVerdict !== 'advance' && implRetries < MAX_FIX_RETRIES) {
    implRetries++;
    implResult = await agent({
      role: 'fixer',
      phase: 'implementation',
      task_id: TASK_ID,
      attempt: implRetries + 1,
      prompt: implPrompt
    });
    implVerdict = extractVerdict(implResult);
  }

  return { item, status: (implVerdict === 'advance' ? 'passed' : 'failed') };
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

export async function run(runner) {
  console.log(`
================================================================================
Release Preparedness Phase 1A + 1B Sprint Composition
Task ID: ${TASK_ID}
Project: ${PROJECT_ROOT}
================================================================================
  `);

  console.log('\n[PHASE 1A] CI Modernization — Running 5 items in parallel');
  const ciResults = await Promise.all([
    phaseCI001(),
    phaseCI002(),
    phaseCI003(),
    phaseCI004(),
    phaseCI005()
  ]);

  console.log('\n[PHASE 1B] Repo Cleanup — Running 3 item groups in parallel');
  const cleanResults = await Promise.all([
    phaseCLEANQuick(),
    phaseCLEAN011(),
    phaseCLEAN013()
  ]);

  const allResults = [...ciResults, ...cleanResults];
  const passedCount = allResults.filter(r => r.status === 'passed').length;
  const failedCount = allResults.filter(r => r.status === 'failed').length;
  const blockedCount = allResults.filter(r => r.status === 'blocked').length;

  console.log(`
================================================================================
SPRINT SUMMARY
================================================================================
Total Items: ${allResults.length}
Passed: ${passedCount}
Failed: ${failedCount}
Blocked: ${blockedCount}

Results:
${allResults.map(r => `  ${r.item}: ${r.status.toUpperCase()}${r.reason ? ' (' + r.reason + ')' : ''}`).join('\n')}

${passedCount === allResults.length ? '✓ ALL ITEMS COMPLETE' : '✗ SOME ITEMS NOT COMPLETE'}
================================================================================
  `);

  return {
    task_id: TASK_ID,
    phase_1a_ci_modernization: ciResults,
    phase_1b_repo_cleanup: cleanResults,
    summary: {
      total: allResults.length,
      passed: passedCount,
      failed: failedCount,
      blocked: blockedCount,
      success: passedCount === allResults.length
    }
  };
}
