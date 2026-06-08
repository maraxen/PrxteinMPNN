/**
 * Release Preparedness Phase 2 Workflow (v2 — oracle-approved)
 *
 * Phase 2: Docs + Release — DOCS-001..010 (10 spec items)
 *
 * Execution order (spec dependency DAG):
 *   Phase 2A (parallel): DOCS-001, DOCS-002, DOCS-003, DOCS-004, DOCS-008
 *   Phase 2B: DOCS-006 parallel; DOCS-005→007→009 serialized (all touch INDEX.md)
 *     — serialization prevents .git/index.lock contention and dirty-tree at DOCS-010
 *   Phase 2C (sequential): DOCS-010
 *     — requires DOCS-009 version bump in committed HEAD before tagging
 *
 * Generated for task_id: 260604_release-prep-sprint-compose
 * Project: prxteinmpnn
 */

export const meta = {
  name: 'release-prep-phase2',
  description: 'Release Prep Phase 2 — Docs + Release (DOCS-001..010), 10 spec items',
  phases: [
    { title: 'Phase 2A', detail: 'DOCS-001/002/003/004/008 in parallel — Phase 1 deps satisfied' },
    { title: 'Phase 2B', detail: 'DOCS-006 parallel; DOCS-005→007→009 serialized (shared INDEX.md)' },
    { title: 'Phase 2C', detail: 'DOCS-010: tag v0.1.0a1 + release notes — requires committed version bump' },
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
// PHASE 2A — PARALLEL
// ============================================================================

async function docsDOCS001() {
  const PHASE = 'Phase 2A';
  const item = 'DOCS-001: README.md full rewrite';
  log(`[DOCS-001] Starting: ${item}`);

  const reconPrompt = `[role: recon] [phase: recon_phase] [task_id: ${TASK_ID}]

Survey the current public API surface of prxteinmpnn for README accuracy.

Project root: ${PROJECT_ROOT}

Steps:
1. Read ${PROJECT_ROOT}/README.md in full — note every claim about API, install instructions, version, benchmarks.
2. Run: grep -rn "^def \\|^class " ${PROJECT_ROOT}/src/prxteinmpnn/ | grep -v "__pycache__" | head -60
3. Read ${PROJECT_ROOT}/src/prxteinmpnn/__init__.py to see what is publicly exported.
4. Check: does current README reference the uv tool install / uvx invocation pattern?
5. Note any stale claims (wrong class names, wrong invocation, wrong Python version requirement).
6. Check ${PROJECT_ROOT}/reports/ and ${PROJECT_ROOT}/outputs/ for Sprint 23 benchmark result files.

Report findings as a structured list:
  - Stale claims to remove or correct
  - Current public API symbols to feature
  - Install invocation (uv tool install prxteinmpnn / uvx prxteinmpnn)
  - Sprint 23 benchmark results if available

End with 'verdict: advance' on its own line.`;

  const reconResult = await agent(reconPrompt, { label: 'recon:docs001', phase: PHASE, agentType: 'recon' });
  const reconVerdict = extractVerdict(reconResult);
  if (reconVerdict !== 'advance') {
    return { item, status: 'blocked', reason: `Recon: ${reconVerdict}` };
  }

  const fixerPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement DOCS-001: Rewrite README.md for prxteinmpnn.

Run git status --short first; if any M or ?? lines exist touching README.md, abort and report.

Epic spec done-when: README reviewed for accuracy against current src/; no stale API references; uv tool install example present.

Prior recon findings (use these as your accuracy guide):
${reconResult}

Write a complete, accurate README.md at ${PROJECT_ROOT}/README.md covering:
1. Project tagline + one-paragraph description of what prxteinmpnn does
2. Installation: uv tool install prxteinmpnn (and uvx for one-shot use)
3. Quick start: minimal working example using only the current public API (confirmed by recon)
4. Composability API: brief overview of the composability primitives surfaced in Sprint 23
5. Sprint 23 benchmarks: include key benchmark numbers if available from recon findings
6. Development setup: uv sync + uv run pytest
7. Links: CONTRIBUTING.md, .praxia/docs/adr/ for decisions

Do NOT include stale API references. Do NOT use pip install for the primary install path.

Verify:
  grep -n 'uv tool install' ${PROJECT_ROOT}/README.md — must be present
  grep -n '3\\.11' ${PROJECT_ROOT}/README.md && echo "STALE VERSION REF" || echo "no 3.11 refs (good)"

End with 'verdict: advance' on its own line.`;

  let fixResult = await agent(fixerPrompt, { label: 'fixer:docs001#0', phase: PHASE, agentType: 'fixer' });
  let fixVerdict = extractVerdict(fixResult);
  let fixRetries = 0;
  while (fixVerdict !== 'advance' && fixRetries < MAX_FIX_RETRIES) {
    fixRetries++;
    fixResult = await agent(fixerPrompt, { label: `fixer:docs001#${fixRetries}`, phase: PHASE, agentType: 'fixer' });
    fixVerdict = extractVerdict(fixResult);
  }

  return { item, status: (fixVerdict === 'advance' ? 'passed' : 'failed') };
}

async function docsDOCS002() {
  const PHASE = 'Phase 2A';
  const item = 'DOCS-002: CLAUDE.md expansion';
  log(`[DOCS-002] Starting: ${item}`);

  const fixerPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement DOCS-002: Expand CLAUDE.md with full project context.

Run git status --short first; if any M or ?? lines exist touching CLAUDE.md, abort and report.

Read ${PROJECT_ROOT}/CLAUDE.md in full first.

Epic spec done-when: CLAUDE.md reflects actual project conventions; commands accurate.

Expand CLAUDE.md to include (add missing sections; update stale ones):
1. Commands table — ensure all four commands are present and accurate:
   uv run ty check | uv run ruff check . | uv run ruff format . | uv run pytest
   Also add: uv run jaxlint check src --no-doc (advisory, not CI gate — per ADR-001)
2. Tech stack — JAX + Equinox (not flax — flax was removed in CLEAN-013)
3. Code style section — add: eqx.Module (not flax.struct.dataclass), filter_jit patterns
4. ADR discipline — brief note: decisions in .praxia/docs/adr/; index at .praxia/docs/INDEX.md
5. Bathos/cluster pointers — cluster jobs via myxcel, experiment tracking via bathos

Do NOT add more than a paragraph per new section — CLAUDE.md is a quick-reference file.

Verify:
  grep -n 'equinox\\|eqx' ${PROJECT_ROOT}/CLAUDE.md — must be present
  grep -n 'flax' ${PROJECT_ROOT}/CLAUDE.md && echo "STALE FLAX REF" || echo "no flax refs (good)"

End with 'verdict: advance' on its own line.`;

  let fixResult = await agent(fixerPrompt, { label: 'fixer:docs002#0', phase: PHASE, agentType: 'fixer' });
  let fixVerdict = extractVerdict(fixResult);
  let fixRetries = 0;
  while (fixVerdict !== 'advance' && fixRetries < MAX_FIX_RETRIES) {
    fixRetries++;
    fixResult = await agent(fixerPrompt, { label: `fixer:docs002#${fixRetries}`, phase: PHASE, agentType: 'fixer' });
    fixVerdict = extractVerdict(fixResult);
  }

  return { item, status: (fixVerdict === 'advance' ? 'passed' : 'failed') };
}

async function docsDOCS003() {
  const PHASE = 'Phase 2A';
  const item = 'DOCS-003: CONTRIBUTING.md review and update';
  log(`[DOCS-003] Starting: ${item}`);

  const reconPrompt = `[role: recon] [phase: recon_phase] [task_id: ${TASK_ID}]

Audit CONTRIBUTING.md for stale instructions.

Read ${PROJECT_ROOT}/CONTRIBUTING.md in full.

Check each of:
1. Python version stated — should be 3.12 (not 3.11 or unversioned)
2. Install instructions — should use uv (not pip venv, not conda)
3. Repo URL — verify it is correct (not a placeholder)
4. Pre-commit hooks — are instructions present and accurate?
5. Test instructions — should be: uv run pytest

Report exactly which lines are stale or incorrect.

End with 'verdict: advance' (stale items found, proceed to fix) or 'verdict: no_changes_needed' (all accurate).`;

  const reconResult = await agent(reconPrompt, { label: 'recon:docs003', phase: PHASE, agentType: 'recon' });
  const reconVerdict = extractVerdict(reconResult);

  if (reconVerdict === 'no_changes_needed') {
    return { item, status: 'passed', reason: 'Recon: no changes needed' };
  }

  const fixerPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement DOCS-003: Update CONTRIBUTING.md based on audit findings.

Run git status --short first; if any M or ?? lines exist touching CONTRIBUTING.md, abort and report.

Audit findings:
${reconResult}

Apply all corrections:
- Python version: ensure 3.12 is stated
- Install: uv sync (not pip install, not conda)
- Tests: uv run pytest
- Fix any stale repo URLs or command references

Verify: grep -n 'uv ' ${PROJECT_ROOT}/CONTRIBUTING.md — should show uv instructions

End with 'verdict: advance' on its own line.`;

  let fixResult = await agent(fixerPrompt, { label: 'fixer:docs003#0', phase: PHASE, agentType: 'fixer' });
  let fixVerdict = extractVerdict(fixResult);
  let fixRetries = 0;
  while (fixVerdict !== 'advance' && fixRetries < MAX_FIX_RETRIES) {
    fixRetries++;
    fixResult = await agent(fixerPrompt, { label: `fixer:docs003#${fixRetries}`, phase: PHASE, agentType: 'fixer' });
    fixVerdict = extractVerdict(fixResult);
  }

  return { item, status: (fixVerdict === 'advance' ? 'passed' : 'failed') };
}

async function docsDOCS004() {
  const PHASE = 'Phase 2A';
  const item = 'DOCS-004: COMPOSITION_GUIDE.md update + Sphinx integration';
  log(`[DOCS-004] Starting: ${item}`);

  const reconPrompt = `[role: recon] [phase: recon_phase] [task_id: ${TASK_ID}]

Survey docs/COMPOSITION_GUIDE.md and docs/source/ Sphinx structure.

Steps:
1. Check if COMPOSITION_GUIDE.md exists: test -f ${PROJECT_ROOT}/docs/COMPOSITION_GUIDE.md && echo "exists" || echo "absent"
2. Read ${PROJECT_ROOT}/docs/source/index.rst
3. Check if COMPOSITION_GUIDE is already linked: grep -n 'COMPOSITION\\|composition' ${PROJECT_ROOT}/docs/source/index.rst
4. List ${PROJECT_ROOT}/docs/source/ to understand the structure
5. Check if myst-parser is configured: grep -n 'myst\\|markdown' ${PROJECT_ROOT}/docs/source/conf.py || echo "myst not configured"
6. List top-level toctree entries in index.rst to understand naming convention

Report:
- Does COMPOSITION_GUIDE.md exist and where?
- Is it linked from index.rst? What toctree entry would be needed?
- Is myst-parser configured (required for .md files in toctree)?
- Any stale API references (flax, old module paths)?

End with 'verdict: advance' on its own line.`;

  const reconResult = await agent(reconPrompt, { label: 'recon:docs004', phase: PHASE, agentType: 'recon' });
  const reconVerdict = extractVerdict(reconResult);
  if (reconVerdict !== 'advance') {
    return { item, status: 'blocked', reason: `Recon: ${reconVerdict}` };
  }

  const fixerPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement DOCS-004: Update COMPOSITION_GUIDE.md and add to Sphinx index.

Run git status --short first; if any M or ?? lines exist touching docs/, abort and report.

Prior recon findings:
${reconResult}

Steps:
1. If COMPOSITION_GUIDE.md has stale API references (flax imports, old class names), correct them.
   Replace any 'from flax' or 'import flax' with eqx.Module equivalents.

2. Ensure ${PROJECT_ROOT}/docs/source/index.rst links to COMPOSITION_GUIDE:
   - If myst-parser IS configured: use '../COMPOSITION_GUIDE' as the toctree entry
     (the guide is at docs/COMPOSITION_GUIDE.md, so path relative to docs/source/ is ../COMPOSITION_GUIDE)
   - If myst-parser is NOT configured: add an .rst stub at docs/source/composition_guide.rst
     with a '.. include::' directive, then add 'composition_guide' to the toctree
   - Match the naming convention of existing toctree entries

3. If COMPOSITION_GUIDE.md does not exist, create a stub at ${PROJECT_ROOT}/docs/COMPOSITION_GUIDE.md
   with a brief overview of composability primitives (StageSet, InferencePlan, eqx.Module patterns).

4. Run a Sphinx dry-run to confirm the toctree resolves (DOCS-005 depends on a warning-free build):
   cd ${PROJECT_ROOT} && uv run python -m sphinx docs/source docs/_build/html-check 2>&1 | grep -E 'WARNING|ERROR|toctree' | head -10 || echo "sphinx not available"

Verify:
  grep -n 'COMPOSITION\\|composition' ${PROJECT_ROOT}/docs/source/index.rst — must be present
  test -f ${PROJECT_ROOT}/docs/COMPOSITION_GUIDE.md && echo "guide exists"

End with 'verdict: advance' on its own line.`;

  let fixResult = await agent(fixerPrompt, { label: 'fixer:docs004#0', phase: PHASE, agentType: 'fixer' });
  let fixVerdict = extractVerdict(fixResult);
  let fixRetries = 0;
  while (fixVerdict !== 'advance' && fixRetries < MAX_FIX_RETRIES) {
    fixRetries++;
    fixResult = await agent(fixerPrompt, { label: `fixer:docs004#${fixRetries}`, phase: PHASE, agentType: 'fixer' });
    fixVerdict = extractVerdict(fixResult);
  }

  return { item, status: (fixVerdict === 'advance' ? 'passed' : 'failed') };
}

async function docsDOCS008() {
  const PHASE = 'Phase 2A';
  const item = 'DOCS-008: Review examples scripts for accuracy';
  log(`[DOCS-008] Starting: ${item}`);

  const auditPrompt = `[role: auditor] [phase: review_phase] [task_id: ${TASK_ID}]

Audit examples scripts for import accuracy. Do NOT execute the scripts — check imports only.

Epic spec done-when: Scripts have no import errors against current API.

Run these SAFE checks (no script execution):
1. Syntax check (compile only, no execution):
   cd ${PROJECT_ROOT} && uv run python -m py_compile examples/read_h5_streaming.py && echo "read_h5: syntax ok" || echo "read_h5: SYNTAX ERROR"
   cd ${PROJECT_ROOT} && uv run python -m py_compile examples/train_with_accumulation.py && echo "train_acc: syntax ok" || echo "train_acc: SYNTAX ERROR"

2. Check for flax imports (flax removed in CLEAN-013 — these would be broken):
   grep -n "import flax\\|from flax" ${PROJECT_ROOT}/examples/read_h5_streaming.py && echo "HAS FLAX (broken)" || echo "no flax in read_h5 (good)"
   grep -n "import flax\\|from flax" ${PROJECT_ROOT}/examples/train_with_accumulation.py && echo "HAS FLAX (broken)" || echo "no flax in train_acc (good)"

3. List all top-level imports in each script:
   grep -n "^import \\|^from " ${PROJECT_ROOT}/examples/read_h5_streaming.py
   grep -n "^import \\|^from " ${PROJECT_ROOT}/examples/train_with_accumulation.py

For each script: are there flax imports (broken)? Any other suspicious imports?

If both scripts pass (no flax, no obvious broken imports): end with 'verdict: advance'
If scripts have flax or broken imports: report each; end with 'verdict: needs_fix'`;

  const auditResult = await agent(auditPrompt, { label: 'auditor:docs008', phase: PHASE, agentType: 'auditor' });
  const auditVerdict = extractVerdict(auditResult);

  if (auditVerdict === 'advance') {
    return { item, status: 'passed' };
  }

  const fixerPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement DOCS-008: Fix import errors in examples scripts.

Run git status --short first; if any M or ?? lines exist touching examples/, abort and report.

Audit findings:
${auditResult}

Fix all import errors identified in the audit:
- Any 'from flax' imports: replace with eqx.Module equivalents or remove
- Any broken module paths
- Do NOT rewrite logic — fix imports only

After fixes, verify:
  cd ${PROJECT_ROOT} && uv run python -m py_compile examples/read_h5_streaming.py && echo "syntax ok"
  grep -n "import flax\\|from flax" ${PROJECT_ROOT}/examples/read_h5_streaming.py && echo "STILL HAS FLAX" || echo "clean"

End with 'verdict: advance' on its own line.`;

  let fixResult = await agent(fixerPrompt, { label: 'fixer:docs008#0', phase: PHASE, agentType: 'fixer' });
  let fixVerdict = extractVerdict(fixResult);
  let fixRetries = 0;
  while (fixVerdict !== 'advance' && fixRetries < MAX_FIX_RETRIES) {
    fixRetries++;
    fixResult = await agent(fixerPrompt, { label: `fixer:docs008#${fixRetries}`, phase: PHASE, agentType: 'fixer' });
    fixVerdict = extractVerdict(fixResult);
  }

  return { item, status: (fixVerdict === 'advance' ? 'passed' : 'failed') };
}

// ============================================================================
// PHASE 2B
// DOCS-006 is standalone (new file only — no INDEX.md, no git-rm) — runs in parallel.
// DOCS-005/007/009 all touch INDEX.md and make git commits — serialized to prevent
//   .git/index.lock contention and ensure DOCS-010 receives a clean working tree.
// ============================================================================

async function docsDOCS005() {
  const PHASE = 'Phase 2B';
  const item = 'DOCS-005: Re-enable fail_on_warning or document remaining Sphinx warnings';
  log(`[DOCS-005] Starting: ${item} (after DOCS-004)`);

  const auditPrompt = `[role: auditor] [phase: review_phase] [task_id: ${TASK_ID}]

Audit Sphinx build for warnings. DOCS-004 has already been applied (COMPOSITION_GUIDE linked).

Steps:
1. Check if sphinx-build is available: which sphinx-build || uv run python -m sphinx --version 2>&1 | head -3
2. If available: cd ${PROJECT_ROOT} && uv run python -m sphinx docs/source docs/_build/html -W --keep-going 2>&1 | tail -30
   The -W flag turns warnings into errors; --keep-going collects all before failing.
3. Collect lines starting with WARNING:
4. Check current .readthedocs.yml: grep -n 'fail_on_warning' ${PROJECT_ROOT}/.readthedocs.yml

If sphinx-build is not available: end with 'verdict: sphinx_not_available'
If build has zero warnings: end with 'verdict: advance'
If build has warnings: list them; end with 'verdict: needs_issue_list'`;

  const auditResult = await agent(auditPrompt, { label: 'auditor:docs005', phase: PHASE, agentType: 'auditor' });
  const auditVerdict = extractVerdict(auditResult);

  if (auditVerdict === 'advance') {
    const enablePrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement DOCS-005 (clean path): Re-enable fail_on_warning in .readthedocs.yml.

Run git status --short first.

Edit ${PROJECT_ROOT}/.readthedocs.yml: set fail_on_warning: true (or ensure not false).

Commit:
  git -C ${PROJECT_ROOT} add .readthedocs.yml
  git -C ${PROJECT_ROOT} commit -m "docs: re-enable fail_on_warning in ReadTheDocs (Sphinx build clean)"

Verify: grep -n 'fail_on_warning' ${PROJECT_ROOT}/.readthedocs.yml

End with 'verdict: advance' on its own line.`;
    let enableResult = await agent(enablePrompt, { label: 'fixer:docs005-enable#0', phase: PHASE, agentType: 'fixer' });
    let enableVerdict = extractVerdict(enableResult);
    let enableRetries = 0;
    while (enableVerdict !== 'advance' && enableRetries < MAX_FIX_RETRIES) {
      enableRetries++;
      enableResult = await agent(enablePrompt, { label: `fixer:docs005-enable#${enableRetries}`, phase: PHASE, agentType: 'fixer' });
      enableVerdict = extractVerdict(enableResult);
    }
    return { item, status: (enableVerdict === 'advance' ? 'passed' : 'failed') };
  }

  if (auditVerdict === 'sphinx_not_available') {
    return { item, status: 'blocked', reason: 'sphinx-build not available in this environment' };
  }

  // warnings found — write issue list
  const issueListPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement DOCS-005 (warnings path): Document remaining Sphinx warnings as tracked issues.

Run git status --short first.

Audit findings (warnings list):
${auditResult}

Step 1 — Create directory:
  mkdir -p ${PROJECT_ROOT}/.praxia/docs/misc

Step 2 — Write issue list to: ${PROJECT_ROOT}/.praxia/docs/misc/260604_sphinx-warning-backlog.md
  Format:
  ---
  title: Sphinx Warning Backlog
  created: 260604
  status: open
  ---
  # Sphinx Build Warnings — Resolution Backlog

  Each entry is a tracked warning. fail_on_warning is deferred until resolved.

  ## Warnings
  (one entry per warning: file:line, warning text, suggested fix)

Step 3 — Update ${PROJECT_ROOT}/.praxia/docs/INDEX.md:
  Check if '## Misc' section exists. If not, add at the end of the file:
    ## Misc
  Then add under ## Misc:
    - [260604_sphinx-warning-backlog](misc/260604_sphinx-warning-backlog.md) — Sphinx warnings deferred from fail_on_warning

Step 4 — Commit:
  git -C ${PROJECT_ROOT} add .praxia/docs/misc/260604_sphinx-warning-backlog.md .praxia/docs/INDEX.md
  git -C ${PROJECT_ROOT} commit -m "docs(misc): add Sphinx warning backlog — deferred fail_on_warning"

Verify: test -f ${PROJECT_ROOT}/.praxia/docs/misc/260604_sphinx-warning-backlog.md && echo ok

End with 'verdict: advance' on its own line.`;

  let fixResult = await agent(issueListPrompt, { label: 'fixer:docs005-list#0', phase: PHASE, agentType: 'fixer' });
  let fixVerdict = extractVerdict(fixResult);
  let fixRetries = 0;
  while (fixVerdict !== 'advance' && fixRetries < MAX_FIX_RETRIES) {
    fixRetries++;
    fixResult = await agent(issueListPrompt, { label: `fixer:docs005-list#${fixRetries}`, phase: PHASE, agentType: 'fixer' });
    fixVerdict = extractVerdict(fixResult);
  }

  return { item, status: (fixVerdict === 'advance' ? 'passed' : 'failed') };
}

async function docsDOCS006() {
  const PHASE = 'Phase 2B';
  const item = 'DOCS-006: New Colab notebook with built-in widgets';
  log(`[DOCS-006] Starting: ${item} (runs in parallel with DOCS-005/007/009)`);

  const fixerPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement DOCS-006: Create a new Colab notebook in examples/ using Colab built-in widgets.

Run git status --short first.

Epic spec done-when: Notebook runs cell-by-cell on Colab free tier; uses current API; widgets functional.

Read ${PROJECT_ROOT}/src/prxteinmpnn/__init__.py first to get the exact public API symbols.
Do NOT invent function signatures — only use what is actually exported.

Create ${PROJECT_ROOT}/examples/colab_inference_demo.ipynb with these cells:
1. Markdown badge cell: [![Open In Colab](...) — use a placeholder URL
2. Code — Install:
   # Requires prxteinmpnn==0.1.0a1 to be published to PyPI (run after release)
   !pip install prxteinmpnn==0.1.0a1
3. Code — Imports: from prxteinmpnn import <actual exported symbols>
4. Code — Widgets (use ipywidgets, available on Colab):
   import ipywidgets as widgets
   seq_input = widgets.Text(description='Sequence:')
   temp_slider = widgets.FloatSlider(value=0.5, min=0.1, max=1.0, step=0.05, description='Temp:')
   mode_dropdown = widgets.Dropdown(options=['design', 'score'], description='Mode:')
   display(seq_input, temp_slider, mode_dropdown)
5. Code — Inference: wire widget .value into the actual public API call
6. Code — Display results

Write the notebook as valid JSON (.ipynb format).
REQUIRED top-level structure:
  {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
      "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.12.0"}
    },
    "cells": [...]
  }
Each cell: {"cell_type": "code"|"markdown", "source": ["line1\\n", "line2"], "metadata": {}, "outputs": [], "execution_count": null}

Verify: uv run python -c "import json; nb=json.load(open('${PROJECT_ROOT}/examples/colab_inference_demo.ipynb')); print(f'valid: nbformat={nb[\"nbformat\"]} cells={len(nb[\"cells\"])}')"

End with 'verdict: advance' on its own line.`;

  let fixResult = await agent(fixerPrompt, { label: 'fixer:docs006#0', phase: PHASE, agentType: 'fixer' });
  let fixVerdict = extractVerdict(fixResult);
  let fixRetries = 0;
  while (fixVerdict !== 'advance' && fixRetries < MAX_FIX_RETRIES) {
    fixRetries++;
    fixResult = await agent(fixerPrompt, { label: `fixer:docs006#${fixRetries}`, phase: PHASE, agentType: 'fixer' });
    fixVerdict = extractVerdict(fixResult);
  }

  return { item, status: (fixVerdict === 'advance' ? 'passed' : 'failed') };
}

async function docsDOCS007() {
  const PHASE = 'Phase 2B';
  const item = 'DOCS-007: Review example notebooks for API accuracy';
  log(`[DOCS-007] Starting: ${item} (sequential after DOCS-005)`);

  const auditPrompt = `[role: auditor] [phase: review_phase] [task_id: ${TASK_ID}]

Audit examples/example_notebook.ipynb and examples/training_example_notebook.ipynb for API accuracy.

Epic spec done-when: Both notebooks either (a) use current API, or (b) are archived.

Steps:
1. Check existence:
   test -f ${PROJECT_ROOT}/examples/example_notebook.ipynb && echo "example_notebook exists"
   test -f ${PROJECT_ROOT}/examples/training_example_notebook.ipynb && echo "training_notebook exists"
2. For each that exists, extract code cells:
   uv run python -c "import json; nb=json.load(open('${PROJECT_ROOT}/examples/example_notebook.ipynb')); src='\\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'); print(src)" 2>&1 | head -60
3. Check for: flax imports (broken — removed), stale class names, removed function signatures

For each notebook, give verdict: 'keep' / 'archive' / 'update'

End with 'verdict: advance' on its own line.`;

  const auditResult = await agent(auditPrompt, { label: 'auditor:docs007', phase: PHASE, agentType: 'auditor' });
  const auditVerdict = extractVerdict(auditResult);
  if (auditVerdict !== 'advance') {
    return { item, status: 'blocked', reason: `Audit: ${auditVerdict}` };
  }

  const fixerPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement DOCS-007: Apply audit decisions to example notebooks.

Run git status --short first.

Audit findings:
${auditResult}

Archive slug mapping (use these exact names):
  examples/example_notebook.ipynb → 260604_example-notebook
  examples/training_example_notebook.ipynb → 260604_training-example-notebook

For each notebook rated 'archive':
  0. mkdir -p ${PROJECT_ROOT}/.praxia/docs/archive
  1. tar -C ${PROJECT_ROOT}/examples -caf ${PROJECT_ROOT}/.praxia/docs/archive/260604_<slug>.tar.zst <notebook-filename>
  2. Write manifest ${PROJECT_ROOT}/.praxia/docs/archive/260604_<slug>.md with frontmatter:
       archive: 260604_<slug>.tar.zst / created: 260604 / source: examples/<filename>
  3. git -C ${PROJECT_ROOT} rm --ignore-unmatch examples/<notebook-filename>

For each notebook rated 'update': apply minimal import/API fixes only.
For each notebook rated 'keep': no action.

Update INDEX.md archives section:
  Check ${PROJECT_ROOT}/.praxia/docs/INDEX.md — find ## Archives section and add manifest entries.
  If ## Archives section is absent, add it before ## ADRs.

Commit all changes:
  git -C ${PROJECT_ROOT} add -A
  git -C ${PROJECT_ROOT} diff --cached --quiet || git -C ${PROJECT_ROOT} commit -m "docs(archive): archive/update example notebooks per DOCS-007 audit"

Verify for archived notebooks: test ! -f ${PROJECT_ROOT}/examples/<notebook-filename> && echo "archived ok"

End with 'verdict: advance' on its own line.`;

  let fixResult = await agent(fixerPrompt, { label: 'fixer:docs007#0', phase: PHASE, agentType: 'fixer' });
  let fixVerdict = extractVerdict(fixResult);
  let fixRetries = 0;
  while (fixVerdict !== 'advance' && fixRetries < MAX_FIX_RETRIES) {
    fixRetries++;
    fixResult = await agent(fixerPrompt, { label: `fixer:docs007#${fixRetries}`, phase: PHASE, agentType: 'fixer' });
    fixVerdict = extractVerdict(fixResult);
  }

  return { item, status: (fixVerdict === 'advance' ? 'passed' : 'failed') };
}

async function docsDOCS009() {
  const PHASE = 'Phase 2B';
  const item = 'DOCS-009: Write versioning ADR + bump pyproject.toml to 0.1.0a1';
  log(`[DOCS-009] Starting: ${item} (sequential after DOCS-007)`);

  const fixerPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement DOCS-009: Write versioning ADR and bump version in pyproject.toml.

Run git status --short first; if any M or ?? lines exist touching files you will edit, abort and report.

Step 1 — Write ADR to: ${PROJECT_ROOT}/.praxia/docs/adr/260604_versioning-strategy.md
  Frontmatter: status: accepted, date: 260604
  Content sections:
  - Decision: version is 0.1.0a1 (alpha). Tagging strategy documented here.
  - Alpha→Beta criteria: README/CONTRIBUTING/COMPOSITION_GUIDE accurate; all public API examples run without error; no known API-breaking issues; CI passes.
  - Beta→Stable criteria: two weeks of alpha use with no critical bugs; all benchmark examples validated; ReadTheDocs clean (or fail_on_warning re-enabled).
  - Rationale: 0.1.0a1 signals honest alpha maturity; avoids premature stability promise.

Step 2 — Edit ${PROJECT_ROOT}/pyproject.toml:
  The CURRENT value in pyproject.toml is exactly: version = "0.1.0"
  Replace it with: version = "0.1.0a1"
  (Use the Edit tool: old_string = 'version = "0.1.0"', new_string = 'version = "0.1.0a1"')

Step 3 — Update ${PROJECT_ROOT}/.praxia/docs/INDEX.md under ## ADRs:
  Add: [260604_versioning-strategy](adr/260604_versioning-strategy.md) — Versioning: 0.1.0a1 → beta → stable

Step 4 — COMMIT all changes (DOCS-010 tags from committed HEAD — this commit must be present):
  git -C ${PROJECT_ROOT} add .praxia/docs/adr/260604_versioning-strategy.md pyproject.toml .praxia/docs/INDEX.md
  git -C ${PROJECT_ROOT} commit -m "docs(adr): add versioning strategy; bump version to 0.1.0a1"

Verify (must check COMMITTED HEAD, not working tree):
  test -f ${PROJECT_ROOT}/.praxia/docs/adr/260604_versioning-strategy.md && echo "ADR ok"
  git -C ${PROJECT_ROOT} show HEAD:pyproject.toml | grep '^version' — must print: version = "0.1.0a1"

End with 'verdict: advance' on its own line.`;

  let fixResult = await agent(fixerPrompt, { label: 'fixer:docs009#0', phase: PHASE, agentType: 'fixer' });
  let fixVerdict = extractVerdict(fixResult);
  let fixRetries = 0;
  while (fixVerdict !== 'advance' && fixRetries < MAX_FIX_RETRIES) {
    fixRetries++;
    fixResult = await agent(fixerPrompt, { label: `fixer:docs009#${fixRetries}`, phase: PHASE, agentType: 'fixer' });
    fixVerdict = extractVerdict(fixResult);
  }

  return { item, status: (fixVerdict === 'advance' ? 'passed' : 'failed') };
}

// ============================================================================
// PHASE 2C — SEQUENTIAL (requires DOCS-009 committed to HEAD)
// ============================================================================

async function docsDOCS010() {
  const PHASE = 'Phase 2C';
  const item = 'DOCS-010: Tag v0.1.0a1 and write release notes';
  log(`[DOCS-010] Starting: ${item} (requires DOCS-009 committed to HEAD)`);

  const fixerPrompt = `[role: fixer] [phase: implementation] [task_id: ${TASK_ID}]

Implement DOCS-010: Write release notes and create local git tag v0.1.0a1.

Run git status --short first; if the tree is not clean (any M or ?? lines), abort.
All Phase 2A and 2B changes must be committed before tagging.

Step 0 — Verify version bump is in COMMITTED HEAD (not just staged/unstaged):
  git -C ${PROJECT_ROOT} show HEAD:pyproject.toml | grep '^version'
  This must print: version = "0.1.0a1"
  If it does not: abort and report "DOCS-009 version bump must be committed before DOCS-010 can tag"
  End the abort message with: verdict: blocked

Step 1 — Create directory:
  mkdir -p ${PROJECT_ROOT}/.praxia/docs/misc

Step 2 — Write release notes to: ${PROJECT_ROOT}/.praxia/docs/misc/260604_release-notes-v0.1.0a1.md
  Structure:
  ---
  title: Release Notes — v0.1.0a1
  date: 260604
  tag: v0.1.0a1
  ---
  # prxteinmpnn v0.1.0a1 — Alpha Release

  ## What's Included
  - Composability primitives: StageSet, eqx.Module-based metrics + data structures
  - Sprint 23 benchmarks: (add key results if available in reports/ or README)
  - Modern CI: Python 3.12, uv run, three workflow files
  - Cleaned repo: legacy scaffolding archived, flax dependency removed

  ## Install
  # Requires PyPI publication of v0.1.0a1 (run after: git push origin v0.1.0a1 + PyPI upload)
  pip install prxteinmpnn==0.1.0a1
  # Or with uv (once published):
  uv tool install prxteinmpnn

  ## Known Limitations
  - Alpha quality: API may change before beta
  - PyPI publish: pending — package must be uploaded after tagging
  - Colab notebook: requires free-tier GPU allocation
  - ReadTheDocs: fail_on_warning may be disabled pending Sphinx cleanup

  ## Upgrade Path to Beta
  See .praxia/docs/adr/260604_versioning-strategy.md for promotion criteria.

Step 3 — Update INDEX.md under ## Misc (create section if absent):
  - [260604_release-notes-v0.1.0a1](misc/260604_release-notes-v0.1.0a1.md) — Release notes for v0.1.0a1 alpha

Step 4 — Commit release notes:
  git -C ${PROJECT_ROOT} add .praxia/docs/misc/260604_release-notes-v0.1.0a1.md .praxia/docs/INDEX.md
  git -C ${PROJECT_ROOT} commit -m "docs: add release notes for v0.1.0a1"

Step 5 — Create LOCAL git tag (do NOT push):
  git -C ${PROJECT_ROOT} tag -a v0.1.0a1 -m "Release v0.1.0a1 — alpha"

NOTE: This creates a local tag only. Do NOT push. User pushes when ready:
  git push origin v0.1.0a1

Verify:
  git -C ${PROJECT_ROOT} tag -l 'v0.1.0a1' — must print v0.1.0a1
  test -f ${PROJECT_ROOT}/.praxia/docs/misc/260604_release-notes-v0.1.0a1.md && echo "notes ok"

End with 'verdict: advance' on its own line.`;

  let fixResult = await agent(fixerPrompt, { label: 'fixer:docs010#0', phase: PHASE, agentType: 'fixer' });
  let fixVerdict = extractVerdict(fixResult);
  let fixRetries = 0;
  while (fixVerdict !== 'advance' && fixRetries < MAX_FIX_RETRIES) {
    fixRetries++;
    fixResult = await agent(fixerPrompt, { label: `fixer:docs010#${fixRetries}`, phase: PHASE, agentType: 'fixer' });
    fixVerdict = extractVerdict(fixResult);
  }

  return { item, status: (fixVerdict === 'advance' ? 'passed' : 'failed') };
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

log('Starting Release Prep Phase 2 — Docs + Release');
log('Phase 2A (parallel): DOCS-001/002/003/004/008');
log('Phase 2B: DOCS-006 parallel; DOCS-005→007→009 serialized (shared INDEX.md)');
log('Phase 2C (sequential): DOCS-010');

// ── Phase 2A ─────────────────────────────────────────────────────────────
phase('Phase 2A');

const [d001, d002, d003, d004, d008] = await parallel([
  () => docsDOCS001(),
  () => docsDOCS002(),
  () => docsDOCS003(),
  () => docsDOCS004(),
  () => docsDOCS008(),
]);

const phase2aResults = [d001, d002, d003, d004, d008].filter(Boolean);
log(`Phase 2A complete: ${phase2aResults.filter(r => r.status === 'passed').length}/${phase2aResults.length} passed`);

// ── Phase 2B ─────────────────────────────────────────────────────────────
// DOCS-006: standalone (creates one new file, no INDEX.md, no git-rm) — fire immediately.
// DOCS-005/007/009: all touch INDEX.md and git-commit — serialized to prevent
//   .git/index.lock races and ensure a clean working tree when DOCS-010 tags.
phase('Phase 2B');

const d006Promise = docsDOCS006();   // fire immediately, runs concurrently with sequential block

const d005 = await docsDOCS005();    // Sphinx audit → issue list or re-enable + commit
const d007 = await docsDOCS007();    // notebook archive/update + git rm + INDEX.md + commit
const d009 = await docsDOCS009();    // ADR + pyproject.toml bump + INDEX.md + COMMIT (required before tag)

const d006 = await d006Promise;      // collect DOCS-006 result

const phase2bResults = [d005, d006, d007, d009].filter(Boolean);
log(`Phase 2B complete: ${phase2bResults.filter(r => r.status === 'passed').length}/${phase2bResults.length} passed`);

// ── Phase 2C ─────────────────────────────────────────────────────────────
phase('Phase 2C');

const d010 = await docsDOCS010();

// ── Summary ───────────────────────────────────────────────────────────────

const allResults = [...phase2aResults, ...phase2bResults, d010].filter(Boolean);
const passedCount = allResults.filter(r => r.status === 'passed').length;
const failedCount = allResults.filter(r => r.status === 'failed').length;
const blockedCount = allResults.filter(r => r.status === 'blocked').length;

return {
  task_id: TASK_ID,
  phase_2a: phase2aResults,
  phase_2b: phase2bResults,
  phase_2c: [d010].filter(Boolean),
  summary: {
    total: allResults.length,
    passed: passedCount,
    failed: failedCount,
    blocked: blockedCount,
    success: failedCount === 0 && blockedCount === 0,
    note: 'DOCS-010 tag is local only — push with: git push origin v0.1.0a1. DOCS-005 may be blocked if sphinx-build unavailable.',
  },
};
