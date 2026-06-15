// Sprint 2 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/260619_rs6-foundation.toml
// Regenerate: praxia dw emit-sprint 260619_rs6-foundation.toml
// task_id: 260619_rs6-foundation   sprint_id: 2
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (C,B,A) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.

export const meta = {
  name: "260619_rs6-foundation",
  description: "RS-6 foundation: xtrax/run/ module, AxisSpec naming gate, FuseFn __all__ cleanup",
  phases: [
    { title: "Track C — R7-1: AxisSpec field mapping table + naming decision (GATE) (#1926)" },
    { title: "Track B — R6-4: Remove FuseFn from xtrax __all__ exports (#1925)" },
    { title: "Track A — R6-1: Create xtrax/run/ module (#1922)" },
  ],
};

const TASK_ID = "260619_rs6-foundation";
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

// Shared context for the writing tracks (from recon, task 260619_rs6-foundation).
const EMITTER_CTX = `Spec: .praxia/docs/specs/260615_runspec-xtrax-run-epic.md (RS-6 section)\n\nKey facts:\n- xtrax/run/ does NOT exist yet — Track A creates it from scratch\n- aminx RunSpec (eqx.Module) at aminx/run/spec.py:114 — fields: io, resource, multistate, ligand, tied, grid, batching, averaging, precision, plan\n- xtrax AxisSpec at xtrax/tiling/plan.py:43 — fields: name, cardinality, batch_size, granularity, heterogeneous, dedup_eligible, bucket_boundaries\n- prolix AxisSpec at prolix/tiling/planner.py:56 — fields: name, axis_index, cardinality, default_batch_size, tile_granularity, heterogeneous, doc\n- Naming conflicts: batch_size vs default_batch_size; granularity vs tile_granularity\n- Track C (R7-1) resolves naming conflicts and gates Track A field names\n- FuseFn class was already removed by H4 (commit bb8187d); __getattr__ shim already in xtrax/stages/__init__.py\n- FuseFn still in __all__ of xtrax/stages/__init__.py (line 6) and xtrax/__init__.py (lines 55, 111)\n- aminx/types/stages.py:68 FuseFn (logit transform, bias param) is DISTINCT and OUT OF SCOPE\n`;

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

// ===== TRACK C — Track C — R7-1: AxisSpec field mapping table + naming decision (GATE) (#1926) =========================
const trackC = () =>
  agent(
    `task_id: ${TASK_ID}. Research task #1926: Produce the AxisSpec canonical field mapping table.\n\nThis is a GATE item — Track A (R6-1) cannot commit RunSpec field names until you have delivered the naming decisions in this document.\n\nOUTPUT FILE: .praxia/docs/research/260616_axisspec-field-map.md\n\nSTEP 1 — Read source definitions:\n- xtrax AxisSpec: xtrax/src/xtrax/tiling/plan.py starting at class AxisSpec (~line 30)\n  Fields: name, cardinality, batch_size, granularity, heterogeneous, dedup_eligible, bucket_boundaries\n- prolix AxisSpec: prolix/src/prolix/tiling/planner.py starting at class AxisSpec (~line 56)\n  Fields: name, axis_index, cardinality, default_batch_size, tile_granularity, heterogeneous, doc\n- Check if aminx has its own AxisSpec definition: grep -r "class AxisSpec" aminx/\n\nSTEP 2 — Build the three-column table with this EXACT structure:\n\n| xtrax field | prolix field | canonical decision |\n|---|---|---|\n| name: str | name: str | name |\n| cardinality: int | cardinality: int | cardinality |\n| batch_size: int | default_batch_size: int | <decide here> |\n| granularity: int = 1 | tile_granularity: int | <decide here> |\n| heterogeneous: bool = False | heterogeneous: bool | heterogeneous |\n| dedup_eligible: bool = False | (absent) | dedup_eligible (xtrax-only) |\n| bucket_boundaries: tuple[int,...] | None = None | (absent) | bucket_boundaries (xtrax-only) |\n| (absent) | axis_index: int | axis_index (prolix-only, ordering metadata) |\n| (absent) | doc: str | doc (prolix-only, free-text) |\n\nSTEP 3 — Make naming decisions for the two conflicts. Choose one name as canonical:\n\nConflict 1 — batch_size vs default_batch_size:\n- xtrax uses batch_size (short; consistent with AxisDecision.batch_size output field)\n- prolix uses default_batch_size (explicit that 0 = vmap; the "default" qualifier prevents confusion with AxisDecision.batch_size)\n- Recommendation: use default_batch_size — clarifies the 0=vmap semantic and avoids collision with AxisDecision.batch_size output field\n- Read both definitions and check AxisDecision in xtrax/tiling/plan.py — does AxisDecision also have a batch_size field? If so, default_batch_size is clearly the right choice to avoid ambiguity.\n- Choose and commit. Do not write "TBD" or "deferred".\n\nConflict 2 — granularity vs tile_granularity:\n- xtrax uses granularity (short)\n- prolix uses tile_granularity (domain-specific; Pallas/safe_map alignment context)\n- Recommendation: use tile_granularity — disambiguates from other granularity concepts (bucket granularity, sequence granularity)\n- Choose and commit. Do not write "TBD" or "deferred".\n\nSTEP 4 — Write the naming gate section (REQUIRED):\n\nThe LAST section of the document must be titled "## RS-6 RunSpec axis field names" and state:\n\nBased on the canonical decisions above, Track A (R6-1) MUST use these constructor kwarg names when creating xtrax/run/spec.py and any AxisSpec instantiation:\n- \`default_batch_size=\` (canonical, replaces batch_size=)\n- \`tile_granularity=\` (canonical, replaces granularity=)\n\nAlso state whether the xtrax AxisSpec class itself needs to be renamed/migrated (SEPARATE WORK — do not do it in this task; just note it as R7-2 future work).\n\nSTEP 5 — Commit:\n  git add .praxia/docs/research/260616_axisspec-field-map.md\n  git commit -m "research(R7-1): AxisSpec field mapping table + naming decisions (#1926)"\n\nAlso update .praxia/docs/INDEX.md to add the new file under the Research section.\n`,
    { agentType: "librarian", label: "research:1926", phase: "Track C — R7-1: AxisSpec field mapping table + naming decision (GATE) (#1926)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK B — Track B — R6-4: Remove FuseFn from xtrax __all__ exports (#1925) =========================
const trackB = () =>
  track(
    "1925",
    "Track B — R6-4: Remove FuseFn from xtrax __all__ exports (#1925)",
    `task_id: ${TASK_ID}. Fix #1925: Remove FuseFn from __all__ in xtrax/stages/__init__.py and xtrax/__init__.py.\n\nCONTEXT:\n- H4 already removed the FuseFn class from protocols.py (commit bb8187d)\n- A __getattr__ shim now exists in xtrax/stages/__init__.py (~line 10) that returns Fuse with DeprecationWarning\n- The shim means \`from xtrax.stages import FuseFn\` still WORKS (triggers warning) without __all__ listing it\n- But FuseFn is still in two __all__ declarations, which means \`from xtrax.stages import *\` imports it silently\n\nSTEP 1 — Remove from xtrax/src/xtrax/stages/__init__.py:\nFind the __all__ list (around line 6). It currently contains "FuseFn" — remove ONLY that string.\nLeave TransformFn, RollingFn, Fuse, Tap, Sink, AxisBoundary, and all other entries.\nDo NOT touch the __getattr__ function below the __all__ list.\n\nSTEP 2 — Remove from xtrax/src/xtrax/__init__.py:\nAround line 55: remove "FuseFn" from the __all__ list.\nAround line 111: remove the "FuseFn": "xtrax.stages" entry from the lazy-import module map.\n\nSTEP 3 — Verify the deprecation path still works (inline check):\n  cd /home/marielle/projects/xtrax && uv run python -c "\n  import warnings\n  with warnings.catch_warnings(record=True) as w:\n      warnings.simplefilter('always')\n      from xtrax.stages import FuseFn\n      assert len(w) == 1 and issubclass(w[0].category, DeprecationWarning), f'Expected DeprecationWarning, got: {w}'\n  print('Deprecation path OK — DeprecationWarning fires correctly')\n  "\n\nSTEP 4 — Verify FuseFn is NOT in star-import:\n  uv run python -c "\n  import xtrax.stages\n  assert 'FuseFn' not in xtrax.stages.__all__, 'FuseFn should not be in __all__'\n  print('FuseFn correctly absent from __all__')\n  "\n\nSTEP 5 — Run tests:\n  cd /home/marielle/projects/xtrax && uv run pytest\n\nSTEP 6 — Commit from xtrax directory:\n  git add src/xtrax/stages/__init__.py src/xtrax/__init__.py\n  git commit -m "refactor(R6-4): remove FuseFn from __all__; deprecated __getattr__ shim remains (#1925)"\n\nSCOPE NOTE: aminx/src/aminx/types/stages.py:68 has a SEPARATE FuseFn (logit transform protocol\nwith a bias: Array parameter). Do NOT touch that file at all.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Review R6-4: FuseFn __all__ cleanup.\n\nVERIFY 1: uv run pytest passes in xtrax (run from /home/marielle/projects/xtrax)\nVERIFY 2: "FuseFn" is NOT present in the __all__ list in xtrax/src/xtrax/stages/__init__.py\nVERIFY 3: "FuseFn" is NOT present in the __all__ list in xtrax/src/xtrax/__init__.py\nVERIFY 4: "FuseFn" is NOT present in the lazy-import module map in xtrax/src/xtrax/__init__.py\nVERIFY 5: \`from xtrax.stages import FuseFn\` still resolves at runtime (deprecated shim) and raises DeprecationWarning\nVERIFY 6: aminx/src/aminx/types/stages.py is UNCHANGED (logit-transform FuseFn is a different protocol, out of scope)\n\nPASS if all 6 VERIFY items are satisfied.\nFAIL if any item is not met.\n`,
  );

// ===== TRACK A — Track A — R6-1: Create xtrax/run/ module (#1922) =========================
const trackA = () =>
  track(
    "1922",
    "Track A — R6-1: Create xtrax/run/ module (#1922)",
    `task_id: ${TASK_ID}. Fix #1922: Create the xtrax/run/ module — the RS-6 foundation.\n\nPREREQUISITE: Track C (R7-1) must be complete. Read .praxia/docs/research/260616_axisspec-field-map.md\nbefore committing any field names. The document declares canonical kwarg names (expected: default_batch_size,\ntile_granularity) — use those names in any AxisSpec instantiation in this task.\n\n--- PART 1: Create xtrax/src/xtrax/run/ ---\n\nCreate the directory and these files:\n\nFILE: xtrax/src/xtrax/run/__init__.py\nContent:\n  """xtrax.run — execution-time configuration layer."""\n  from xtrax.run.spec import RunSpec\n  from xtrax.run.resolver import FeatureBatch, InputResolver, RuntimeBundle\n  from xtrax.run.sink import SinkSpec\n\n  __all__ = ["RunSpec", "InputResolver", "RuntimeBundle", "FeatureBatch", "SinkSpec"]\n\nFILE: xtrax/src/xtrax/run/spec.py\nContent:\n  from __future__ import annotations\n  from typing import Any\n  import equinox as eqx\n  from xtrax.stages.boundaries import AxisBoundary\n  from xtrax.tiling import AxisSpec\n\n  class RunSpec(eqx.Module):\n      """Base execution config. aminx.run.RunSpec (eqx.Module) extends this."""\n      seed: int\n      axes: list[AxisSpec]\n      carry_specs: dict[str, Any] = eqx.field(default_factory=dict)\n      boundaries: list[AxisBoundary] | None = None\n\nFILE: xtrax/src/xtrax/run/resolver.py\nContent:\n  from __future__ import annotations\n  from dataclasses import dataclass\n  from typing import Any, NewType\n  from typing import Protocol, runtime_checkable\n  import equinox as eqx\n  from xtrax.run.spec import RunSpec\n\n  FeatureBatch = NewType("FeatureBatch", dict)\n\n  @dataclass\n  class RuntimeBundle:\n      """Materialized execution context (produced before InputResolver fires)."""\n      iterator: Any\n      model: eqx.Module\n\n  @runtime_checkable\n  class InputResolver(Protocol):\n      """Map (spec, materialized bundle) to a feature batch.\n\n      Do NOT make this a generic Protocol[S, T] over two TypeVars.\n      Use @functools.singledispatch for subclass-specific implementations.\n      """\n      def __call__(self, spec: RunSpec, bundle: RuntimeBundle) -> FeatureBatch: ...\n\nFILE: xtrax/src/xtrax/run/sink.py\nContent:\n  from __future__ import annotations\n  from dataclasses import dataclass, field\n  from pathlib import Path\n  from typing import Literal\n\n  @dataclass\n  class SinkSpec:\n      """Routing config for output sinks."""\n      output_dir: Path | None = None\n      format: Literal["jsonl", "h5", "none"] = "jsonl"\n      flush_every: int = 1\n\nFILE: xtrax/src/xtrax/run/protocols.py\nContent:\n  from __future__ import annotations\n  from typing import Protocol, runtime_checkable\n\n  @runtime_checkable\n  class AggregationFn(Protocol):\n      def __call__(self, *args: object) -> object: ...\n\n  @runtime_checkable\n  class DecodeFn(Protocol):\n      def __call__(self, *args: object) -> object: ...\n\n  @runtime_checkable\n  class NoiseFn(Protocol):\n      def __call__(self, *args: object) -> object: ...\n\n--- PART 2: Update aminx RunSpec to extend xtrax.run.RunSpec ---\n\nFILE: aminx/src/aminx/run/spec.py\nCurrent line 114: class RunSpec(eqx.Module):\n\nChange ONLY the class definition line and add ONE import.\nAdd near the top of the file (with other xtrax imports or after existing imports):\n  from xtrax.run import RunSpec as _XtraxRunSpec\n\nChange line 114 from:\n  class RunSpec(eqx.Module):\nto:\n  class RunSpec(_XtraxRunSpec):\n\nNo other changes — the existing fields (io, resource, multistate, ligand, tied, grid, batching, averaging, precision, plan) remain exactly as-is.\neqx.Module inheritance is valid — subclasses simply add fields; eqx.tree_flatten recurses correctly.\n\n--- PART 3: Tests ---\n\nCreate xtrax/tests/run/test_spec.py:\n  from xtrax.run import RunSpec, InputResolver, RuntimeBundle, FeatureBatch, SinkSpec\n\n  def test_run_spec_constructs():\n      spec = RunSpec(seed=0, axes=[], carry_specs={}, boundaries=None)\n      assert spec.seed == 0\n\n  def test_sink_spec_defaults():\n      s = SinkSpec()\n      assert s.format == "jsonl"\n      assert s.flush_every == 1\n\n  def test_input_resolver_protocol():\n      class MyResolver:\n          def __call__(self, spec, bundle):\n              return FeatureBatch({})\n      assert isinstance(MyResolver(), InputResolver)\n\nRun: cd /home/marielle/projects/xtrax && uv run pytest tests/run/\n\nAlso run all aminx tests to confirm no regression:\n  cd /home/marielle/projects/aminx && uv run pytest\n\n--- PART 4: Commit ---\n\nFrom xtrax directory:\n  git add src/xtrax/run/\n  git commit -m "feat(RS-6): add xtrax.run module — RunSpec, InputResolver, RuntimeBundle, SinkSpec (#1922)"\n\nFrom aminx directory:\n  git add src/aminx/run/spec.py\n  git commit -m "feat(RS-6): aminx RunSpec extends xtrax.run.RunSpec (#1922)"\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Review R6-1: xtrax/run/ module creation and aminx RunSpec inheritance.\n\nVERIFY 1: uv run pytest passes in xtrax (cd /home/marielle/projects/xtrax && uv run pytest)\nVERIFY 2: uv run pytest passes in aminx (cd /home/marielle/projects/aminx && uv run pytest)\nVERIFY 3: from xtrax.run import RunSpec, InputResolver, RuntimeBundle, FeatureBatch, SinkSpec — all resolve without ImportError\nVERIFY 4: RunSpec(seed=0, axes=[], carry_specs={}, boundaries=None) constructs; eqx.tree_flatten does not raise\nVERIFY 5: InputResolver is a @runtime_checkable Protocol; its __call__ is (self, spec: RunSpec, bundle: RuntimeBundle) -> FeatureBatch; NOT a generic over 2 TypeVars\nVERIFY 6: issubclass(aminx_RunSpec, xtrax_RunSpec) is True (import both and check)\nVERIFY 7: The xtrax/run/ tests in tests/run/test_spec.py exist and pass\nVERIFY 8: aminx/src/aminx/types/stages.py is UNCHANGED (out of scope)\n\nPASS if all 8 VERIFY items are satisfied.\nFAIL if any item is not met.\n`,
  );

// ---- orchestrate: writing chain (C -> B -> A, sequential) ----
log("260619_rs6-foundation: writing chain (C -> B -> A, sequential)");
const c = await trackC();
const b = await trackB();
const a = await trackA();

return {
  task_id: TASK_ID,
  sprint_id: 2,
  verdicts: {
    "1926": c,
    "1925": b,
    "1922": a
  },
};
