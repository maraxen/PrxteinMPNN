// Sprint 1 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/260620_rs6-noise-fn.toml
// Regenerate: praxia dw emit-sprint 260620_rs6-noise-fn.toml
// task_id: 260620_rs6-noise-fn   sprint_id: 1
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (E,D,A,B,C) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.

export const meta = {
  name: "260620_rs6-noise-fn",
  description: "RS-6 continuation: FeatureNoiseBundle, Literal→Callable fn migration, double-fire guard, xtrax.run typing",
  phases: [
    { title: "Track E — R6-2 gate: noise field mapping table (#1953)" },
    { title: "Track D — Tighten xtrax.run typing: replace Any with concrete types (#1952)" },
    { title: "Track A — Fix _sync_run_spec double-fire via guard-flag (#1906)" },
    { title: "Track B — R6-2: RunSpecification logical grouping + FeatureNoiseBundle (#1923)" },
    { title: "Track C — R6-3: Literal→Callable fn migration on RunSpec subclasses (#1924)" },
  ],
};

const TASK_ID = "260620_rs6-noise-fn";
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

// Shared context for the writing tracks (from recon, task 260620_rs6-noise-fn).
const EMITTER_CTX = `RS-6 epic context:\n  Spec: .praxia/docs/specs/260615_runspec-xtrax-run-epic.md\n\nKey facts carried from sprint 260619:\n- xtrax.run/ now exists: RunSpec (eqx.Module), InputResolver, RuntimeBundle, SinkSpec, FeatureBatch\n- aminx RunSpec (eqx.Module) at aminx/src/aminx/run/spec.py now extends xtrax.run.RunSpec\n- RunSpecification is a plain @dataclass (NOT eqx.Module) at aminx/src/aminx/run/specs.py\n- FuseFn removed from xtrax __all__ (R6-4 done)\n- aminx/types/stages.py:68 FuseFn (logit transform protocol, bias param) is DISTINCT and always OUT OF SCOPE\n\nSprint 260620 scope:\n- Track E (#1953): Research gate — produce noise field mapping table (gates Track B)\n- Track D (#1952): Tighten xtrax/run/ typing (carry_specs: dict→list[CarrySpec], iterator: Any→union, FeatureBatch comment)\n- Track A (#1906): Fix _sync_run_spec double-fire via guard flag\n- Track B (#1923): R6-2 — FeatureNoiseBundle + logical grouping (depends on Track E gate doc)\n- Track C (#1924): R6-3 — Literal→Callable fn migration (depends on Track B)\n`;

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

// ===== TRACK E — Track E — R6-2 gate: noise field mapping table (#1953) =========================
const trackE = () =>
  agent(
    `task_id: ${TASK_ID}. Research task #1953: Produce the noise field mapping table that gates FeatureNoiseBundle implementation.\n\nThis is a GATE item — Track B (R6-2 FeatureNoiseBundle) cannot start code until you deliver this document.\n\nOUTPUT FILE: .praxia/docs/research/260616_noise-field-map.md\n\nSTEP 1 — Read the 8 noise fields from aminx/src/aminx/run/specs.py:\nRead the RunSpecification class definition. Find these 8 fields and note their types, defaults, and docstrings:\n  - backbone_noise: Sequence[float] | float (default (0.0,))\n  - backbone_noise_mode: Literal["direct", "thermal"] (default "direct")\n  - estat_noise: Sequence[float] | float | None (default None)\n  - estat_noise_mode: Literal["direct", "thermal"] (default "direct")\n  - vdw_noise: Sequence[float] | float | None (default None)\n  - vdw_noise_mode: Literal["direct", "thermal"] (default "direct")\n  - use_electrostatics: bool (default False)\n  - use_vdw: bool (default False)\n\nAlso check:\n  - aminx/src/aminx/run/specs.py __post_init__ (around line 228) to see how these are normalized\n  - grep for all usage sites: grep -rEn "backbone_noise|estat_noise|vdw_noise|use_electrostatics|use_vdw" aminx/src/\n\nSTEP 2 — Build the mapping table with this EXACT structure:\n\n| old field | type | default | role | FeatureNoiseBundle slot |\n|---|---|---|---|---|\n| backbone_noise | Sequence[float] | float | (0.0,) | backbone geometry perturbation | noise_levels |\n| backbone_noise_mode | Literal["direct","thermal"] | "direct" | thermal vs direct noise injection | mode |\n| estat_noise | Sequence[float] | float | None | None | electrostatic potential noise | noise_levels (if enabled) |\n| estat_noise_mode | Literal["direct","thermal"] | "direct" | ... | mode |\n| vdw_noise | Sequence[float] | float | None | None | van der Waals noise | noise_levels (if enabled) |\n| vdw_noise_mode | Literal["direct","thermal"] | "direct" | ... | mode |\n| use_electrostatics | bool | False | whether estat noise is active | feature_type discriminator |\n| use_vdw | bool | False | whether vdw noise is active | feature_type discriminator |\n\nAdjust the slot assignments based on what you read from the actual field docstrings and __post_init__ logic.\n\nSTEP 3 — Design FeatureNoiseBundle as a frozen dataclass:\n\nBased on the mapping, propose the FeatureNoiseBundle design:\n  - feature_type: Literal["backbone", "electrostatic", "vdw"]  ← discriminates the 3 noise domains\n  - noise_levels: tuple[float, ...]  ← replaces {backbone,estat,vdw}_noise (normalized to tuple in __post_init__)\n  - mode: Literal["direct", "thermal"]  ← replaces {backbone,estat,vdw}_noise_mode\n  - enabled: bool = True  ← replaces use_electrostatics / use_vdw\n\nWrite the proposed dataclass definition in the doc. Note that FeatureNoiseBundle is a frozen @dataclass\n(NOT eqx.Module) because it lives on RunSpecification (serializable plain @dataclass).\n\nSTEP 4 — Migration notes:\n\nAdd a section "## Migration notes" that covers:\n  - How to convert old RunSpecification construction to new \`noise=[...]\` construction\n  - How build_run_spec() should transform list[FeatureNoiseBundle] → the existing host/model inputs\n  - Whether use_electrostatics / use_vdw should become deprecated aliases (yes — keep them as init=True kwargs that auto-populate noise= list for backward compat)\n\nSTEP 5 — Commit:\n  git add .praxia/docs/research/260616_noise-field-map.md\n  git commit -m "research(R6-2): noise field mapping table + FeatureNoiseBundle design (#1953)"\n\nAlso update .praxia/docs/INDEX.md to add the new file under the Research section.\n`,
    { agentType: "librarian", label: "research:1953", phase: "Track E — R6-2 gate: noise field mapping table (#1953)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK D — Track D — Tighten xtrax.run typing: replace Any with concrete types (#1952) =========================
const trackD = () =>
  track(
    "1952",
    "Track D — Tighten xtrax.run typing: replace Any with concrete types (#1952)",
    `task_id: ${TASK_ID}. Fix #1952: Tighten the xtrax/run/ module to replace \`Any\` with concrete types.\n\nWORKING DIRECTORY: /home/marielle/projects/xtrax\n\nFiles to edit:\n  src/xtrax/run/spec.py\n  src/xtrax/run/resolver.py\n  aminx/src/aminx/run/spec.py (build_run_spec default fallback)\n\n--- FIX 1: RunSpec.carry_specs: dict[str, Any] → list[CarrySpec] ---\n\nFile: xtrax/src/xtrax/run/spec.py\n\nCurrent:\n  carry_specs: dict[str, Any] = eqx.field(default_factory=dict)\n\nChange to:\n  carry_specs: list[CarrySpec] = eqx.field(default_factory=list)\n\nAdd import at top (with other xtrax imports):\n  from xtrax.tiling import CarrySpec\n\nRationale: aminx RunSpecification.carry_specs is list[CarrySpec]; BatchPlanner.carry_specs is\nlist[CarrySpec] | None. dict[str, Any] was a placeholder — the domain type is list[CarrySpec].\n\nAlso update aminx/src/aminx/run/spec.py — in build_run_spec(), the fallback for carry_specs is\ncurrently getattr(spec, 'carry_specs', {}) — change {} to [] (empty list):\n  carry_specs=getattr(spec, 'carry_specs', []),\n\n--- FIX 2: RuntimeBundle.iterator: Any → concrete union ---\n\nFile: xtrax/src/xtrax/run/resolver.py\n\nThe xtrax iterator types (all from xtrax.tiling):\n  VmapIterator, SafeMapIterator, JaxScanIterator, BucketIterator, MapIterator, ScanIterator\n\nAdd import:\n  from xtrax.tiling import (\n      BucketIterator,\n      JaxScanIterator,\n      MapIterator,\n      SafeMapIterator,\n      ScanIterator,\n      VmapIterator,\n  )\n\nChange RuntimeBundle field:\n  iterator: Any\nto:\n  iterator: VmapIterator | SafeMapIterator | JaxScanIterator | BucketIterator | MapIterator | ScanIterator | None\n\n--- FIX 3: FeatureBatch — leave Any but add explanatory comment ---\n\nFile: xtrax/src/xtrax/run/resolver.py\n\nCurrent:\n  FeatureBatch = NewType("FeatureBatch", dict[str, Any])\n\nKeep as-is but add a one-line comment ABOVE the NewType line:\n  # Values are JAX arrays, numpy arrays, or scalars — heterogeneous, so Any is intentional.\n  FeatureBatch = NewType("FeatureBatch", dict[str, Any])\n\n--- VERIFY ---\n\nSTEP 1 — Run xtrax tests:\n  cd /home/marielle/projects/xtrax && uv run pytest\n\nSTEP 2 — Run aminx tests:\n  cd /home/marielle/projects/aminx && uv run pytest\n\nSTEP 3 — Smoke-check the types resolve:\n  cd /home/marielle/projects/xtrax && uv run python -c "\n  from xtrax.run import RunSpec, RuntimeBundle\n  from xtrax.tiling import CarrySpec, VmapIterator, SafeMapIterator\n  spec = RunSpec(seed=0, axes=[], carry_specs=[], boundaries=None)\n  print('carry_specs type:', type(spec.carry_specs))\n  assert isinstance(spec.carry_specs, list), 'carry_specs must be list'\n  print('OK — carry_specs is list[CarrySpec]')\n  "\n\n--- COMMIT ---\n\nFrom xtrax directory:\n  git add src/xtrax/run/spec.py src/xtrax/run/resolver.py\n  git commit -m "refactor(RS-6): tighten xtrax.run typing — carry_specs→list[CarrySpec], iterator→union, FeatureBatch comment (#1952)"\n\nFrom aminx directory:\n  git add src/aminx/run/spec.py\n  git commit -m "fix(RS-6): update build_run_spec carry_specs fallback to [] (matches list[CarrySpec]) (#1952)"\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Review #1952: xtrax.run typing tightening.\n\nVERIFY 1: xtrax/src/xtrax/run/spec.py — carry_specs field is list[CarrySpec] (not dict[str, Any]); imports CarrySpec from xtrax.tiling\nVERIFY 2: xtrax/src/xtrax/run/resolver.py — RuntimeBundle.iterator is the concrete iterator union (VmapIterator | SafeMapIterator | JaxScanIterator | BucketIterator | MapIterator | ScanIterator | None), NOT Any\nVERIFY 3: xtrax/src/xtrax/run/resolver.py — FeatureBatch stays dict[str, Any] NewType but has an explanatory comment above it\nVERIFY 4: aminx/src/aminx/run/spec.py — build_run_spec() uses getattr(spec, 'carry_specs', []) (empty list, not empty dict)\nVERIFY 5: uv run pytest passes in xtrax (cd /home/marielle/projects/xtrax && uv run pytest)\nVERIFY 6: uv run pytest passes in aminx (cd /home/marielle/projects/aminx && uv run pytest)\nVERIFY 7: RunSpec(seed=0, axes=[], carry_specs=[], boundaries=None) constructs without error\n\nPASS if all 7 VERIFY items are satisfied.\nFAIL if Any remains in carry_specs or iterator without a comment explaining why.\n`,
  );

// ===== TRACK A — Track A — Fix _sync_run_spec double-fire via guard-flag (#1906) =========================
const trackA = () =>
  track(
    "1906",
    "Track A — Fix _sync_run_spec double-fire via guard-flag (#1906)",
    `task_id: ${TASK_ID}. Fix #1906: RunSpecification._sync_run_spec fires twice per construction due to both the base\nclass __post_init__ and each subclass __post_init__ calling it.\n\nFILE: aminx/src/aminx/run/specs.py\n\nSTEP 1 — Recon: understand the current call pattern.\nRead specs.py and find:\n  - class RunSpecification (base dataclass) — its __post_init__ and _sync_run_spec method\n  - class ScoringSpecification, SamplingSpecification, JacobianSpecification, InspectionSpecification\n  - For each subclass: does its __post_init__ call both super().__post_init__() AND _sync_run_spec?\n\nThe expected pattern you will find:\n  RunSpecification.__post_init__ (around line 228) calls self._sync_run_spec()\n  Each subclass __post_init__ calls super().__post_init__() then self._sync_run_spec() again\n  → build_run_spec fires twice per construction\n\nSTEP 2 — Add the guard flag to RunSpecification.\n\nIn the RunSpecification dataclass, add ONE new field (after the last user-visible field, before\nany field(init=False) fields that exist):\n  _run_spec_synced: bool = field(init=False, default=False)\n\nSTEP 3 — Modify _sync_run_spec to short-circuit.\n\nFind the _sync_run_spec method in RunSpecification. Wrap its body:\n\n  def _sync_run_spec(self) -> None:\n      if self._run_spec_synced:\n          return\n      object.__setattr__(self, "_run_spec_synced", True)\n      # ... existing body unchanged ...\n\nSTEP 4 — Reset the flag at the start of each subclass __post_init__.\n\nFor EACH of the 4 subclasses (ScoringSpecification, SamplingSpecification,\nJacobianSpecification, InspectionSpecification), add ONE line at the very start of their\n__post_init__, BEFORE the super().__post_init__() call:\n\n  def __post_init__(self) -> None:\n      object.__setattr__(self, "_run_spec_synced", False)  # reset so base fires, then we skip\n      super().__post_init__()\n      # ... remaining subclass body unchanged ...\n\nThis ensures: subclass resets → super().__post_init__() fires _sync_run_spec (flag → True)\n→ subclass finishes → if subclass called _sync_run_spec directly, it short-circuits.\n\nSTEP 5 — Write a test.\n\nCreate (or add to) aminx/tests/run/test_sync_run_spec.py:\n\n  from unittest.mock import patch\n  from aminx.run.specs import SamplingSpecification\n\n  def test_sync_run_spec_fires_once():\n      with patch("aminx.run.specs.build_run_spec") as mock_build:\n          mock_build.return_value = None\n          SamplingSpecification(inputs=["dummy.pdb"], num_samples=4)\n      assert mock_build.call_count == 1, f"Expected 1 call, got {mock_build.call_count}"\n\n  def test_run_spec_populated_after_construction():\n      spec = SamplingSpecification(inputs=["dummy.pdb"], num_samples=4)\n      # run_spec attribute must exist and not raise AttributeError\n      assert hasattr(spec, "run_spec")\n\nRun: cd /home/marielle/projects/aminx && uv run pytest tests/run/test_sync_run_spec.py -v\n\nSTEP 6 — Run full aminx test suite:\n  cd /home/marielle/projects/aminx && uv run pytest\n\nSTEP 7 — Commit:\n  git add src/aminx/run/specs.py tests/run/test_sync_run_spec.py\n  git commit -m "fix(RS): guard _sync_run_spec double-fire with _run_spec_synced flag (#1906)"\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Review #1906: _sync_run_spec double-fire guard.\n\nVERIFY 1: RunSpecification has a _run_spec_synced: bool = field(init=False, default=False) field\nVERIFY 2: _sync_run_spec short-circuits (returns early) if _run_spec_synced is True; sets it True before running\nVERIFY 3: All 4 subclasses (Scoring, Sampling, Jacobian, Inspection) reset _run_spec_synced to False at the start of their __post_init__ before calling super()\nVERIFY 4: Test test_sync_run_spec_fires_once passes (mock_build.call_count == 1)\nVERIFY 5: uv run pytest passes in aminx (cd /home/marielle/projects/aminx && uv run pytest)\n\nPASS if all 5 VERIFY items are satisfied.\nFAIL if double-fire is still possible (i.e., any subclass __post_init__ can cause two build_run_spec calls).\n`,
  );

// ===== TRACK B — Track B — R6-2: RunSpecification logical grouping + FeatureNoiseBundle (#1923) =========================
const trackB = () =>
  track(
    "1923",
    "Track B — R6-2: RunSpecification logical grouping + FeatureNoiseBundle (#1923)",
    `task_id: ${TASK_ID}. Fix #1923: Replace 8 scattered noise fields on RunSpecification with a structured FeatureNoiseBundle list.\n\nPREREQUISITE: Track E (R6-2 gate) must be complete. Read .praxia/docs/research/260616_noise-field-map.md\nfor the definitive FeatureNoiseBundle design BEFORE writing any code. Use the dataclass definition\nfrom that document exactly.\n\nFILE: aminx/src/aminx/run/specs.py\n\n--- PART 1: Define FeatureNoiseBundle ---\n\nAdd near the top of specs.py (before RunSpecification), as a frozen @dataclass:\n(Use the exact fields from .praxia/docs/research/260616_noise-field-map.md — expected shape:)\n\n  @dataclass(frozen=True)\n  class FeatureNoiseBundle:\n      feature_type: Literal["backbone", "electrostatic", "vdw"]\n      noise_levels: tuple[float, ...]\n      mode: Literal["direct", "thermal"] = "direct"\n      enabled: bool = True\n\n(Adjust to match the research doc if it differs.)\n\nAlso add to aminx/src/aminx/run/__init__.py exports (or aminx's public __init__.py wherever\nRunSpecification is exported):\n  from aminx.run.specs import FeatureNoiseBundle\n\n--- PART 2: Replace the 8 noise fields ---\n\nIn RunSpecification, remove these 8 fields:\n  backbone_noise, backbone_noise_mode, estat_noise, estat_noise_mode,\n  vdw_noise, vdw_noise_mode, use_electrostatics, use_vdw\n\nReplace with ONE field:\n  noise: list[FeatureNoiseBundle] = field(default_factory=list)\n\nDefault behavior: empty list means no feature noise applied.\n\nFor backward compatibility, add deprecated init=False aliases in __post_init__:\n  # Deprecated single-field accessors — set from noise list for backward compat\n  object.__setattr__(self, "backbone_noise", _extract_noise_levels(self.noise, "backbone"))\n  object.__setattr__(self, "use_electrostatics", _has_enabled(self.noise, "electrostatic"))\n  object.__setattr__(self, "use_vdw", _has_enabled(self.noise, "vdw"))\n\nAdd helper functions (module-level, not class methods):\n  def _extract_noise_levels(bundles, ftype):\n      for b in bundles:\n          if b.feature_type == ftype and b.enabled:\n              return b.noise_levels\n      return (0.0,)\n\n  def _has_enabled(bundles, ftype):\n      return any(b.feature_type == ftype and b.enabled for b in bundles)\n\n--- PART 3: Add sub-config dataclasses ---\n\nAdd these two frozen dataclasses before RunSpecification:\n\n  @dataclass(frozen=True)\n  class ModelConfig:\n      precision: str = "float32"         # moves aminx RunSpec precision field (bridge only; keep existing field too)\n\n  @dataclass(frozen=True)\n  class ResourceConfig:\n      num_devices: int = 1\n      memory_fraction: float = 1.0\n\nAdd optional fields to RunSpecification (do NOT remove existing precision/resource fields yet — bridge only):\n  model_config: ModelConfig | None = None\n  resource_config: ResourceConfig | None = None\n\n--- PART 4: SinkConfig bridge ---\n\nAdd:\n  @dataclass(frozen=True)\n  class SinkConfig:\n      output_path: Path | None = None\n      format: Literal["jsonl", "h5", "none"] = "jsonl"\n      flush_every: int = 1\n\nAdd to each subclass that has output_h5_path:\n  sink: SinkConfig | None = None\n\nIn __post_init__ of those subclasses, populate from output_h5_path if sink is None:\n  if self.sink is None and self.output_h5_path is not None:\n      object.__setattr__(self, "sink", SinkConfig(output_path=Path(self.output_h5_path), format="h5"))\n\n(Check which subclasses have output_h5_path by reading specs.py.)\n\n--- PART 5: Tests ---\n\nCreate aminx/tests/run/test_noise_bundle.py:\n\n  from aminx.run.specs import FeatureNoiseBundle, RunSpecification\n\n  def test_feature_noise_bundle_constructs():\n      b = FeatureNoiseBundle(feature_type="backbone", noise_levels=(0.1, 0.2))\n      assert b.feature_type == "backbone"\n      assert b.noise_levels == (0.1, 0.2)\n      assert b.mode == "direct"\n      assert b.enabled is True\n\n  def test_run_spec_noise_defaults_empty():\n      # Default construction has no noise bundles\n      spec = RunSpecification(inputs=["dummy.pdb"])\n      assert spec.noise == []\n\n  def test_run_spec_backbone_noise_bundle():\n      b = FeatureNoiseBundle(feature_type="backbone", noise_levels=(0.2, 0.5))\n      spec = RunSpecification(inputs=["dummy.pdb"], noise=[b])\n      # backward-compat accessor\n      assert spec.backbone_noise == (0.2, 0.5)\n\nRun: cd /home/marielle/projects/aminx && uv run pytest tests/run/test_noise_bundle.py -v\n\nAlso run full test suite:\n  cd /home/marielle/projects/aminx && uv run pytest\n\n--- PART 6: Commit ---\n  git add src/aminx/run/specs.py tests/run/test_noise_bundle.py\n  git commit -m "feat(R6-2): replace 8 noise fields with FeatureNoiseBundle list; add ModelConfig/ResourceConfig/SinkConfig (#1923)"\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Review R6-2: FeatureNoiseBundle and RunSpecification logical grouping.\n\nVERIFY 1: FeatureNoiseBundle is a frozen @dataclass with at least feature_type, noise_levels, mode, enabled fields\nVERIFY 2: RunSpecification.noise: list[FeatureNoiseBundle] field exists; all 8 old noise fields are REMOVED from the field declarations\nVERIFY 3: Backward-compat accessor for backbone_noise (and use_electrostatics / use_vdw) is populated in __post_init__ from the noise list\nVERIFY 4: ModelConfig and ResourceConfig dataclasses exist and are added as optional fields to RunSpecification\nVERIFY 5: SinkConfig dataclass exists; relevant subclasses have a sink: SinkConfig | None field\nVERIFY 6: test_noise_bundle.py exists and all tests pass\nVERIFY 7: uv run pytest passes in aminx (cd /home/marielle/projects/aminx && uv run pytest)\n\nPASS if all 7 VERIFY items are satisfied.\nFAIL if any of the 8 old noise fields remain as first-class fields (they may exist as computed attributes but not as __init__ parameters).\n`,
  );

// ===== TRACK C — Track C — R6-3: Literal→Callable fn migration on RunSpec subclasses (#1924) =========================
const trackC = () =>
  track(
    "1924",
    "Track C — R6-3: Literal→Callable fn migration on RunSpec subclasses (#1924)",
    `task_id: ${TASK_ID}. Fix #1924: Replace Literal-typed averaging and decode fields on RunSpec subclasses with typed callables.\n\nPREREQUISITE: Track B (R6-2) must be complete — this touches the same subclasses.\n\nFILE: aminx/src/aminx/run/specs.py\nAlso: xtrax/src/xtrax/run/spec.py (add from_spec classmethod)\n\nSTEP 1 — Read the current averaging and decode field definitions:\n\nGrep: grep -En "average_encoding_mode|decode_fn|AveragingMode|AggregationFn" aminx/src/aminx/run/specs.py\n\nYou will find:\n  - ScoringSpecification.average_encoding_mode: Literal["inputs", "noise_levels", "inputs_and_noise"] = "inputs_and_noise" (~line 287)\n  - SamplingSpecification.average_encoding_mode (~line 331) and decode_fn: Any | None (~line 346)\n  - JacobianSpecification.average_encoding_mode (~line 424)\n\nSTEP 2 — Define AveragingMode enum + to_fn() shim:\n\nAdd near the top of specs.py (before RunSpecification):\n\n  import jax.numpy as jnp\n  from enum import Enum\n  from typing import Callable\n  import jax\n\n  class AveragingMode(Enum):\n      INPUTS = "inputs"\n      NOISE_LEVELS = "noise_levels"\n      INPUTS_AND_NOISE = "inputs_and_noise"\n\n      def to_fn(self) -> Callable:\n          if self == AveragingMode.INPUTS:\n              return lambda encodings, _: jnp.mean(encodings, axis=0)\n          elif self == AveragingMode.NOISE_LEVELS:\n              return lambda _, noised: jnp.mean(noised, axis=0)\n          else:  # INPUTS_AND_NOISE\n              return lambda encodings, noised: jnp.mean(\n                  jnp.concatenate([encodings, noised], axis=0), axis=0\n              )\n\nSTEP 3 — Define AggregationFn and DecodeFn protocols:\n\nSCOPE NOTE: Do NOT use AggregationFn or DecodeFn from xtrax/run/protocols.py — that file was\nDELETED in sprint 260619 (auditor cleanup). Define them locally in specs.py or import from\na new aminx/run/protocols.py if preferred.\n\n  from typing import Protocol, runtime_checkable\n\n  @runtime_checkable\n  class AggregationFn(Protocol):\n      def __call__(self, encodings: Any, noised: Any) -> Any: ...\n\n  @runtime_checkable\n  class DecodeFn(Protocol):\n      def __call__(self, output: Any) -> Any: ...\n\nSTEP 4 — Migrate average_encoding_mode in all 3 subclasses:\n\nFor ScoringSpecification, SamplingSpecification, JacobianSpecification:\n\nRemove:\n  average_encoding_mode: Literal["inputs", "noise_levels", "inputs_and_noise"] = "inputs_and_noise"\n\nAdd:\n  encoding_aggregation_fn: AggregationFn = field(\n      default_factory=lambda: AveragingMode.INPUTS_AND_NOISE.to_fn()\n  )\n\nFor backward compat, also add in each subclass __post_init__:\n  # Deprecated: average_encoding_mode as Literal → encoding_aggregation_fn\n  # Accept average_encoding_mode kwarg via __init_subclass__ shim — NOT needed here;\n  # if callers pass average_encoding_mode=..., add a deprecated classmethod:\n\nAdd a classmethod on RunSpecification (or as a module-level helper):\n  @classmethod\n  def with_averaging_mode(cls, mode: str | AveragingMode, **kwargs):\n      if isinstance(mode, str):\n          mode = AveragingMode(mode)\n      return cls(encoding_aggregation_fn=mode.to_fn(), **kwargs)\n\nSTEP 5 — Migrate decode_fn in SamplingSpecification:\n\nRemove:\n  decode_fn: Any | None = None\n\nAdd:\n  decode_fn: DecodeFn | None = None\n\n(Same field name, just tightened type — no logic change needed.)\n\nSTEP 6 — Add RunSpec.from_spec classmethod:\n\nIn xtrax/src/xtrax/run/spec.py, add:\n\n  @classmethod\n  def from_spec(cls, spec: "RunSpec") -> "RunSpec":\n      # Identity for already-built RunSpec; subclasses override to build from RunSpecification.\n      return spec\n\n(This is the base version — aminx subclass will override this in aminx/run/spec.py in a future sprint.)\n\nSTEP 7 — Tests:\n\nCreate aminx/tests/run/test_fn_migration.py:\n\n  from aminx.run.specs import AveragingMode, ScoringSpecification, SamplingSpecification\n\n  def test_averaging_mode_enum():\n      fn = AveragingMode.INPUTS_AND_NOISE.to_fn()\n      import jax.numpy as jnp\n      x = jnp.ones((4, 8))\n      result = fn(x, x)\n      assert result.shape == (8,)\n\n  def test_scoring_spec_has_aggregation_fn():\n      spec = ScoringSpecification(inputs=["dummy.pdb"])\n      assert callable(spec.encoding_aggregation_fn)\n\n  def test_sampling_spec_decode_fn_typed():\n      spec = SamplingSpecification(inputs=["dummy.pdb"], num_samples=4)\n      assert spec.decode_fn is None  # default\n\n  def test_with_averaging_mode_classmethod():\n      spec = ScoringSpecification.with_averaging_mode("inputs", inputs=["dummy.pdb"])\n      assert callable(spec.encoding_aggregation_fn)\n\nRun: cd /home/marielle/projects/aminx && uv run pytest tests/run/test_fn_migration.py -v\n\nAlso run full test suite:\n  cd /home/marielle/projects/aminx && uv run pytest\n\nSTEP 8 — Commits:\n\nFrom aminx directory:\n  git add src/aminx/run/specs.py tests/run/test_fn_migration.py\n  git commit -m "feat(R6-3): Literal→Callable fn migration — AveragingMode enum, AggregationFn/DecodeFn protocols, encoding_aggregation_fn field (#1924)"\n\nFrom xtrax directory:\n  git add src/xtrax/run/spec.py\n  git commit -m "feat(R6-3): add RunSpec.from_spec() identity classmethod (#1924)"\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Review R6-3: Literal→Callable fn migration.\n\nVERIFY 1: AveragingMode enum exists in specs.py with INPUTS, NOISE_LEVELS, INPUTS_AND_NOISE values and a to_fn() method\nVERIFY 2: ScoringSpecification, SamplingSpecification, JacobianSpecification all have encoding_aggregation_fn: AggregationFn (NOT average_encoding_mode: Literal)\nVERIFY 3: SamplingSpecification.decode_fn is DecodeFn | None (NOT Any | None)\nVERIFY 4: AggregationFn and DecodeFn are @runtime_checkable Protocols (NOT imported from xtrax.run.protocols — that file was deleted)\nVERIFY 5: RunSpec.from_spec() classmethod exists in xtrax/src/xtrax/run/spec.py\nVERIFY 6: test_fn_migration.py exists and all tests pass\nVERIFY 7: uv run pytest passes in aminx (cd /home/marielle/projects/aminx && uv run pytest)\nVERIFY 8: uv run pytest passes in xtrax (cd /home/marielle/projects/xtrax && uv run pytest)\n\nPASS if all 8 VERIFY items are satisfied.\nFAIL if average_encoding_mode: Literal remains as a first-class field on any subclass, or if protocols.py is imported (it was deleted).\n`,
  );

// ---- orchestrate: writing chain (E -> D -> A -> B -> C, sequential) ----
log("260620_rs6-noise-fn: writing chain (E -> D -> A -> B -> C, sequential)");
const e = await trackE();
const d = await trackD();
const a = await trackA();
const b = await trackB();
const c = await trackC();

return {
  task_id: TASK_ID,
  sprint_id: 1,
  verdicts: {
    "1953": e,
    "1952": d,
    "1906": a,
    "1923": b,
    "1924": c
  },
};
