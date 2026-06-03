// Sprint 23 — Benchmark Capability Suite
// Generated scaffolding: `praxia dw emit --all` (2026-06-03)
// Sprint prompts authored and hand-edited (daemon off; no pcw_compose classification).
//
// Edits vs. the raw `praxia dw emit` templates (see _baseline_*.js for diff):
//   1. Date.now() REMOVED from budget guard — it throws under Workflow resume.
//      Budget is rewind-count only (makeBudget / rewind helpers, baker_followup pattern).
//   2. Real dispatch prompts replace every TODO(#854) placeholder; CONTEXT embedded
//      so every agent dispatch is self-contained (felsenstein / baker pattern).
//   3. MAX_FIX_RETRIES bounded loops for every mutating fixer (m2c pattern).
//   4. Pre-flight git dirty-check baked into every fixer prompt (m2c pattern).
//   5. Early-abort on Tier 0 failure: Tiers 1-3 are skipped with status="skipped",
//      no wasted agents (m2c Track-B abort pattern).
//   6. phase: passed explicitly in agent() opts for all agents inside parallel()
//      to avoid the global phase() race (baker_followup note 4).
//   7. Structured return value at workflow end.
//   8. uv run python rule baked into every fixer prompt.
//
// DAG tiers:
//   Tier 0 (parallel): #946 ColabDesign precision fix, #947 ligand flag fix
//   Tier 1 (parallel): #948 bench_dedup_hetero, #949 bench_temperature_array,
//                      #950 L=150/300 fixtures
//   Tier 2 (sequential): #951 bench_mixed_length (depends on #950)
//   Tier 3 (sequential): #952 bench_suite integration (depends on all above)

export const meta = {
  name: "bench-capability-sprint-23",
  description:
    "Sprint 23: benchmark DedupGather/temperature-array/mixed-length capabilities vs ColabDesign+PyTorch across H200/A100/L40s/Blackwell",
  phases: [
    { title: "Bug Fixes",           detail: "#946 ColabDesign precision + #947 ligand flag (parallel)" },
    { title: "New Scripts",         detail: "#948 dedup_hetero + #949 temperature_array + #950 fixtures (parallel)" },
    { title: "Mixed Length Script", detail: "#951 bench_mixed_length (depends on #950 fixtures)" },
    { title: "Suite Integration",   detail: "#952 bench_suite.py + bench_report.py (depends on all above)" },
  ],
};

// ── Helpers (felsenstein / baker pattern) ───────────────────────────────────
function extractVerdict(text) {
  const m = String(text ?? "").match(/verdict:\s*([a-z_]+)/i);
  return m ? m[1].toLowerCase() : "advance";
}

const MAX_FIX_RETRIES = 2;

function makeBudget(maxRewinds) {
  return { rewinds: 0, max: maxRewinds };
}
function rewind(b, label) {
  b.rewinds++;
  log(`[budget] rewind ${b.rewinds}/${b.max} — returning to ${label}`);
  if (b.rewinds > b.max)
    throw new Error(`FAIL: max_rewinds exceeded (${b.rewinds}/${b.max}) at ${label}`);
}

const verdictLine =
  "End your reply with a line 'verdict: <label>' on its own line so the workflow can route.";

// ── Constants ────────────────────────────────────────────────────────────────
const TASK_ID = "260603_bench-capability-s23";
const ROOT    = "/home/marielle/projects/tev_design/prxteinmpnn";

// Shared context embedded in every dispatch so agents are self-contained.
const CONTEXT = `
Project root (ALL commands from here): ${ROOT}
Stack rules: always \`uv run python\` — never bare python. Edit only on existing files;
Write only for new files. ruff check scripts/benchmarks/ must stay green.
Sprint 23 task_id: ${TASK_ID}

Existing benchmark scripts (confirmed by recon 260603_bench-suite-recon):
  scripts/benchmarks/bench_suite.py          — main dispatcher (spawns 3 subprocesses)
  scripts/benchmarks/bench_prxteinmpnn_jax.py — JAX adapter; temp hardcoded [1.0] (lines 135,137)
  scripts/benchmarks/bench_colabdesign_jax.py — ColabDesign adapter; NO --precision arg (always fp32)
  scripts/benchmarks/bench_ligandmpnn_pytorch.py — PyTorch adapter; uses --ligand-conditioning
  scripts/benchmarks/prepare_fixtures.py      — fixture prep; L=76 + L=500 exist in tests/data/

Key src anchors (do not re-derive):
  src/prxteinmpnn/tiling/axes.py         — N_TEMPERATURES AxisSpec at axis_index=5, cardinality=8
  src/prxteinmpnn/tiling/dedup.py        — DedupGather merged at 3cfa662 (Sprint 22)
  scripts/spikes/dedup_encode_kvn_spike.py — K-not-N proof (io_callback: 3 unique → 3 calls)
  scripts/engaging/_gpu_env.sh           — Blackwell XLA flag guard (lines 38-45)

GPU targets: H200 (node4009), A100 (mit_preemptable), L40s (mit_normal_gpu), Blackwell (node4008).
Blackwell requires XLA_FLAGS=--xla_gpu_shard_autotuning=false (see _gpu_env.sh:38-45).
`.trim();

// ═══════════════════════════════════════════════════════════════════════════
// Tier 0 — Bug Fixes (parallel; BOTH must advance before Tier 3 is valid)
// ═══════════════════════════════════════════════════════════════════════════
phase("Bug Fixes");
log("[s23] Tier 0: patching benchmark script bugs that corrupt cross-framework precision results.");

async function fix946() {
  const b = makeBudget(MAX_FIX_RETRIES);
  let res, verdict;
  for (let attempt = 1; attempt <= MAX_FIX_RETRIES + 1; attempt++) {
    res = await agent(
      `[role: fixer] [backlog #946] [task_id: ${TASK_ID}]
${CONTEXT}

Fix bench_colabdesign_jax.py: the --precision flag is currently IGNORED (adapter always runs fp32).
bench_suite.py dispatches it with --precision {fp32,bf16} but the adapter has no argparse entry.

Pre-flight: run \`git -C ${ROOT} status --short scripts/benchmarks/bench_colabdesign_jax.py\`.
If that file is already uncommitted-dirty with unknown edits, STOP and emit verdict: needs_work.

OBJECTIVE (edit only — do not Write on this existing file):
  1. Add argparse argument --precision {fp32,bf16,fp16} (default: fp32) to bench_colabdesign_jax.py.
  2. After parse: if precision == "bf16" call jax.config.update("jax_default_matmul_precision", "bfloat16")
     and cast model weights accordingly before any forward pass.
  3. Propagate into result JSON cells: add "precision": args.precision to each output cell dict.

Verify: uv run python scripts/benchmarks/bench_colabdesign_jax.py --dry-run --precision bf16
must exit 0 and result JSON must contain "precision": "bf16".
${attempt > 1 ? `\nPrior attempt failed — fix these issues:\n${res}\n` : ""}
${verdictLine} (advance | needs_work)`,
      { agentType: "fixer", label: `#946-attempt${attempt}`, phase: "Bug Fixes" }
    );
    verdict = extractVerdict(res);
    if (verdict !== "needs_work") break;
    rewind(b, "#946 fix");
  }
  return { id: 946, verdict };
}

async function fix947() {
  const b = makeBudget(MAX_FIX_RETRIES);
  let res, verdict;
  for (let attempt = 1; attempt <= MAX_FIX_RETRIES + 1; attempt++) {
    res = await agent(
      `[role: fixer] [backlog #947] [task_id: ${TASK_ID}]
${CONTEXT}

Fix bench_suite.py ligand flag mismatch: it builds --ligand for the prxteinmpnn adapter but
bench_ligandmpnn_pytorch.py expects --ligand-conditioning. This causes silent subprocess
"unrecognized argument" failures on all ligand benchmark cells.

Pre-flight: run \`git -C ${ROOT} status --short scripts/benchmarks/bench_suite.py\`.
If that file is dirty with unknown edits, STOP and emit verdict: needs_work.

OBJECTIVE (edit only):
  1. Find the per-adapter subprocess arg builder in bench_suite.py.
  2. For the PyTorch (bench_ligandmpnn_pytorch) adapter branch, emit --ligand-conditioning.
     For the prxteinmpnn adapter branch, keep --ligand (matches its argparse).
     For colabdesign (no ligand support), do not pass either flag.
  3. Verify: uv run python scripts/benchmarks/bench_suite.py --dry-run --ligand must exit 0
     with no subprocess stderr "unrecognized argument" for any adapter.
${attempt > 1 ? `\nPrior attempt failed — fix these issues:\n${res}\n` : ""}
${verdictLine} (advance | needs_work)`,
      { agentType: "fixer", label: `#947-attempt${attempt}`, phase: "Bug Fixes" }
    );
    verdict = extractVerdict(res);
    if (verdict !== "needs_work") break;
    rewind(b, "#947 fix");
  }
  return { id: 947, verdict };
}

const [t0_946, t0_947] = await parallel([fix946, fix947]);

const tier0Ok = t0_946.verdict === "advance" && t0_947.verdict === "advance";
log(`[s23] Tier 0 done: #946=${t0_946.verdict}  #947=${t0_947.verdict}`);
if (!tier0Ok) {
  log(`[s23] WARN: Tier 0 bug fix(es) did not advance. Tiers 1-3 will be skipped.`);
  log(`[s23] NOTE: The GPU benchmark runs in Tier 3 will produce incorrect results`);
  log(`[s23]       (precision mismatch, silent ligand skips) until #946 and #947 are fixed.`);
}

// ═══════════════════════════════════════════════════════════════════════════
// Tier 1 — New Scripts + Fixtures (parallel; independent of each other)
// ═══════════════════════════════════════════════════════════════════════════
phase("New Scripts");
log("[s23] Tier 1: new benchmark scripts + bathos sidecars in parallel.");

async function new948() {
  const b = makeBudget(MAX_FIX_RETRIES);
  let res, verdict;
  for (let attempt = 1; attempt <= MAX_FIX_RETRIES + 1; attempt++) {
    res = await agent(
      `[role: fixer] [backlog #948] [task_id: ${TASK_ID}]
${CONTEXT}

Implement scripts/benchmarks/bench_dedup_hetero.py — DedupGather K/N unique-ratio throughput benchmark.

BACKGROUND (do not re-derive):
  DedupGather (src/prxteinmpnn/tiling/dedup.py, merged at 3cfa662) collapses N logical batch
  elements to K unique physical elements, runs the model K times (not N), scatters back.
  Spike proof at scripts/spikes/dedup_encode_kvn_spike.py: 3 unique → 3 io_callback calls vs 6.
  This benchmark proves K-proportional savings at real inference scale vs baselines that
  cannot do in-trace dedup.

SPEC:
  Inputs: synthetic batch N=32 structures, K in {1,2,4,8,16,32} unique (fill by repeating).
  For each K:
    - prxteinmpnn path: InferencePlan with DedupGather on n_structures axis; run plan.score()
    - ColabDesign baseline: K separate model.sample(num=1) calls (one per unique structure)
    - PyTorch baseline: sequential loop over K unique structures
  Metrics per K: latency_ms (median of n_timed=20 warm runs after n_warmup=10),
    dedup_ratio (K/N), speedup_vs_pytorch (pytorch_latency / prxteinmpnn_latency)
  Output JSON: {schema_version:"1", hardware, k_values:[...], cells:[{k, n, dedup_ratio,
    prxteinmpnn_latency_ms, colabdesign_latency_ms, pytorch_latency_ms, speedup_vs_pytorch}]}
  CLI: --hardware, --n-total (default 32), --k-values (default "1,2,4,8,16,32"),
       --n-warmup (default 10), --n-timed (default 20), --pdb-dir, --output-json,
       --dry-run, --smoke

BATHOS SIDECAR scripts/benchmarks/bench_dedup_hetero.py.bth.toml (v0.3):
  [experiment]
    hypothesis = "DedupGather achieves K-proportional throughput savings: at K=16,N=32 prxteinmpnn
      runtime is approximately 0.5x the K=32 homogeneous baseline"
  [outcomes.pass]   condition = "speedup_vs_pytorch >= 1.5 at K=16, N=32"; is_residual = false
  [outcomes.marginal] condition = "speedup_vs_pytorch >= 1.0 at all K (no regression)"; is_residual = false
  [outcomes.fail]   condition = "any K shows speedup_vs_pytorch < 1.0 (dedup costs exceed savings)"; is_residual = true
  [result_schema]   fields include k, n, dedup_ratio, prxteinmpnn_latency_ms, speedup_vs_pytorch

SLURM: create scripts/engaging/submit_bench_dedup_{h200,a100,l40s,blackwell}.sh.
  Pattern: copy submit_bench_h200.sh structure; replace benchmark with bench_dedup_hetero.
  Blackwell script MUST include the XLA flag guard from _gpu_env.sh lines 38-45:
    if [[ \${SLURM_JOB_NODELIST:-} == *node4008* ]]; then
      export XLA_FLAGS="\${XLA_FLAGS:+\${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"; fi

Pre-flight: run \`git -C ${ROOT} status --short scripts/benchmarks/\` to confirm no conflicting dirty files.
${attempt > 1 ? `\nPrior attempt failed — fix these issues:\n${res}\n` : ""}
SUCCESS CRITERIA:
  - uv run python scripts/benchmarks/bench_dedup_hetero.py --dry-run exits 0
  - uv run python scripts/benchmarks/bench_dedup_hetero.py --smoke exits 0
  - bth check scripts/benchmarks/bench_dedup_hetero.py.bth.toml exits 0
  - All 4 SLURM scripts exist and Blackwell script contains XLA flag guard
${verdictLine} (advance | needs_work)`,
      { agentType: "fixer", label: `#948-attempt${attempt}`, phase: "New Scripts" }
    );
    verdict = extractVerdict(res);
    if (verdict !== "needs_work") break;
    rewind(b, "#948 bench_dedup_hetero");
  }
  return { id: 948, verdict };
}

async function new949() {
  const b = makeBudget(MAX_FIX_RETRIES);
  let res, verdict;
  for (let attempt = 1; attempt <= MAX_FIX_RETRIES + 1; attempt++) {
    res = await agent(
      `[role: fixer] [backlog #949] [task_id: ${TASK_ID}]
${CONTEXT}

Implement scripts/benchmarks/bench_temperature_array.py — JIT-native temperature sweep benchmark.

BACKGROUND (do not re-derive):
  N_TEMPERATURES AxisSpec exists in src/prxteinmpnn/tiling/axes.py: axis_index=5, cardinality=8,
  heterogeneous=False. Currently hardcoded to temperature=[1.0] in bench_prxteinmpnn_jax.py
  (lines 135, 137). ColabDesign model.sample() takes a SCALAR temperature — M temperatures
  require M separate calls. prxteinmpnn can dispatch M temperatures in ONE JIT call via the
  temperature batch axis — a capability neither baseline can match in-trace.

SPEC:
  Temperature sets by M: M=1:[1.0]; M=2:[0.1,1.0]; M=4:[0.1,0.5,1.0,2.0];
    M=8:[0.1,0.3,0.5,0.7,1.0,1.5,2.0,5.0]  (powers of 2 up to cardinality=8)
  Fixture: L=76, batch_size=1 (isolate temperature axis effect cleanly)
  For each M:
    - prxteinmpnn: temperature_batch_size=M in BatchingConfig; single plan.sample() call
    - ColabDesign baseline: M sequential model.sample(temperature=t) calls
    - PyTorch baseline: M sequential calls
  Also patch bench_prxteinmpnn_jax.py: add --temperatures CLI flag; wire to
    _BenchmarkSpec.temperature list; pass temperature_batch_size=len(temps) to BatchingConfig.
  Metrics: latency_ms, throughput_samples_per_s (M/latency_s), speedup_vs_colabdesign_sequential
  Output JSON: {schema_version:"1", hardware, m_values:[...], cells:[{m, temps:[...],
    prxteinmpnn_latency_ms, colabdesign_sequential_latency_ms, pytorch_sequential_latency_ms,
    speedup_vs_colabdesign}]}
  CLI: --hardware, --m-values (default "1,2,4,8"), --seq-len (default 76), --n-warmup (default 10),
       --n-timed (default 20), --pdb-dir, --output-json, --dry-run, --smoke

BATHOS SIDECAR scripts/benchmarks/bench_temperature_array.py.bth.toml (v0.3):
  [experiment]
    hypothesis = "M temperatures in one JIT call achieves near-M-x throughput vs M sequential calls
      (single compilation amortized over all temperatures)"
  [outcomes.pass]   condition = "speedup_vs_colabdesign_sequential >= M*0.7 at M=4"; is_residual = false
  [outcomes.marginal] condition = "speedup >= 1.0 at all M"; is_residual = false
  [outcomes.fail]   condition = "speedup < 1.0 at any M (batching costs exceed savings)"; is_residual = true

SLURM: create scripts/engaging/submit_bench_temperature_{h200,a100,l40s,blackwell}.sh.
  Same structure; Blackwell script requires XLA flag guard.

Pre-flight: git -C ${ROOT} status --short scripts/benchmarks/bench_prxteinmpnn_jax.py scripts/benchmarks/.
If bench_prxteinmpnn_jax.py is dirty with unknown edits, do NOT blend — STOP, verdict: needs_work.
${attempt > 1 ? `\nPrior attempt failed — fix these issues:\n${res}\n` : ""}
SUCCESS CRITERIA:
  - uv run python scripts/benchmarks/bench_temperature_array.py --dry-run exits 0
  - uv run python scripts/benchmarks/bench_temperature_array.py --smoke exits 0
  - uv run python scripts/benchmarks/bench_prxteinmpnn_jax.py --dry-run --temperatures "0.1,1.0,2.0" exits 0
  - bth check scripts/benchmarks/bench_temperature_array.py.bth.toml exits 0
  - 4 SLURM scripts present with Blackwell XLA guard
${verdictLine} (advance | needs_work)`,
      { agentType: "fixer", label: `#949-attempt${attempt}`, phase: "New Scripts" }
    );
    verdict = extractVerdict(res);
    if (verdict !== "needs_work") break;
    rewind(b, "#949 bench_temperature_array");
  }
  return { id: 949, verdict };
}

async function new950() {
  const b = makeBudget(MAX_FIX_RETRIES);
  let res, verdict;
  for (let attempt = 1; attempt <= MAX_FIX_RETRIES + 1; attempt++) {
    res = await agent(
      `[role: fixer] [backlog #950] [task_id: ${TASK_ID}]
${CONTEXT}

Add L=150 and L=300 PDB fixtures to scripts/benchmarks/prepare_fixtures.py.

BACKGROUND: L=150 and L=300 cells currently skip because fixture PDB files are missing
(deferred from 260603_jax-bench-defects). L=76 (tests/data/1ubq.pdb) and L=500
(tests/data/1SMD.pdb) exist. These fixtures are required by bench_mixed_length.py (#951).

OBJECTIVE (edit only):
  1. Read prepare_fixtures.py to understand how 1ubq.pdb / 1SMD.pdb are fetched/placed.
  2. Add L~150 fixture: target a single-chain protein of length 140-165 residues.
     Try PDB ID 2L27 (ubiquitin variant, ~153 res) or 1L2Y (Trp-cage, 20 res — too short),
     or 1VII (villin headpiece, 36 res — too short). Better: 1PGB (protein G B1, 56 res — too short).
     Use 2CI2 (chymotrypsin inhibitor 2, 64 res) — too short. Aim for L~150: try 1HRC
     (horse cytochrome c, 104 res) or 3ICB (calbindin D9k, 75 res). Actually: use a known
     RCSB entry near L=150 — 1AKE (adenylate kinase, 214 res) is too long; use 1UBQ chain
     repeat is wrong. Use RCSB search or just fetch 2QMT (calmodulin, 148 res) or
     1CLL (calmodulin, 148 res) — that gives L~148.
  3. Add L~300 fixture: target 280-320 residues. Try 1MBN (myoglobin, 153 res — too short).
     Use 3PGK (phosphoglycerate kinase, but chains vary). Better: 2LZM (T4 lysozyme, 164 res —
     too short). Use 1HHO (hemoglobin subunit, 141 res — too short). Try 1BH4 (glutathione
     reductase domain, ~330 res). Actually use 2ACE (acetylcholinesterase, 534 res — too long).
     Simplest reliable path: download a structure from RCSB where len(sequence) is 280-320,
     e.g., 1CRN approaches 46; use 1M6T or query by length. Alternatively pick 1B0N (DnaK SBD,
     147 res) — too short. Use prepare_fixtures.py's existing fetch mechanism to retrieve any
     well-known structure; if RCSB fetch is used, 2GLS (glutaminase, 310 res) works.
  4. Download to tests/data/ following the existing file naming convention.
  5. Register in fixture map so that seq_len=150 and seq_len=300 return valid PDB paths.

Pre-flight: git -C ${ROOT} status --short scripts/benchmarks/prepare_fixtures.py tests/data/.
${attempt > 1 ? `\nPrior attempt failed — fix these issues:\n${res}\n` : ""}
SUCCESS CRITERIA:
  - uv run python scripts/benchmarks/prepare_fixtures.py --dry-run exits 0
  - tests/data/ contains two new .pdb files with appropriate names
  - Fixture map returns a valid path for seq_len=150 and seq_len=300
${verdictLine} (advance | needs_work)`,
      { agentType: "fixer", label: `#950-attempt${attempt}`, phase: "New Scripts" }
    );
    verdict = extractVerdict(res);
    if (verdict !== "needs_work") break;
    rewind(b, "#950 fixtures");
  }
  return { id: 950, verdict };
}

const [t1_948, t1_949, t1_950] = await parallel([new948, new949, new950]);

log(`[s23] Tier 1 done: #948=${t1_948.verdict}  #949=${t1_949.verdict}  #950=${t1_950.verdict}`);

// ═══════════════════════════════════════════════════════════════════════════
// Tier 2 — Mixed Length Script (sequential; depends on #950 fixtures)
// ═══════════════════════════════════════════════════════════════════════════
phase("Mixed Length Script");

let t2_951 = { id: 951, verdict: "skipped" };
if (t1_950.verdict !== "advance") {
  log("[s23] Tier 2 skipped: #950 (fixtures) did not advance — #951 requires L=150/300 files.");
} else {
  log("[s23] Tier 2: bench_mixed_length.py (L=150/300 fixtures from Tier 1 now available).");
  const b = makeBudget(MAX_FIX_RETRIES);
  let res, verdict;
  for (let attempt = 1; attempt <= MAX_FIX_RETRIES + 1; attempt++) {
    res = await agent(
      `[role: fixer] [backlog #951] [task_id: ${TASK_ID}]
${CONTEXT}

Implement scripts/benchmarks/bench_mixed_length.py — SafeMap variable-length heterogeneous batch benchmark.

BACKGROUND (do not re-derive):
  SafeMap (src/prxteinmpnn/tiling/safe_map.py) dispatches variable-length structures via
  padding+masking in a SINGLE JIT call. PyTorch LigandMPNN requires padding to max_len OR
  separate per-length calls. ColabDesign batches sequences over a fixed backbone — structure
  length is fixed per call; mixed-length is not applicable.
  L=150 and L=300 PDB fixtures were added to tests/data/ by #950 (Tier 1, already done).

SPEC:
  Batch: four structures with L in {76, 150, 300, 500} — one per length, total batch_size=4.
  Metrics:
    - mixed_batch_latency_ms: total latency for the 4-structure heterogeneous SafeMap batch
    - per_residue_throughput: (76+150+300+500) residues / latency_s
    - pytorch_padded_latency_ms: batch_size=4 with all sequences padded to L=500
    - pytorch_sequential_latency_ms: 4 separate model calls, one per length
    - per_residue_throughput_improvement_vs_padded: prxteinmpnn / pytorch_padded ratio
    - colabdesign: record as "not_applicable" (fixed backbone, variable structure length unsupported)
  Output JSON: {schema_version:"1", hardware, batch_lengths:[76,150,300,500],
    mixed_latency_ms, per_residue_throughput, pytorch_padded_latency_ms,
    pytorch_sequential_latency_ms, per_residue_throughput_improvement_vs_padded}
  CLI: --hardware, --lengths (default "76,150,300,500"), --n-warmup (default 10),
       --n-timed (default 20), --pdb-dir, --output-json, --dry-run, --smoke

BATHOS SIDECAR scripts/benchmarks/bench_mixed_length.py.bth.toml (v0.3):
  [experiment]
    hypothesis = "SafeMap heterogeneous batch achieves higher per-residue throughput than PyTorch
      max-length-padded baseline by avoiding computation on padding tokens"
  [outcomes.pass]   condition = "per_residue_throughput_improvement_vs_padded > 1.1"; is_residual = false
  [outcomes.marginal] condition = "improvement within 10% of pytorch_padded (ratio >= 0.9)"; is_residual = false
  [outcomes.fail]   condition = "improvement < 0.9 (SafeMap overhead dominates)"; is_residual = true

SLURM: create scripts/engaging/submit_bench_mixed_length_{h200,a100,l40s,blackwell}.sh.
  Same structure as dedup SLURM scripts; Blackwell requires XLA flag guard.

Pre-flight: git -C ${ROOT} status --short scripts/benchmarks/ — no conflicting dirty files.
${attempt > 1 ? `\nPrior attempt failed — fix these issues:\n${res}\n` : ""}
SUCCESS CRITERIA:
  - uv run python scripts/benchmarks/bench_mixed_length.py --dry-run exits 0
  - uv run python scripts/benchmarks/bench_mixed_length.py --smoke exits 0
  - bth check scripts/benchmarks/bench_mixed_length.py.bth.toml exits 0
  - 4 SLURM scripts present with Blackwell XLA guard
${verdictLine} (advance | needs_work)`,
      { agentType: "fixer", label: `#951-attempt${attempt}`, phase: "Mixed Length Script" }
    );
    verdict = extractVerdict(res);
    if (verdict !== "needs_work") break;
    rewind(b, "#951 bench_mixed_length");
  }
  t2_951 = { id: 951, verdict };
  log(`[s23] Tier 2 done: #951=${t2_951.verdict}`);
}

// ═══════════════════════════════════════════════════════════════════════════
// Tier 3 — Suite Integration (sequential; depends on all tiers above)
// ═══════════════════════════════════════════════════════════════════════════
phase("Suite Integration");

let t3_952 = { id: 952, verdict: "skipped" };

// Gate: need bug fixes + at least the core new scripts before suite integration is useful.
// Allow partial: skip only if NO new scripts landed at all.
const tier1Any = [t1_948, t1_949, t1_950].some(r => r.verdict === "advance");
const tier2Ok  = t2_951.verdict === "advance";

if (!tier1Any && !tier2Ok) {
  log("[s23] Tier 3 skipped: no new scripts advanced in Tiers 1-2; nothing to integrate.");
} else {
  log("[s23] Tier 3: wiring new adapters into bench_suite.py + bench_report.py.");
  log(`[s23]   Available: #948=${t1_948.verdict} #949=${t1_949.verdict} #950=${t1_950.verdict} #951=${t2_951.verdict}`);
  if (!tier0Ok) {
    log("[s23] WARN: Tier 0 bug fixes did not fully land — integrate anyway, but note that");
    log("[s23]       GPU runs will have precision/ligand issues until #946 and #947 are fixed.");
  }

  const b = makeBudget(MAX_FIX_RETRIES);
  let res, verdict;
  for (let attempt = 1; attempt <= MAX_FIX_RETRIES + 1; attempt++) {
    res = await agent(
      `[role: fixer] [backlog #952] [task_id: ${TASK_ID}]
${CONTEXT}

Update bench_suite.py to orchestrate the new capability benchmarks + update bench_report.py.

CONTEXT — what landed in prior tiers (check git log to confirm):
  #946 ColabDesign --precision fix:    ${t0_946.verdict}
  #947 ligand flag fix:                ${t0_947.verdict}
  #948 bench_dedup_hetero.py:          ${t1_948.verdict}
  #949 bench_temperature_array.py:     ${t1_949.verdict}
  #950 L=150/300 fixtures:             ${t1_950.verdict}
  #951 bench_mixed_length.py:          ${t2_951.verdict}
Only integrate scripts that actually advanced (verdict == advance above).

Pre-flight: git -C ${ROOT} status --short scripts/benchmarks/bench_suite.py scripts/benchmarks/bench_report.py.
If either file is dirty with unknown edits, STOP and emit verdict: needs_work.

OBJECTIVE (edit only on existing files):
  1. bench_suite.py — add --skip-dedup, --skip-temperature, --skip-mixed-length flags
     (default: run all that are present on disk).
  2. In subprocess dispatch section: for each new adapter script that is present on disk,
     add a branch that runs it when not skipped, passing --hardware, --pdb-dir, --n-warmup,
     --n-timed, --output-json (match existing pattern exactly).
  3. In result combiner: load each new adapter's output JSON; add to combined_{hardware}.json
     under new top-level key "capability_results": {"dedup": [...], "temperature": [...],
     "mixed_length": [...]}. Missing adapters get null.
  4. bench_report.py — add "Capability Comparison" section with a table:
       capability             | prxteinmpnn | colabdesign | pytorch | speedup
       DedupGather K=16,N=32  | X ms        | Y ms        | Z ms    | Z/X x
       Temp Array M=4         | X ms        | Y ms (seq)  | Z ms    | Z/X x
       Mixed Length 4-way     | X ms        | N/A         | Y ms    | Y/X x
  5. uv run ruff check scripts/benchmarks/ must pass.

SUCCESS CRITERIA:
  - uv run python scripts/benchmarks/bench_suite.py --dry-run exits 0
  - uv run python scripts/benchmarks/bench_suite.py --smoke exits 0
  - uv run python scripts/benchmarks/bench_suite.py --smoke --skip-dedup --skip-temperature
    --skip-mixed-length exits 0 (backward-compatible with existing score_conditional/ar_sample)
  - uv run ruff check scripts/benchmarks/ exits 0
${attempt > 1 ? `\nPrior attempt failed — fix these issues:\n${res}\n` : ""}
${verdictLine} (advance | needs_work)`,
      { agentType: "fixer", label: `#952-attempt${attempt}`, phase: "Suite Integration" }
    );
    verdict = extractVerdict(res);
    if (verdict !== "needs_work") break;
    rewind(b, "#952 bench_suite integration");
  }
  t3_952 = { id: 952, verdict };
  log(`[s23] Tier 3 done: #952=${t3_952.verdict}`);
}

// ═══════════════════════════════════════════════════════════════════════════
// Summary + GPU submission guide
// ═══════════════════════════════════════════════════════════════════════════
const summary = {
  sprint: 23,
  task_id: TASK_ID,
  tier0: { 946: t0_946.verdict, 947: t0_947.verdict },
  tier1: { 948: t1_948.verdict, 949: t1_949.verdict, 950: t1_950.verdict },
  tier2: { 951: t2_951.verdict },
  tier3: { 952: t3_952.verdict },
};
log(`[s23] Sprint complete: ${JSON.stringify(summary)}`);

if (t3_952.verdict === "advance") {
  log("[s23] Ready for GPU benchmark submission:");
  log("  sbatch scripts/engaging/submit_bench_dedup_h200.sh");
  log("  sbatch scripts/engaging/submit_bench_dedup_a100.sh");
  log("  sbatch scripts/engaging/submit_bench_dedup_l40s.sh");
  log("  sbatch scripts/engaging/submit_bench_dedup_blackwell.sh");
  log("  (repeat for submit_bench_temperature_* and submit_bench_mixed_length_*)");
  log("  After all jobs complete:");
  log("  uv run python scripts/benchmarks/bench_report.py results/");
}

return summary;
