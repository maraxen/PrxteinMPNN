export const meta = {
  name: 'jax-score-perf-investigation',
  description: 'Map missing filter_jit boundaries causing 30-60x score_conditional slowdown; design and dispatch fix',
  phases: [
    { title: 'Recon', detail: 'Parallel: full JIT boundary map, encode_fn/decode_fn construction, static-arg retrace risk, pytorch baseline comparison' },
    { title: 'Synthesis', detail: 'Ranked fix targets with evidence; static-arg strategy per function' },
    { title: 'Plan', detail: 'Fixer dispatch spec: exact file+line+decorator for each missing filter_jit' },
  ],
}

// ─── context ──────────────────────────────────────────────────────────────────
// Confirmed: InferencePlan is a HOST-SIDE dataclass (not eqx.Module, not JIT-traced).
// plan.score() is pure Python dispatch. If encode_fn/decode_fn inside
// InferenceComponents are not individually wrapped with eqx.filter_jit, every
// model layer op dispatches as a separate GPU kernel → 30-60x overhead vs PyTorch.
//
// Benchmark evidence (H200 fp32 B=1):
//   prxteinmpnn_jax  score_conditional: L=76 → 164ms, L=495 → 596ms
//   ligandmpnn_pytorch score_conditional: L=76 → 4.3ms, L=495 → 5.6ms
//   ratio: 38-106×
//
// ar_sample is even worse: L=76 → 1854ms, L=495 → 51s (PyTorch: 87ms / 543ms)
// ─────────────────────────────────────────────────────────────────────────────

const W = '/home/marielle/projects/tev_design/prxteinmpnn'
const TASK_ID = '260602_benchmark-staging'

const FINDING_SCHEMA = {
  type: 'object',
  required: ['agent', 'jit_map', 'unjitted_hot_paths', 'static_arg_risks', 'verdict'],
  properties: {
    agent: { type: 'string' },
    jit_map: {
      type: 'array',
      description: 'Every filter_jit / jax.jit decoration found, with file:line',
      items: {
        type: 'object',
        required: ['file', 'line', 'function', 'decorator', 'notes'],
        properties: {
          file: { type: 'string' },
          line: { type: 'number' },
          function: { type: 'string' },
          decorator: { type: 'string' },
          notes: { type: 'string' },
        }
      }
    },
    unjitted_hot_paths: {
      type: 'array',
      description: 'Functions in the score/sample hot path that lack filter_jit',
      items: {
        type: 'object',
        required: ['file', 'line', 'function', 'called_by', 'fix'],
        properties: {
          file: { type: 'string' },
          line: { type: 'number' },
          function: { type: 'string' },
          called_by: { type: 'string' },
          fix: { type: 'string' },
        }
      }
    },
    static_arg_risks: {
      type: 'array',
      description: 'Python scalars/objects that would cause retrace if passed as dynamic args',
      items: { type: 'string' }
    },
    verdict: { type: 'string' },
  }
}

// ─────────────────────────────────────────────────────────────────────────────
phase('Recon')
log('Parallel recon: JIT boundary map · encode_fn construction · ar_sample path · pytorch baseline · static arg risks')

const recons = await parallel([

  // ── A: Complete JIT boundary map across the full inference stack ───────────
  () => agent(`
TASK_ID: ${TASK_ID}
WORKSPACE: ${W}

You are mapping every eqx.filter_jit / jax.jit boundary in the prxteinmpnn inference stack.

KNOWN CONTEXT:
- InferencePlan is a host-side dataclass (NOT eqx.Module, never JIT-traced itself)
- plan.score() / plan.sample() call self.components.encode_fn and self.components.decode_fn
- The ONLY filter_jit in host/plan.py is on _run_packer_vmap (line ~528), NOT on score/encode/decode
- This means every model layer executes as a separate untraced kernel unless encode_fn/decode_fn are themselves jitted

Step 0: code_index_workspace(workspace_root: "${W}")

Read these files completely and record every @eqx.filter_jit / @jax.jit / eqx.filter_jit(fn) decoration:
1. src/prxteinmpnn/host/plan.py           — InferencePlan, make_inference_plan, make_encode_fn
2. src/prxteinmpnn/inference/encode.py    — encode_fn factory
3. src/prxteinmpnn/inference/driver.py    — inference driver (called by encode_fn or decode_fn?)
4. src/prxteinmpnn/inference/decode/      — decode_fn factories (list files, read each)
5. src/prxteinmpnn/host/runner.py         — runner (if it wraps calls)

For each function in the score_conditional hot path, answer:
- Is it decorated with filter_jit?
- If not, is it called from inside a filter_jit boundary (i.e., inlined into one)?
- Does it have Python-level loops that would prevent fusion?

Return JSON matching the schema. agent = "jit-boundary-map"
  `, { label: 'jit-boundary-map', phase: 'Recon', schema: FINDING_SCHEMA }),

  // ── B: encode_fn and decode_fn construction — are they jitted at factory time? ──
  () => agent(`
TASK_ID: ${TASK_ID}
WORKSPACE: ${W}

You are checking whether encode_fn and decode_fn in InferenceComponents are
wrapped with eqx.filter_jit at the time they are constructed by make_inference_plan.

Step 0: code_index_workspace(workspace_root: "${W}")

1. Read src/prxteinmpnn/host/plan.py: find make_inference_plan() and make_encode_fn().
   Trace what encode_fn is — is it a bare lambda/partial, or is it wrapped with filter_jit?

2. Find how decode_fn is constructed. Check InferenceComponents namedtuple/dataclass.
   Is decode_fn a raw model method or a jitted wrapper?

3. Read src/prxteinmpnn/inference/encode.py fully.
   Does make_encode_fn return eqx.filter_jit(something) or just something?

4. Check the ar_sample path too (sample_step / ar_decode): does it have its own filter_jit?

5. Critical: if encode_fn is NOT jitted, what does plan.encode() do?
   Does it call the raw model.encoder.__call__() without any JIT boundary?
   Count the number of individual JAX ops that would dispatch separately at L=76.

Return JSON matching the schema. agent = "encode-decode-construction"
  `, { label: 'encode-decode-construction', phase: 'Recon', schema: FINDING_SCHEMA }),

  // ── C: static_argnames / InferenceConfig — retrace risk ───────────────────
  () => agent(`
TASK_ID: ${TASK_ID}
WORKSPACE: ${W}

You are checking whether InferenceConfig contains Python scalars that would
cause eqx.filter_jit to retrace on every call with a different config.

Step 0: code_index_workspace(workspace_root: "${W}")

1. Find the InferenceConfig type (likely src/prxteinmpnn/types/configs.py or similar).
   List every field. Which fields are Python ints/floats/bools/strings (would be traced
   as static in filter_jit)? Which are JAX arrays?

2. Check how the benchmark (bench_prxteinmpnn_jax.py lines 620-750) constructs the config.
   Is the same config object reused across warmup and timed calls, or is it recreated?

3. Check whether InferenceConfig is an eqx.Module. If yes, eqx.filter_jit treats
   non-array fields as static and WILL retrace if they change. Does the benchmark
   change any fields between calls?

4. For ar_sample: does InferenceConfig contain n_steps or sequence_length as a Python int?
   If so, every unique (seq_len, n_steps) combo triggers a fresh compilation.

5. Check the MultiStateSpec in bench_prxteinmpnn_jax.py (lines 125-140). Does it
   contain Python scalars that get captured into the JIT closure? Would changing
   seq_len cause a retrace?

Return JSON matching the schema. agent = "static-arg-retrace-risk"
  `, { label: 'static-arg-retrace', phase: 'Recon', schema: FINDING_SCHEMA }),

  // ── D: PyTorch adapter — what it actually measures ────────────────────────
  () => agent(`
TASK_ID: ${TASK_ID}
WORKSPACE: ${W}

You are auditing what ligandmpnn_pytorch score_conditional actually measures,
to confirm the comparison is apples-to-apples.

Step 0: code_index_workspace(workspace_root: "${W}")

1. Read scripts/benchmarks/bench_ligandmpnn_pytorch.py.
   Find measure_warm_latency_score(). What is timed? Does it call model.score()
   for a full encode+decode or only a partial path?

2. The data shows B=1 → 4.3ms and B=16 → 6.4ms on H200 (only 1.5× for 16× batch).
   Does the pytorch adapter actually vmap/batch? Or does it run a single sequence
   regardless of batch_size, and batch_size is ignored?

3. What model architecture does proteinmpnn_pytorch use for the no-ligand case?
   How many encoder/decoder layers? Hidden dim?

4. If pytorch is NOT batching, then the 38× gap is actually:
   (164ms JAX batched-16) vs (4.3ms PyTorch single-sequence)
   = 38× gap for a batch of 16 vs single sequence = ~2.4× per sequence gap.
   Is this consistent with the B=1 comparison too? Check H200 B=1: 164ms vs 4.3ms.

5. Check the prxteinmpnn_jax adapter: does it vmap over batch_size? What does
   benchmark_cell() do when batch_size > 1 — does it tile via vmap or loop?

Return JSON matching the schema. agent = "pytorch-baseline-audit"
  `, { label: 'pytorch-baseline', phase: 'Recon', schema: FINDING_SCHEMA }),

])

const valid = recons.filter(Boolean)
log('Recon done: ' + valid.length + '/4 agents returned findings')

// ─────────────────────────────────────────────────────────────────────────────
phase('Synthesis')

const SYNTHESIS_SCHEMA = {
  type: 'object',
  required: ['confirmed_root_causes', 'fix_targets', 'static_arg_strategy', 'ar_sample_diagnosis', 'lesson'],
  properties: {
    confirmed_root_causes: {
      type: 'array',
      items: {
        type: 'object',
        required: ['rank', 'description', 'evidence', 'impact'],
        properties: {
          rank: { type: 'number' },
          description: { type: 'string' },
          evidence: { type: 'string' },
          impact: { type: 'string' },
        }
      }
    },
    fix_targets: {
      type: 'array',
      description: 'Exact functions that need filter_jit added, in application order',
      items: {
        type: 'object',
        required: ['file', 'function', 'decorator', 'static_argnames', 'rationale'],
        properties: {
          file: { type: 'string' },
          function: { type: 'string' },
          decorator: { type: 'string' },
          static_argnames: { type: 'array', items: { type: 'string' } },
          rationale: { type: 'string' },
        }
      }
    },
    static_arg_strategy: { type: 'string' },
    ar_sample_diagnosis: { type: 'string' },
    lesson: {
      type: 'object',
      required: ['title', 'body'],
      properties: {
        title: { type: 'string' },
        body: { type: 'string' },
      }
    },
  }
}

const synthesis = await agent(`
TASK_ID: ${TASK_ID}

You are synthesizing recon findings about missing eqx.filter_jit boundaries
causing 30-60x score_conditional slowdown in prxteinmpnn_jax.

CONFIRMED CONTEXT:
- InferencePlan is a HOST-SIDE dataclass: plan.score() / plan.encode() / plan.decode()
  are NEVER inside a JIT boundary unless encode_fn/decode_fn are individually jitted.
- @eqx.filter_jit exists only on _run_packer_vmap in host/plan.py.
- Without filter_jit wrapping encode_fn/decode_fn, each JAX op dispatches as a
  separate kernel: for a transformer with L=76, that's O(n_layers × n_ops_per_layer)
  separate kernel launches per call = massive overhead.
- Benchmark data: H200 B=1 fp32: JAX 164ms vs PyTorch 4.3ms = 38×.

RECON FINDINGS:
${JSON.stringify(valid.map(r => ({
  agent: r.agent,
  unjitted_hot_paths: r.unjitted_hot_paths,
  static_arg_risks: r.static_arg_risks,
  verdict: r.verdict,
})), null, 2)}

Synthesize into:
1. Ranked confirmed root causes (from recon evidence, not speculation)
2. Exact fix_targets: for each function that needs filter_jit, specify:
   - file path (relative to workspace root)
   - function name
   - decorator form: "@eqx.filter_jit" or "eqx.filter_jit(fn, ...)"
   - static_argnames: list any Python-scalar args that must be declared static
     (int dims, bool flags, string modes) to prevent retrace on value change
3. static_arg_strategy: one paragraph on how to handle InferenceConfig if it
   contains Python scalars — declare them eqx.field(static=True), or use
   filter_jit with a custom filter, or extract scalars into static_argnames
4. ar_sample_diagnosis: why ar_sample is EVEN slower (lax.while_loop in an
   unjitted context would force trace overhead per step? or something else?)
5. A lesson to log to praxia about this class of bug

Return JSON matching the schema.
`, { label: 'synthesis', phase: 'Synthesis', schema: SYNTHESIS_SCHEMA })

log('Root causes confirmed: ' + (synthesis ? synthesis.confirmed_root_causes.length : 0))
log('Fix targets: ' + (synthesis ? synthesis.fix_targets.length : 0))

// ─────────────────────────────────────────────────────────────────────────────
phase('Plan')

const PLAN_SCHEMA = {
  type: 'object',
  required: ['fixer_dispatch', 'verification_steps', 'expected_speedup'],
  properties: {
    fixer_dispatch: {
      type: 'object',
      required: ['task_id', 'objective', 'changes', 'success_criteria', 'do_not'],
      properties: {
        task_id: { type: 'string' },
        objective: { type: 'string' },
        changes: {
          type: 'array',
          items: {
            type: 'object',
            required: ['file', 'change_type', 'target', 'instruction'],
            properties: {
              file: { type: 'string' },
              change_type: { type: 'string', enum: ['add_decorator', 'wrap_function', 'add_static_field', 'modify_factory'] },
              target: { type: 'string' },
              instruction: { type: 'string' },
            }
          }
        },
        success_criteria: { type: 'array', items: { type: 'string' } },
        do_not: { type: 'array', items: { type: 'string' } },
      }
    },
    verification_steps: { type: 'array', items: { type: 'string' } },
    expected_speedup: { type: 'string' },
  }
}

const plan = await agent(`
TASK_ID: ${TASK_ID}
WORKSPACE: ${W}

You are writing a precise fixer dispatch plan to add missing eqx.filter_jit
boundaries to the prxteinmpnn inference path.

SYNTHESIS:
${synthesis ? JSON.stringify(synthesis, null, 2) : 'synthesis unavailable — default to: wrap encode_fn and decode_fn with eqx.filter_jit in make_inference_plan / make_encode_fn'}

CONSTRAINTS FROM COMPOSABLE-JAX SKILL:
- eqx.filter_jit on an eqx.Module __call__ works: the module itself is the first arg,
  filter_jit treats arrays as dynamic and non-arrays (Python scalars, etc.) as static.
- For functions with InferenceConfig args: if InferenceConfig is an eqx.Module,
  its non-array fields are automatically treated as static by filter_jit.
  If it's a plain dataclass, use static_argnames to declare Python-scalar fields.
- Do NOT add filter_jit inside a function that is already called from inside filter_jit
  (double-jit is a no-op but adds confusion).
- Do NOT change the public API of InferencePlan.score() / .sample() / .encode() / .decode().

Write a fixer dispatch plan with:
1. A list of exact file+change_type+target+instruction for each edit
2. success_criteria: measurable checks (uv run pytest, latency target)
3. do_not: list of things the fixer must NOT do (e.g. break public API, add redundant jit)
4. expected_speedup: based on the gap, what latency should score_conditional achieve post-fix?

The fixer should be a SINGLE agent dispatched with this plan. Changes ≤ 6 files.

Return JSON matching the schema.
`, { label: 'fixer-plan', phase: 'Plan', schema: PLAN_SCHEMA })

// ─────────────────────────────────────────────────────────────────────────────
// Log lesson to praxia
if (synthesis && synthesis.lesson) {
  log('Lesson: ' + synthesis.lesson.title)
}

return {
  task_id: TASK_ID,
  recon: valid,
  synthesis,
  plan,
  lesson_to_log: synthesis ? synthesis.lesson : null,
}
