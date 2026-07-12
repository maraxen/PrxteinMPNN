---
draft_status: NOT YET FILED — pending review
target: https://github.com/jax-ml/jax/issues (per their bug-report.yml template + NVIDIA GPU-bug best practices)
---

# Draft bug report: jax.grad crashes with 'scf.if' control-flow shape mismatch — regression between jaxlib 0.9.2 and 0.10.2

> This file is a draft for external submission to `jax-ml/jax`'s issue tracker. It is **not filed yet**.
> Everything below the `---` is intended to become the GitHub issue body almost verbatim; the two
> fields match `jax-ml/jax`'s own `bug-report.yml` template (`Description`, `System info`), plus one
> extra section (`Version bisection`) that isn't part of their template but is exactly the kind of
> evidence their own "Ten simple rules" / NVIDIA-forum best-practices guidance asks reporters to
> provide up front, since it saves maintainers the first round of triage.
>
> **Open item before filing** (flagging honestly, not resolved in this draft): the repro below
> imports `aminx.ebm.model.ProteinEBMModel` from this project's own (currently **unpushed**, local-only)
> `worktree-proteinebm-decomposition` branch. A JAX maintainer can't run this as-is. Before filing we
> need to either (a) push this branch (or a minimal extract of just the `ProteinEBMModel`/trunk
> classes) to a public remote and link a fixed commit SHA, or (b) fully inline the ~150-200 lines of
> `DiffusionTransformer`/`AdaLN`/`AttentionPairBias` trunk code so the repro has zero external
// dependencies. Not done here — a real scoping decision, not skipped by oversight.

---

## Title

`jax.grad` crash on GPU: `'scf.if' op along control flow edge ... successor operand type` shape mismatch — regression between jaxlib 0.9.2 and 0.10.2

## Description

Taking `jax.grad` of a scalar energy function over a small transformer-style trunk (attention with
pair bias + AdaLN conditioning, wrapped in `eqx.filter_jit`) crashes at XLA compile time on every
GPU architecture I've tested (Blackwell, Hopper, Ampere, Ada) with jaxlib `0.10.2`, but **not** with
`0.8.0`, `0.8.3`, `0.9.0`, or `0.9.2`. The plain forward pass (no `grad`) is unaffected at every
version and on every GPU tested. Turing-generation GPUs (`sm_75`) also do not hit this even on
`0.10.2` — only Ampere-and-newer architectures do.

### Error

```
jax.errors.JaxRuntimeError: UNKNOWN: <unknown>:0: error: loc("add_any.819.1"): 'scf.if' op along
control flow edge from Operation scf.yield to parent: successor operand type #0
'tensor<1x1x1xf32>' should match successor input type #0 'tensor<1x256x64xf32>'
<unknown>:0: note: loc("add_any.819.1"): region branch point
```

The `add_any` op name suggests this is happening inside JAX's cotangent-accumulation primitive
(used when a value contributes to the output gradient via multiple paths) — the shapes involved
(`1x1x1` vs. `1x256x64`) look like a scalar/keepdims-reduced cotangent failing to broadcast back to
a per-token tensor shape inside a compiler-generated conditional, but I haven't traced this further
into XLA's HLO/StableHLO lowering myself.

### Minimal(ish) repro

Reproduces with the **smallest architecture my library supports** (2 transformer layers, `token_s=16`)
at sequence length 64 — it does *not* reproduce at length 16, regardless of model size (see
"Additional findings" below), so length crossing some internal threshold appears to be the actual
trigger, not model size.

```python
import jax, jax.numpy as jnp
import equinox as eqx
from aminx.ebm.model import ProteinEBMModel  # see note above: currently unpushed, TODO before filing

key = jax.random.PRNGKey(0)
model = ProteinEBMModel(
    token_s=16, token_z=8, dim_fourier=12,
    conditioning_transition_layers=1, transformer_depth=2, transformer_heads=2,
    key=key,
)

N = 64  # crashes at 64 and 256; does NOT crash at 16 (see below)
coords = jax.random.normal(jax.random.PRNGKey(1), (N, 3)) * 0.1
aatype = jnp.zeros((N,), dtype=jnp.int32)
mask = jnp.ones((N,), dtype=bool)
t = jnp.array(0.05)

@eqx.filter_jit
def score_fn(m, c, a, tt, mm):
    return m.score(c, a, tt, mm)  # score(c,a,tt,mm) == -jax.grad(energy)(c) internally

out = score_fn(model, coords, aatype, t, mask)
jax.block_until_ready(out)  # <-- crashes here at compile time
```

`model.score` is a thin wrapper: `score = -jax.grad(lambda c: energy(c, aatype, t, mask))(coords)`,
where `energy` runs the coords through a small `DiffusionTransformer` trunk (attention with a pair
bias term + AdaLN-style conditioning, doubly-nested `jax.vmap` over an `(L, L, D)` pairwise tensor)
and reduces to a scalar. The plain forward pass (`model.energy(...)`, no `grad`) never crashes at
any length or model size tested.

### Additional findings (from a longer investigation, included for triage speed)

**Not model-size dependent — only sequence-length dependent.** Swept 4 model sizes (2-layer/16-dim
up to 16-layer/384-dim, ~85M params) × 3 sequence lengths (16, 64, 256) on the same GPU: **every**
model size passes at length 16 and **every** model size fails at 64 and 256. This rules out "just a
big/complex graph" as the trigger and points at something length- (or shape-) dependent instead.

**Not an autotuning/scheduling/fusion flag issue.** Tried 6 `XLA_FLAGS` combinations against the
crash (`--xla_gpu_shard_autotuning=false` baseline, `--xla_gpu_autotune_level=0`,
`--xla_gpu_enable_triton_gemm=false`, `--xla_gpu_enable_command_buffer=` disabled,
`--xla_gpu_enable_latency_hiding_scheduler=false`, and all four combined) — all six produce the
identical error. Doesn't look like an optional-codegen-path issue.

**Reproduces on every modern GPU architecture tested, not just one.** Blackwell (RTX PRO 6000
Server Edition), H100, A100, and L40S all crash with the same error class under jaxlib `0.10.2`
(H100/L40S: byte-identical shapes to Blackwell in the error message; A100: same bug, a different
intermediate shape — `tensor<1x128x128xf32>` — consistent with per-architecture tiling, not a
different bug). Turing-generation cards (TITAN RTX, `sm_75`) do **not** crash, even on the same
jaxlib `0.10.2` — this may point at a newer/Ampere+-specific GPU codegen path in XLA that Turing
doesn't route through.

## Version bisection

Ran the exact repro above (length 64, tiny model) via `uv run --with "jax[cuda12]==<ver>" --with
"jaxlib==<ver>"` against 5 versions on the same Blackwell node, no other changes:

| jax / jaxlib version | Result |
| :-- | :-- |
| `0.8.0` | PASS |
| `0.8.3` | PASS |
| `0.9.0` | PASS |
| `0.9.2` | PASS |
| `0.10.2` | **FAIL** (error above) |

This is a real regression introduced somewhere between `0.9.2` and `0.10.2`, not an inherent
limitation of this computation graph — every version from `0.8.0` through `0.9.2` compiles and runs
the identical `jax.grad` call cleanly on the identical hardware.

## System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.10.2
jaxlib: 0.10.2
numpy:  2.4.6
python: 3.14.3 (main, Feb 12 2026, 00:42:54) [Clang 21.1.4]
device info: NVIDIA RTX PRO 6000 Blackwell Server Edition-1, 1 local devices
process_count: 1
platform: uname_result(system='Linux', node='node4008', release='4.18.0-553.83.1.el8_10.x86_64', ...)
XLA_FLAGS=--xla_gpu_shard_autotuning=false
```

```
$ nvidia-smi
NVIDIA RTX PRO 6000 Blackwell Server Edition, Driver Version: 590.48.01, CUDA Version: 13.1
```

Also reproduced (identical error class, see "Additional findings" above) on:
- H100 (`mit_preemptable` cluster partition, gres `gpu:h100:1`)
- A100 (same partition, gres `gpu:a100:1`)
- L40S (same partition, gres `gpu:l40s:1`)

Does **not** reproduce on: 4× NVIDIA TITAN RTX (Turing, `sm_75`), same jaxlib `0.10.2`.

## Environment / how this was found

Found while benchmarking a JAX port of a published energy-based protein model (score-matching
diffusion) against its PyTorch reference implementation — `jax.grad` of the energy function is the
JAX side of a decoy-ranking / stability-prediction throughput benchmark. Not an isolated one-off:
blocks two of four planned GPU benchmarks entirely until either fixed upstream or worked around by
pinning to `0.9.2` (which we've done for now — see version bisection above for why that specific
pin is well-supported by evidence).

---

*(End of draft issue body. Internal notes below this line are NOT part of the intended GitHub issue.)*

## Internal notes (not for the issue)

- Full investigation trail, including the jobs cited above (job IDs, exact commands, cluster
  partition names) lives in `.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md` §7–§11 —
  useful for anyone reproducing this investigation later, but too much aminx-specific/cluster-specific
  detail for an external bug report.
- Considered filing directly against `openxla/xla` instead of `jax-ml/jax` (this is really an
  XLA:GPU compiler bug, not a JAX Python-level issue) — went with `jax-ml/jax` per the precedent in
  `jax-ml/jax#25759` (a structurally similar grad+vmap+scan XLA layout bug): the reporter filed on
  `jax-ml/jax`, and JAX maintainers did the minimization down to raw HLO and routed the actual fix to
  `openxla/xla#21511` themselves. Filing on `jax-ml/jax` first, not `openxla/xla` directly, matches
  how that got resolved.
- Still open before filing:
  1. Resolve the "unpushed private branch" issue in the repro (push a public extract, or inline the
     trunk code).
  2. Optionally attempt a fully framework-independent repro (no `aminx` import at all) by
     hand-writing a small doubly-nested-`vmap` + `jax.grad` synthetic function with a similar shape
     profile — not attempted; the model-size sweep above already shows model complexity isn't the
     driver, so a synthetic minimal repro is plausible but unconfirmed.
  3. Double-check whether `jax-ml/jax`'s issue search already has something close once actually
     searched from a logged-in account / with different query phrasing than the unauthenticated
     `gh search issues`/WebSearch queries used during this investigation turned up.
