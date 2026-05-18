# PrxteinMPNN Composition Guide

How to build new inference paths by assembling existing stages — without touching kernel math.

---

## Mental model

The pipeline has three layers:

```
Spec / host layer        StageSet (what to do)      JAX-traced kernel
─────────────────        ──────────────────────      ────────────────────
SamplingSpecification ──► make_inference_plan ──► driver.decode
InferencePlan                 │                         │
  .sample()                   ▼                         ▼
  .score()               StageSet slots           infer_topology(stage_set)
                              │                    ├─ AR scan
                         encode_fn                 ├─ conditional vmap
                         logit_transform           └─ unconditional vmap
                         ar_logit_transform
                         decode_step
                         sample_step
```

**`StageSet` is the composition interface.** Setting or leaving `None` its four slots fully determines which decode path runs and what logic executes at each step.

**`InferencePlan` is the recommended entry point.** It wraps model + stages + encoding strategy into a single callable object with `.sample()` and `.score()` methods. Use the factory `make_inference_plan(model, spec)` to auto-resolve stages from a specification, or build one manually for full control.

---

## StageSet slot reference

| Slot | Type | Default | Controls |
|------|------|---------|----------|
| `logit_transform` | `BatchLogitFn` | `ArithmeticMeanLogits` | State fusion `(S, L, V) → (L, V)` |
| `ar_logit_transform` | `BatchLogitFn` | `ARLogitFuse` | Per-position AR fusion `(S, V) → (V)` |
| `decode_step` | `ConditionalDecodeStep \| UnconditionalDecodeStep \| None` | `None` | Which decoder variant runs (None = conditional fallback) |
| `sample_step` | any \| `None` | `None` | Sampling function; `None` means scoring mode |

**Topology is inferred at call time** by `driver.infer_topology(stage_set)`:

```python
if stage_set.sample_step is not None:          → TOPOLOGY_AR
elif isinstance(decode_step, UnconditionalDecodeStep): → TOPOLOGY_UNCONDITIONAL
else:                                          → TOPOLOGY_CONDITIONAL_SCORE
```

---

## The four extension points

### 1. New state-fusion strategy

Implement a new `BatchLogitFn` as an `eqx.Module`. The `weights` array must be a **traced leaf** (not `eqx.field(static=True)`) so JIT can differentiate through it.

```python
# inference/logits.py or your experiment module
@LOGIT_STRATEGIES.register("harmonic_mean")
class HarmonicMeanLogits(eqx.Module):
    weights: Float[Array, "S"]

    def __call__(self, per_state: Float[Array, "S ... V"], bias=None) -> Float[Array, "... V"]:
        w = self.weights / self.weights.sum()
        dims = per_state.ndim - 1
        w = w.reshape((per_state.shape[0],) + (1,) * dims)
        result = 1.0 / jnp.sum(w / (per_state + 1e-9), axis=0)  # harmonic mean
        return result if bias is None else result + bias
```

Wire it:
```python
stage_set = StageSet(logit_transform=HarmonicMeanLogits(weights=state_weights))
```

---

### 2. New per-position AR fusion

For a custom per-step fuse across S states (shape `(S, V) → (V,)`):

```python
class TemperatureARFuse(eqx.Module):
    temperature: float = eqx.field(static=True)

    def __call__(self, logits: Float[Array, "S V"]) -> Float[Array, "V"]:
        return jax.scipy.special.logsumexp(logits / self.temperature, axis=0) - jnp.log(logits.shape[0])
```

The driver vmaps this over `L` positions automatically:
```python
# in driver._decode_ar:
combined = jax.vmap(stage_set.ar_logit_transform, in_axes=1, out_axes=0)(logits)
```

---

### 3. New decode variant

Subclass `eqx.Module` with the appropriate signature.

**Conditional** (takes sequence):
```python
class MyConditionalDecode(eqx.Module):
    decoder: Any
    w_s_embed: Any
    dropout_rate: float = eqx.field(static=True)

    def __call__(self, node_f, edge_f, nei, mask, ar_mask, seq_oh, *, key, inference):
        return self.decoder.call_conditional(
            node_f, edge_f, nei, mask, ar_mask, seq_oh, self.w_s_embed,
            key=key, inference=inference,
            # custom arg:
            dropout_rate=0.0 if inference else self.dropout_rate,
        )
```

**Unconditional** (no sequence):
```python
class MyUnconditionalDecode(UnconditionalDecodeStep):
    # override __call__ if needed
    pass
```

---

### 4. New sampling method (changes topology to AR)

Anything set as `sample_step` (non-None) triggers the AR scan topology.

```python
class GumbelTopKStep(eqx.Module):
    tau: float = eqx.field(static=True)
    k: int = eqx.field(static=True)

    def __call__(self, logits: Float[Array, "V"], key) -> Int[Array, ""]:
        gumbel = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape) + 1e-9))
        return jnp.argmax((logits + gumbel) / self.tau)
```

> **Note:** the current `_decode_ar` in `driver.py` uses `jax.random.categorical` directly. Wiring `sample_step` into the AR scan body as a fully composable delegate is a potential future enhancement. For now, `sample_step` presence (non-None) flags AR topology; the sampling method itself uses `jax.random.categorical`.

---

## Putting it together: full example

### Option 1: Factory (Recommended)

For the common case where you just want to change fusion strategy via spec:

```python
from prxteinmpnn.host.plan import make_inference_plan
from prxteinmpnn.run.specs import SamplingSpecification

spec = SamplingSpecification(
    inputs="structure.pdb",
    num_samples=10,
    multi_state_strategy="geometric_mean",
    state_weights=[1.0, 0.8, 0.6],
)

# Reads strategy, state_weights, rolling_state from spec automatically
plan = make_inference_plan(model, spec)

# Use the plan
result = plan.sample(bundle, key, config)     # SampleResult(sequence, logits)
logits = plan.score(bundle, key, config)      # (L, 21)
```

### Option 2: Manual Component Assembly

For fine-grained control over stages and encoding:

```python
import jax
import jax.numpy as jnp
from prxteinmpnn.host.plan import InferencePlan, InferenceComponents, make_inference_plan
from prxteinmpnn.inference.encode import make_encode_fn
from prxteinmpnn.inference import driver
from prxteinmpnn.inference.logits import GeometricMeanLogits, ARLogitFuse
from prxteinmpnn.types.stages import StageSet

# --- Custom experiment: geometric mean fusion with custom temperature ---

state_weights = jnp.array([1.0, 0.8, 0.6])   # 3 conformational states

stage_set = StageSet(
    logit_transform=GeometricMeanLogits(weights=state_weights, temperature=1.2),
    ar_logit_transform=ARLogitFuse(),           # or your custom fuse
    decode_step=None,                           # conditional fallback
    sample_step=None,                           # scoring topology
)

components = InferenceComponents(
    encode_fn=make_encode_fn(model, use_rolling_state=False),
    driver=driver.decode,
    stage_set=stage_set,
)

plan = InferencePlan(model=model, components=components)

# Use the plan
result = plan.sample(bundle, key, config)     # SampleResult(sequence, logits)
logits = plan.score(bundle, key, config)      # (L, 21)
```

---

## What NOT to touch

These are the JAX-traced invariants. Modifying them breaks JIT retracing contracts:

| Invariant | Why |
|-----------|-----|
| `InferenceBundle` and sub-bundles | JIT boundary — shapes must be static at trace time |
| `state_weights` as traced leaf | Must flow through AD; never mark `static=True` |
| Scatter/scan layout in `driver._decode_ar` | Rewriting changes numerical output |
| `SamplerFn` / `ScoreFn` top-level signatures | External callers (runner, averaging) depend on these |

---

## Quick reference: which file to edit

| Goal | File |
|------|------|
| New fusion strategy | `inference/logits.py` — add eqx.Module, register with `@LOGIT_STRATEGIES.register` |
| New encode strategy | `inference/encode.py` — extend `make_encode_fn` flags |
| New decode variant | `types/stages.py` — add eqx.Module implementation |
| New host dispatch path | `host/kernel_dispatch.py` — add case to `resolve_kernel_fn` |
| New experiment plan | `host/plan.py` — extend `make_inference_plan` or build components directly |
| Topology routing | `inference/driver.py` — extend `infer_topology` and add `_decode_*` function |
