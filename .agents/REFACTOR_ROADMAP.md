# prxteinmpnn Refactor Roadmap (FINAL)

> **Target file count delta**: net +0 to +20 files; net **−2500 LoC** in `src/prxteinmpnn/`.
> **Parity contract** (qualified — see §10 *What we are NOT promising*): numerical outputs of `model.__call__`, `make_score_fn(...)(...)`, `make_*_logits_fn(...)(...)`, and `make_sample_sequences(...)(...)` are within `get_tolerances("float32")` before/after every phase, verified by `parity_fast` in CI and `parity_heavy` as a manual per-phase release gate.
> **Status**: final — supersedes the draft after defender + critic adversarial review. Open Questions are now in the Resolution Log (§13) with default decisions; they no longer block Phase 0.

> **Decisions recorded (2026-05-05, by repo owner):**
> - **Q1 / §3.6 Vendor-vs-Depend matrix: CONFIRMED** as written. Vendor `utils/callbacks`, `utils/testing`, `utils/typing`; depend on `core/profiling`, `utils/mapping`, `utils/io`, `core/safety`, `jax_io/sources`. No further open question.
> - **Q6 / Phase 0a SPIKE: CONFIRMED proceed.** Owner has previously tested `state_vmap_exact` against `jax.vmap(single_state_path)` informally and observed numerical agreement; the SPIKE remains mandatory to formalize the result (capture HLO bytes + parity-fixture coverage + recorded decision in PR). **Default expectation upgraded from "route, not unify" to "expect unify, verify formally."** Phase 4 plans the unification path; spike outcome only downgrades to routing if HLO byte-count or parity-fixture comparison surfaces an unexpected divergence.

---

## 1. Executive Summary

`prxteinmpnn` ships a numerically-correct, parity-pinned LigandMPNN port in JAX/Equinox, but the surface area between the four parity-pinned callables has accreted faster than its structure. `src/prxteinmpnn/model/mpnn.py` is **3933 LoC** with two near-duplicate Equinox modules (`PrxteinMPNN`, `PrxteinLigandMPNN`) that reach into each other's private static methods. `src/prxteinmpnn/run/sampling.py` is **1718 LoC** with four near-duplicate streaming variants (`_sample_streaming`, `_sample_streaming_arrayrecord`, `_sample_streaming_averaged`, in-memory) and parallels in scoring/jacobian/conformational_inference. `RunSpecification` and four subclasses carry **50+ flat fields** with at least five provably dead (`output_path`, `average_logits`, `score_batch_size`, `gmm_min_iters`, `combine_noise_batch_size`).

The **seven** legacy `strategy_map` literals are **routed** through `registry._COMBINE_INDEX` / `combine_strategy_to_index`. **Multistate** branching at the four roadmap call surfaces (`PrxteinMPNN` / `PrxteinLigandMPNN.__call__`, `make_sample_sequences`, `make_score_fn`) now reads **immutable** `MultistateModeDescriptor` rows from `registry.MULTISTATE_MODES` (Python / `static_argnames` only — no traced registry lookups). STE ``optimize_sequence`` validates ``multistate_mode`` via ``assert_known_multistate_mode`` at entry.

Phase 1 **import-time** ``multiprocessing.set_start_method`` removal and **hardcoded debug-log** deletion from ``mpnn.py`` are **landed** on main; ``configure_multiprocessing()`` is the supported opt-in (``runtime.py``). Remaining §11 risk is **review-level** (HLO diffs, Equinox static-field warnings), not committed debug paths.

We refactor now because (a) **TECHNICAL_DEBT §9** (mpnn.py split) and **§10** (Protocols/contracts) gate the typed-dispatch work that **§14** (io_callback streaming) requires; (b) **§11** (StableHLO) still needs disciplined HLO baselines and tracer-hygiene as ``mpnn.py`` evolves; (c) the parallel `state_vmap_exact` re-implementation is about to grow a third variant for grid lineage and will permanently fork the codebase if not unified now.

The guiding principle is **"parity is pinned at four callable boundaries; everything inside is fluid"**: we move dataclasses, dispatch tables, file boundaries, and registries freely while keeping the four callables within `get_tolerances("float32")` at every merge. We adopt patterns from the sibling `jaxbeans` repo selectively — see the **VENDOR vs DEPEND matrix (§3.6)** — because jaxbeans has already debugged the JAX-specific edges (jaxtyping cost, Equinox static fields, `stop_gradient` around `io_callback`).

### 1.1 Rename: "Engine/Trainer" → "Host orchestrator / JIT-pure step"

The draft borrowed jaxbeans' `Engine`/`Trainer` class names. Critic correctly noted these are training-loop terms wrong-cast onto a sampling codebase: prxteinmpnn has no `LossFunction`, no `DataModule`, no `checkpoint_dir`. We adopt the **separation principle** — host-side orchestration vs JIT-pure step function — but rename to **`SamplingDriver`** (host: data, callbacks, sinks, signals, ArrayRecord writes) consuming **JIT-pure sampler functions** (registered via `SAMPLERS` in §3.3). The training path (`training/trainer.py`) keeps its existing structure; only sampling/scoring/jacobian/conformational paths are unified under `SamplingDriver`. See Phase 5 for migration detail.

---

## 2. Guiding Principles

- **Parity is pinned at four callables.** `model.__call__`, `make_score_fn(...)(...)`, `make_*_logits_fn(...)(...)`, `make_sample_sequences(...)(...)`. Anything inside (file layout, dataclasses, dispatch, even module class boundaries) is fluid. Every phase ends with `tests/parity` (the `parity_fast` subset) green in CI; `parity_heavy` is a manual release gate before the phase tag is cut.
- **Additive before subtractive.** Introduce the new primitive (Protocol, payload dataclass, registry) alongside existing code; migrate call-sites one PR at a time; delete old code only when zero call-sites remain.
- **Registries only with extension intent.** Criterion (explicit): use a registry only when (a) ≥2 future entries are expected, OR (b) extension-by-separate-file is a stated UX goal. Registries: `SAMPLERS`, `MULTISTATE_MODES`, `OUTPUT_SINKS`. **Demoted to a frozen `OrderedDict` constant**: `_COMBINE_INDEX = OrderedDict([("arithmetic_mean", 0), ("geometric_mean", 1), ("product", 2)])` — only 3 stable values, no extension pressure. `decoding_approach` stays as `lax.switch` (parity-pinned).
- **Host orchestrator / JIT-pure step separation** (renamed from Engine/Trainer; §1.1). Host-side code (sinks, callbacks, signals, ArrayRecord writes) lives in `SamplingDriver`. JIT-pure functions remain stateless and are registered into `SAMPLERS`. Pattern-source: jaxbeans `core/engine.py:78-88, 187-196` and `core/trainer.py:40-52, 96-99` — separation principle only; we do not adopt their `LossFunction`/`DataModule`/`checkpoint_dir` API.
- **HLO snapshots are review artifacts, not numeric gates.** Phase 0 captures baseline HLO bytes for each parity-pinned callable. PRs touching JIT-relevant code surface a diff in review; only block on diffs exceeding a separately-justified threshold per call site (no blanket %). The review artifact lives at `tests/profiling/baseline_hlo/`; CI runs `assert_zero_copy_overhead` (jaxbeans `core/profiling.py:41-99`) for **detection**, with a per-callable allowlist of intentional regressions.
- **Every phase independently mergeable and parity-green.** No phase depends on a later phase to compile, type-check, or pass `parity_fast`. PRs may be reordered within a phase but not across phases.
- **Typed boundaries beat runtime introspection.** `inspect.signature(model.__call__)` and `if TYPE_CHECKING: ... else: Callable[..., Any]` are debt; the replacement is `runtime_checkable` Protocols + `ModelCapabilities(eqx.Module)` static fields.

---

## 3. Architectural Primitives

### 3.1 `runtime_checkable` Protocols at callable boundaries

Pattern source: jaxbeans `core/losses.py:11-32`, `core/callbacks.py:8-38`.

```python
from typing import Protocol, runtime_checkable
from jaxtyping import Array, Float, Int

@runtime_checkable
class ConditionalLogitsFn(Protocol):
    def __call__(
        self,
        prng_key: PRNGKey,
        structure: ProteinStructure,
        decoding_order: Int[Array, "L"],
    ) -> Float[Array, "L 21"]: ...

@runtime_checkable
class StateVmapExactLogitsFn(Protocol):
    def __call__(
        self,
        prng_key: PRNGKey,
        structure_stack: MultistateStackPayload,  # see §3.2
        decoding_order: Int[Array, "L"],
    ) -> Float[Array, "S L 21"]: ...
```

Replaces `if TYPE_CHECKING: ... else: Callable[..., Any]` at `conditional_logits.py:48-66`, `unconditional_logits.py:39-56`. The lying `cast(ScoringFn, ...)` at `score.py:302` becomes `cast(StateVmapExactScoreFn, ...)` (separate Protocol — different signature, different contract).

Protocols introduced (Phase 2): `ConditionalLogitsFn`, `UnconditionalLogitsFn`, `StateVmapExactLogitsFn`, `SamplerFn`, `ScoreFn`, `StateVmapExactScoreFn`. **`DesignSink` is deferred to Phase 5** (when the sink boundary actually consolidates). **`BiasHook` is dropped** (speculative — no current call-site demands it; reintroduce when a real consumer arrives).

### 3.2 Frozen pytree dataclasses (Equinox modules)

Pattern source: jaxbeans `core/state.py:13-34`, `core/padding.py:22-69`. Use `eqx.field(static=True)` for shape ints / string keys; tracer-friendly arrays as regular fields.

The motivation is **tuple-unpacking pain at the call boundary**, not specific shape numbers. Today, `state_vmap_exact` callers pass 5–7 positional arrays (coords, mask, chain_idx, residue_idx, ligand_coords, ligand_mask, ...) and several `_lm` variants pass yet another superset. Static-field shape integers exist as parameters to `DesignArrayRecordWriter` and the multistate code paths but are scattered as call-site constants; collapsing them into a single payload removes the whole "did I pass these in the right order?" hazard. Specific values (e.g. `n_canonical`, `n_states`) are **not** asserted in this roadmap — they live wherever the call-sites currently compute them.

```python
import equinox as eqx
from jaxtyping import Array, Float, Int

class MultistateStackPayload(eqx.Module):
    coords: Float[Array, "S L 4 3"]
    mask:   Float[Array, "S L"]
    chain_idx: Int[Array, "S L"]
    n_states: int = eqx.field(static=True)
    n_canonical: int = eqx.field(static=True)

    def replace(self, **kw) -> "MultistateStackPayload":
        return eqx.tree_at(lambda s: tuple(getattr(s, k) for k in kw), self, tuple(kw.values()))
```

Eight payloads introduced: `MultistateStackPayload`, `LigandStack`, `LigandContext`, `SamplingControls`, `MultistateContext`, `EncodedFeatures`, `SampleResult`, `GridLineage`.

### 3.3 Per-axis decorator registries

Pattern source: jaxbeans `aprt/sensors.py:96-129` (strategy-by-instance registry).

```python
from typing import TypeVar, Callable
T = TypeVar("T")

class Registry[T]:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}
    def register(self, key: str) -> Callable[[T], T]:
        def deco(item: T) -> T:
            self._items[key] = item
            return item
        return deco
    def get(self, key: str) -> T: return self._items[key]
    def keys(self) -> list[str]: return list(self._items)

SAMPLERS = Registry[SamplerFn]("samplers")
MULTISTATE_MODES = Registry[MultistateModeDescriptor]("multistate_modes")
OUTPUT_SINKS = Registry[DesignSink]("output_sinks")  # Phase 5
```

| Axis | Mechanism | Rationale |
|---|---|---|
| `combine_strategy` | **frozen `_COMBINE_INDEX = OrderedDict(...)` constant** | 3 stable values; no extension pressure; explicit ordering for `lax.switch` index |
| `samplers` | `Registry[SamplerFn]` | Multiple variants, more expected (jacobian, ste, conformational) |
| `multistate_modes` | `Registry[MultistateModeFn]` | Extension intent (`state_vmap_exact`, `averaging`, future grid variants) |
| `output_sinks` | `Registry[DesignSink]` (Phase 5) | UX goal: contributors add sinks by separate file |
| `decoding_approach` | `lax.switch` over fixed-cardinality enum | Parity-pinned; not contributor-extensible |

Replaces the **7 duplicated `strategy_map` literals** (above) and the **4 `multistate_mode` dispatch surfaces** (`model/mpnn.py` both `__call__`, `sampling/sample.py`, `scoring/score.py`) with **`multistate_mode_descriptor()` → `MultistateModeDescriptor`** (host-only flags; JIT boundaries unchanged).

### 3.4 `ModelCapabilities(eqx.Module)` static field

Replaces `inspect.signature(model.__call__)` introspection at the **3 verified sites**: `sampling/sample.py:77`, `scoring/score.py:342`, `run/averaging.py:58`.

```python
class ModelCapabilities(eqx.Module):
    accepts_ligand: bool = eqx.field(static=True)
    accepts_state_stack: bool = eqx.field(static=True)
    accepts_tied_positions: bool = eqx.field(static=True)
    accepts_bias: bool = eqx.field(static=True)
    accepts_fixed_positions: bool = eqx.field(static=True)
    output_logit_shape: tuple[str, ...] = eqx.field(static=True)

class PrxteinMPNN(eqx.Module):
    ...
    capabilities: ModelCapabilities = eqx.field(static=True)
```

Call-sites become `if model.capabilities.accepts_ligand: ...` — JIT-static, no runtime introspection.

### 3.5 Composed `RunSpec` from sub-configs

Replaces `RunSpecification` + 4 subclasses with 50+ flat fields. Pattern source: jaxbeans `core/trainer.py:40-52`.

```python
class IOConfig(eqx.Module):
    output_dir: pathlib.Path = eqx.field(static=True)
    sink_kind: str = eqx.field(static=True)            # registry key
    manifest_path: pathlib.Path | None = eqx.field(static=True)

class ResourceConfig(eqx.Module):
    n_devices: int = eqx.field(static=True)
    sample_batch_size: int = eqx.field(static=True)
    structure_batch_size: int = eqx.field(static=True)

class MultistateConfig(eqx.Module):
    mode: str = eqx.field(static=True)                  # registry key
    n_states: int = eqx.field(static=True)
    combine_strategy: str = eqx.field(static=True)      # registry key (string, NOT integer index)

class RunSpec(eqx.Module):
    io: IOConfig
    resource: ResourceConfig
    multistate: MultistateConfig
    ligand: LigandConfig
    tied: TiedPositionsConfig
    grid: GridLineageConfig
    batching: BatchingConfig
    averaging: AveragingConfig
    precision: PrecisionConfig
```

**ArrayRecord stores `combine_strategy: str` (the registry key), NOT the integer index.** This makes records reorder-safe: if `_COMBINE_INDEX` ordering ever changes, the integer would change but the string key would not. Records persist outside the process; the integer is a JIT-trace artifact.

Dead fields dropped during migration: `output_path`, `average_logits`, `score_batch_size`, `gmm_min_iters`, `combine_noise_batch_size` (with one-minor-version deprecation shim per §13 Q4).

### 3.6 VENDOR vs DEPEND matrix (jaxbeans patterns)

Critic point accepted: not every jaxbeans utility justifies a hard dep. Matrix:

| Pattern | Decision | Source size | Rationale |
|---|---|---|---|
| `assert_zero_copy_overhead`, `analyze_memory`, `export_hlo` | **DEPEND** | jaxbeans `core/profiling.py` (~99 LoC) | Stable; CI-only; large pattern; jaxbeans is the canonical consumer |
| `safe_map` (vmap↔lax.map dispatch) | **DEPEND** | jaxbeans `utils/mapping.py` | Stable, well-tested; semantic core of resource-bounded paths |
| `PreemptionHandler` (SIGUSR1/SIGTERM) | **DEPEND** | jaxbeans `core/safety.py:20-63` | Cluster-platform glue; jaxbeans already debugged |
| `atomic_write`, `MultiPartWriter` | **DEPEND** | jaxbeans `utils/io.py:15-83` | I/O correctness primitives |
| `BinaryDatasetWriter` schema-as-dict | **DEPEND** | jaxbeans `jax_io/sources.py:82-104` | ArrayRecord schema convention |
| `async_indexed_stream`, `BoundedCallbackHandler` | **VENDOR** | jaxbeans `utils/callbacks.py` (~70 LoC) | We need to customize: add `effects_barrier` at sink boundary + shutdown semantics jaxbeans lacks (their I/O scale is smaller). Vendor lives at `prxteinmpnn/utils/_vendored_callbacks.py` with header pointing to upstream commit hash. |
| `get_tolerances` test helper | **VENDOR** | jaxbeans `utils/testing.py:6-10` | 5 LoC — depending is heavier than copying |
| `PRXTEINMPNN_VERIFY` jaxtyping+beartype decorator | **VENDOR** | jaxbeans `utils/typing.py:17-31` | Renamed env var (`JAXBEANS_VERIFY` → `PRXTEINMPNN_VERIFY`); thin wrapper |
| Engine/Trainer class API | **NEITHER** | — | Adopt the **separation principle only** (§1.1). Concrete `SamplingDriver` is written from scratch against prxteinmpnn's sampling needs. |
| `aprt/*` (FSM/sensors/judge-jury) | **NEITHER** | — | Too heavy, not needed |
| `quantization/module.py` | **NEITHER** | — | Uses `NamedTuple` for a model module — wrong pattern |

---

## 4. Phase-by-phase plan

### Phase 0 — Pre-flight (baseline + scaffolding)

| | |
|---|---|
| **Goal** | Capture baselines, add CI gates, vendor + depend on jaxbeans pieces, no behavior change. |
| **Parity risk** | Zero |
| **PRs** | 2 |
| **Pre-conditions** | None. |

**Tasks:**
- Add `jaxbeans` to `pyproject.toml [tool.uv.sources]` as editable workspace dep (default per §13 Q1/Q2: workspace member during refactor; PyPI when jaxbeans hits 0.1.0).
- Add `[tool.jaxlint] select = ["JL"]` to `pyproject.toml` for **optional local runs**. **jaxlint** is on **PyPI** (`jaxlint>=0.1.0a1` in `prxteinmpnn[dev]`). **Default CI must not gate** on jaxlint: the checker is still maturing and may emit false positives; use it as an **advisory** signal only (local / optional pre-commit). Blocking merge on jaxlint clearance is **explicitly out of scope** until the project chooses a separate policy.
- Capture HLO baselines as **review artifacts** at `tests/profiling/baseline_hlo/{model_call,score,sample,logits}.txt` via `jax.jit(...).lower(...).compile().runtime_executable().hlo_modules()`. CI runs `assert_zero_copy_overhead` for **detection** only; threshold lives in a per-callable allowlist file with rationale strings.
- Vendor `prxteinmpnn/utils/testing.py::get_tolerances` and `prxteinmpnn/utils/typing.py::PRXTEINMPNN_VERIFY` (5 + ~15 LoC, per §3.6).
- Add `ty.toml` `[allowed-unresolved-imports]` for proxide / prolix / optional deps.

**Phase 0a SPIKE — `state_vmap_exact == vmap(single_state)` formal verification gate.** Critic correctly flagged that the Phase 4 unification claim was **not formally verified** in the draft. Owner has prior informal evidence that the equality holds; the SPIKE formalizes that evidence under `tests/sampling/spikes/test_state_vmap_exact_spike.py` and:

- Runs both implementations (the existing `state_vmap_exact` and a freshly-constructed `jax.vmap(single_state_path)` over the same payload) on every parity input fixture (parity_fast and, where reference assets are available, parity_heavy).
- Compares numerical outputs at `get_tolerances("float32")`.
- Compares HLO byte counts and op-count summaries; records both into the PR description.

**Decision rule (recorded in PR; default expectation = unify):**
- **Go (numeric ✓ AND HLO within allowlist) — expected outcome:** Phase 4 collapses `state_vmap_exact` into `MULTISTATE_MODES.register("state_vmap_exact")(jax.vmap(...))`. The parallel re-implementation is deleted.
- **No-go (either gate fails) — fallback:** `state_vmap_exact` stays as a **registry entry that dispatches to the existing implementation** (registry as routing layer, not unification). The duplication is preserved with explicit annotation; the rest of Phase 4 (the 7 strategy_map sites, the 4 multistate_mode ladders) proceeds unchanged.

The spike is mandatory before Phase 4 PRs may merge. Until then, draft Phase 4 PRs may exist but cannot be merged with the unification claim.

**CI gates added:** `pytest tests/profiling/test_hlo_baseline.py` (advisory in Phase 0, detection-blocking from Phase 1 onward — meaning it must run, but only blocks on diffs that exceed allowlisted thresholds).
**Tech-debt closed:** none yet (scaffolding).
**Migration cost:** see §6.

---

### Phase 1 — Hygiene + dead-code (subtractive, mechanical)

| | |
|---|---|
| **Goal** | Delete dead code, hardcoded debug-log blocks, `mp.set_start_method` at import time. |
| **Parity risk** | Zero |
| **PRs** | 2 (delete + import-time hygiene). Doc-drift PRs **moved out** to cross-cutting workstream §7.2. |
| **Pre-conditions** | Phase 0 baselines captured. Phase 0a SPIKE not required (independent path). |

**Tasks:**
- **DELETE** the two hardcoded debug-log blocks at `model/mpnn.py:2428` and `:3098` (the `_logp = "/home/marielle/projects/tev_design/.cursor/debug-5a01b7.log"` lines). These block StableHLO export per **§11**.
- Delete provably dead `RunSpecification` fields: `output_path` (base, never read), `average_logits`, `score_batch_size`, `gmm_min_iters`, `combine_noise_batch_size`. Keep deprecation shim accepting + warning for one minor version if the field is kwarg-named in any `scripts/` (run `rg` against the entire `scripts/` tree as part of the PR).
- **Two `mp.set_start_method("spawn", force=True)` sites must both go:**
  - `prxteinmpnn/__init__.py` — top-level statement.
  - `run/specs.py:15` — top-level statement.
  - Both are replaced by a single explicit `prxteinmpnn.runtime.configure_multiprocessing()` opt-in. Callers (notebooks, scripts) call once at startup.
- **PEP 562 lazy `__init__.py` is demoted to opt-in experiment**, not a Phase 1 mandate. Defender's point is correct: import time is dominated by transitive deps (JAX/Equinox), so the gain from lazy module attributes is small. We will measure cold-import time after the `mp.set_start_method` removal; if it remains >500ms cold, revisit lazy `__getattr__` in a separate scoped PR.
- **Doc-drift cleanup is removed from Phase 1.** It ships as standalone PRs in cross-cutting workstream §7.2.

**CI gates added:** Cold `import prxteinmpnn` time measured (no hard threshold yet — measurement only).
**Tech-debt closed:** §11 (StableHLO unblocked).
**Migration cost:** see §6.

---

### Phase 2 — Protocols + ModelCapabilities (typed boundaries)

| | |
|---|---|
| **Goal** | Replace `Callable[..., Any]` and `inspect.signature` with `runtime_checkable` Protocols and static `ModelCapabilities`. **Trimmed**: `BiasHook` dropped, `DesignSink` deferred. |
| **Parity risk** | Zero (type-only) |
| **PRs** | 3 (Protocols, ModelCapabilities, call-site migration) |
| **Pre-conditions** | Phase 0 (HLO baseline captured). |

**Tasks:**
- Introduce `src/prxteinmpnn/protocols.py` with the **6 callable-boundary Protocols** from §3.1: `ConditionalLogitsFn`, `UnconditionalLogitsFn`, `StateVmapExactLogitsFn`, `SamplerFn`, `ScoreFn`, `StateVmapExactScoreFn`. **Drop `BiasHook`** (speculative — no current consumer; the critic-trim says descope). **Defer `DesignSink`** to Phase 5 where it's actually wired into the sink registry.
- Replace `if TYPE_CHECKING: ... else: Callable[..., Any]` in `conditional_logits.py:48-66` and `unconditional_logits.py:39-56`.
- Add `ModelCapabilities(eqx.Module)` static field on `PrxteinMPNN` and `PrxteinLigandMPNN`. Concrete capability instances live next to each model class.
- Migrate the **3 verified `inspect.signature` sites** (`sampling/sample.py:77`, `scoring/score.py:342`, `run/averaging.py:58`) to `model.capabilities.accepts_*`.
- Fix the lying `cast(ScoringFn, ...)` at `score.py:302` to `cast(StateVmapExactScoreFn, ...)`.
- Make `uv run ty check` blocking in CI. **jaxlint:** **advisory only** — install from PyPI via `prxteinmpnn[dev]`, run locally when useful; **do not add blocking CI** on jaxlint (false positives / tool bugs are expected during adoption).

**Critic point partially accepted (rejected the descope to a 30-line PR):** the critic suggested replacing all 3 `inspect.signature` sites with one explicit `is_ligand_mpnn: bool` parameter. We reject the descope because (a) it doesn't solve the `Callable[..., Any]` problem and (b) `accepts_state_stack` and `accepts_tied_positions` are independent capability axes, not collapsible onto a single LigandMPNN/MPNN bit. We do however adopt the **trim**: `BiasHook` is out, `DesignSink` is deferred.

**CI gates added:** `ty check` strict (no `Callable[..., Any]` in module API surface). **jaxlint** remains **non-blocking** (advisory local / optional hook only).
**Tech-debt closed:** §10.
**Migration cost:** see §6.

---

### Phase 3 — Pytree payloads + RunSpec composition

| | |
|---|---|
| **Goal** | Introduce 8 Equinox payloads + composed `RunSpec`; migrate call-sites; unblock §2 resource wiring. |
| **Parity risk** | Low (tuple-unpacking → struct field is mechanical) |
| **PRs** | **6** (replace harness tests; payloads + RunSpec; sampling/scoring/prep migrations split across PRs; **`scripts/` audit**; pickle migration) — see §14 sprint doc for ordered PR1–PR6; execution may still batch merges where safe. |
| **Pre-conditions** | Phase 2 (Protocols typed). |

**Tasks:**
- Introduce `src/prxteinmpnn/payloads.py` with 8 Equinox modules (§3.2). Each carries a `replace()` via `eqx.tree_at`.
- Replace ad-hoc tuple-passing with `MultistateStackPayload` at all `state_vmap_exact` call sites and at `DesignArrayRecordWriter`. **Execution split (§14):** land payloads + `RunSpec` shim first; complete tuple migration at writers / all hot paths in follow-on PRs once carriers are stable (same phase, ordered merges).
- Compose `RunSpec` from the 9 sub-configs (§3.5). Migrate `RunSpecification` + 4 subclasses to thin shims that build a `RunSpec`.
- Wire `compute_resource_allocation` into `run/prep.py` (currently exists but never called — **§2**).
- Land `PrecisionConfig` and route training-time `dtype`/`policy` through it (closes **§1**).
- **NEW: `scripts/` audit and bulk-update PR.** Run `rg "RunSpecification\("` against the entire `scripts/` tree (including `scripts/engaging/`). For every call-site, update to construct via the new `RunSpec` shim. Maintain a count in the PR description; gate the migration on this audit being complete. Critic's point #8 is fully accepted.
- **`RunSpec` pickle stability for SLURM.** Critic's #7 / Open Q4. Default decision (logged in §13): **explicit break with migration script.** `RunSpecification` shim accepts kwargs, emits `DeprecationWarning`, and delegates to `RunSpec`. **Pickled `RunSpecification` instances from in-flight SLURM jobs are NOT supported across the boundary.** A migration script `scripts/migrate_run_spec.py` reads old pickles (via the shim) and re-emits as `RunSpec` pickles. Documented in CHANGELOG; engaging-cluster integration test verifies the migration script on a representative pickle corpus before Phase 3 merges.

**CI gates added:** Existing parity suite + new `tests/payloads/test_replace_roundtrip.py`. `assert_zero_copy_overhead` review-artifact diff inspected.
**Tech-debt closed:** §1 (precision casting), §2 (resource allocation wiring).
**Migration cost:** see §6.

---

### Phase 4 — Registries + multistate unification (collapse the duplication)

| | |
|---|---|
| **Goal** | Collapse `strategy_map` (7 sites) and `multistate_mode` ladders (4 sites); unify `state_vmap_exact` **conditionally on Phase 0a SPIKE outcome**. |
| **Parity risk** | Medium (`state_vmap_exact` outcome controlled by spike) |
| **PRs** | 4 (registry infra + `_COMBINE_INDEX` constant, multistate_modes registry, samplers registry, `state_vmap_exact` unification or registry-routing) |
| **Pre-conditions** | Phase 3 (payloads + RunSpec available); **Phase 0a SPIKE merged** with go/no-go decision recorded. |

**Tasks:**
- Introduce `src/prxteinmpnn/registry.py` with `Registry[T]` (§3.3) and the **frozen `_COMBINE_INDEX = OrderedDict([("arithmetic_mean", 0), ("geometric_mean", 1), ("product", 2)])`** module-level constant (NOT a registry — only 3 stable values).
- **Migrate the 7 verified `strategy_map` literals** to import `_COMBINE_INDEX`:
  - `model/mpnn.py:1457, 1581, 2322 (the _lm variant), 2615`
  - `scoring/score.py:75, 127`
  - `sampling/sample.py:340`
  - The `lax.switch` index is `_COMBINE_INDEX[spec.multistate.combine_strategy]` at trace time.
- **Migrate the 4 `multistate_mode` if/elif ladders** (`model/mpnn.py` both `__call__`, `sampling/sample.py`, `scoring/score.py`) **→ done (descriptor registry, 2026-05):** host-side `MultistateModeDescriptor` rows in `MULTISTATE_MODES` + `multistate_mode_descriptor()` replace string equality while preserving JIT boundaries. **Still open:** optional `jax.vmap` unification of `state_vmap_exact` vs registry-only routing (Phase 0a **GO** recorded in §13.2).
- **`state_vmap_exact` outcome (controlled by Phase 0a SPIKE):**
  - **If go:** becomes `@MULTISTATE_MODES.register("state_vmap_exact")` calling `jax.vmap` over the existing single-state path; the parallel re-implementation is deleted.
  - **If no-go:** `state_vmap_exact` becomes a registry entry that dispatches to the existing implementation. The duplication is preserved; the registry layer still unifies the dispatch surface (callers see a single `MULTISTATE_MODES.get(mode)` API).
- Convert `decoding_approach` if/elif to a fixed-cardinality `lax.switch` (NOT a registry).
- Migrate samplers (`sample`, `score`, etc.) to the `SAMPLERS` registry.

**CI gates added:** new `tests/sampling/test_state_vmap_exact_routing.py` asserting (a) numeric equivalence to pre-Phase-4 outputs and (b) **if go**: numeric equivalence to `jax.vmap(single_state)(stack)` element-wise; **if no-go**: equivalence to the preserved direct call (regression guard only).
**Tech-debt closed:** none new (sets up §9).
**Migration cost:** see §6.

---

### Phase 5 — `mpnn.py` split + SamplingDriver + io_callback streaming + ensemble→jaxbeans

| | |
|---|---|
| **Goal** | Split 3933-LoC `mpnn.py`; introduce `SamplingDriver`; replace 4 streaming variants with `async_indexed_stream` + `BoundedCallbackHandler`; relocate `ensemble/` to jaxbeans. |
| **Parity risk** | Medium |
| **PRs** | **4–8** (mpnn-split is decomposed, see below) |
| **Pre-conditions** | Phase 4 (registries; multistate unified or routed). |

**mpnn.py-split sub-PRs (5 sub-PRs):**

| # | Sub-PR | Target file | Approx LoC moved |
|---|---|---|---|
| 5a | Extract encoder | `model/encoder.py` | ~600 |
| 5b | Expand existing decoder, move shared decoder body in | `model/decoder.py` | ~700 |
| 5c | Extract shared MPNN message-passing primitives | `model/mpnn_core.py` | ~500 |
| 5d | Extract shared cross-class private static methods | `model/_shared.py` | ~400 |
| 5e | `model/mpnn.py` reduced to `PrxteinMPNN` only (~400 LoC); `model/ligand_mpnn.py` houses `PrxteinLigandMPNN` only (~500 LoC) | both | ~1700 (delete + redistribute) |

Each sub-PR is parity-pinned and individually mergeable. Closes **§9**.

**Other Phase 5 sub-PRs (3):**
- 5f: `SamplingDriver` (host) parameterized by `DesignSink` Protocol (introduced here, deferred from Phase 2). Consumes JIT-pure registered samplers from §3.3. Replaces the 4 near-duplicate streaming functions in `run/sampling.py` and the parallels in `run/scoring.py`, `run/jacobian.py`, `run/conformational_inference.py`. **Does not** copy jaxbeans's in-place `datamodule.batch_size` mutation — batching flows via `RunSpec.batching`.
- 5g: Plumb `async_indexed_stream` (vendored at `prxteinmpnn/utils/_vendored_callbacks.py`, per §3.6) using `stop_gradient` + `ordered=False` for hot paths; `ordered=True` only for debug. Wrap with `BoundedCallbackHandler` for backpressure. **Add `jax.effects_barrier()` at every sink boundary and at epoch end** — this is the customization that motivates vendoring (jaxbeans omits it; their I/O scale is smaller). `io_callback` lives **outside** any `checkify` region. Adopt `atomic_write` + `MultiPartWriter` (DEPEND), `safe_map` (DEPEND), `PreemptionHandler` (DEPEND), `BinaryDatasetWriter` schema-as-dict (DEPEND).
- 5h: Relocate `ensemble/dbscan.py` and `ensemble/pca.py` to jaxbeans (separate jaxbeans-side PR; in this repo we add a shim that imports from jaxbeans). Closes **§12**.

**JIT-cache fragmentation measurement (critic #6, accepted).** The mpnn split risks fragmenting the JIT compile cache because trace contexts now span more modules. Sub-PR 5e adds a benchmark to `tests/profiling/test_cold_start.py` that:
- Measures cold `jax.jit(model.__call__).lower(...).compile()` wall time before and after the split.
- Records the result in the PR description.
- Threshold: any regression > 20% on cold-start wall time triggers an investigation (likely cause: missing `eqx.field(static=True)` on a moved attribute, splitting traces). No hard CI gate (cold-start wall time is noisy on GH runners) — review-artifact only.

**CI gates added:** `tests/streaming/test_io_callback_ordering.py` (designs arrive in deterministic order despite `ordered=False` + barrier); HLO baseline review-artifact diffs reviewed.
**Tech-debt closed:** §9, §12, §14.
**Migration cost:** see §6.

---

### Phase 6 — Proxide / Prolix migration + final cleanup

| | |
|---|---|
| **Goal** | Migrate residual structure utilities to proxide; trajectory/MD wrappers to prolix; final docstring sweep; remove deprecation shims from Phases 1+3. |
| **Parity risk** | Low |
| **PRs** | 2–3 |
| **Pre-conditions** | Phase 5 (`SamplingDriver` split — proxide consumers are now in the host orchestrator, not threaded through hot paths). |

**Tasks:**
- Migrate structure/IO utilities currently duplicated between `prxteinmpnn` and `proxide` to proxide (**§7**).
- Migrate trajectory/MD wrappers to prolix.
- Final docstring sweep across all parity-pinned callables and Protocol definitions (**§6**).
- Remove deprecation shims for the dead `RunSpecification` fields from Phase 1 and the `RunSpec` shim from Phase 3 (after a one-minor-version window per §13 Q4).

**CI gates added:** docstring coverage threshold (`interrogate --fail-under=80` on `src/prxteinmpnn/`).
**Tech-debt closed:** §6, §7.

---

## 5. Phase ordering rationale

The order is **additive-first, then mechanical extractions, then behavioral changes, finally external migrations**. Phase 0 is pure scaffolding plus the **0a SPIKE** that gates Phase 4's unification claim. Phase 1 is pure deletion of dead/debug code (zero parity risk, immediately unblocks §11 StableHLO; both `mp.set_start_method` sites — `__init__.py` and `run/specs.py:15` — are addressed together). Phase 2 introduces typed boundaries *additively* — old `Callable[..., Any]` aliases coexist with new Protocols during the migration window. Phase 3 introduces pytree payloads and the composed `RunSpec` *before* registries because the registries dispatch on those payloads' static fields (combine_strategy, multistate.mode); reversing would force a second migration of registry signatures. Phase 4 collapses duplication only after Phase 3 has unified the data shapes and after the spike has decided whether `state_vmap_exact` can be unified or merely routed. Phase 5 is the highest-risk phase (file split + `SamplingDriver` + io_callback) and depends on every prior phase; the mpnn split is decomposed into 5 sub-PRs (5a–5e) plus 3 more (5f–5h). Phase 6 (proxide/prolix) goes last because the host-orchestrator split clarifies which utilities belong outside the hot path.

---

## 6. Migration cost ledger (per phase)

Estimates from `rg`-driven counts at HEAD; refine in each phase's first PR.

| Phase | Files in `src/prxteinmpnn/` touched | Files in `scripts/` touched | New files | Deleted files | Notes |
|---|---|---|---|---|---|
| 0   | 0 (config only) | 0 | 2 (`utils/testing.py`, `utils/typing.py`) + baseline HLO artifacts | 0 | Plus the 0a SPIKE test |
| 0a  | 0 | 0 | 1 (`tests/sampling/spikes/test_state_vmap_exact_spike.py`) | 0 | Notebook artifact in PR description |
| 1   | ~5 (`__init__.py`, `run/specs.py`, `model/mpnn.py`, doc files via §7.2 separately) | audit-only via `rg`, no edits expected (kwargs to dead fields are the failure mode) | 1 (`runtime.py` for `configure_multiprocessing`) | 0 | Two debug-log blocks deleted in-place |
| 2   | ~6 (`conditional_logits.py`, `unconditional_logits.py`, `model/mpnn.py`, `sampling/sample.py`, `scoring/score.py`, `run/averaging.py`) | 0 | 1 (`protocols.py`) | 0 | |
| 3   | ~12 (8 payload sites + 9 sub-config sites + `RunSpecification` shim) | **expected ~10–25** under `scripts/engaging/` and `scripts/`; full audit is its own PR | 2 (`payloads.py`, `run/spec.py`) + migration script | 0 (`RunSpecification` shimmed) | Pickle migration script must run on representative engaging pickles |
| 4   | ~8 (the 7 strategy_map sites + 4 multistate_mode sites + `state_vmap_exact` route/unify) | 0 | 1 (`registry.py`) | 0 or 1 (depending on spike) | |
| 5   | ~30+ (mpnn split touches every importer of `model.mpnn`; sampling/scoring/jacobian/conformational `run/*.py`) | likely some `scripts/` import path updates | 6 (`model/encoder.py`, `model/ligand_mpnn.py`, `model/mpnn_core.py`, `model/_shared.py`, `run/sampling_driver.py`, `utils/_vendored_callbacks.py`) | 0 (shim retains old paths) | Highest churn phase |
| 6   | ~10 (proxide/prolix consumers) | 0 | 0 | several (deprecation shims removed) | |

**`scripts/engaging/` audit (Phase 3 PR description must include):** the output of `rg -l "RunSpecification\(" scripts/` with each file annotated as updated, deferred, or out-of-scope.

---

## 7. Cross-cutting workstreams

These run alongside phases, not in serial.

### 7.1 jaxlint adoption
- **jaxlint** ships on **PyPI** (`jaxlint`); list it under **`prxteinmpnn[dev]`** for convenient `uv run jaxlint check …`. **Policy:** treat jaxlint as **advisory** — **no default CI merge gate** on jaxlint clearance (the tool may still mis-fire while it matures). Optional pre-commit or maintainer workflows may run it locally. Revisit a stricter gate only if/when the team agrees false-positive rates are acceptable.
- **JL001** flags the scatter/gather anti-pattern documented in jaxbeans `docs/BEST_PRACTICES.md §7`. Apply to touched files only initially; broaden coverage in dev workflows after Phase 4.

### 7.2 Doc-drift cleanup (STANDALONE PRs, moved out of Phase 1)

Critic point #9 accepted: doc-drift PRs ship independently, not as part of any phase. They have no parity risk and their cadence shouldn't be coupled to refactor phases.

- Refresh `docs/FULL_FUNCTIONALITY_TODO.md` (drop "Not Implemented" on STE/fixed-positions; they are implemented).
- Refresh `docs/TODO_BLOCKED_MODULES.md` (drop "deleted" on `conditional_logits` / `unconditional_logits`; they exist).
- Audit `docs/PHYSICS_*.md` against current code; remove references to removed APIs.
- Update `TECHNICAL_DEBT.md` after each phase merges.

Each is its own PR; collectively ~3–5 PRs over the refactor window.

### 7.3 Hardcoded debug-log block deletion (Phase 1, mandatory) — **completed on main**

- Former `model/mpnn.py` debug-log paths and import-time `mp.set_start_method` on the package surface were removed per Phase 1; see §1 executive summary. Retain this subsection as historical rationale for §11.

### 7.4 jaxbeans dependency adoption

Tracked separately from phase PRs; lands incrementally per the **VENDOR vs DEPEND matrix (§3.6)**:

| Pattern | Mode | Phase needed |
|---|---|---|
| `core/profiling` (`assert_zero_copy_overhead`, etc.) | DEPEND | 0 |
| `utils/typing` (`PRXTEINMPNN_VERIFY` decorator) | VENDOR | 0 |
| `utils/testing` (`get_tolerances`) | VENDOR | 0 |
| `utils/callbacks` (`async_indexed_stream`, `BoundedCallbackHandler`) | VENDOR (effects_barrier customization) | 5 |
| `utils/mapping` (`safe_map`) | DEPEND | 5 |
| `utils/io` (`atomic_write`, `MultiPartWriter`) | DEPEND | 5 |
| `core/safety` (`PreemptionHandler`) | DEPEND | 5 |
| `jax_io/sources` (`BinaryDatasetWriter`) | DEPEND | 5 |
| Engine/Trainer class API | NEITHER (separation principle only) | — |
| `aprt/*`, `quantization/module.py` | NEITHER | — |

---

## 8. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Phase 0a SPIKE shows `state_vmap_exact ≠ vmap(single_state)` numerically | **Low** (owner has prior informal evidence of agreement; spike formalizes) | Medium | Exactly the case the gate handles: `state_vmap_exact` becomes a routed registry entry, not unified. Phase 4 still proceeds with the 7 strategy_map sites and 4 multistate_mode ladders. |
| Phase 4 `_COMBINE_INDEX` ordering shift breaks parity for in-flight ArrayRecords | Low | High | ArrayRecord stores **the registry key string**, not the integer. Dedicated test asserts `OrderedDict` insertion order matches the legacy `strategy_map` ordering at the registry-introduction PR. |
| `state_vmap_exact` unification (if go) introduces compile-time HLO regression | Medium | Medium | HLO baseline review-artifact diff inspected in PR; allowlist threshold per call site; if intolerable, fall back to "no-go" route (registry entry → existing impl). |
| io_callback `ordered=False` + `effects_barrier` deadlocks under preemption | Low | High | `PreemptionHandler` drains queue on SIGUSR1 before exit; integration test forces SIGUSR1 mid-stream. |
| `ModelCapabilities` static-field changes force JIT recompiles in user notebooks | Low | Low | Capabilities constructed once per model class; document in CHANGELOG. |
| `SamplingDriver` (Phase 5) exposes a previously hidden race in checkpoint write paths | Low | Medium | `atomic_write` + `MultiPartWriter` serialize manifest; integration test parallel-writes 16 checkpoints. |
| Proxide migration (Phase 6) breaks because proxide diverges during phases 1-5 | Medium | Medium | Pin proxide commit at Phase 0; revisit at Phase 6 start. |
| `parity_heavy` reference assets at `$REFERENCE_PATH` rotate or change format | Low | High | Phase 0 captures content-addressed snapshot of reference assets; manual release gate uses snapshot. |
| `RunSpec` pickle migration script misses an in-flight engaging pickle format | Medium | High | Phase 3 PR description includes the engaging pickle corpus (sampled), and migration script is exercised against it before merge. Hard cutover policy is communicated to cluster users 1 minor version in advance. |
| JIT compile-cache fragmentation from mpnn-split degrades cold start | Medium | Low | Sub-PR 5e benchmark; 20% threshold for investigation. |
| Lazy `__init__.py` (PEP 562) breaks downstream `from prxteinmpnn import X` | — | — | Demoted to opt-in experiment per critic; not in scope unless cold-import measurement justifies it. |

---

## 9. Out of scope

Explicitly NOT in this roadmap:
- **Rewriting `model/diffusion_mpnn.py`** — separate effort.
- **Replacing JAX with anything else.**
- **Touching `training/trainer.py` semantics** — Phase 3 wires `PrecisionConfig` (§1), Phase 5 reshapes I/O (§14), but the loss function and gradient flow are unchanged. The host/JIT-pure split (§1.1) is **not** retrofitted onto the training path; the rename clarifies that `SamplingDriver` is sampling-only.
- **Performance optimization beyond HLO-invariance.** No fused kernels, no `pallas`. Phase gates assert *no regression*, not *improvement*.
- **The jaxbeans-side ensemble PR** is its own effort tracked in jaxbeans.
- **Adopting jaxbeans `aprt/` or `quantization/module.py`** — explicitly avoided.
- **Public API rename** — Protocol names are new; `RunSpecification` is shimmed for one minor version. Renaming `make_score_fn` etc. is out.
- **Replacing pytest, GPU kernel work, full migration guide for downstream consumers.**
- **Lazy `__init__.py` PEP 562 mandate** — demoted to optional experiment.
- **`BiasHook` Protocol** — speculative; reintroduce when a real consumer arrives.

---

## 10. What we are NOT promising

This section qualifies the parity contract and other claims in plain language.

- **`parity_heavy` is NOT in CI.** Per `CLAUDE.md`, `parity_heavy` requires `REFERENCE_PATH` pointing at a local-only `ligandmpnn_reference_assets` directory. CI runs **`parity_fast` only**. `parity_heavy` is a **manual per-phase release gate**: the phase tag is not cut until `parity_heavy` is run locally and recorded in the PR description (output captured per the Verification Visibility Protocol in `AGENTS.md`).
- **HLO byte counts are NOT a numeric pass/fail gate.** They are review artifacts captured at Phase 0. PRs touching JIT-relevant code surface a diff that reviewers inspect. CI `assert_zero_copy_overhead` runs per parity-pinned callable; allowlisted thresholds with rationale are checked into the repo. There is no blanket ±5% rule.
- **Pickle stability of `RunSpec` is NOT promised across the Phase 3 boundary.** Prefer **JSON** (`prxteinmpnn.run.spec_json`, `prxteinmpnn spec validate`) for durable configs and CLI. Legacy pickle-based workflows are unsupported unless you maintain a private adapter.
- **Cold `import prxteinmpnn` time is measured, not asserted.** Phase 1 records the number after `mp.set_start_method` removal. No hard ms threshold is in DoD; the lazy-import experiment is opt-in.
- **`state_vmap_exact == vmap(single_state)` is NOT asserted unconditionally.** It is a hypothesis tested by the Phase 0a SPIKE. The roadmap's Phase 4 outcome adapts to the spike result.
- **Specific shape numbers** like `n_canonical`, `n_states` are not pinned in this document. Payload static fields carry whatever the call-sites already compute.

---

## 11. Definition of done

The roadmap is complete when **all** of the following are measurably true:

1. **`parity_fast` green at every commit on `main`.** **`parity_heavy` recorded green in each phase tag's release notes** (manual gate, not CI).
2. **`src/prxteinmpnn/model/mpnn.py` ≤ 600 LoC**, with `PrxteinMPNN` and `PrxteinLigandMPNN` in separate files, no cross-class private static method calls.
3. **Zero `Callable[..., Any]`** in module API surface (enforced by `ty check` strict). Zero `inspect.signature` calls. Zero hardcoded debug-log paths. Zero top-level `mp.set_start_method` calls.
4. **`uv run jaxlint check src`** (or repo root) with `JL*` enabled is **recommended on maintainer machines** when touching JIT-heavy paths; **default CI does not require jaxlint** and must **not** block merges on jaxlint (advisory-only policy; see §7.1).
5. **`assert_zero_copy_overhead`** runs for all four parity-pinned callables; any regressions sit in the allowlist with a rationale string.
6. **StableHLO export** of `model.__call__` succeeds (validates §11 closure).
7. **Cold-start wall-time benchmark** runs in CI (advisory); no regression in the mpnn-split benchmark exceeds 20% without an explanation in the PR.
8. **`scripts/engaging/` audit complete** (Phase 3): every `RunSpecification(...)` call-site updated or explicitly waived in writing.
9. **Spec interchange:** JSON round-trip tests and `prxteinmpnn spec` CLI exercised in CI-relevant paths (pickle migration script **optional** / descoped if JSON-only policy holds).
10. **Phase 0a SPIKE outcome documented** with go/no-go decision and matching Phase 4 implementation. **Split acceptance:** the spike (numeric + HLO evidence + recorded go/no-go in a PR) may complete **before** Phase 4; the clause *“matching Phase 4 implementation”* is satisfied only when Phase 4 registry/unification (or routing) PRs merge. Track the spike slice under sprint **Phase 3b PR1** (`.agents/SPRINT_refactor-phase3b-20260506.md`).

---

## 12. Mapping to existing tech-debt items

| Tech-debt § | Title (short) | Phase that closes it | Notes |
|---|---|---|---|
| §1 | Precision casting (training) | Phase 3 | `PrecisionConfig` carrier in composed `RunSpec`. |
| §2 | Resource allocation wiring | Phase 3 | `compute_resource_allocation` invoked from `run/prep.py`. |
| §6 | Docstring preservation | Phase 6 (with partial in Phase 2) | Protocol docstrings land in Phase 2; full sweep in Phase 6. |
| §7 | Proxide/Prolix migration | Phase 6 | Depends on `SamplingDriver` split (Phase 5). |
| §9 | mpnn.py split | Phase 5 (sub-PRs 5a–5e) | Requires registries (Phase 4) to avoid replicating duplication. |
| §10 | Protocols/contracts | Phase 2 | 6 callable-boundary Protocols; `DesignSink` deferred to Phase 5; `BiasHook` dropped. |
| §11 | StableHLO export | Phase 1 | Hardcoded debug-log blocks deleted; ongoing review-artifact diffs. |
| §12 | ensemble → jaxbeans | Phase 5 (sub-PR 5h) | `ensemble/dbscan.py`, `ensemble/pca.py` relocated. |
| §13 | Doc hygiene | Cross-cutting §7.2 (standalone) | No longer Phase-1-coupled. |
| §14 | io_callback streaming | Phase 5 (sub-PR 5g) | `async_indexed_stream` (vendored) + `BoundedCallbackHandler` + `effects_barrier`. |

---

## 13. Open Question Resolution Log

Each Open Question now has a **default decision** that holds unless a triggering phase explicitly overturns it. Open Questions no longer block Phase 0.

| # | Question | Default decision | Trigger phase | Confirmation artifact |
|---|---|---|---|---|
| Q1 | Vendor or depend on jaxbeans pieces? | **CONFIRMED 2026-05-05.** Mixed per §3.6 matrix. VENDOR for `utils/callbacks` (need `effects_barrier` customization), `utils/testing`, `utils/typing`. DEPEND for `core/profiling`, `utils/mapping`, `utils/io`, `core/safety`, `jax_io/sources`. | Phase 0 (vendor pieces); Phase 5 (depend pieces) | `pyproject.toml [tool.uv.sources]` and `prxteinmpnn/utils/_vendored_callbacks.py` header |
| Q2 | jaxbeans distribution model? | **`uv` workspace member during refactor; PyPI release when jaxbeans hits 0.1.0.** | Phase 0 | `pyproject.toml` |
| Q3 | Where do `ensemble/*` live in jaxbeans? | **Default: jaxbeans `ml/clustering/`** (new submodule). Confirm with jaxbeans maintainer before sub-PR 5h opens. | Phase 5 sub-PR 5h | jaxbeans-side PR link |
| Q4 | `RunSpecification` deprecation window? | **Hard cutover for pickled instances** + one-minor-version kwarg shim with `DeprecationWarning`. **JSON** is the supported interchange for new tooling (Typer CLI: `prxteinmpnn spec`). | Phase 3 | JSON round-trip tests + `spec validate` |
| Q5 | `PRXTEINMPNN_VERIFY` default? | **Off in CI fast tests; on in `tests/parity/` via fixture; configurable in `parity_heavy`.** | Phase 0 | README + fixture in `tests/parity/conftest.py` |
| Q6 | (NEW) `state_vmap_exact` unification feasibility? | **CONFIRMED 2026-05-05: proceed with SPIKE; expectation upgraded to UNIFY.** Owner has prior informal evidence that `state_vmap_exact == jax.vmap(single_state_path)` numerically. SPIKE remains mandatory to formalize (capture HLO + parity fixtures + recorded PR decision). Phase 4 plans toward unification; only downgrades to routing if the formal SPIKE surfaces an unexpected divergence. | Phase 0a / Phase 4 | SPIKE PR records numeric + HLO comparison; Phase 4 unification PR (or routing PR on no-go) |
| Q7 | (NEW) Lazy `__init__.py` (PEP 562)? | **Deferred / opt-in experiment.** Revisit only if Phase 1 cold-import measurement is unacceptable. | Post-Phase-1 (optional) | Cold-import benchmark in PR |
| Q8 | (NEW) HLO threshold per call site? | **Allowlist file at `tests/profiling/hlo_allowlist.toml`** with rationale strings. No blanket %. | Phase 0 | Allowlist file with rationale entries |

### 13.1 `parity_heavy` targeted repro closure (Engaging, 2026-05-06)

**Symptom:** On GPU (`JAX_PLATFORMS=cuda`), the full `parity_heavy` slice showed **four** failures—protein projected edge `allclose`, packer **mean** (then concentration/mix at risk), ligand **`y_edges`** `allclose`—while encoder/decoder/AR parity checks largely **passed**.

**Diagnosis:** Reference PyTorch runs **`device='cpu'`** for packer and uses float32 GEMM semantics; JAX on an **A100** used default GPU matmul accumulation paths. Protein diag showed **Pearson 1.0** vs reference `W_e(features)` but **~1.1e-2 max_abs** under tight `rtol=1e-5`—consistent with **precision policy**, not wrong weights (`eqx` vs `pt_convert` finals were identical).

**Fixes merged (tests + scripts):**

- **`pytest.mark.parity_targeted`** on the four repro nodes; **`scripts/engaging/submit_parity_targeted.sh`** (~4 tests, short wall clock) vs the full heavy submit script.
- **`scripts/diag_protein_feature_parity.py`**, **`diag_ligand_feature_parity.py`**, **`diag_packer_parity.py`** run before pytest in both Slurm scripts unless **`PRXTEIN_SKIP_DIAG=1`**.
- **Protein / ligand** feature parity tests: **`jax.config.update("jax_default_matmul_precision", "highest")`** for the assertion block.
- **Packer:** **`_forward_jax_packer_for_parity`** — same matmul setting, then **`jax.default_device(cpu)`** when a CPU device exists so JAX matches the CPU reference packer.
- **Reference env:** `dm-tree` + `biopython` in `prxteinmpnn[tests]` and tev_design **`[dependency-groups] dev`** so `sc_utils` / `Bio` import on Engaging.

**Recorded green (targeted gate):** Slurm job **`13440956`** — **`4 passed`**, pytest **~23 s**, Slurm **`COMPLETED` `ExitCode 0:0`**. Diag tail: packer **mean** max_abs **~1.7e-6**; ligand **`y_nodes` / `y_edges` / `y_m`** at **≤ ~1.6e-6** vs PyTorch on the fixed path.

**Release posture:** Treat **`parity_targeted`** as a **fast GPU smoke** for the former red quartet. **`parity_heavy` full suite** (`submit_parity_heavy_ligand.sh`) remains the **manual release gate** per §11 / DoD—re-run it once per release or when JIT/feature code near these paths changes.

**Full `parity_heavy` gate (closed):** Slurm job **`13441413`** — **`COMPLETED`**, **`ExitCode 0:0`**, wall **~19m 27s** (ended **2026-05-06** 14:46 cluster time). Pytest: **`24 passed`**, **`55 deselected`**, **`3 warnings`**, **`1067.40 s`** (~**17m 47s**) on **`tests/parity` + `tests/model/test_ligandmpnn_equivalence.py`** with **`-m parity_heavy`**. Stack tail in log: **JAX 0.10.0** / **jaxlib 0.10.0**, **torch 2.11.0+cu130**. This satisfies the §11 manual gate for the **whole** heavy matrix (not only **`parity_targeted`**).

**Post-run notes (non-fatal):** log shows **DeprecationWarning** (Haiku, e3nn_jax) and a **Equinox `UserWarning`** (JAX array marked static in `mpnn.py` during ligand feature parity)—worth a follow-up issue/PR but **did not fail** the gate.

### 13.2 Phase 0a spike — recorded **GO** (numeric + dual HLO advisory, 2026-05-07)

**Verdict: GO** for Phase 4 *entry* on the **unconditional ProteinMPNN** `state_vmap_exact` vs explicit `jax.vmap` reference stack (synthetic payloads in `tests/sampling/spikes/test_state_vmap_exact_spike.py`; not wired to `tests/parity` LigandMPNN fixtures).

| Gate | Result |
| :--- | :--- |
| **Numeric** | `jnp.allclose(logits_sv, logits_ref, rtol/atol=get_tolerances(float32))` — **PASS** (`parity_fast`: `n_states=2`, `n_can=6`, key 101). |
| **HLO (advisory)** | Both paths lowered via `export_hlo`; `UserWarning` metrics logged per path. Example local run: **`state_vmap_exact`** — bytes **145368**, newlines **2348**, `custom_call_markers` **0**; **`explicit_vmap_ref`** — bytes **145369**, newlines **2348**, `custom_call_markers` **0** (byte delta **1**; no allowlist assertions—process evidence only). |
| **`parity_heavy` slice** | Engaging Slurm **`13445172`** (**2026-05-07**) with **`REFERENCE_PATH=/home/maarxaru/repos/LigandMPNN`**: **`COMPLETED` `ExitCode 0:0`**, wall **~65 s**. Pytest: **`parity_fast`** — `1 passed, 1 deselected`; **`parity_heavy`** — `1 passed, 1 deselected` (~5.5 s each); HLO advisory lines on heavy run: **`state_vmap_exact`** bytes **145368** / **`explicit_vmap_ref`** bytes **145369** (same pattern as §13.2 fast row). |

**NO-GO would apply if:** numeric `allclose` failed, or the team later mandates HLO ceilings in CI (then extend `tests/profiling/hlo_allowlist.toml` + assert).

**Reference checkout:** Engaging default parity scripts use **`REFERENCE_PATH=/home/maarxaru/repos/LigandMPNN`** (LigandMPNN clone with `model_params/`). Locally that corresponds to **`~/repos/LigandMPNN`** when your home layout matches; export **`REFERENCE_PATH`** explicitly if paths differ.

---

## 14. Sprint status (Phase 0a GO + PR2b; Phase 4 prep)

| Field | Value |
| :--- | :--- |
| **task_id** | `refactor-sprint-20260507-phase0a-go-pr2-sample` (active sprint); OODA cycle `refactor-sprint-20260507-ooda`; prior `refactor-phase4-pr2-20260506`, `refactor-phase4-entry-20260505`, `refactor-phase3b-sprint-20260506`, `refactor-phase3-sprint-20260505` |
| **Last update** | 2026-05-05 — Phase 4 **multistate dispatch slice**: `MultistateModeDescriptor` rows + `multistate_mode_descriptor()` wired through `mpnn.__call__`, `make_sample_sequences`, `make_score_fn`; STE validates mode at entry; `logs/` gitignored. §13.2 Phase 0a **GO** unchanged. |
| **Current phase** | **Phase 3b signed off** on `main`. **Phase 4 prep:** `_COMBINE_INDEX` + **multistate descriptor registry** landed; **`state_vmap_exact` jax.vmap unification** vs registry-only routing remains the next behavioral Phase 4 decision (§13.2). **PR2b** `sample.py` tuple branches when `multistate_stack is None` remain in the separate sprint doc. |
| **Still open** | **Phase 4:** `state_vmap_exact` unify vs registry-route; `SAMPLERS` / `SamplingDriver` (Phase 5). **PR2b:** `sample.py` tuple branches when `multistate_stack is None`. **Defer:** portable RunSpec JSON v3 follow-ups (Jacobian / CIF / Inspection round-trips). |
| **Plan** | **Active:** `.agents/SPRINT_refactor-phase0a-go-pr2-sample-20260507.md`. **Prior / superseded plan body:** `.agents/SPRINT_refactor-phase3c-0a-pr2-20260506.md` (retain for PR2a history). **Prior (retained):** `.agents/SPRINT_refactor-phase4-entry-20260505.md`. **Prior / closed (do not delete):** `.agents/SPRINT_refactor-phase3b-20260506.md`, `.agents/SPRINT_refactor-phase3-20260505.md`. |
| **Prior landed (Phase 2)** | `protocols.py`, `model/capabilities.py`, introspection removal at sample/score/averaging, honest casts on score paths; sprint `refactor-phase2-sprint-20260505`, plan `.agents/SPRINT_refactor-phase2-20260505.md`. |
| **Prior phase** | Phase 1: `task_id` `refactor-phase1-sprint-20260505` (§14 prior row archived in git history). |

---

*End of roadmap. Phase 2 typed boundaries are landed; Phase 3b portable RunSpec JSON v2 + hygiene signed off; active sprint — §14 and `.agents/SPRINT_refactor-phase0a-go-pr2-sample-20260507.md` (prior Phase 3c / PR2a doc `.agents/SPRINT_refactor-phase3c-0a-pr2-20260506.md` retained; Phase 4 entry `.agents/SPRINT_refactor-phase4-entry-20260505.md` retained).*
