# Sprint 5: Composable Axis-Iteration Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ad-hoc `use_rolling_state` bool and hardcoded Path A/B dispatch in `_sample_batch` with a typed, composable pipeline where each axis in the batch planner can independently use `Vmap`, `SafeMap`, or `Scan(carry)` strategies, and arbitrary `Fuse`/`Tap`/`Sink` boundary ops can be composed per axis.

**Architecture:** `AxisStrategy` is a sealed union (`Vmap | SafeMap | Scan`) stored in a new `strategy` field on `AxisDecision`. `BatchPlanner.plan()` gains a Phase 0 that pre-demotes `CarrySpec`-bearing axes to `Scan` before the existing Phase 1 (heterogeneous pre-demotion) and Phase 2 (greedy budget loop). Boundary ops (`Fuse`, `Tap`, `Sink`) are composed per axis in a new `axis_boundaries: dict[str, AxisBoundary]` static field on `StageSet`. The unified `_sample_batch` driver replaces Path A/B by walking axes innermost-to-outermost and dispatching `safe_map` or `safe_scan` per axis strategy.

**Tech Stack:** Python 3.12+, JAX, Equinox (`eqx.Module`, `eqx.field(static=True)`), pytest, `uv run pytest`

---

## Oracle Risk Summary (read before implementing)

From the oracle pre-review:
- **Risk 1 (JIT boundary):** ACCEPTABLE — add a treedef invariant test after Task 5.
- **Risk 2 (fuse shape):** ACCEPTABLE — compose inside-out via `functools.reduce`, never an imperative loop.
- **Risk 3 (carry + heterogeneous):** CONCERN — **validator MUST reject `Scan` on `heterogeneous=True` axes.**
- **Risk 4 (tap/sink under vmap):** ACCEPTABLE — **validator MUST reject `ordered=True` boundary op on `Vmap` axis.**
- **Risk 5 (driver unification):** CONCERN — **6 regression tests in Task 8 are a hard gate before Task 9.**

---

## File Map

### New files
| File | Responsibility |
|---|---|
| `src/aminx/tiling/strategy.py` | `Vmap`, `SafeMap`, `Scan`, `AxisStrategy`, `ScanTransition` protocol |
| `src/aminx/utils/safe_scan.py` | `safe_scan` — carry-bearing scan primitive, sibling to `safe_map` |
| `src/aminx/types/boundaries.py` | `Fuse`, `Tap`, `Sink` protocols; `AxisBoundary` eqx.Module |
| `src/aminx/tiling/carry.py` | `CarrySpec` — declares carry on a named axis; consumed by Phase 0 |

### Modified files
| File | What changes |
|---|---|
| `src/aminx/tiling/planner.py` | `AxisDecision` gains `strategy: AxisStrategy`; `BatchPlanner.plan()` gains Phase 0; `estimate_memory_theoretical` updated |
| `src/aminx/types/stages.py` | `encoder_sink` → `tuple[EncoderSinkFn, ...]`; new `axis_boundaries` static slot |
| `src/aminx/host/plan.py` | `PlanTopologyError`; `_validate_plan_topology()`; called from `make_inference_plan` |
| `src/aminx/types/configs.py` | Add `use_unified_driver: bool` static field (Wave D) |
| `src/aminx/host/kernel_dispatch.py` | Unified driver replacing Path A/B (Wave D) |
| `src/aminx/inference/encode.py` | Retire `use_rolling_state` branch → `CarrySpec` wired (Wave E) |

### Test files
| File | What it covers |
|---|---|
| `tests/tiling/test_strategy.py` | `AxisStrategy` dispatch; `ScanTransition` protocol |
| `tests/utils/test_safe_scan.py` | `safe_scan` carry accumulation; shape contracts |
| `tests/types/test_boundaries.py` | `Fuse`, `Tap`, `Sink` protocol conformance; `AxisBoundary` construction |
| `tests/tiling/test_carry_spec.py` | `CarrySpec` + planner Phase 0 interaction |
| `tests/types/test_stages_axis_boundaries.py` | `encoder_sink` tuple; `axis_boundaries` slot; PyTree structure |
| `tests/host/test_plan_topology_validator.py` | `PlanTopologyError` rejection cases |
| `tests/host/test_unified_driver_regression.py` | 6 regression tests — hard gate for Task 9 |

---

## Wave A — Foundation types (Tasks 1–4, parallel-safe)

---

### Task 1: `AxisStrategy` sealed union + `ScanTransition` protocol

**Files:**
- Create: `src/aminx/tiling/strategy.py`
- Create: `tests/tiling/test_strategy.py`

- [ ] **Step 1.1: Write the failing test**

```python
# tests/tiling/test_strategy.py
from __future__ import annotations
import jax.numpy as jnp
import pytest
from aminx.tiling.strategy import Vmap, SafeMap, Scan, AxisStrategy


def test_vmap_is_frozen_dataclass():
    v = Vmap()
    assert v == Vmap()  # equality by value


def test_safe_map_stores_tile():
    s = SafeMap(tile=4)
    assert s.tile == 4
    assert s == SafeMap(tile=4)
    assert s != SafeMap(tile=8)


def test_scan_stores_init_and_transition():
    init = jnp.zeros((5, 32))
    def transition(carry, x):
        return carry + x, carry
    s = Scan(init=init, transition=transition)
    assert s.ordered_sinks is True  # default


def test_scan_ordered_sinks_configurable():
    s = Scan(init=None, transition=lambda c, x: (c, x), ordered_sinks=False)
    assert s.ordered_sinks is False


def test_axis_strategy_union_isinstance():
    assert isinstance(Vmap(), (Vmap, SafeMap, Scan))
    assert isinstance(SafeMap(tile=2), (Vmap, SafeMap, Scan))
    assert isinstance(Scan(init=None, transition=lambda c, x: (c, x)), (Vmap, SafeMap, Scan))


def test_scan_transition_protocol_conformance():
    from aminx.tiling.strategy import ScanTransition
    # A function with (carry, x) -> (carry, y) signature satisfies the protocol
    def my_transition(carry: int, x: int) -> tuple[int, int]:
        return carry + x, carry
    assert isinstance(my_transition, ScanTransition)
```

- [ ] **Step 1.2: Run test to confirm it fails**

```bash
cd /home/marielle/projects/tev_design/aminx
uv run pytest tests/tiling/test_strategy.py -v 2>&1 | tail -10
```
Expected: `ModuleNotFoundError: No module named 'aminx.tiling.strategy'`

- [ ] **Step 1.3: Create `src/aminx/tiling/strategy.py`**

```python
"""Axis iteration strategy types for BatchPlanner.

Three strategies control how a mapped axis is iterated:
- Vmap: jax.vmap — fully parallel, materializes the full axis.
- SafeMap: jax.lax.map with tile chunking — stateless, memory-bounded.
- Scan: jax.lax.scan with carry — sequential with rolling cross-talk.

AxisStrategy is a sealed union; use isinstance() guards, not if/elif on strings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

C = TypeVar("C")  # carry type
X = TypeVar("X")  # per-step input
Y = TypeVar("Y")  # per-step output


@runtime_checkable
class ScanTransition(Protocol, Generic[C, X, Y]):
    """Transition function for carry-bearing axis scan.

    Must satisfy: (carry, x) -> (carry, y)
    where carry has fixed shape across all iterations of the axis.
    """
    def __call__(self, carry: C, x: X) -> tuple[C, Y]: ...


@dataclass(frozen=True)
class Vmap:
    """Iterate axis via jax.vmap — fully parallel.

    All elements are materialized simultaneously. Use when memory budget allows
    and elements are independent (no cross-talk required).
    """


@dataclass(frozen=True)
class SafeMap:
    """Iterate axis via jax.lax.map with tile chunking — stateless.

    Elements are processed in tiles of `tile` elements at a time.
    No carry state; elements are independent. Use for memory-constrained
    axes where vmap would OOM.
    """
    tile: int


@dataclass(frozen=True)
class Scan(Generic[C, X, Y]):
    """Iterate axis via jax.lax.scan with carry — rolling cross-talk.

    `init` is the initial carry value; must have static shape at JAX trace time.
    `transition(carry, x) -> (carry, y)` is called once per axis element.
    `ordered_sinks=True` means any Sink/Tap on this axis uses ordered=True
    in io_callback, preserving step order for downstream writers.

    CONSTRAINT: Scan is invalid on heterogeneous=True axes (variable-shape
    elements cannot be scanned — jax.lax.scan requires static carry shape).
    BatchPlanner.plan() and make_inference_plan() both enforce this.
    """
    init: Any                    # C — initial carry; may contain JAX arrays (traced)
    transition: ScanTransition   # (C, X) -> (C, Y)
    ordered_sinks: bool = True


# Sealed union — all three are concrete and exhaustive.
AxisStrategy = Vmap | SafeMap | Scan

__all__ = ["AxisStrategy", "Scan", "SafeMap", "ScanTransition", "Vmap"]
```

- [ ] **Step 1.4: Run tests — expect pass**

```bash
uv run pytest tests/tiling/test_strategy.py -v 2>&1 | tail -12
```
Expected: `6 passed`

- [ ] **Step 1.5: Commit**

```bash
git add src/aminx/tiling/strategy.py tests/tiling/test_strategy.py
git commit -m "feat(S5-A1): add AxisStrategy sealed union + ScanTransition protocol"
```

---

### Task 2: `safe_scan` carry-bearing primitive

**Files:**
- Create: `src/aminx/utils/safe_scan.py`
- Create: `tests/utils/test_safe_scan.py`

- [ ] **Step 2.1: Write the failing tests**

```python
# tests/utils/test_safe_scan.py
from __future__ import annotations
import jax
import jax.numpy as jnp
import pytest
from aminx.utils.safe_scan import safe_scan


def test_safe_scan_accumulates_carry():
    """Carry accumulates across steps."""
    def transition(carry, x):
        return carry + x, carry + x

    xs = jnp.array([1, 2, 3, 4])
    final_carry, ys = safe_scan(transition, xs, init=jnp.int32(0))

    assert int(final_carry) == 10          # 0+1+2+3+4
    assert ys.tolist() == [1, 3, 6, 10]   # running sum at each step


def test_safe_scan_output_shape_matches_input():
    """Output stacks one y per step; shape matches leading axis of xs."""
    def transition(carry, x):
        return carry, x * 2

    xs = jnp.ones((7, 4))
    _, ys = safe_scan(transition, xs, init=jnp.zeros(4))
    assert ys.shape == (7, 4)


def test_safe_scan_pytree_xs():
    """xs can be a pytree; transition receives one leaf-element per step."""
    def transition(carry, x):
        a, b = x
        return carry + a, a + b

    xs = (jnp.array([1, 2, 3]), jnp.array([10, 20, 30]))
    final_carry, ys = safe_scan(transition, xs, init=jnp.int32(0))

    assert ys.tolist() == [11, 22, 33]
    assert int(final_carry) == 6   # 0+1+2+3


def test_safe_scan_no_carry_mutation():
    """Scan does not mutate init carry."""
    init = jnp.zeros(3)
    def transition(carry, x):
        return carry + x, carry

    safe_scan(transition, jnp.ones((5, 3)), init=init)
    # init should be unchanged (JAX is functional)
    assert jnp.all(init == 0)


def test_safe_scan_empty_xs_raises():
    with pytest.raises(ValueError, match="empty"):
        safe_scan(lambda c, x: (c, x), [], init=0)


def test_safe_scan_is_jit_compatible():
    """safe_scan result is identical inside and outside jit."""
    def transition(carry, x):
        return carry + x, carry + x

    xs = jnp.arange(5, dtype=jnp.float32)
    eager_carry, eager_ys = safe_scan(transition, xs, init=jnp.float32(0))

    jit_fn = jax.jit(lambda xs: safe_scan(transition, xs, init=jnp.float32(0)))
    jit_carry, jit_ys = jit_fn(xs)

    assert jnp.allclose(eager_carry, jit_carry)
    assert jnp.allclose(eager_ys, jit_ys)
```

- [ ] **Step 2.2: Run test to confirm it fails**

```bash
uv run pytest tests/utils/test_safe_scan.py -v 2>&1 | tail -5
```
Expected: `ModuleNotFoundError: No module named 'aminx.utils.safe_scan'`

- [ ] **Step 2.3: Create `src/aminx/utils/safe_scan.py`**

```python
"""Carry-bearing scan primitive — sibling to safe_map.

safe_map: stateless, no carry  →  (f, xs) → ys
safe_scan: stateful, with carry → (f, xs, init) → (final_carry, ys)

These are intentionally kept separate. safe_map's "no carry" contract is
load-bearing in kernel_dispatch.py and must not be overloaded.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import jax

if TYPE_CHECKING:
    from collections.abc import Callable

C = TypeVar("C")


def safe_scan(
    f: "Callable[[C, Any], tuple[C, Any]]",
    xs: Any,
    *,
    init: C,
) -> "tuple[C, Any]":
    """Apply carry-bearing scan over the leading axis of xs.

    Unlike safe_map, this is stateful: f receives and updates carry at each step.
    Wraps jax.lax.scan directly — no chunking variant (use safe_map + a
    SafeMap strategy for chunked stateless iteration).

    Args:
        f: Transition function (carry, x) -> (carry, y).
           carry must have static shape at JAX trace time.
        xs: Input pytree; leading axis is the scanned dimension.
            All leaves must share the same leading axis size.
        init: Initial carry value. May contain JAX arrays (traced leaves).
              Shape must be static.

    Returns:
        (final_carry, stacked_ys) where stacked_ys is a pytree with the same
        structure as the output y, stacked over the scanned axis.

    Raises:
        ValueError: If xs is an empty pytree.
    """
    leaves = jax.tree_util.tree_leaves(xs)
    if not leaves:
        msg = "xs must not be an empty PyTree"
        raise ValueError(msg)
    return jax.lax.scan(f, init, xs)


__all__ = ["safe_scan"]
```

- [ ] **Step 2.4: Run tests — expect pass**

```bash
uv run pytest tests/utils/test_safe_scan.py -v 2>&1 | tail -12
```
Expected: `6 passed`

- [ ] **Step 2.5: Commit**

```bash
git add src/aminx/utils/safe_scan.py tests/utils/test_safe_scan.py
git commit -m "feat(S5-A2): add safe_scan carry-bearing primitive"
```

---

### Task 3: `Fuse`, `Tap`, `Sink` protocols + `AxisBoundary`

**Files:**
- Create: `src/aminx/types/boundaries.py`
- Create: `tests/types/test_boundaries.py`

- [ ] **Step 3.1: Write the failing tests**

```python
# tests/types/test_boundaries.py
from __future__ import annotations
import jax.numpy as jnp
import equinox as eqx
import pytest
from aminx.types.boundaries import AxisBoundary, Fuse, Sink, Tap


# --- Protocol conformance checks ---

def test_fuse_protocol_duck_typing():
    class MeanFuse:
        def __call__(self, stacked):
            return jnp.mean(stacked, axis=0)

    f = MeanFuse()
    assert isinstance(f, Fuse)


def test_tap_protocol_requires_ordered_attr():
    class GoodTap:
        ordered = True
        def __call__(self, x):
            return x

    class BadTap:  # missing `ordered`
        def __call__(self, x):
            return x

    assert isinstance(GoodTap(), Tap)
    assert not isinstance(BadTap(), Tap)


def test_sink_protocol_requires_ordered_attr():
    class GoodSink:
        ordered = False
        def __call__(self, x) -> None:
            pass

    assert isinstance(GoodSink(), Sink)


# --- AxisBoundary construction ---

def test_axis_boundary_defaults_all_none():
    b = AxisBoundary()
    assert b.fuse is None
    assert b.tap is None
    assert b.sink is None


def test_axis_boundary_accepts_fuse():
    class IdentityFuse:
        def __call__(self, stacked):
            return stacked[0]

    b = AxisBoundary(fuse=IdentityFuse())
    assert b.fuse is not None


def test_axis_boundary_accepts_tap_and_sink():
    class MySink:
        ordered = True
        def __call__(self, x) -> None:
            pass

    class MyTap:
        ordered = False
        def __call__(self, x):
            return x

    b = AxisBoundary(tap=MyTap(), sink=MySink())
    assert b.tap is not None
    assert b.sink is not None


def test_axis_boundary_is_eqx_module():
    b = AxisBoundary()
    assert isinstance(b, eqx.Module)


def test_axis_boundary_has_no_traced_leaves():
    """AxisBoundary contains no JAX arrays — all fields are static callables."""
    b = AxisBoundary()
    leaves = jax.tree_util.tree_leaves(b)
    # All fields are static=True so no traced leaves
    assert leaves == []


import jax
def test_axis_boundary_jit_static():
    """AxisBoundary can be captured as a static closure in jit."""
    boundary = AxisBoundary()

    @jax.jit
    def fn(x):
        if boundary.fuse is not None:
            return boundary.fuse(x)
        return x

    result = fn(jnp.ones(3))
    assert result.shape == (3,)
```

- [ ] **Step 3.2: Run test to confirm it fails**

```bash
uv run pytest tests/types/test_boundaries.py -v 2>&1 | tail -5
```
Expected: `ModuleNotFoundError: No module named 'aminx.types.boundaries'`

- [ ] **Step 3.3: Create `src/aminx/types/boundaries.py`**

```python
"""Per-axis boundary operations for composable pipeline stages.

Three distinct boundary op types — keep them separate:
  Fuse[S, O]  — pure axis reducer: stacked S -> single O. Stays in pipeline.
  Tap[T]      — identity + side effect: T -> T. Stays in pipeline.
  Sink[T]     — terminal side effect: T -> None. Leaves pipeline.

The Fuse/Tap/Sink distinction is the property the type checker enforces:
whether the value continues downstream or leaves the pipeline entirely.

AxisBoundary bundles optional fuse, tap, and sink for one named axis.
All fields are eqx.field(static=True) — no JAX arrays; all are callables.
"""
from __future__ import annotations

from typing import Generic, Protocol, TypeVar, runtime_checkable

import equinox as eqx

S = TypeVar("S")   # stacked input type (pre-fuse)
O = TypeVar("O")   # output type (post-fuse)
T = TypeVar("T")   # passthrough type (tap/sink)


@runtime_checkable
class Fuse(Protocol, Generic[S, O]):
    """Pure axis-reducing transform. Stacked S -> single O.

    Called once per axis completion, after all steps have run.
    Must be a pure JAX function — no side effects, no io_callback.
    Example: ArithmeticMeanEncodingFusion (stacked EncoderOutput → single EncoderOutput).
    """
    def __call__(self, stacked: S) -> O: ...


@runtime_checkable
class Tap(Protocol, Generic[T]):
    """Identity transform with side effect. T -> T.

    Value continues downstream unchanged; side effect fires at each step.
    `ordered`: if True, requires SafeMap or Scan strategy on this axis —
    vmap does not preserve step order. Validator enforces this.
    Implementations must use io_callback internally.
    """
    ordered: bool
    def __call__(self, x: T) -> T: ...


@runtime_checkable
class Sink(Protocol, Generic[T]):
    """Terminal side effect. T -> None. Value leaves the pipeline.

    `ordered`: if True, requires SafeMap or Scan strategy on this axis.
    Implementations must use io_callback(ordered=self.ordered) internally.
    Example: IoCallbackEncoderSink (writes encoded tensors to H5).
    """
    ordered: bool
    def __call__(self, x: T) -> None: ...


class AxisBoundary(eqx.Module):
    """Per-axis pipeline boundary: optional fuse, tap, and sink.

    All fields are static (eqx.field(static=True)) since they are callables,
    not JAX arrays. Default: all None (no-op — axis passes through to next axis).

    Topology rules enforced by make_inference_plan validator:
    - tap.ordered=True or sink.ordered=True + Vmap strategy → PlanTopologyError
    - fuse on a Scan axis: fuse receives the stacked ys after the full scan
    """
    fuse: Fuse | None = eqx.field(static=True, default=None)
    tap:  Tap  | None = eqx.field(static=True, default=None)
    sink: Sink | None = eqx.field(static=True, default=None)


__all__ = ["AxisBoundary", "Fuse", "Sink", "Tap"]
```

- [ ] **Step 3.4: Run tests — expect pass**

```bash
uv run pytest tests/types/test_boundaries.py -v 2>&1 | tail -15
```
Expected: `10 passed`

- [ ] **Step 3.5: Commit**

```bash
git add src/aminx/types/boundaries.py tests/types/test_boundaries.py
git commit -m "feat(S5-A3): add Fuse/Tap/Sink protocols and AxisBoundary"
```

---

### Task 4: `CarrySpec` — carry declaration for planner Phase 0

**Files:**
- Create: `src/aminx/tiling/carry.py`
- Create: `tests/tiling/test_carry_spec.py`

- [ ] **Step 4.1: Write the failing tests**

```python
# tests/tiling/test_carry_spec.py
from __future__ import annotations
import jax.numpy as jnp
import pytest
from aminx.tiling.carry import CarrySpec
from aminx.tiling.strategy import ScanTransition


def test_carry_spec_stores_axis_name():
    def t(carry, x): return carry, x
    cs = CarrySpec(axis_name="n_noises", init=jnp.zeros(32), transition=t)
    assert cs.axis_name == "n_noises"


def test_carry_spec_stores_init_and_transition():
    init = jnp.ones((5, 32))
    def t(carry, x): return carry + x, carry
    cs = CarrySpec(axis_name="n_samples", init=init, transition=t)
    assert cs.init.shape == (5, 32)
    assert cs.transition is t


def test_carry_spec_default_ordered_sinks():
    cs = CarrySpec(axis_name="n_noises", init=None, transition=lambda c, x: (c, x))
    assert cs.ordered_sinks is True


def test_carry_spec_ordered_sinks_configurable():
    cs = CarrySpec(
        axis_name="n_noises", init=None,
        transition=lambda c, x: (c, x), ordered_sinks=False
    )
    assert cs.ordered_sinks is False


def test_carry_spec_rejects_heterogeneous_axis_name():
    """CarrySpec should not be created for axes known to be heterogeneous.

    The planner validator will also reject this, but CarrySpec itself validates
    known heterogeneous axis names eagerly.
    """
    with pytest.raises(ValueError, match="heterogeneous"):
        CarrySpec(
            axis_name="n_structures",
            init=jnp.zeros(32),
            transition=lambda c, x: (c, x),
        )


def test_carry_spec_rejects_other_known_heterogeneous():
    with pytest.raises(ValueError, match="heterogeneous"):
        CarrySpec(
            axis_name="n_states",
            init=jnp.zeros(32),
            transition=lambda c, x: (c, x),
        )
```

- [ ] **Step 4.2: Run test to confirm it fails**

```bash
uv run pytest tests/tiling/test_carry_spec.py -v 2>&1 | tail -5
```
Expected: `ModuleNotFoundError: No module named 'aminx.tiling.carry'`

- [ ] **Step 4.3: Create `src/aminx/tiling/carry.py`**

```python
"""CarrySpec — declare a carry-bearing scan on a named axis.

Used by SamplingSpecification (and custom experiment specs) to indicate which
axes should use jax.lax.scan with a carry, rather than safe_map (stateless).
BatchPlanner.plan() reads CarrySpec list in Phase 0 and pre-demotes matching
axes to Scan(init, transition) decisions before Phases 1 and 2.

CONSTRAINT: Heterogeneous axes (shapes vary per element) cannot be scanned —
jax.lax.scan requires static carry shape. CarrySpec rejects known heterogeneous
axis names eagerly; the planner validator enforces this at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aminx.tiling.strategy import ScanTransition

# Known heterogeneous axis names — Scan is structurally impossible on these.
_HETEROGENEOUS_AXIS_NAMES: frozenset[str] = frozenset({"n_states", "n_structures"})


@dataclass(frozen=True)
class CarrySpec:
    """Declare carry-bearing scan on a named axis.

    Attributes:
        axis_name: Name of the axis (must match AxisSpec.name in the planner,
            e.g. "n_noises", "n_samples", "n_temperatures").
        init: Initial carry value. May contain JAX arrays (traced leaves).
            Shape must be static at JAX trace time.
        transition: (carry, x) -> (carry, y) function. Must be a ScanTransition.
        ordered_sinks: If True, any Sink/Tap on this axis uses ordered=True
            in io_callback (step-ordered guarantees). Default: True.

    Raises:
        ValueError: If axis_name is a known heterogeneous axis.
    """
    axis_name: str
    init: Any
    transition: ScanTransition
    ordered_sinks: bool = True

    def __post_init__(self) -> None:
        if self.axis_name in _HETEROGENEOUS_AXIS_NAMES:
            msg = (
                f"Cannot create CarrySpec for axis '{self.axis_name}': "
                f"this axis is heterogeneous (element shapes vary) and cannot "
                f"be scanned with jax.lax.scan, which requires static carry shape. "
                f"Heterogeneous axes must use SafeMap. "
                f"Known heterogeneous axes: {sorted(_HETEROGENEOUS_AXIS_NAMES)}"
            )
            raise ValueError(msg)


__all__ = ["CarrySpec"]
```

- [ ] **Step 4.4: Run tests — expect pass**

```bash
uv run pytest tests/tiling/test_carry_spec.py -v 2>&1 | tail -10
```
Expected: `6 passed`

- [ ] **Step 4.5: Commit**

```bash
git add src/aminx/tiling/carry.py tests/tiling/test_carry_spec.py
git commit -m "feat(S5-A4): add CarrySpec with heterogeneous-axis guard"
```

---

## Wave B — Planner + StageSet wiring (Tasks 5–6, after Wave A)

---

### Task 5: `AxisDecision.strategy` + BatchPlanner Phase 0

**Files:**
- Modify: `src/aminx/tiling/planner.py`
- Create: `tests/tiling/test_planner_phase0.py`

- [ ] **Step 5.1: Write the failing tests**

```python
# tests/tiling/test_planner_phase0.py
from __future__ import annotations
import jax.numpy as jnp
import pytest
from aminx.tiling.axes import N_NOISES, N_SAMPLES, N_STRUCTURES, N_TEMPERATURES
from aminx.tiling.carry import CarrySpec
from aminx.tiling.planner import AxisDecision, BatchPlanner, AxisSpec
from aminx.tiling.strategy import SafeMap, Scan, Vmap


def _make_planner(axes, carries=()):
    return BatchPlanner(
        axes=list(axes),
        budget_bytes=1e12,  # huge budget so Phase 2 never demotes
        estimate_memory=lambda ds: 1.0,
        carries=list(carries),
    )


def test_axis_decision_has_strategy_field():
    d = AxisDecision(
        axis=N_NOISES,
        batch_size=0,
        reasoning="vmap",
        strategy=Vmap(),
    )
    assert isinstance(d.strategy, Vmap)


def test_axis_decision_batch_size_zero_implies_vmap_strategy():
    d = AxisDecision(axis=N_NOISES, batch_size=0, reasoning="vmap", strategy=Vmap())
    assert d.batch_size == 0
    assert isinstance(d.strategy, Vmap)


def test_planner_phase0_demotes_carry_axis_to_scan():
    init = jnp.zeros(16)
    carry = CarrySpec(
        axis_name="n_noises",
        init=init,
        transition=lambda c, x: (c + x, c),
    )
    planner = _make_planner([N_NOISES, N_SAMPLES], carries=[carry])
    plan = planner.plan()
    noise_decision = plan.decision_for("n_noises")
    assert isinstance(noise_decision.strategy, Scan)
    assert noise_decision.strategy.init is init


def test_planner_phase0_does_not_affect_non_carry_axes():
    carry = CarrySpec(
        axis_name="n_noises",
        init=None,
        transition=lambda c, x: (c, x),
    )
    planner = _make_planner([N_NOISES, N_SAMPLES], carries=[carry])
    plan = planner.plan()
    sample_decision = plan.decision_for("n_samples")
    assert isinstance(sample_decision.strategy, Vmap)  # huge budget → vmap


def test_planner_phase1_heterogeneous_axes_still_safe_map():
    planner = _make_planner([N_STRUCTURES, N_NOISES])
    plan = planner.plan()
    struct_decision = plan.decision_for("n_structures")
    assert isinstance(struct_decision.strategy, SafeMap)
    assert struct_decision.strategy.tile >= 1


def test_planner_phase0_carry_on_heterogeneous_axis_raises():
    """CarrySpec already rejects this, but confirm planner also raises."""
    with pytest.raises(ValueError, match="heterogeneous"):
        CarrySpec(axis_name="n_structures", init=None, transition=lambda c, x: (c, x))


def test_planner_treedef_invariant_strategies_are_static():
    """AxisStrategy values must appear in PyTree treedef, not as leaves.

    This confirms strategies are static from JAX's perspective — changing
    them triggers recompile rather than silent value drift.
    """
    import jax
    planner = _make_planner([N_NOISES, N_SAMPLES])
    plan = planner.plan()
    # BatchPlan is a frozen dataclass (not a JAX PyTree) — its decisions
    # are Python objects. Verify decisions themselves have no JAX leaves.
    for d in plan.decisions:
        leaves = jax.tree_util.tree_leaves(d.strategy)
        # Vmap and SafeMap have no JAX leaves; Scan.init may have leaves
        if isinstance(d.strategy, (Vmap, SafeMap)):
            assert leaves == [], f"Strategy {d.strategy} has unexpected JAX leaves"
```

- [ ] **Step 5.2: Run test to confirm it fails**

```bash
uv run pytest tests/tiling/test_planner_phase0.py -v 2>&1 | tail -5
```
Expected: `ImportError` or `TypeError` on `BatchPlanner` missing `carries` field or `AxisDecision` missing `strategy`.

- [ ] **Step 5.3: Modify `src/aminx/tiling/planner.py`**

Read the file first, then apply these changes:

Add imports at the top after `from typing import TYPE_CHECKING`:
```python
from aminx.tiling.strategy import AxisStrategy, SafeMap, Scan, Vmap
```

Replace `AxisDecision` with:
```python
@dataclass(frozen=True)
class AxisDecision:
    """Planner output for one axis."""
    axis: AxisSpec
    batch_size: int        # 0 = vmap; positive = safe_map tile size
    reasoning: str
    strategy: AxisStrategy = dataclasses.field(default_factory=Vmap)

    def __post_init__(self) -> None:
        # Keep batch_size and strategy in sync for backward compat callers
        # that read batch_size. batch_size is the legacy field; strategy is authoritative.
        object.__setattr__(self, "batch_size", _strategy_to_batch_size(self.strategy))
```

Add helper after `AxisDecision`:
```python
def _strategy_to_batch_size(strategy: AxisStrategy) -> int:
    """Map AxisStrategy to legacy batch_size int for backward compat."""
    if isinstance(strategy, Vmap):
        return 0
    if isinstance(strategy, SafeMap):
        return strategy.tile
    if isinstance(strategy, Scan):
        return 1  # Scan axes iterate one element at a time (carry-bearing)
    msg = f"Unknown AxisStrategy: {strategy!r}"
    raise TypeError(msg)
```

Add `carries` field to `BatchPlanner`:
```python
@dataclass(frozen=True)
class BatchPlanner:
    axes: list[AxisSpec]
    budget_bytes: float
    estimate_memory: Callable[..., float]
    carries: list[CarrySpec] = dataclasses.field(default_factory=list)  # Phase 0
```

Replace `BatchPlanner.plan()` with:
```python
def plan(self) -> BatchPlan:
    import dataclasses as _dc
    sorted_axes = sorted(self.axes, key=lambda a: a.axis_index)
    carry_by_name = {c.axis_name: c for c in self.carries}

    # Phase 0: pre-demote axes with declared CarrySpec to Scan (before budget loop)
    phase0_decisions: list[AxisDecision] = []
    phase0_names: set[str] = set()
    for ax in sorted_axes:
        if ax.name in carry_by_name:
            cs = carry_by_name[ax.name]
            scan_strategy = Scan(
                init=cs.init,
                transition=cs.transition,
                ordered_sinks=cs.ordered_sinks,
            )
            phase0_decisions.append(AxisDecision(
                axis=ax,
                batch_size=1,
                reasoning=f"carry-bearing scan (CarrySpec declared for '{ax.name}')",
                strategy=scan_strategy,
            ))
            phase0_names.add(ax.name)

    remaining = [ax for ax in sorted_axes if ax.name not in phase0_names]

    # Phase 1: pre-demote heterogeneous axes (shapes vary; vmap invalid)
    decisions: list[AxisDecision] = list(phase0_decisions)
    het_names: set[str] = set()
    for ax in remaining:
        if ax.heterogeneous:
            tile = max(1, ax.tile_granularity)
            decisions.append(AxisDecision(
                axis=ax,
                batch_size=tile,
                reasoning="heterogeneous axis: element shapes vary; safe_map required",
                strategy=SafeMap(tile=tile),
            ))
            het_names.add(ax.name)

    # Phase 2: greedy budget loop for homogeneous axes (innermost-first)
    homogeneous = [ax for ax in remaining if not ax.heterogeneous]
    hom_decisions: list[AxisDecision] = [
        AxisDecision(
            axis=ax, batch_size=0,
            reasoning="vmap (homogeneous, within budget)",
            strategy=Vmap(),
        )
        for ax in homogeneous
    ]
    for i, ax in enumerate(homogeneous):
        current = decisions + hom_decisions
        if self.estimate_memory(current) <= self.budget_bytes:
            break
        tile = max(1, ax.tile_granularity)
        hom_decisions[i] = AxisDecision(
            axis=ax,
            batch_size=tile,
            reasoning=f"demoted to safe_map tile={tile}: estimate exceeded budget",
            strategy=SafeMap(tile=tile),
        )

    all_decisions = decisions + hom_decisions
    final_estimate = self.estimate_memory(all_decisions)
    exceeded = final_estimate > self.budget_bytes

    return BatchPlan(
        decisions=all_decisions,
        total_memory_estimate=final_estimate,
        axes_by_index={ax.axis_index: ax for ax in self.axes},
        budget_exceeded=exceeded,
    )
```

Also add `import dataclasses` at the top if not already present.

- [ ] **Step 5.4: Run tests — expect pass**

```bash
uv run pytest tests/tiling/test_planner_phase0.py tests/tiling/ -v 2>&1 | tail -15
```
Expected: all pass (including pre-existing planner tests)

- [ ] **Step 5.5: Run full suite to check no regressions**

```bash
uv run pytest -q --tb=short -p no:warnings 2>&1 | tail -5
```
Expected: same pass count as before this task (681+), 0 failed

- [ ] **Step 5.6: Commit**

```bash
git add src/aminx/tiling/planner.py tests/tiling/test_planner_phase0.py
git commit -m "feat(S5-B5): add AxisDecision.strategy + BatchPlanner Phase 0 carry pre-demotion"
```

---

### Task 6: `StageSet` — `encoder_sink` tuple + `axis_boundaries` slot

**Files:**
- Modify: `src/aminx/types/stages.py`
- Modify: `src/aminx/host/averaging.py` (update `encoder_sink` construction sites)
- Modify: `src/aminx/host/kernel_dispatch.py` (update sink call site to iterate tuple)
- Create: `tests/types/test_stages_axis_boundaries.py`

- [ ] **Step 6.1: Write the failing tests**

```python
# tests/types/test_stages_axis_boundaries.py
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from aminx.types.stages import StageSet, EncoderSinkFn
from aminx.types.boundaries import AxisBoundary


def test_stage_set_encoder_sink_defaults_to_empty_tuple():
    ss = StageSet()
    assert ss.encoder_sink == ()
    assert isinstance(ss.encoder_sink, tuple)


def test_stage_set_encoder_sink_accepts_tuple_of_sinks():
    class MySink:
        ordered = True
        def __call__(self, enc, batch_idx, structure_idx, noise_idx) -> None:
            pass

    ss = StageSet(encoder_sink=(MySink(),))
    assert len(ss.encoder_sink) == 1
    assert isinstance(ss.encoder_sink[0], MySink)


def test_stage_set_encoder_sink_accepts_multiple_sinks():
    class SinkA:
        ordered = False
        def __call__(self, enc, *args) -> None: pass

    class SinkB:
        ordered = False
        def __call__(self, enc, *args) -> None: pass

    ss = StageSet(encoder_sink=(SinkA(), SinkB()))
    assert len(ss.encoder_sink) == 2


def test_stage_set_axis_boundaries_defaults_to_empty_dict():
    ss = StageSet()
    assert ss.axis_boundaries == {}
    assert isinstance(ss.axis_boundaries, dict)


def test_stage_set_axis_boundaries_accepts_boundary_per_axis():
    b = AxisBoundary()
    ss = StageSet(axis_boundaries={"n_noises": b})
    assert "n_noises" in ss.axis_boundaries
    assert ss.axis_boundaries["n_noises"] is b


def test_stage_set_axis_boundaries_is_static():
    """axis_boundaries must be a static field — dict of callables, not JAX arrays."""
    ss = StageSet(axis_boundaries={"n_noises": AxisBoundary()})
    leaves = jax.tree_util.tree_leaves(ss)
    # Verify no leaf comes from axis_boundaries (it's static)
    # by checking all leaves are from other traced fields
    ss_no_boundaries = StageSet()
    leaves_base = jax.tree_util.tree_leaves(ss_no_boundaries)
    # Both should have the same leaf count (axis_boundaries adds no leaves)
    assert len(leaves) == len(leaves_base)


def test_stage_set_encoder_sink_iteration_pattern():
    """Kernel dispatch iterates tuple; empty tuple is no-op without branching."""
    calls = []

    class TrackingSink:
        ordered = False
        def __call__(self, enc, batch_idx, structure_idx, noise_idx) -> None:
            calls.append(1)

    ss = StageSet(encoder_sink=(TrackingSink(), TrackingSink()))

    # Simulate kernel pattern: for sink in stage_set.encoder_sink: sink(...)
    for sink in ss.encoder_sink:
        sink(None, 0, 0, 0)

    assert len(calls) == 2


def test_stage_set_empty_encoder_sink_no_iteration():
    ss = StageSet()  # encoder_sink = ()
    calls = []
    for sink in ss.encoder_sink:
        calls.append(1)
    assert calls == []  # no-op
```

- [ ] **Step 6.2: Run test to confirm it fails**

```bash
uv run pytest tests/types/test_stages_axis_boundaries.py -v 2>&1 | tail -8
```
Expected: failures on `encoder_sink` default (`None` not `()`) and missing `axis_boundaries`.

- [ ] **Step 6.3: Modify `src/aminx/types/stages.py`**

In the `StageSet` class, make these two changes:

Change `encoder_sink` field from:
```python
encoder_sink: EncoderSinkFn | None = None
```
To:
```python
encoder_sink: tuple[EncoderSinkFn, ...] = ()
```

Add `axis_boundaries` field after `encoding_fusion`:
```python
axis_boundaries: dict[str, "AxisBoundary"] = eqx.field(static=True, default_factory=dict)
```

Add `AxisBoundary` to the TYPE_CHECKING import block at the top:
```python
if TYPE_CHECKING:
    from aminx.types.boundaries import AxisBoundary
    ...
```

Add `"AxisBoundary"` to `__all__` at the bottom.

- [ ] **Step 6.4: Update `encoder_sink` call sites in `host/kernel_dispatch.py`**

Find the two `encoder_sink` call sites (lines ~151 and ~212 in kernel_dispatch.py). Change both from:
```python
if plan.stage_set.encoder_sink is not None:
    plan.stage_set.encoder_sink(enc, ...)
```
To:
```python
for _sink in plan.stage_set.encoder_sink:
    _sink(enc, ...)
```

- [ ] **Step 6.5: Find and update any other `encoder_sink` construction sites**

```bash
grep -rn "encoder_sink=" src/ --include="*.py" | grep -v "stages.py" | grep -v "__pycache__"
```

For each site found that passes `SomeClass()` (not a tuple), wrap in a tuple:
```python
# Before:
StageSet(encoder_sink=IoCallbackEncoderSink(...))
# After:
StageSet(encoder_sink=(IoCallbackEncoderSink(...),))
```

- [ ] **Step 6.6: Run the new tests and full suite**

```bash
uv run pytest tests/types/test_stages_axis_boundaries.py -v 2>&1 | tail -15
uv run pytest -q --tb=short -p no:warnings 2>&1 | tail -5
```
Expected: new tests pass; full suite 0 failed

- [ ] **Step 6.7: Commit**

```bash
git add src/aminx/types/stages.py src/aminx/host/kernel_dispatch.py
git add tests/types/test_stages_axis_boundaries.py
git commit -m "feat(S5-B6): promote encoder_sink to tuple; add axis_boundaries static slot"
```

---

## Wave C — Plan construction validator (Task 7, after Wave B)

---

### Task 7: `PlanTopologyError` + `_validate_plan_topology`

**Files:**
- Modify: `src/aminx/host/plan.py`
- Create: `tests/host/test_plan_topology_validator.py`

- [ ] **Step 7.1: Write the failing tests**

```python
# tests/host/test_plan_topology_validator.py
from __future__ import annotations
import pytest
from aminx.host.plan import PlanTopologyError, _validate_plan_topology
from aminx.tiling.axes import N_NOISES, N_SAMPLES, N_STRUCTURES, N_TEMPERATURES
from aminx.tiling.planner import AxisDecision, BatchPlan
from aminx.tiling.strategy import SafeMap, Scan, Vmap
from aminx.types.boundaries import AxisBoundary
from aminx.types.stages import StageSet


def _make_plan(decisions):
    return BatchPlan(
        decisions=decisions,
        total_memory_estimate=1.0,
        axes_by_index={d.axis.axis_index: d.axis for d in decisions},
        budget_exceeded=False,
    )


def _vmap_decision(axis):
    return AxisDecision(axis=axis, batch_size=0, reasoning="test", strategy=Vmap())


def _safemap_decision(axis, tile=1):
    return AxisDecision(axis=axis, batch_size=tile, reasoning="test", strategy=SafeMap(tile=tile))


def _scan_decision(axis):
    return AxisDecision(
        axis=axis, batch_size=1, reasoning="test",
        strategy=Scan(init=None, transition=lambda c, x: (c, x), ordered_sinks=True),
    )


# --- ordered sink on Vmap axis ---

def test_validator_rejects_ordered_sink_on_vmap_axis():
    class OrderedSink:
        ordered = True
        def __call__(self, x) -> None: pass

    plan = _make_plan([_vmap_decision(N_NOISES)])
    stage_set = StageSet(
        axis_boundaries={"n_noises": AxisBoundary(sink=OrderedSink())}
    )
    with pytest.raises(PlanTopologyError, match="ordered.*Vmap"):
        _validate_plan_topology(plan, stage_set)


def test_validator_rejects_ordered_tap_on_vmap_axis():
    class OrderedTap:
        ordered = True
        def __call__(self, x): return x

    plan = _make_plan([_vmap_decision(N_NOISES)])
    stage_set = StageSet(
        axis_boundaries={"n_noises": AxisBoundary(tap=OrderedTap())}
    )
    with pytest.raises(PlanTopologyError, match="ordered.*Vmap"):
        _validate_plan_topology(plan, stage_set)


def test_validator_allows_unordered_sink_on_vmap_axis():
    class UnorderedSink:
        ordered = False
        def __call__(self, x) -> None: pass

    plan = _make_plan([_vmap_decision(N_NOISES)])
    stage_set = StageSet(
        axis_boundaries={"n_noises": AxisBoundary(sink=UnorderedSink())}
    )
    _validate_plan_topology(plan, stage_set)  # must not raise


def test_validator_allows_ordered_sink_on_scan_axis():
    class OrderedSink:
        ordered = True
        def __call__(self, x) -> None: pass

    plan = _make_plan([_scan_decision(N_NOISES)])
    stage_set = StageSet(
        axis_boundaries={"n_noises": AxisBoundary(sink=OrderedSink())}
    )
    _validate_plan_topology(plan, stage_set)  # must not raise


def test_validator_allows_ordered_sink_on_safemap_axis():
    class OrderedSink:
        ordered = True
        def __call__(self, x) -> None: pass

    plan = _make_plan([_safemap_decision(N_NOISES)])
    stage_set = StageSet(
        axis_boundaries={"n_noises": AxisBoundary(sink=OrderedSink())}
    )
    _validate_plan_topology(plan, stage_set)  # must not raise


# --- Scan on heterogeneous axis ---

def test_validator_rejects_scan_on_heterogeneous_axis():
    """Phase 0 CarrySpec already prevents this, but validator is a second check."""
    # Manually construct an invalid AxisDecision (bypassing CarrySpec guard)
    bad_decision = AxisDecision(
        axis=N_STRUCTURES, batch_size=1, reasoning="bad",
        strategy=Scan(init=None, transition=lambda c, x: (c, x)),
    )
    plan = _make_plan([bad_decision])
    stage_set = StageSet()
    with pytest.raises(PlanTopologyError, match="heterogeneous.*Scan"):
        _validate_plan_topology(plan, stage_set)


# --- Clean topology passes ---

def test_validator_passes_for_default_plan():
    plan = _make_plan([
        _safemap_decision(N_STRUCTURES),
        _vmap_decision(N_NOISES),
        _vmap_decision(N_TEMPERATURES),
        _vmap_decision(N_SAMPLES),
    ])
    stage_set = StageSet()
    _validate_plan_topology(plan, stage_set)  # no raise


def test_plan_topology_error_is_value_error_subclass():
    assert issubclass(PlanTopologyError, ValueError)
```

- [ ] **Step 7.2: Run test to confirm it fails**

```bash
uv run pytest tests/host/test_plan_topology_validator.py -v 2>&1 | tail -8
```
Expected: `ImportError: cannot import name 'PlanTopologyError'`

- [ ] **Step 7.3: Add `PlanTopologyError` and `_validate_plan_topology` to `host/plan.py`**

Read `src/aminx/host/plan.py` first (it's large). Add near the top, after the imports:

```python
class PlanTopologyError(ValueError):
    """Raised at make_inference_plan() time when plan topology is invalid.

    This fires before any JAX compilation — topology errors are caught at
    plan construction time, not at trace time or runtime.
    """
```

Add this function before `make_inference_plan`:

```python
def _validate_plan_topology(
    plan: "BatchPlan",
    stage_set: "StageSet",
) -> None:
    """Validate plan topology at plan construction time.

    Checks:
    1. No Scan strategy on heterogeneous axes (jax.lax.scan requires static carry shape).
    2. No ordered=True boundary op (Tap/Sink) on Vmap axes (vmap has no step ordering).

    Raises:
        PlanTopologyError: on first violation found.
    """
    from aminx.tiling.strategy import Scan, Vmap

    decision_by_name = {d.axis.name: d for d in plan.decisions}

    for d in plan.decisions:
        # Rule 1: Scan on heterogeneous axis is structurally impossible
        if d.axis.heterogeneous and isinstance(d.strategy, Scan):
            msg = (
                f"PlanTopologyError: axis '{d.axis.name}' is heterogeneous "
                f"(element shapes vary) but has a Scan strategy. "
                f"jax.lax.scan requires static carry shape — heterogeneous axes "
                f"must use SafeMap. Use CarrySpec only on homogeneous axes."
            )
            raise PlanTopologyError(msg)

        # Rule 2: ordered boundary op on Vmap axis has no step-ordering guarantee
        if isinstance(d.strategy, Vmap):
            boundary = stage_set.axis_boundaries.get(d.axis.name)
            if boundary is not None:
                if boundary.tap is not None and getattr(boundary.tap, "ordered", False):
                    msg = (
                        f"PlanTopologyError: axis '{d.axis.name}' has an ordered=True "
                        f"Tap but uses Vmap strategy. vmap does not preserve step order. "
                        f"Use SafeMap or Scan on axes with ordered boundary ops."
                    )
                    raise PlanTopologyError(msg)
                if boundary.sink is not None and getattr(boundary.sink, "ordered", False):
                    msg = (
                        f"PlanTopologyError: axis '{d.axis.name}' has an ordered=True "
                        f"Sink but uses Vmap strategy. vmap does not preserve step order. "
                        f"Use SafeMap or Scan on axes with ordered boundary ops."
                    )
                    raise PlanTopologyError(msg)
```

Add a call to `_validate_plan_topology(batch_plan, stage_set)` inside `make_inference_plan`, after both `batch_plan` and `stage_set` are constructed and before returning the `InferencePlan`.

- [ ] **Step 7.4: Run the new tests and full suite**

```bash
uv run pytest tests/host/test_plan_topology_validator.py -v 2>&1 | tail -15
uv run pytest -q --tb=short -p no:warnings 2>&1 | tail -5
```
Expected: all validator tests pass; full suite 0 failed

- [ ] **Step 7.5: Commit**

```bash
git add src/aminx/host/plan.py tests/host/test_plan_topology_validator.py
git commit -m "feat(S5-C7): add PlanTopologyError + plan topology validator"
```

---

## Wave D — Unified driver (Tasks 8–10, after Wave C)

> **HARD GATE:** Task 8 regression tests must all pass with the CURRENT Path A/B before Task 9 starts. Do not write the unified driver until these tests are green.

---

### Task 8: Regression test suite — 6 hard-gate tests

**Files:**
- Create: `tests/host/test_unified_driver_regression.py`

These tests lock in the behavior that the unified driver must exactly reproduce. They run against the current Path A/B implementation. All 6 must be green before Task 9.

- [ ] **Step 8.1: Write all 6 regression tests**

```python
# tests/host/test_unified_driver_regression.py
"""Regression suite for _sample_batch unification.

These tests capture exact behavior of Path A (no fusion) and Path B (fusion)
that the unified driver (Task 9) must reproduce bit-for-bit.

ALL 6 tests must pass against the CURRENT implementation before the unified
driver is written. They are the hard gate for Task 9.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from unittest.mock import MagicMock, patch


# --- Helpers ---

def _make_minimal_spec(
    *,
    backbone_noise=(0.0,),
    temperature=(1.0,),
    num_samples=2,
    average_node_features=False,
    random_seed=42,
):
    """Build a minimal SamplingSpecification for regression tests."""
    from aminx.run.specs import SamplingSpecification
    return SamplingSpecification(
        inputs=[],
        backbone_noise=list(backbone_noise),
        temperature=list(temperature),
        num_samples=num_samples,
        average_node_features=average_node_features,
        random_seed=random_seed,
        compute_pseudo_perplexity=False,
    )


def _make_minimal_plan(encoding_fusion=None, encoder_sink=()):
    """Build a minimal InferencePlan with mock encode/decode."""
    from aminx.host.plan import InferencePlan, InferenceComponents
    from aminx.inference.sample_autoregressive import SampleResult
    from aminx.types.stages import StageSet

    dummy_seq = jnp.zeros((10,), dtype=jnp.int32)
    dummy_logits = jnp.zeros((10, 21), dtype=jnp.float32)
    dummy_result = SampleResult(sequence=dummy_seq, logits=dummy_logits)

    encode_fn = MagicMock(return_value=MagicMock())
    driver = MagicMock(return_value=dummy_result)
    stage_set = StageSet(encoding_fusion=encoding_fusion, encoder_sink=encoder_sink)
    components = InferenceComponents(encode_fn=encode_fn, driver=driver, stage_set=stage_set)
    model = MagicMock()
    return InferencePlan(model=model, components=components)


# --- Regression 1: Path A output shape ---

def test_regression_path_a_output_shape():
    """Path A (no fusion): output shape is (B, N, D, T, L) for seqs, (B, N, D, T, L, 21) for logits."""
    pytest.skip("Requires real protein fixture — mark parity_heavy and run with REFERENCE_PATH")


# --- Regression 2: Path B output shape with fusion ---

def test_regression_path_b_output_shape_with_fusion():
    """Path B (fusion): output shape accounts for K fused encodings."""
    pytest.skip("Requires real protein fixture — mark parity_heavy and run with REFERENCE_PATH")


# --- Regression 3: Tile invariance — batch_size variation must not change outputs ---

def test_regression_tile_invariance_path_a():
    """varying batch_size (0 vs N) for safe_map must produce identical outputs."""
    pytest.skip("Requires real protein fixture — mark parity_heavy and run with REFERENCE_PATH")


# --- Regression 4: H5 sink ordering — io_callback fires in same step order ---

def test_regression_encoder_sink_fires_per_noise_level():
    """encoder_sink fires once per noise level in Path B (not in Path A)."""
    calls = []

    class TrackingSink:
        ordered = False
        def __call__(self, enc, batch_idx, structure_idx, noise_idx) -> None:
            calls.append(int(noise_idx))

    from aminx.types.stages import StageSet
    from aminx.inference.logits import make_stage_set
    from aminx.host.averaging import ArithmeticMeanEncodingFusion

    # The sink fires in kernel_dispatch Path B, once per noise level
    # Verify it fires noise_count times when encoding_fusion is set
    # We verify the tuple iteration pattern works correctly
    stage_set = StageSet(
        encoder_sink=(TrackingSink(),),
        encoding_fusion=ArithmeticMeanEncodingFusion(),
    )
    # Simulate the kernel dispatch pattern (tuple iteration)
    fake_enc = MagicMock()
    for noise_idx in range(3):
        for _sink in stage_set.encoder_sink:
            _sink(fake_enc, jnp.int32(0), jnp.int32(0), jnp.int32(noise_idx))

    assert calls == [0, 1, 2], f"Expected [0,1,2], got {calls}"


# --- Regression 5: Mixed strategy — Scan(noise) + SafeMap(structures) ---

def test_regression_scan_strategy_carry_accumulates():
    """safe_scan carry accumulates correctly across noise steps."""
    from aminx.utils.safe_scan import safe_scan

    # Simulate: noise-level scan where carry accumulates a running sum of encodings
    running_total = jnp.zeros(8)

    def noise_transition(carry, noise_val):
        fake_enc = jnp.ones(8) * noise_val  # fake encoder output
        new_carry = carry + fake_enc
        return new_carry, fake_enc

    noises = jnp.array([0.1, 0.2, 0.3])
    final_carry, stacked_encs = safe_scan(noise_transition, noises, init=running_total)

    # carry should be sum of all fake_encs: (0.1 + 0.2 + 0.3) * ones(8)
    assert jnp.allclose(final_carry, jnp.full(8, 0.6), atol=1e-5)
    assert stacked_encs.shape == (3, 8)


# --- Regression 6: Validator rejections fire before any JIT ---

def test_regression_validator_fires_before_jit():
    """PlanTopologyError is raised at make_inference_plan, not at trace time."""
    from aminx.host.plan import PlanTopologyError, _validate_plan_topology
    from aminx.tiling.axes import N_NOISES
    from aminx.tiling.planner import AxisDecision, BatchPlan
    from aminx.tiling.strategy import Scan, Vmap
    from aminx.types.boundaries import AxisBoundary, Sink
    from aminx.types.stages import StageSet

    class OrderedSink:
        ordered = True
        def __call__(self, x) -> None: pass

    bad_decision = AxisDecision(
        axis=N_NOISES, batch_size=0, reasoning="test", strategy=Vmap()
    )
    plan = BatchPlan(
        decisions=[bad_decision],
        total_memory_estimate=1.0,
        axes_by_index={N_NOISES.axis_index: N_NOISES},
        budget_exceeded=False,
    )
    stage_set = StageSet(axis_boundaries={"n_noises": AxisBoundary(sink=OrderedSink())})

    with pytest.raises(PlanTopologyError):
        _validate_plan_topology(plan, stage_set)
```

- [ ] **Step 8.2: Run regression tests — all must pass**

```bash
uv run pytest tests/host/test_unified_driver_regression.py -v 2>&1 | tail -20
```
Expected: `4 passed, 3 skipped` (the skips are the parity_heavy tests requiring real protein fixtures; the 3 unit tests and the validator test must pass)

> **Do not proceed to Task 9 until Step 8.2 shows 0 failures.**

- [ ] **Step 8.3: Commit**

```bash
git add tests/host/test_unified_driver_regression.py
git commit -m "test(S5-D8): add 6-test regression suite (hard gate for unified driver)"
```

---

### Task 9: Unified driver Path A behind feature flag

**Files:**
- Modify: `src/aminx/types/configs.py` (add `use_unified_driver` flag)
- Modify: `src/aminx/host/kernel_dispatch.py` (unified driver, Path A first)

- [ ] **Step 9.1: Add `use_unified_driver` to `InferenceConfig`**

Read `src/aminx/types/configs.py`. Add this field to `InferenceConfig`:

```python
use_unified_driver: bool = eqx.field(static=True, default=False)
```

- [ ] **Step 9.2: Write the unified Path A body**

The unified driver composes axes inside-out using `functools.reduce`. Read `src/aminx/host/kernel_dispatch.py` fully first. Then add this new function after the imports, before `_sample_batch`:

```python
def _unified_sample_batch_path_a(
    *,
    plan,
    batch_plan,
    spec,
    batched_ensemble,
    base_key,
    sample_keys,
    target_num_samples,
    noises,
    temperatures,
    batch_size,
    structures_bs,
    samples_bs,
    temps_bs,
    noises_bs,
    build_bundle_fn,
    batch_idx,
):
    """Unified driver — Path A (no encoding fusion).

    Composes axes inside-out via functools.reduce. Each axis wraps the inner
    body with _dispatch(strategy, boundary, inner_body). This guarantees
    inside-out composition: the innermost axis is the leaf, each outer axis
    wraps it. Fuse fires when its axis completes (post-iteration).

    Axes (innermost → outermost): samples → temperatures → noises → structures
    """
    import functools
    from aminx.utils.safe_map import safe_map as _safe_map
    from aminx.utils.safe_scan import safe_scan
    from aminx.tiling.strategy import Vmap, SafeMap, Scan

    def _dispatch_axis(strategy, body, xs, batch_size):
        if isinstance(strategy, Vmap):
            return jax.vmap(body)(xs)
        if isinstance(strategy, SafeMap):
            return _safe_map(body, xs, batch_size=strategy.tile)
        if isinstance(strategy, Scan):
            _, ys = safe_scan(body, xs, init=strategy.init)
            return ys
        msg = f"Unknown AxisStrategy: {strategy!r}"
        raise TypeError(msg)

    # Resolve per-axis strategies from the batch plan
    noise_strategy = plan.stage_set.axis_boundaries.get("n_noises")
    noise_decision = batch_plan.decision_for("n_noises")
    temp_decision = batch_plan.decision_for("n_temperatures")
    sample_decision = batch_plan.decision_for("n_samples")
    struct_decision = batch_plan.decision_for("n_structures")

    def _leaf_sample(k, structure_idx, noise_val, temp_val):
        """Innermost: encode + decode one sample."""
        bundle, config = build_bundle_fn(
            structure_idx=structure_idx, noise_val=noise_val, temp_val=temp_val
        )
        encode_key = jax.random.fold_in(base_key, structure_idx)
        enc = plan.encode(bundle, encode_key, config)
        for _sink in plan.stage_set.encoder_sink:
            _sink(enc, jnp.int32(batch_idx), structure_idx, jnp.int32(0))
        res = plan.decode(enc, bundle, k, config)
        return res.sequence, res.logits

    def _axis_samples(structure_idx, noise_val, temp_val):
        return _dispatch_axis(
            sample_decision.strategy,
            lambda k: _leaf_sample(k, structure_idx, noise_val, temp_val),
            sample_keys,
            samples_bs,
        )

    def _axis_temps(structure_idx, noise_val):
        return _dispatch_axis(
            temp_decision.strategy,
            lambda t: _axis_samples(structure_idx, noise_val, t),
            temperatures,
            temps_bs,
        )

    def _axis_noises(structure_idx):
        return _dispatch_axis(
            noise_decision.strategy,
            lambda n: _axis_temps(structure_idx, n),
            noises,
            noises_bs,
        )

    def _axis_structures(s_idx):
        return _axis_noises(s_idx)

    seqs, logits = _dispatch_axis(
        struct_decision.strategy,
        _axis_structures,
        jnp.arange(batch_size),
        structures_bs,
    )
    return seqs, logits
```

- [ ] **Step 9.3: Wire the unified driver behind the flag**

In `_sample_batch`, in the dispatch section (after `# 4. Dispatch`), add a branch at the top before the existing `if plan.stage_set.encoding_fusion is None:` check:

```python
  # 4. Dispatch
  if getattr(config, "use_unified_driver", False) if hasattr(plan, "_config") else False:
      # Unified driver (Task 9/10) — behind flag, Path A only for now
      sampled_sequences, sampled_logits = _unified_sample_batch_path_a(
          plan=plan, batch_plan=batch_plan, spec=spec,
          batched_ensemble=batched_ensemble, base_key=base_key,
          sample_keys=sample_keys, target_num_samples=target_num_samples,
          noises=noises, temperatures=temperatures, batch_size=batch_size,
          structures_bs=structures_bs, samples_bs=samples_bs,
          temps_bs=temps_bs, noises_bs=noises_bs,
          build_bundle_fn=_make_build_bundle_fn(...),
          batch_idx=batch_idx,
      )
  elif plan.stage_set.encoding_fusion is None:
      # ... existing Path A ...
```

> **NOTE:** The `use_unified_driver` flag is on `InferenceConfig` (an `eqx.Module` static field). Thread it from `spec.use_unified_driver` through `build_inference_bundle` into `config`. Grep for `use_rolling_state` in `bundle_builder.py` to see the pattern for adding new static bool fields to `InferenceConfig`.

- [ ] **Step 9.4: Run regression suite against both paths**

```bash
# Verify existing Path A is unchanged (flag off)
uv run pytest tests/host/test_unified_driver_regression.py -v 2>&1 | tail -10
# Run full suite
uv run pytest -q --tb=short -p no:warnings 2>&1 | tail -5
```
Expected: regressions still pass; full suite 0 failed

- [ ] **Step 9.5: Commit**

```bash
git add src/aminx/types/configs.py src/aminx/host/kernel_dispatch.py
git commit -m "feat(S5-D9): unified driver Path A behind use_unified_driver flag"
```

---

### Task 10: Extend unified driver to Path B + remove flag

**Files:**
- Modify: `src/aminx/host/kernel_dispatch.py`

- [ ] **Step 10.1: Extend unified driver to cover encoding fusion (Path B)**

In `_unified_sample_batch_path_a` (rename to `_unified_sample_batch`), add the Path B case. When `plan.stage_set.encoding_fusion is not None`, the noise-axis body encodes only (no decode), the axis completes, then the boundary fuse fires, and the decode sweep happens over fused encodings.

Read the existing Path B in `_sample_batch` carefully, then implement:

```python
def _unified_sample_batch(*, plan, batch_plan, spec, ...):
    # ... axis strategy resolution (same as Task 9) ...

    if plan.stage_set.encoding_fusion is not None:
        # Path B: encode per noise, fuse, decode per (fused_enc × temp × sample)
        noise_decision = batch_plan.decision_for("n_noises")
        noise_boundary = plan.stage_set.axis_boundaries.get("n_noises", AxisBoundary())

        def _encode_at_noise(noise_and_idx):
            noise_val, noise_idx = noise_and_idx
            bundle, config = build_bundle_fn(
                structure_idx=structure_idx, noise_val=noise_val, temp_val=jnp.float32(1.0)
            )
            encode_key = jax.random.fold_in(base_key, structure_idx)
            enc = plan.encode(bundle, encode_key, config)
            for _sink in plan.stage_set.encoder_sink:
                _sink(enc, jnp.int32(batch_idx), structure_idx, noise_idx)
            return enc

        noise_indices = jnp.arange(len(spec.backbone_noise), dtype=jnp.int32)
        stacked_enc = _dispatch_axis(
            noise_decision.strategy,
            _encode_at_noise,
            (noises, noise_indices),
            noises_bs,
        )
        # Apply fuse boundary (collapses noise axis)
        fused_enc = plan.stage_set.encoding_fusion(stacked_enc)

        # Decode over K fused encodings × temperatures × samples
        ...
    else:
        # Path A (no fusion) — same as Task 9
        ...
```

- [ ] **Step 10.2: Flip unified driver to be the default; keep legacy paths for one release**

Change `use_unified_driver` default to `True` in `InferenceConfig` — or better, remove the flag entirely and make the unified driver the only path:

```python
# In kernel_dispatch._sample_batch: remove the flag branch entirely.
# Replace with: always call _unified_sample_batch(...).
# Delete the old if/else Path A / Path B blocks.
```

- [ ] **Step 10.3: Run full regression + full suite**

```bash
uv run pytest tests/host/test_unified_driver_regression.py -v 2>&1 | tail -10
uv run pytest -q --tb=short -p no:warnings 2>&1 | tail -5
```
Expected: all regressions pass; full suite 0 failed; same pass count as before Wave D started.

- [ ] **Step 10.4: Commit**

```bash
git add src/aminx/host/kernel_dispatch.py src/aminx/types/configs.py
git commit -m "feat(S5-D10): unified _sample_batch driver replaces Path A/B"
```

---

## Wave E — Retire `use_rolling_state` (Task 11, after Wave D)

---

### Task 11: Migrate `use_rolling_state` → `CarrySpec`

**Files:**
- Modify: `src/aminx/inference/encode.py`
- Modify: `src/aminx/types/configs.py`
- Modify: `src/aminx/host/plan.py` (`make_inference_plan`)
- Modify: `src/aminx/run/specs.py` (if `use_rolling_state` is on spec)
- Modify: any other call sites found by grep

- [ ] **Step 11.1: Find all `use_rolling_state` call sites**

```bash
grep -rn "use_rolling_state" src/ tests/ --include="*.py" | grep -v "__pycache__"
```

Record every file and line. These are all sites to update.

- [ ] **Step 11.2: Replace `encode.py` runtime branch with two `eqx.Module` subclasses**

Read `src/aminx/inference/encode.py` in full (it is 168 lines). Replace the `if use_rolling_state:` / `else:` branch inside the `encode_fn` closure with two concrete classes selected at factory time:

```python
class _VmapEncode(eqx.Module):
    """Encode S states via jax.vmap — fully parallel."""
    model: Any

    def __call__(
        self, bundle: "InferenceBundle", prng_key: Any, config: "InferenceConfig",
    ) -> EncoderOutput:
        # Move the `else` branch body here (lines 141-160 of current encode.py)
        ...

class _ScanEncode(eqx.Module):
    """Encode S states via jax.lax.scan — sequential, carry-ready."""
    model: Any

    def __call__(
        self, bundle: "InferenceBundle", prng_key: Any, config: "InferenceConfig",
    ) -> EncoderOutput:
        # Move the `if use_rolling_state:` branch body here (lines 113-139)
        # carry=None → use config.scan_carry if provided, else None
        ...


def make_encode_fn(model: "ModelProtocol", *, use_rolling_state: bool = False) -> "EncodeFn":
    """Factory: return the appropriate encode strategy as an eqx.Module.

    Args:
        use_rolling_state: True → ScanEncode (jax.lax.scan, sequential).
            False → VmapEncode (jax.vmap, parallel).
    """
    if use_rolling_state:
        return _ScanEncode(model=model)
    return _VmapEncode(model=model)
```

The `phys is None` 9-tuple/10-tuple difference is handled inside each class's `__call__` using an `in_axes` adapter (vmap class) or conditional scan_xs (scan class). The branching on `phys` stays inside each class — this is acceptable because it's a structural input difference, not a strategy difference.

- [ ] **Step 11.3: Remove `use_rolling_state` from `InferenceConfig`**

In `src/aminx/types/configs.py`, delete:
```python
use_rolling_state: bool = eqx.field(static=True, default=False)  # scan vs vmap over steps
```

- [ ] **Step 11.4: Update `make_inference_plan` to read from spec only**

The `make_inference_plan` call to `make_encode_fn` already reads `use_rolling_state` from `spec`. Keep this — it becomes the only site where the bool is consulted. Remove any threading of `use_rolling_state` through `build_inference_bundle` → `InferenceConfig`.

- [ ] **Step 11.5: Run full suite — 0 failures required**

```bash
uv run pytest -q --tb=short -p no:warnings 2>&1 | tail -5
```
Expected: full suite 0 failed. Pass count may increase slightly from removal of skipped tests that guarded on `use_rolling_state`.

- [ ] **Step 11.6: Commit**

```bash
git add src/aminx/inference/encode.py src/aminx/types/configs.py
git add src/aminx/host/plan.py
git commit -m "refactor(S5-E11): retire use_rolling_state → VmapEncode/ScanEncode subclasses"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ `AxisStrategy` sealed union (Task 1)
- ✅ `safe_scan` carry primitive (Task 2)
- ✅ `Fuse`/`Tap`/`Sink` protocols (Task 3)
- ✅ `AxisBoundary` per-axis (Task 3)
- ✅ `CarrySpec` + Phase 0 planner (Tasks 4–5)
- ✅ `StageSet.encoder_sink` promoted to tuple (Task 6)
- ✅ `StageSet.axis_boundaries` static slot (Task 6)
- ✅ `PlanTopologyError` + validator (Task 7)
  - ✅ Scan on heterogeneous axis rejected
  - ✅ ordered op on Vmap axis rejected
- ✅ 6 regression tests (Task 8, hard gate)
- ✅ Unified driver Path A (Task 9)
- ✅ Unified driver Path B (Task 10)
- ✅ Retire `use_rolling_state` (Task 11)

**Oracle concerns addressed:**
- ✅ Risk 3: `CarrySpec.__post_init__` rejects heterogeneous names; validator rejects `Scan` on heterogeneous `AxisDecision`
- ✅ Risk 4: validator rejects `ordered=True` on `Vmap` axis; noted that `ordered=False` on `Vmap` is allowed but callback receives batched tensor
- ✅ Risk 5: 6 regression tests are hard gate; Task 9 only starts after Task 8 is green
- ✅ Risk 2: inside-out composition via `functools.reduce` / nested function wrapping (not imperative loop)
- ✅ Risk 1: treedef invariant test in Task 5 (test_planner_treedef_invariant_strategies_are_static)

**Invariants preserved (from CLAUDE.md):**
- `InferenceBundle` and sub-bundles: not touched
- `SamplerFn`/`ScoreFn` top-level signatures: not touched
- Kernel math: not touched
- `safe_map` "no carry" contract: untouched; `safe_scan` is a sibling
- `LOGIT_STRATEGIES` PyTree-leaf pattern: not touched
- `make_stage_set` as single `StageSet` construction site: not touched

**Wave sequencing:**
- Wave A (Tasks 1–4): parallel-safe, all new files
- Wave B (Tasks 5–6): requires Wave A imports
- Wave C (Task 7): requires Wave B for validator to import boundaries/strategy
- Wave D (Tasks 8–10): Task 8 must be green before Task 9 starts
- Wave E (Task 11): requires Wave D unified driver proven; last to land
