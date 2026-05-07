# Phase 6 Track A — Batch Layout Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a host-side batch planner that assigns vmap vs safe_map tile sizes per axis based on memory budget, with advisory logging at sampling/scoring entry points and active enforcement for heterogeneous axes (n_structures).

**Architecture:** Three layers: (1) `utils/batching.py` — pure-Python frozen dataclasses implementing AxisSpec/BatchPlan/BatchPlanner with a three-phase greedy algorithm (pre-demote heterogeneous axes → greedy budget loop → tile-size warping); (2) `utils/batching_registry.py` — 10 canonical AxisSpec instances covering all BatchingConfig fields; (3) wiring at run-layer entry points — advisory DEBUG logging for sweep axes, active safe_map dispatch for heterogeneous n_structures.

**Tech Stack:** Python `dataclasses` (frozen=True, no JAX), JAX `vmap`/`lax.map`, existing `utils/safe_map.py`.

**Spec:** `docs/superpowers/specs/2026-05-07-phase6-batch-layout-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/prxteinmpnn/utils/batching.py` | **Create** | AxisSpec, AxisDecision, BatchPlan, BatchPlanner, `ceil_to_granularity`, `estimate_memory_theoretical` |
| `src/prxteinmpnn/utils/batching_registry.py` | **Create** | 10 canonical AxisSpec constants (N_RESIDUES … N_APC_PAIRS) |
| `src/prxteinmpnn/utils/safe_map.py` | **Modify** | Add `batch_size=0` as vmap sentinel alongside `None` (line 49) |
| `src/prxteinmpnn/run/sampling.py` | **Modify** | Advisory BatchPlanner logging at `_sample_batch` (~line 744); active n_structures safe_map dispatch (~line 961) |
| `src/prxteinmpnn/run/scoring.py` | **Modify** | Advisory BatchPlanner logging at `score` (~line 110); active n_structures safe_map dispatch (~line 281) |
| `tests/utils/test_safe_map.py` | **Modify** | Add `batch_size=0` sentinel tests (file already exists) |
| `tests/utils/test_batching.py` | **Create** | Unit tests for core types, greedy algorithm, tile-size warping, memory formula |
| `tests/utils/test_batching_registry.py` | **Create** | Registry invariant tests |

---

## Parity Gate (run after every task's commit)

All commands run from the `prxteinmpnn/` directory:

```bash
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/model/test_ligand_wave_parallel.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -q
```

All four files must PASS before proceeding to the next task.

---

## Task 1: safe_map.py — add batch_size=0 vmap sentinel

**Files:**
- Modify: `src/prxteinmpnn/utils/safe_map.py` (line 49)
- Modify: `tests/utils/test_safe_map.py`

The existing dispatcher (line 49) treats `batch_size is None` as "use vmap." The planner produces `int` values where `0` means vmap. Add `batch_size == 0` to the existing condition. This is backward-compatible: existing callers passing `None` are unaffected.

- [ ] **Step 1: Add tests for batch_size=0 to existing test file**

Append to `tests/utils/test_safe_map.py`:

```python
def test_batch_size_zero_routes_to_vmap():
    """batch_size=0 must dispatch to vmap, identical output to batch_size=None."""
    xs = jnp.arange(8)
    result_zero = safe_map(lambda x: x * 3, xs, batch_size=0)
    result_none = safe_map(lambda x: x * 3, xs, batch_size=None)
    assert jnp.allclose(result_zero, result_none)

def test_batch_size_zero_pytree():
    xs = {"a": jnp.ones((4, 8)), "b": jnp.zeros((4, 8))}
    result = safe_map(lambda d: d["a"] + d["b"], xs, batch_size=0)
    assert result.shape == (4,)
```

- [ ] **Step 2: Run to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/utils/test_safe_map.py::test_batch_size_zero_routes_to_vmap -v
```

Expected: FAIL (batch_size=0 currently routes to lax.map because `0 <= 0` is True but `num_elements <= 0` is False for non-empty xs — the condition fails and falls through to lax.map).

- [ ] **Step 3: Edit safe_map.py line 49**

Change line 49 from:
```python
  if batch_size is None or num_elements <= batch_size:
```
to:
```python
  if batch_size is None or batch_size == 0 or num_elements <= batch_size:
```

- [ ] **Step 4: Run all safe_map tests**

```bash
PYTHONPATH=src uv run pytest tests/utils/test_safe_map.py -v
```

Expected: all PASS.

- [ ] **Step 5: Parity gate**

```bash
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/model/test_ligand_wave_parallel.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -q
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/prxteinmpnn/utils/safe_map.py tests/utils/test_safe_map.py
git commit -m "feat(safe_map): treat batch_size=0 as vmap sentinel alongside None"
```

---

## Task 2: Create utils/batching.py — core planner types

**Files:**
- Create: `src/prxteinmpnn/utils/batching.py`
- Create: `tests/utils/test_batching.py`

Pure Python, zero JAX imports, zero prxteinmpnn imports. All types `@dataclass(frozen=True)`.

Key invariants to preserve:
- `batch_size=0` → vmap; `batch_size>0` → safe_map with that tile size
- Memory formula: `base × ∏(cardinality_i if batch_size==0 else batch_size_i) × multiplier`
- Demotion target is `tile_granularity` (not `cardinality` — demoting to cardinality saves zero memory)
- Heterogeneous axes are pre-demoted to `tile_granularity` before the budget loop

- [ ] **Step 1: Write failing tests**

Create `tests/utils/test_batching.py`:

```python
import pytest
from prxteinmpnn.utils.batching import (
    AxisDecision,
    AxisSpec,
    BatchPlan,
    BatchPlanner,
    ceil_to_granularity,
    estimate_memory_theoretical,
)


def _axis(name, index, cardinality, default_bs, tile_gran, heterogeneous=False):
    return AxisSpec(
        name=name,
        axis_index=index,
        cardinality=cardinality,
        default_batch_size=default_bs,
        tile_granularity=tile_gran,
        heterogeneous=heterogeneous,
        doc="",
    )


# --- ceil_to_granularity ---

def test_ceil_already_aligned():
    assert ceil_to_granularity(128, 128) == 128

def test_ceil_unaligned():
    assert ceil_to_granularity(129, 128) == 256

def test_ceil_granularity_one():
    assert ceil_to_granularity(7, 1) == 7

def test_ceil_zero():
    assert ceil_to_granularity(0, 128) == 0


# --- estimate_memory_theoretical ---

def test_estimate_all_vmap():
    ax = _axis("a", 0, 8, 0, 1)
    d = AxisDecision(axis=ax, batch_size=0, reasoning="vmap")
    assert estimate_memory_theoretical([d], 1.0, 1.0) == 8.0

def test_estimate_safe_map_tile_1():
    ax = _axis("a", 0, 8, 1, 1)
    d = AxisDecision(axis=ax, batch_size=1, reasoning="safe_map")
    assert estimate_memory_theoretical([d], 1.0, 1.0) == 1.0

def test_estimate_safe_map_tile_equals_cardinality_same_as_vmap():
    ax = _axis("a", 0, 8, 8, 1)
    d_vmap = AxisDecision(axis=ax, batch_size=0, reasoning="vmap")
    d_safe = AxisDecision(axis=ax, batch_size=8, reasoning="safe_map cardinality")
    assert estimate_memory_theoretical([d_vmap], 1.0, 1.0) == \
           estimate_memory_theoretical([d_safe], 1.0, 1.0)

def test_estimate_mixed_two_axes():
    ax_states = _axis("n_states", 0, 4, 0, 1)
    ax_temps = _axis("n_temperatures", 1, 4, 1, 1)
    decisions = [
        AxisDecision(axis=ax_states, batch_size=0, reasoning="vmap"),
        AxisDecision(axis=ax_temps, batch_size=1, reasoning="safe_map"),
    ]
    # base=1, product=4×1=4, multiplier=2.5 → 10.0
    assert estimate_memory_theoretical(decisions, 1.0, 2.5) == pytest.approx(10.0)

def test_estimate_multiplier_applied():
    ax = _axis("a", 0, 4, 0, 1)
    d = AxisDecision(axis=ax, batch_size=0, reasoning="vmap")
    assert estimate_memory_theoretical([d], 2.0, 3.0) == pytest.approx(24.0)


# --- BatchPlanner.plan() ---

def _planner(axes, budget):
    return BatchPlanner(
        axes=axes,
        budget_bytes=budget,
        estimate_memory=lambda ds: estimate_memory_theoretical(ds, 1.0, 1.0),
    )


def test_planner_no_demotion_needed():
    ax = _axis("n_samples", 0, 8, 0, 1)
    plan = _planner([ax], budget=1000.0).plan()
    assert plan.decision_for("n_samples").batch_size == 0
    assert not plan.exceeded_budget()

def test_planner_demotes_innermost_first():
    # Two axes; budget forces demotion of ax0 (innermost, index=0)
    ax0 = _axis("n_states", 0, 4, 0, 1)
    ax1 = _axis("n_samples", 1, 8, 0, 1)
    # budget=8: vmap both → 4×8=32 > 8; demote ax0 → 1×8=8 <= 8
    plan = _planner([ax0, ax1], budget=8.0).plan()
    assert plan.decision_for("n_states").batch_size == 1   # demoted
    assert plan.decision_for("n_samples").batch_size == 0  # still vmap

def test_planner_tile_granularity_respected():
    ax = _axis("n_residues", 0, 1200, 0, 128)
    # budget=64: vmap → 1200 > 64; demote → tile=128
    plan = _planner([ax], budget=64.0).plan()
    assert plan.decision_for("n_residues").batch_size == 128

def test_planner_heterogeneous_always_safe_map_ignores_budget():
    ax = _axis("n_structures", 0, 32, 0, 1, heterogeneous=True)
    # huge budget: heterogeneous still gets safe_map
    plan = _planner([ax], budget=1e9).plan()
    assert plan.decision_for("n_structures").batch_size == 1

def test_planner_exceeded_budget_flag():
    # Even with minimum tile, estimate exceeds budget
    ax = _axis("n_residues", 0, 1200, 0, 128)
    # budget=0.5: even tile=128 → 128 > 0.5
    plan = _planner([ax], budget=0.5).plan()
    assert plan.exceeded_budget()

def test_planner_exceeded_budget_logs_warning_not_raises():
    ax = _axis("n_residues", 0, 1200, 0, 128)
    plan = _planner([ax], budget=0.5).plan()
    # exceeded_budget() returns bool, never raises
    result = plan.exceeded_budget()
    assert isinstance(result, bool)

def test_plan_decision_for_unknown_raises():
    ax = _axis("n_samples", 0, 8, 0, 1)
    plan = _planner([ax], budget=1000.0).plan()
    with pytest.raises(KeyError):
        plan.decision_for("nonexistent")
```

- [ ] **Step 2: Run to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/utils/test_batching.py::test_ceil_already_aligned -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create src/prxteinmpnn/utils/batching.py**

```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def ceil_to_granularity(n: int, g: int) -> int:
    """Round n up to the nearest multiple of g. If g <= 1, returns n unchanged."""
    if g <= 1 or n == 0:
        return n
    return ((n + g - 1) // g) * g


def estimate_memory_theoretical(
    decisions: list[AxisDecision],
    base_shape_bytes: float,
    activation_multiplier: float,
) -> float:
    """Estimate peak memory for a set of axis decisions.

    Axes with batch_size=0 (vmap) contribute their full cardinality to the
    memory product. Axes with batch_size>0 (safe_map) contribute only their
    tile size — one tile is live at a time.

    activation_multiplier must be supplied by the caller (no default) because
    it depends on execution context (inference vs training vs checkpointed).
    """
    product = 1
    for d in decisions:
        if d.batch_size == 0:
            product *= d.axis.cardinality
        else:
            product *= d.batch_size
    return base_shape_bytes * product * activation_multiplier


@dataclass(frozen=True)
class AxisSpec:
    """Describes one mappable axis for the batch planner."""
    name: str
    axis_index: int        # lower = innermost; greedy loop demotes in ascending order
    cardinality: int       # typical/max size of this axis
    default_batch_size: int  # 0 = vmap; positive = safe_map tile size
    tile_granularity: int  # safe_map tile sizes are rounded up to multiples of this
    heterogeneous: bool    # if True, element shapes may vary; vmap invalid; always safe_map
    doc: str


@dataclass(frozen=True)
class AxisDecision:
    """Planner output for one axis."""
    axis: AxisSpec
    batch_size: int        # 0 = vmap; positive = safe_map tile size
    reasoning: str


@dataclass(frozen=True)
class BatchPlan:
    """Planner output for a full set of axes."""
    decisions: list[AxisDecision]
    total_memory_estimate: float
    axes_by_index: dict[int, AxisSpec]
    budget_exceeded: bool

    def exceeded_budget(self) -> bool:
        """True when even minimum-tile decisions exceed the budget. Never raises."""
        return self.budget_exceeded

    def decision_for(self, name: str) -> AxisDecision:
        for d in self.decisions:
            if d.axis.name == name:
                return d
        raise KeyError(name)

    def log_summary(self) -> None:
        parts = [f"{d.axis.name}=bs:{d.batch_size}" for d in self.decisions]
        status = "EXCEEDED" if self._exceeded_budget else "ok"
        logger.debug("BatchPlan [%s]: %s | estimate=%.1f bytes", status, ", ".join(parts), self.total_memory_estimate)


@dataclass(frozen=True)
class BatchPlanner:
    """Host-side planner: decides vmap vs safe_map tile size per axis.

    estimate_memory is injected so the theoretical estimator can be swapped
    for an HLO-backed empirical model without changing this class.
    """
    axes: list[AxisSpec]
    budget_bytes: float
    estimate_memory: Callable[..., float]

    def plan(self) -> BatchPlan:
        sorted_axes = sorted(self.axes, key=lambda a: a.axis_index)

        # Phase 1: pre-demote heterogeneous axes (shapes vary; vmap invalid)
        decisions: list[AxisDecision] = []
        for ax in sorted_axes:
            if ax.heterogeneous:
                tile = max(1, ax.tile_granularity)
                decisions.append(AxisDecision(
                    axis=ax,
                    batch_size=tile,
                    reasoning="heterogeneous axis: element shapes vary; safe_map required",
                ))

        # Phase 2: greedy budget loop for homogeneous axes (innermost-first)
        homogeneous = [ax for ax in sorted_axes if not ax.heterogeneous]
        hom_decisions: list[AxisDecision] = [
            AxisDecision(axis=ax, batch_size=0, reasoning="vmap (homogeneous, within budget)")
            for ax in homogeneous
        ]
        for i, ax in enumerate(homogeneous):
            current = decisions + hom_decisions
            if self.estimate_memory(current) <= self.budget_bytes:
                break  # fits with current assignments
            tile = ceil_to_granularity(max(1, ax.tile_granularity), ax.tile_granularity)
            hom_decisions[i] = AxisDecision(
                axis=ax,
                batch_size=tile,
                reasoning=f"demoted to safe_map tile={tile}: estimate exceeded budget",
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

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/utils/test_batching.py -v
```

Expected: all PASS.

- [ ] **Step 5: Parity gate**

```bash
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/model/test_ligand_wave_parallel.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -q
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/prxteinmpnn/utils/batching.py tests/utils/test_batching.py
git commit -m "feat(batching): add AxisSpec/BatchPlan/BatchPlanner with greedy tile-size planner"
```

---

## Task 3: Create utils/batching_registry.py — canonical axis specs

**Files:**
- Create: `src/prxteinmpnn/utils/batching_registry.py`
- Create: `tests/utils/test_batching_registry.py`

Ten canonical `AxisSpec` instances. Cardinalities are representative max values derived from BatchingConfig defaults and model constraints. Deferred axes (N_JACOBIAN_PAIRS, N_COMBINE, N_APC_PAIRS) are included in the registry for completeness but not wired at call sites in this phase.

- [ ] **Step 1: Write failing tests**

Create `tests/utils/test_batching_registry.py`:

```python
from prxteinmpnn.utils.batching_registry import (
    ALL_AXES,
    N_APC_PAIRS,
    N_COMBINE,
    N_JACOBIAN_PAIRS,
    N_LIGAND_ATOMS,
    N_NOISES,
    N_RESIDUES,
    N_SAMPLES,
    N_STATES,
    N_STRUCTURES,
    N_TEMPERATURES,
)


def test_all_axes_present():
    assert len(ALL_AXES) == 10

def test_axis_indices_unique():
    indices = [ax.axis_index for ax in ALL_AXES]
    assert len(indices) == len(set(indices))

def test_axis_indices_contiguous():
    indices = sorted(ax.axis_index for ax in ALL_AXES)
    assert indices == list(range(len(ALL_AXES)))

def test_heterogeneous_axes():
    assert N_STRUCTURES.heterogeneous is True
    assert N_STATES.heterogeneous is True

def test_homogeneous_axes():
    for ax in [N_RESIDUES, N_LIGAND_ATOMS, N_SAMPLES, N_TEMPERATURES, N_NOISES,
               N_JACOBIAN_PAIRS, N_COMBINE, N_APC_PAIRS]:
        assert ax.heterogeneous is False, f"{ax.name} should not be heterogeneous"

def test_residues_tile_granularity():
    assert N_RESIDUES.tile_granularity == 128

def test_vmap_defaults():
    for ax in [N_RESIDUES, N_LIGAND_ATOMS]:
        assert ax.default_batch_size == 0, f"{ax.name} should default to vmap"

def test_safe_map_defaults():
    for ax in [N_STRUCTURES, N_SAMPLES, N_TEMPERATURES, N_NOISES,
               N_STATES, N_JACOBIAN_PAIRS, N_COMBINE, N_APC_PAIRS]:
        assert ax.default_batch_size > 0, f"{ax.name} should default to safe_map"

def test_positive_cardinalities():
    for ax in ALL_AXES:
        assert ax.cardinality > 0, f"{ax.name}.cardinality must be positive"

def test_positive_tile_granularities():
    for ax in ALL_AXES:
        assert ax.tile_granularity >= 1, f"{ax.name}.tile_granularity must be >= 1"
```

- [ ] **Step 2: Run to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/utils/test_batching_registry.py::test_all_axes_present -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create src/prxteinmpnn/utils/batching_registry.py**

```python
"""Canonical AxisSpec registry for all BatchingConfig-mapped axes.

axis_index ordering (innermost = 0, outermost = 9):
  0: n_residues      — innermost; residue dimension within a computation
  1: n_ligand_atoms  — per-residue atom count
  2: n_states        — multistate stack (heterogeneous)
  3: n_structures    — batch of proteins (heterogeneous)
  4: n_samples       — sample sweep
  5: n_temperatures  — temperature sweep
  6: n_noises        — backbone noise sweep
  7: n_jacobian_pairs — residue-pair products (deferred)
  8: n_combine        — multistate combine (deferred)
  9: n_apc_pairs      — all-pair contact scoring (deferred)
"""
from prxteinmpnn.utils.batching import AxisSpec

N_RESIDUES = AxisSpec(
    name="n_residues",
    axis_index=0,
    cardinality=1200,          # max LENGTH_BUCKET in padding.py
    default_batch_size=0,      # vmap — innermost axis, vectorisation is primary benefit
    tile_granularity=128,      # tensor core alignment
    heterogeneous=False,       # fixed shape after LENGTH_BUCKETS binning
    doc="Residue/position dimension within a single structure. Fixed after bucketing.",
)

N_LIGAND_ATOMS = AxisSpec(
    name="n_ligand_atoms",
    axis_index=1,
    cardinality=64,            # typical max ligand atom count
    default_batch_size=0,      # vmap — per-residue atom count is small and fixed
    tile_granularity=1,
    heterogeneous=False,
    doc="Ligand atom dimension (ligand_mpnn.py:437 triple-vmap). Fixed per structure.",
)

N_STATES = AxisSpec(
    name="n_states",
    axis_index=2,
    cardinality=64,            # typical max multistate cardinality
    default_batch_size=1,      # safe_map — states may have different sequence lengths
    tile_granularity=1,        # single-element iteration; no intra-tile padding needed
    heterogeneous=True,        # per-state sequence lengths can differ
    doc="Multistate stack axis (MultistateStackPayload.n_states). Shapes vary across states.",
)

N_STRUCTURES = AxisSpec(
    name="n_structures",
    axis_index=3,
    cardinality=32,            # typical batch_size
    default_batch_size=1,      # safe_map — proteins differ in length before binning
    tile_granularity=1,
    heterogeneous=True,        # proteins in a batch have different sequence lengths
    doc="Batch of protein structures (BatchingConfig.batch_size). Lengths vary before LENGTH_BUCKETS.",
)

N_SAMPLES = AxisSpec(
    name="n_samples",
    axis_index=4,
    cardinality=128,           # typical samples_batch_size upper bound
    default_batch_size=1,      # safe_map — output accumulates; avoids tiling sample axis
    tile_granularity=1,
    heterogeneous=False,
    doc="Sequence sample sweep (BatchingConfig.samples_batch_size, samples_chunk_size).",
)

N_TEMPERATURES = AxisSpec(
    name="n_temperatures",
    axis_index=5,
    cardinality=8,             # typical temperature sweep length
    default_batch_size=1,      # safe_map — scalar sweep; no memory benefit to vmap
    tile_granularity=1,
    heterogeneous=False,
    doc="Temperature sweep axis (BatchingConfig.temperature_batch_size).",
)

N_NOISES = AxisSpec(
    name="n_noises",
    axis_index=6,
    cardinality=8,             # typical backbone_noise sweep length
    default_batch_size=1,      # safe_map — scalar sweep
    tile_granularity=1,
    heterogeneous=False,
    doc="Backbone noise sweep axis (BatchingConfig.noise_batch_size).",
)

N_JACOBIAN_PAIRS = AxisSpec(
    name="n_jacobian_pairs",
    axis_index=7,
    cardinality=10000,         # residue-pair product can be very large
    default_batch_size=1,
    tile_granularity=1,
    heterogeneous=False,
    doc="Residue-pair axis for Jacobian computation (BatchingConfig.jacobian_batch_size). DEFERRED.",
)

N_COMBINE = AxisSpec(
    name="n_combine",
    axis_index=8,
    cardinality=64,
    default_batch_size=1,
    tile_granularity=1,
    heterogeneous=False,
    doc="Multistate combine step (BatchingConfig.combine_batch_size). DEFERRED.",
)

N_APC_PAIRS = AxisSpec(
    name="n_apc_pairs",
    axis_index=9,
    cardinality=10000,
    default_batch_size=1,
    tile_granularity=1,
    heterogeneous=False,
    doc="All-pair contact scoring (BatchingConfig.apc_batch_size, apc_residue_batch_size). DEFERRED.",
)

ALL_AXES: list[AxisSpec] = [
    N_RESIDUES, N_LIGAND_ATOMS, N_STATES, N_STRUCTURES,
    N_SAMPLES, N_TEMPERATURES, N_NOISES,
    N_JACOBIAN_PAIRS, N_COMBINE, N_APC_PAIRS,
]
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/utils/test_batching_registry.py -v
```

Expected: all PASS.

- [ ] **Step 5: Parity gate**

```bash
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/model/test_ligand_wave_parallel.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -q
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/prxteinmpnn/utils/batching_registry.py tests/utils/test_batching_registry.py
git commit -m "feat(batching): add canonical axis registry with 10 AxisSpec instances"
```

---

## Task 4: Advisory wiring — run/sampling.py

**Files:**
- Modify: `src/prxteinmpnn/run/sampling.py`

Wire `BatchPlanner` at `_sample_batch` (~line 744). The planner runs before any JIT dispatch. Its output is logged at DEBUG level only — **no execution paths change**. Axes wired: N_TEMPERATURES, N_NOISES, N_SAMPLES.

Budget computation: `jax.devices()[0].memory_stats()["bytes_limit"] * 0.80 - param_bytes`. If `memory_stats()` is unavailable (CPU), fall back to `4 * 1024**3 * 0.80` (4 GB × 80%).

- [ ] **Step 1: Read the function signature and imports at the top of sampling.py**

Read `src/prxteinmpnn/run/sampling.py` lines 1–50 (imports) and lines 740–760 (`_sample_batch` signature). Note what parameters the function receives — you need `spec: BatchingConfig` (or equivalent) to read `samples_batch_size`, `temperature_batch_size`, `noise_batch_size`.

- [ ] **Step 2: Add imports to sampling.py**

In the imports block of `sampling.py`, add:

```python
import logging

from prxteinmpnn.utils.batching import BatchPlanner, estimate_memory_theoretical
from prxteinmpnn.utils.batching_registry import N_NOISES, N_SAMPLES, N_TEMPERATURES

_batch_logger = logging.getLogger(__name__ + ".batch_plan")
```

- [ ] **Step 3: Add _make_sampling_planner helper**

Add this helper directly before `_sample_batch` (find the exact line number after reading Step 1):

```python
def _make_sampling_planner(
    spec,  # BatchingConfig
    param_bytes: float = 0.0,
    headroom: float = 0.80,
    activation_multiplier: float = 2.5,  # inference-only default; see spec §Memory estimation
) -> BatchPlanner:
    try:
        import jax
        limit = jax.devices()[0].memory_stats()["bytes_limit"]
    except Exception:
        limit = 4 * 1024**3  # 4 GB fallback for CPU / unavailable
    budget = limit * headroom - param_bytes

    import dataclasses
    axes = [
        dataclasses.replace(N_SAMPLES, cardinality=max(1, spec.samples_batch_size or 128)),
        dataclasses.replace(N_TEMPERATURES, cardinality=max(1, len(getattr(spec, "temperature", [1.0])))),
        dataclasses.replace(N_NOISES, cardinality=max(1, len(getattr(spec, "backbone_noise", [0.0])))),
    ]
    return BatchPlanner(
        axes=axes,
        budget_bytes=budget,
        estimate_memory=lambda ds: estimate_memory_theoretical(ds, 1.0, activation_multiplier),
    )
```

- [ ] **Step 4: Wire the planner call at the top of _sample_batch**

Inside `_sample_batch`, at the very beginning of the function body (before any JAX calls), add:

```python
_planner = _make_sampling_planner(spec)
_plan = _planner.plan()
_plan.log_summary()
if _plan.exceeded_budget():
    _batch_logger.warning(
        "_sample_batch: BatchPlan exceeded budget even at minimum tiles. "
        "Consider reducing batch sizes or enabling PR-C safe_map adoption."
    )
```

- [ ] **Step 5: Verify the import and wiring compile (smoke test)**

```bash
PYTHONPATH=src uv run python -c "
from prxteinmpnn.run.sampling import _sample_batch
print('import ok')
"
```

Expected: `import ok` with no errors.

- [ ] **Step 6: Parity gate**

```bash
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/model/test_ligand_wave_parallel.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -q
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/prxteinmpnn/run/sampling.py
git commit -m "feat(sampling): wire BatchPlanner advisory logging at _sample_batch (PR-A)"
```

---

## Task 5: Advisory wiring — run/scoring.py

**Files:**
- Modify: `src/prxteinmpnn/run/scoring.py`

Same pattern as Task 4. Axis wired: N_NOISES only (scoring consumes `noise_batch_size` at line ~347; no temperature or sample sweep at the `score` entry point).

- [ ] **Step 1: Read scoring.py entry point**

Read `src/prxteinmpnn/run/scoring.py` lines 100–130 (`score` function signature) and lines 340–360 (where `noise_batch_size` is consumed). Confirm `spec` parameter name.

- [ ] **Step 2: Add imports to scoring.py**

```python
import logging

from prxteinmpnn.utils.batching import BatchPlanner, estimate_memory_theoretical
from prxteinmpnn.utils.batching_registry import N_NOISES

_batch_logger = logging.getLogger(__name__ + ".batch_plan")
```

- [ ] **Step 3: Add _make_scoring_planner helper**

Add before `score`:

```python
def _make_scoring_planner(
    spec,
    param_bytes: float = 0.0,
    headroom: float = 0.80,
    activation_multiplier: float = 2.5,
) -> BatchPlanner:
    import dataclasses
    try:
        import jax
        limit = jax.devices()[0].memory_stats()["bytes_limit"]
    except Exception:
        limit = 4 * 1024**3
    budget = limit * headroom - param_bytes

    axes = [
        dataclasses.replace(N_NOISES, cardinality=max(1, len(getattr(spec, "backbone_noise", [0.0])))),
    ]
    return BatchPlanner(
        axes=axes,
        budget_bytes=budget,
        estimate_memory=lambda ds: estimate_memory_theoretical(ds, 1.0, activation_multiplier),
    )
```

- [ ] **Step 4: Wire the planner at the top of score**

At the beginning of `score`'s function body:

```python
_planner = _make_scoring_planner(spec)
_plan = _planner.plan()
_plan.log_summary()
if _plan.exceeded_budget():
    _batch_logger.warning(
        "score: BatchPlan exceeded budget even at minimum tiles."
    )
```

- [ ] **Step 5: Smoke test**

```bash
PYTHONPATH=src uv run python -c "
from prxteinmpnn.run.scoring import score
print('import ok')
"
```

Expected: `import ok`.

- [ ] **Step 6: Parity gate**

```bash
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/model/test_ligand_wave_parallel.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -q
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/prxteinmpnn/run/scoring.py
git commit -m "feat(scoring): wire BatchPlanner advisory logging at score (PR-A)"
```

---

## Task 6: Active n_structures routing — sampling.py (PR-B)

**Files:**
- Modify: `src/prxteinmpnn/run/sampling.py`

Replace `jax.vmap(internal_sample, ...)` (~line 961) with `safe_map`-driven dispatch for n_structures. The planner's `batch_size` for N_STRUCTURES is always 1 (heterogeneous=True, tile_granularity=1), so this processes one structure at a time. Bucketing and padding within each structure call are unchanged.

- [ ] **Step 1: Read the vmap_structures call site**

Read `src/prxteinmpnn/run/sampling.py` lines 950–975. Identify:
- The `jax.vmap(internal_sample, in_axes=(...))` call
- Which arguments are batched (in_axes=0) vs shared (in_axes=None)
- The variable names passed to `vmap_structures(...)`

- [ ] **Step 2: Add N_STRUCTURES to the sampling planner**

In `_make_sampling_planner` (added in Task 4), add N_STRUCTURES to the axes list:

```python
import dataclasses
axes = [
    dataclasses.replace(N_STRUCTURES, cardinality=max(1, spec.batch_size or 1)),
    dataclasses.replace(N_SAMPLES, cardinality=max(1, spec.samples_batch_size or 128)),
    dataclasses.replace(N_TEMPERATURES, cardinality=max(1, len(getattr(spec, "temperature", [1.0])))),
    dataclasses.replace(N_NOISES, cardinality=max(1, len(getattr(spec, "backbone_noise", [0.0])))),
]
```

- [ ] **Step 3: Replace vmap_structures with safe_map dispatch**

After reading Step 1, the vmap call has the pattern:

```python
vmap_structures = jax.vmap(internal_sample, in_axes=(0, 0, 0, 0, None, ...))
result = vmap_structures(batched_arg0, batched_arg1, batched_arg2, batched_arg3, shared_arg, ...)
```

Replace it with:

```python
from prxteinmpnn.utils.safe_map import safe_map as _safe_map
import functools

_structures_plan = _plan.decision_for("n_structures")
_structures_bs = _structures_plan.batch_size  # always 1 for heterogeneous

# Restructure multi-argument call as a single-pytree safe_map
_batched_inputs = (batched_arg0, batched_arg1, batched_arg2, batched_arg3)  # adjust names from Step 1
_call_one = functools.partial(
    lambda args, *shared: internal_sample(args[0], args[1], args[2], args[3], *shared),
    # shared args captured via partial:
)
# Actual replacement — fill in shared_arg, ... from Step 1:
result = _safe_map(
    lambda args: internal_sample(args[0], args[1], args[2], args[3], shared_arg, ...),
    _batched_inputs,
    batch_size=_structures_bs,
)
```

**Important:** the exact argument names and count must come from your Step 1 read. The pattern above is the template — substitute the real variable names. Ensure `_plan` (from Task 4's wiring) is in scope; if `_sample_batch` has sub-closures, pass `_structures_bs` explicitly.

- [ ] **Step 4: Parity gate**

```bash
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/model/test_ligand_wave_parallel.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -q
```

Expected: all PASS. Output must be numerically identical to before — safe_map and vmap produce the same floats for independent-per-element functions.

- [ ] **Step 5: Commit**

```bash
git add src/prxteinmpnn/run/sampling.py
git commit -m "feat(sampling): active n_structures safe_map dispatch via BatchPlanner (PR-B)"
```

---

## Task 7: Active n_structures routing — scoring.py (PR-B)

**Files:**
- Modify: `src/prxteinmpnn/run/scoring.py`

Same pattern as Task 6 applied to `scoring.py:~281` (`vmap_structures = jax.vmap(vmap_noises, ...)`).

- [ ] **Step 1: Read the vmap_structures call site in scoring.py**

Read `src/prxteinmpnn/run/scoring.py` lines 270–295. Identify batched vs shared arguments in the `jax.vmap(vmap_noises, in_axes=(...))` call.

- [ ] **Step 2: Add N_STRUCTURES to the scoring planner**

In `_make_scoring_planner` (Task 5), add:

```python
import dataclasses
axes = [
    dataclasses.replace(N_STRUCTURES, cardinality=max(1, spec.batch_size or 1)),
    dataclasses.replace(N_NOISES, cardinality=max(1, len(getattr(spec, "backbone_noise", [0.0])))),
]
```

- [ ] **Step 3: Replace vmap_structures with safe_map dispatch**

Using the argument names from Step 1:

```python
from prxteinmpnn.utils.safe_map import safe_map as _safe_map

_structures_bs = _plan.decision_for("n_structures").batch_size

result = _safe_map(
    lambda args: vmap_noises(args[0], args[1], args[2], args[3], shared_arg, ...),
    (batched_arg0, batched_arg1, batched_arg2, batched_arg3),
    batch_size=_structures_bs,
)
```

Substitute real variable names from Step 1.

- [ ] **Step 4: Parity gate**

```bash
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/model/test_ligand_wave_parallel.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -q
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/prxteinmpnn/run/scoring.py
git commit -m "feat(scoring): active n_structures safe_map dispatch via BatchPlanner (PR-B)"
```

---

## Notes for Implementers

**`dataclasses.replace` vs `._replace`:** `AxisSpec` is a `@dataclass(frozen=True)`. Use `dataclasses.replace(axis, cardinality=N)` — frozen dataclasses do not have `._replace()` (that's namedtuple syntax).

**`_plan` scope in Tasks 6–7:** The plan object (`_plan`) must be computed before the vmap call site. In Task 4 it is wired at the top of `_sample_batch`. Ensure the vmap replacement in Task 6 references the same `_plan` object.

**PR-C (optional — active budget-driven routing for homogeneous axes):** Only open this if `exceeded_budget=True` appears in >10% of cluster runs over a 7-day window after Tasks 4–5 land. See spec §PR-C trigger for monitoring guidance.

**n_states active routing (deferred):** The n_states axis (multistate state stacking) involves vmaps inside model layers (`mpnn_scoring_state_vmap_exact_ligand.py`). Threading the planner's tile size into model internals requires a separate recon pass and is deferred to a follow-on PR within Phase 6.
