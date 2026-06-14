---
sprint_id: "260615"
task_id: 260615_potts-gates-runspec
status: planning
created: 260614
items: [1295, 1296, 1550, 1621, 1552, 1553]
---

# Sprint 260615: Potts gates, G1, RS-2, T2.2–T2.4

## Sprint discovery: T2.1 already complete

Prior handoff marked T2.1 (#1551) as a failed dispatch. **It actually landed**: three commits in xtrax
(`f9847a9` add Scan.init, `eff90d4` make transition optional, `a24dd06` export Scan). The
`dispatch.py` carry-init fallback at lines 71–76 is wired. #1551 marked complete; #1552 and #1553
are now unblocked.

## Sprint slice (6 tracks, all executable)

| Track | ID | Title | Agent | Repo | Priority |
|-------|----|-------|-------|------|----------|
| A | #1295 | P-06: sampling.py JAX-native | fixer | aminx | P1 |
| B | #1296 | P-05: poe.py JAX-native | fixer | aminx | P1 |
| C | #1550 | G1: training parity gate | reviewer | aminx | P1 |
| D | #1621 | RS-2: PlannerTopology to RunSpec | fixer | aminx | P1 |
| E | #1552 | T2.2-2.3: CarrySpec + DedupSpec | fixer | xtrax | P2 |
| F | #1553 | T2.4: make_axis_dispatch factory | fixer | xtrax | P2 |

Tracks A–D are fully independent and can be dispatched in parallel. E and F are both in
`/home/marielle/projects/xtrax` and can also be dispatched in parallel (different files).

---

## Track A — P-06: sampling.py JAX-native (#1295)

**Goal:** Replace two Python loops in `_parallel_tempering_exchange` with JAX-native constructs
so the function is fully traceable under `jax.jit`.

**File:** `src/aminx/potts/sampling.py:202–225`

**Current code (lines 206–224):**
```python
for parity in range(2):
    start = parity
    i = start
    while i + 1 < k_rep:
        key_cur, sk = jax.random.split(key_cur)
        seq_cur, key_cur, acc = _attempt_adjacent_swap(sk, seq_cur, jnp.int32(i), ...)
        accept_edge = accept_edge.at[i].set(acc)
        i += 2
```

**Key constraint:** `k_rep = int(seqs.shape[0])` is statically known at trace time — safe to use
as a compile-time constant.

**Approach:**
1. For each parity (0=even, 1=odd), build static edge-index arrays:
   `even_edges = jnp.arange(0, k_rep-1, 2)` and `odd_edges = jnp.arange(1, k_rep-1, 2)`.
2. Within each parity group, swaps are **non-overlapping** (even parity: pairs (0,1),(2,3),...;
   odd parity: pairs (1,2),(3,4),...). This makes them safe to vmap.
3. Pre-split keys per edge: `keys = jax.random.split(key_cur, n_edges)` then split again for
   the updated `key_cur` chain.
4. `jax.vmap(_attempt_adjacent_swap_scalar, in_axes=(0, None, 0, None, None, ...))(keys, seq_cur, edge_indices, ...)`
   — note: `_attempt_adjacent_swap` likely needs a helper that takes a scalar edge index rather
   than mutating `seq_cur`. Check its signature and adapt accordingly.
5. Scatter accepted swaps back into `seq_cur` using `jax.lax.dynamic_update_slice` or
   `.at[indices].set(...)` with the vmapped result.
6. Accumulate `accept_edge` directly from the vmap output (no `.at[i].set` loop needed).

**Verify:** `uv run pytest tests/potts/ -v --tb=short` passes + `uv run ty check` clean.

---

## Track B — P-05: poe.py JAX-native (#1296)

**Goal:** Replace two Python for-loops in `PoeModel` with JAX-native constructs.

**File:** `src/aminx/potts/poe.py`

### Subchange B1 — `__call__` backbone loop (line 172)

Current:
```python
all_marginals = []
for i in range(self.n_backbones):
    backbone = self.backbones[i]
    ...
    marginals, _, _, _ = backbone(edge_knn=edge_knn_b, nei=nei_b, mask=mask, key=key_b)
    all_marginals.append(marginals)
per_backbone_marginals = jnp.stack(all_marginals, axis=0)
```

Reference pattern: `infer_all_params` at line 247+ already uses `eqx.filter_vmap` for the same
pattern of vmapping across `self.backbones`. Follow that pattern exactly.

Replace with `eqx.filter_vmap` over the backbone pytree:
- `self.backbones` must be a stacked pytree (list of identical-structure modules)
- `eqx.filter_vmap` handles the Equinox module correctly

### Subchange B2 — `joint_energy` params loop (line 234)

Current:
```python
total_energy = 0.0
for h, j, w in params_list:
    energy_b = PottsModel.log_prob(seq, h, j, w)
    total_energy = total_energy + energy_b
return jnp.asarray(total_energy)
```

Replace with:
```python
h_stack = jnp.stack([p[0] for p in params_list])   # (B, n, q)
j_stack = jnp.stack([p[1] for p in params_list])   # (B, n, n, q, q)
w_stack = jnp.stack([p[2] for p in params_list])   # (B, n, n)
energies = jax.vmap(PottsModel.log_prob, in_axes=(None, 0, 0, 0))(seq, h_stack, j_stack, w_stack)
return jnp.sum(energies)
```

Note: stacking in `params_list` is a Python-level operation at trace time (params_list is a tuple
of static-shape arrays). Confirm shape compatibility before stacking.

**Verify:** `uv run pytest tests/potts/ -v --tb=short` passes + `uv run ty check` clean.

---

## Track C — G1: Training parity gate (#1550)

**Goal:** Close the G1 gate — verify all training infrastructure tests pass and the 50-step
overfit smoke demonstrates functional training.

**Agent:** reviewer (needs Bash)

**Gate criteria (all 3 must pass):**

**G1.1 — pytest training suite:**
```bash
uv run pytest tests/training/ -v --tb=short
```
Expect: ≥8 tests pass (test_checkpoint.py has 5+ tests, test_resumable_state.py has 2+ tests).
Zero failures acceptable; xfail is OK.

**G1.2 — Checkpoint round-trip smoke:**
- Save a `ResumableState` at step=42 via `get_checkpoint_manager`
- Load it back; assert `jnp.allclose(model_weights_before, model_weights_after, atol=1e-7)`
- Use `tests/training/test_checkpoint.py:test_save_and_load_roundtrip` as a base fixture or
  write an inline script if the fixture isn't parameterized for this.

**G1.3 — 50-step overfit smoke:**
- Use `src/aminx/training/trainer.py` with a tiny dummy batch
- Run 50 gradient steps; assert `loss[49] / loss[0] < 0.9` (≥10% loss reduction)
- The trainer uses `eqx.nn.Linear + cross_entropy_loss` — a 4-residue mock protein is sufficient

**Output:** Write gate results to `.praxia/docs/research/260614_g1-training-parity-gate.md`
with sections: test run output, checkpoint round-trip result, overfit smoke result, verdict.

---

## Track D — RS-2: PlannerTopology to RunSpec (#1621)

**Goal:** Add `PlannerTopology` sub-config to `RunSpec` and `build_run_spec` as an aminx-specific
wrapper over the aminx planner. Scoped to `use_unified_driver` only for RS-2.

**Placement rationale:** `PlannerTopology` lives in aminx, not xtrax.
- `use_unified_driver` is an aminx kernel flag (kernel_dispatch.py) with no xtrax equivalent
- aminx's `BatchPlanner` (multi-phase, `aminx.tiling.planner`) is more mature than xtrax's
  (single-phase, `xtrax.tiling.plan`) — they are not yet interchangeable
- Once T2.5 ships (xtrax multi-phase BatchPlanner reaching aminx parity), `PlannerTopology` will
  gain an `execution_profile: xtrax.ExecutionProfile` field to wrap the xtrax planner config
- For RS-2: thin wrapper only — do not try to incorporate xtrax types yet

**File:** `src/aminx/run/spec.py`

**Current RunSpec fields (line 102–114):**
`io, resource, multistate, ligand, tied, grid, batching, averaging, precision`

**Step 1 — Add `PlannerTopology` class** (after `PrecisionConfig` at line 96):
```python
class PlannerTopology(eqx.Module):
    """aminx kernel dispatch topology. Wraps aminx.tiling.planner config for RunSpec.

    Note: will gain an xtrax.ExecutionProfile field in RS-2+T2.5 once xtrax
    BatchPlanner reaches multi-phase parity with aminx.tiling.BatchPlanner.
    """
    use_unified_driver: bool = eqx.field(static=True)
```

**Step 2 — Add `plan: PlannerTopology` field to `RunSpec`.**

**Step 3 — Add `topology_hash` function** (module-level, not a method):
```python
def topology_hash(plan: PlannerTopology) -> str:
    import hashlib, json
    payload = {"use_unified_driver": plan.use_unified_driver}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
```
Keep as a module-level function — no methods on eqx.Module needed.

**Step 4 — Update `build_run_spec` (line 218+)** to populate `plan`:
```python
plan = PlannerTopology(
    use_unified_driver=bool(getattr(spec, "use_unified_driver", True)),
    # Correct default is True — RS-5 fix in 26c9bb5; False was wrong
)
```
Pass `plan=plan` to the `RunSpec(...)` constructor.

**Step 5 — Add topology_hash golden test.** Find or create `tests/run/test_run_spec.py`:
- Assert `topology_hash(PlannerTopology(use_unified_driver=True)) == "<expected_hex>"`
- Assert determinism: call twice, assert equal

**Verify:** `uv run ty check` + `uv run pytest tests/run/ -v --tb=short`.

---

## Track E — T2.2-2.3: CarrySpec + DedupSpec enrichment in xtrax (#1552)

**Goal:** Port `CarrySpec`, `CarryShape`, and `DedupSpec` FROM aminx.tiling INTO xtrax.tiling,
and enrich xtrax's `BatchPlanner.plan()` with Phase 0 (CarrySpec pre-demote) and Phase 0b
(DedupSpec pre-demote) to match aminx's multi-phase planner.

**This is a port task, not a new design.** The source of truth is aminx; do not invent new APIs.

**Repo:** `/home/marielle/projects/xtrax`

**Source files to read first (aminx reference implementations):**
- `aminx/src/aminx/tiling/carry.py` — `CarrySpec` (axis_name, init, transition, ordered_sinks)
- `aminx/src/aminx/tiling/carry_shape.py` — `CarryShape` (name, shape, dtype, materialize())
- `aminx/src/aminx/tiling/dedup.py` — `DedupSpec` (unique_indices, index_map, k, dedup_fn, gather_fn, get_k_bucket)
- `aminx/src/aminx/tiling/planner.py:104–165` — Phase 0 and Phase 0b pre-demote logic in BatchPlanner

**Files to create/modify in xtrax:**
- `src/xtrax/tiling/carry.py` — NEW: port `CarrySpec` from aminx (adapt imports: ScanTransition is already in xtrax.tiling.strategy)
- `src/xtrax/tiling/carry_shape.py` — NEW: port `CarryShape` from aminx (no aminx deps; pure dataclass + jax.numpy)
- `src/xtrax/tiling/dedup.py` — NEW: port `DedupSpec` and `get_k_bucket` from aminx (adapt DedupGather import to xtrax.tiling.strategy)
- `src/xtrax/tiling/plan.py` — MODIFY: add Phase 0 (CarrySpec list param) and Phase 0b (DedupSpec list param) to `BatchPlanner.__init__` and `BatchPlanner.plan()`, following aminx planner lines 104–165

**`BatchPlanner.plan()` enrichment (key diff from current xtrax):**
Current xtrax planner has a single-pass loop. aminx adds two pre-demote phases BEFORE the
budget loop:
- Phase 0: for each `CarrySpec`, find matching `AxisSpec` by name and force strategy to `Scan(transition=cs.transition, init=cs.init)`
- Phase 0b: for each `DedupSpec`, find matching `AxisSpec` by name and force strategy to `DedupGather(...)`
- Then the existing cardinality/budget loop runs over remaining axes

**Export:** Update `src/xtrax/tiling/__init__.py` to export `CarrySpec`, `CarryShape`, `DedupSpec`, `get_k_bucket`.

**Verify:** `uv run pytest /home/marielle/projects/xtrax/tests/ -v` passes. Add a test for Phase 0 pre-demote if none exists.

---

## Track F — T2.4: make_axis_dispatch factory refactor in xtrax (#1553)

**Goal:** Refactor xtrax's eager `make_axis_dispatch` to match aminx's factory-style dispatch
pattern (returns iterator object, not immediate result).

**This is a port task, not a new design.** aminx's dispatch is already factory-style and is
the reference.

**Repo:** `/home/marielle/projects/xtrax`
**File:** `src/xtrax/tiling/dispatch.py`

**Source to read first (aminx reference):**
- `aminx/src/aminx/tiling/dispatch.py` — aminx's `make_axis_dispatch(strategy, *, axis) -> iterator`
- `aminx/src/aminx/tiling/iterator.py` — `VmapIterator`, `SafeMapIterator`, `JaxScanIterator`, `MapIterator`

**Key difference:**
- aminx: `make_axis_dispatch(strategy, *, axis="state") -> VmapIterator | SafeMapIterator | JaxScanIterator`
  — returns a typed iterator; execution happens when the iterator is called with `(fn, xs)`
- xtrax current: `make_axis_dispatch(strategy, fn, xs, init=None) -> Any`
  — immediately executes `jax.vmap(fn)(xs)` etc.

**Approach (keep xtrax independent of aminx.tiling.iterator — don't import aminx types):**
1. Create `src/xtrax/tiling/iterator.py`: port `VmapIterator`, `SafeMapIterator`,
   `JaxScanIterator` from aminx (pure xtrax deps — `safe_map`, `safe_scan` already in xtrax.transforms)
2. Refactor `make_axis_dispatch(strategy, *, axis="")` → returns iterator; reject `DedupGather`
   (it's handled by `BatchPlanner + _dispatch_axis`, same as aminx convention)
3. Keep the old eager call pattern available as a compatibility shim:
   `axis_dispatch(strategy, fn, xs, init=None)` that wraps `make_axis_dispatch(strategy)(fn, xs, init)`

**Do not break existing xtrax tests.** Audit `tests/` for `make_axis_dispatch` call sites before
changing the signature; add the shim first if needed.

**4 invariants to preserve (from aminx):**
1. Vmap iterator: result shape matches `jax.vmap(fn)(xs)` output
2. SafeMap iterator: result is chunk-order-stable (same as `safe_map`)
3. Scan iterator: returns `(final_carry, stacked_outputs)` matching `safe_scan` convention
4. DedupGather: NOT handled here — raise `DispatchRejected` (same as aminx)

**Verify:** `uv run pytest /home/marielle/projects/xtrax/tests/ -v` passes.

---

## Dispatch order

```
Session start:
  [ ] Check git status --short on both aminx and xtrax (must be clean)
  [ ] Tracks A, B, C, D → dispatch in parallel (4 independent fixers/reviewers in aminx)
  [ ] Tracks E, F → dispatch in parallel (2 fixers in xtrax)

After fixers complete:
  [ ] git diff HEAD on aminx and xtrax before any commit
  [ ] Run full pytest on modified files
  [ ] Commit each track separately with atomic messages

Blockers resolved by this sprint:
  - #1295 done → enables P-08 (#1299, the end-to-end run loop)
  - #1296 done → enables P-00d (#1305, PoE sanity check) and P-08 (#1299)
  - #1550 done → closes G1 gate on EPIC #1541
  - #1621 done → enables RS-3 (#1622), RS-6 (#1625), RS-7 (#1626)
  - #1552 done → enables T2.4 (#1553) if not done in parallel; enables #1554
  - #1553 done → enables #1554, #1555

Next sprint (260616) expected items:
  - P-08 (#1299): end-to-end potts run loop (blocked on #1295+#1296+#1297; #1297 done)
  - P-00d (#1305): PoE sanity check (blocked on #1296)
  - RS-6 (#1625): phased host migration (blocked on RS-2 #1621)
  - P-10 (#1300): alphabet alignment research (independent, P1)
```

---

## Recon summary (260614 pre-sprint)

**aminx.tiling is the mature library; xtrax.tiling is being uplifted.**

| Concept | aminx | xtrax | Status |
|---------|-------|-------|--------|
| `BatchPlanner` | `aminx.tiling.planner` (multi-phase: Phase 0 CarrySpec, Phase 0b DedupSpec, then budget loop) | `xtrax.tiling.plan` (single-phase, budget loop only) | xtrax needs Phase 0/0b (T2.2-2.3) |
| `AxisSpec` | `aminx.tiling.planner.AxisSpec` | `xtrax.tiling.plan.AxisSpec` | both exist; compatible |
| `Vmap/SafeMap/Scan/DedupGather/Bucket` | `aminx.tiling.strategy` (typed generics) | `xtrax.tiling.strategy` (simpler) | both exist; need to verify compat |
| `CarrySpec` | `aminx.tiling.carry` ✓ | NOT IN XTRAX | T2.2-2.3 ports to xtrax |
| `CarryShape` | `aminx.tiling.carry_shape` ✓ | NOT IN XTRAX | T2.2-2.3 ports to xtrax |
| `DedupSpec` | `aminx.tiling.dedup` ✓ | NOT IN XTRAX | T2.2-2.3 ports to xtrax |
| `make_axis_dispatch` | `aminx.tiling.dispatch` (factory → iterator) | `xtrax.tiling.dispatch` (eager → result) | T2.4 ports factory pattern to xtrax |
| `BatchPlan` | both | both | compatible |
| `PlannerTopology` | `aminx.run.spec` (to be added RS-2) | N/A — not an xtrax concept | stays in aminx; wraps aminx BatchPlanner |

**T2.2-2.3 and T2.4 are port tasks:** Copy-adapt from `aminx.tiling/` into `xtrax/tiling/` with
no aminx imports. Do not re-invent the API — aminx is the reference implementation.

**PlannerTopology stays in aminx** for RS-2. It will gain an `xtrax.ExecutionProfile` field in
T2.5+ once xtrax's BatchPlanner reaches aminx multi-phase parity.

## Open questions to resolve at dispatch time

1. **P-06 exchange vmap:** Does `_attempt_adjacent_swap` accept a scalar edge index `i` or
   is `i` a JAX array? If it uses `jnp.int32(i)` it's already device-array — vmap-safe. Check
   return shape to confirm scatter is straightforward.

2. **P-05 `eqx.filter_vmap` over `self.backbones`:** Confirm the backbones list can be stacked
   as a pytree. If `self.backbones` is a Python list of `eqx.Module`, use
   `eqx.filter_vmap(lambda b, knn, nei: b(edge_knn=knn, nei=nei, mask=mask, key=key))` with
   an explicit pytree stack.

3. **T2.2-2.3 ScanTransition import:** aminx's `CarrySpec` imports `ScanTransition` from
   `aminx.tiling.strategy`. When porting to xtrax, use `xtrax.tiling.strategy.ScanTransition`
   (already exists). Verify the xtrax `ScanTransition` protocol signature matches before porting.

4. **T2.4 iterator shim:** Check xtrax test suite for `make_axis_dispatch(strategy, fn, xs)`
   eager call sites before changing signature — add backward-compat shim first if tests use
   the old 3-arg form.
