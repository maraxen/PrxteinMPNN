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

**Goal:** Add `PlannerTopology` (ExecutionProfile) sub-config to `RunSpec` and `build_run_spec`,
with a deterministic `topology_hash` golden.

**File:** `src/aminx/run/spec.py`

**Current RunSpec fields (line 102–114):**
`io, resource, multistate, ligand, tied, grid, batching, averaging, precision`

**Step 1 — Add `PlannerTopology` class** (after `PrecisionConfig` at line 96):
```python
class PlannerTopology(eqx.Module):
    """Execution topology for the kernel dispatch planner."""
    use_unified_driver: bool = eqx.field(static=True)
```

**Step 2 — Add `plan: PlannerTopology` field to `RunSpec`.**

**Step 3 — Add `topology_hash` function** (pure, deterministic):
```python
def topology_hash(plan: PlannerTopology) -> str:
    """Stable hash for topology cache-key derivation."""
    import hashlib, json
    payload = {"use_unified_driver": plan.use_unified_driver}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
```

**Step 4 — Update `build_run_spec` (line 218+)** to populate `plan`:
```python
plan = PlannerTopology(
    use_unified_driver=bool(getattr(spec, "use_unified_driver", True)),
    # Note: correct default is True (per RS-5 fix in 26c9bb5; False was wrong)
)
```
And pass `plan=plan` to the `RunSpec(...)` constructor.

**Step 5 — Add topology_hash golden test.** Find or create `tests/run/test_run_spec.py`:
- Assert `topology_hash(PlannerTopology(use_unified_driver=True)) == "<expected_hex>"`
- Assert determinism: call twice, assert equal

**Verify:** `uv run ty check` + `uv run pytest tests/run/ -v --tb=short`.

---

## Track E — T2.2-2.3: CarrySpec + DedupSpec enrichment in xtrax (#1552)

**Goal:** Add `CarrySpec`/`CarryShape` types to xtrax tiling and enrich `DedupGather` to match
aminx parity needs.

**Repo:** `/home/marielle/projects/xtrax`
**File:** `src/xtrax/tiling/strategy.py`

**Step 0 (fixer recon — read before implementing):**
- Read `strategy.py` current `DedupGather` definition (line ~52-58)
- Grep aminx for `DedupGather|CarrySpec|CarryShape` usage to understand what "parity" means:
  `grep -rn "DedupGather\|CarrySpec\|CarryShape" /home/marielle/projects/aminx/src/`
- Read xtrax `dispatch.py` DedupGather dispatch path (lines ~70-80)

**Expected additions:**
- `CarrySpec(frozen=True)`: metadata dataclass describing expected carry shape/dtype, e.g.
  `fields: dict[str, tuple[int, ...]]` and `dtypes: dict[str, str]`
- `CarryShape`: type alias or companion for `CarrySpec`
- Enrich `DedupGather` if aminx grep reveals additional fields needed (e.g., `pad_value`, `max_unique`)

**Verify:** `uv run pytest /home/marielle/projects/xtrax/tests/ -v` passes.

---

## Track F — T2.4: make_axis_dispatch factory refactor (#1553)

**Goal:** Refactor `make_axis_dispatch` from eager-dispatch to factory-over-eager, add aminx
wrapper preserving 4 invariants.

**Repo:** `/home/marielle/projects/xtrax`
**File:** `src/xtrax/tiling/dispatch.py`

**Step 0 (fixer recon — read before implementing):**
- Read `dispatch.py` current `make_axis_dispatch` signature + full implementation
- Grep aminx for `make_axis_dispatch` call sites to understand the 4 invariants:
  `grep -rn "make_axis_dispatch" /home/marielle/projects/aminx/src/`
- Check xtrax tests for `make_axis_dispatch` to understand expected interface

**Factory-over-eager pattern (expected):**
Current signature: `make_axis_dispatch(strategy, fn, xs, init=None) -> Any`
Factory signature: `make_axis_dispatch(strategy) -> Callable[[Callable, Any, Any], Any]`

This lets callers compose: `dispatch = make_axis_dispatch(Scan(transition=f, init=c))` then
`result = dispatch(fn, xs)` — the strategy is bound once, the compute is called later.

**Aminx wrapper:** If aminx has call sites that use the current eager API, add a thin wrapper
`axis_dispatch(strategy, fn, xs, init=None)` in aminx that adapts the factory to the old
eager-call pattern. This preserves backward compat.

**4 invariants** (to be confirmed from aminx grep, but expected):
1. Vmap result shape matches eager `jax.vmap(fn)(xs)` output shape
2. SafeMap result is chunk-order-stable
3. Scan final carry is accessible from the return value
4. DedupGather result matches sequential map on the same inputs (up to gather reordering)

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

## Open questions to resolve at dispatch time

1. **P-06 exchange vmap:** Does `_attempt_adjacent_swap` accept a scalar edge index `i` or
   is `i` a JAX array? If it uses `jnp.int32(i)` it's already device-array — vmap-safe. Check
   return shape to confirm scatter is straightforward.

2. **P-05 `eqx.filter_vmap` over `self.backbones`:** Confirm the backbones list can be stacked
   as a pytree. If `self.backbones` is a Python list of `eqx.Module`, use
   `eqx.filter_vmap(lambda b, knn, nei: b(edge_knn=knn, nei=nei, mask=mask, key=key))` with
   an explicit pytree stack.

3. **RS-2 `topology_hash` location:** Could live in `spec.py` as a module-level function or as
   a method on `PlannerTopology`. Prefer module-level to keep `PlannerTopology` a pure data
   class (no methods on eqx.Module needed for this).

4. **T2.2-2.3 DedupSpec "aminx parity":** The exact fields needed are unknown until recon.
   Fixer agent should grep aminx for `DedupGather` usage as step 0.
