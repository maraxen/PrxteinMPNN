# Sprint: Phase 2 — Protocols + ModelCapabilities

| Field | Value |
| :--- | :--- |
| **task_id** | `refactor-phase2-sprint-20260505` |
| **Roadmap** | `.agents/REFACTOR_ROADMAP.md` §271–292 |
| **Plan audit** | NEEDS_WORK → **accepted with amendments** (2026-05-05): Protocol shapes anchored to real factories; CI `ty` waiver documented; per-PR verification explicit. |

## Preconditions

- Phase 0 / Phase 1 complete per roadmap §14.
- **CI:** `.github/workflows/ci.yml` already runs **`ty check`** in `quality-checks`. Phase 2 roadmap bullet “make ty blocking” = **verify only**, no standalone CI PR unless broken.

## Amendment (plan-auditor)

- **`protocols.py` `__call__` signatures** must match **current** `TYPE_CHECKING` `Callable[[...], ...]` shapes in `conditional_logits.py` / `unconditional_logits.py` and actual score/sampler/state-vmap factory contracts — **not** a literal copy of roadmap §3.1 illustrative snippet (`ProteinStructure`, `MultistateStackPayload` there are Phase 3+).
- **`averaging.py`:** use `model.capabilities` field aligned with encode split (e.g. `encode_fn_supports_structure_mapping: bool`) or a single shared helper used when building the split — **no** `inspect.signature(encode_fn)` after this sprint.

## Work packages (single merge acceptable; logical 3-PR split retained)

### A — Protocols

1. Add `src/prxteinmpnn/protocols.py`: `@runtime_checkable` — `ConditionalLogitsFn`, `UnconditionalLogitsFn`, `StateVmapExactLogitsFn`, `SamplerFn`, `ScoreFn`, `StateVmapExactScoreFn`. **No** imports from `prxteinmpnn.model`.
2. `sampling/conditional_logits.py`, `unconditional_logits.py`: remove `TYPE_CHECKING` / `Callable[..., Any]` split; import Protocol types from `protocols`.
3. `run/jacobian.py`: import `ConditionalLogitsFn` from `prxteinmpnn.protocols`.
4. **Verify PR A:** `uv run ty check`, `uv run ruff check .`, `uv run pytest tests/parity -m parity_fast -q`.

### B — ModelCapabilities

1. Add `ModelCapabilities(eqx.Module)` (roadmap §3.4 + extensions): cover **all** axes inferred today in `sample.py` (~77–81), `score.py` (~341–343), and `averaging.py` encode path (~57–58). Include at minimum: `accepts_ligand`, `accepts_state_stack`, `accepts_tied_positions`, `accepts_bias`, `accepts_fixed_positions`, `output_logit_shape`; plus explicit fields for multi-state temperature / state weights / fixed tokens as needed; **`encode_fn_supports_structure_mapping`** (or equivalent) for averaging.
2. `model/mpnn.py`: `capabilities: ModelCapabilities = eqx.field(static=True)` on `PrxteinMPNN` and `PrxteinLigandMPNN` with **correct per-class instances**.
3. Export `ModelCapabilities` from `model/__init__.py` if consistent with public surface.
4. **Verify PR B:** same as A (behavior-neutral if call-sites not yet migrated — either keep introspection until C or migrate in same commit; **prefer one atomic commit** for this sprint: A+B+C together is OK).

### C — Call sites + casts

1. `sampling/sample.py`: remove `inspect.signature`; use `model.capabilities`.
2. `scoring/score.py`: remove `inspect.signature`; use capabilities; `cast(StateVmapExactScoreFn, ...)` at state-vmap path (~302); `cast(ScoreFn, ...)` at flat path (~444) per contract.
3. `run/averaging.py`: remove `inspect.signature(encode_fn)`; use capabilities (or shared helper per amendment).
4. `rg "inspect\\.signature"` on the three files → **no matches** in `src/`.
5. **Verify PR C:** `uv run ty check`, `uv run ruff check .`, `uv run pytest tests/parity -m parity_fast -q`, and targeted sampling tests from parent `CLAUDE.md` if feasible:

```bash
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/model/test_ligand_wave_parallel.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -q
```

## Definition of Done (sprint)

- No `Callable[..., Any]` on logits public type surface in `conditional_logits.py` / `unconditional_logits.py`.
- No `inspect.signature` in `sample.py`, `scoring/score.py`, `run/averaging.py` (production `src/`).
- Casts honest at score boundaries (`StateVmapExactScoreFn` / `ScoreFn`).
- `uv run ty check`, `uv run ruff check .`, `parity_fast` green; full `uv run pytest` before merge if time permits.

## Out of scope

- `BiasHook`, `DesignSink`, payloads, registries, `mpnn.py` split (later phases).
