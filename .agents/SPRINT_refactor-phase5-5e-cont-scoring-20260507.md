# Sprint `refactor-phase5-5e-cont-scoring-20260507` (Phase 5e-cont — Ligand scoring `state_vmap_exact` extract)

**Signed scope:** Move `PrxteinLigandMPNN.score_unconditional_state_vmap_exact` and `score_conditional_state_vmap_exact` bodies into `model/mpnn_scoring_state_vmap_exact_ligand.py` as `run_score_unconditional_state_vmap_exact_ligand` / `run_score_conditional_state_vmap_exact_ligand`; move `_ligand_slice_pad_*` and `ligand_score_*_one_chunk` into the same module. Class methods delegate via lazy import. `mpnn.py` re-exports chunk/slice helpers from the new module (after `ligand_mpnn` import) to preserve `from prxteinmpnn.model.mpnn import …` paths. **No top-level** `ligand_encode_stack_row` import in the scoring module (lazy import inside runners) to avoid `model/__init__` ↔ `mpnn` circular import.

**Non-goals:** SamplingDriver (5f); scoring sink unify; further mpnn split beyond this slice.

**Plan-auditor:** PASS (extract-only; parity-preserving move).

**Verification:** `.agents/verification_logs/sprint_phase5_5e_cont_scoring_ligand_20260507.filtered.txt` (full raw: same stem `.pytest.raw.txt`). Pytest: `tests/sampling/test_state_vmap_exact_jit.py`, `test_state_vmap_exact_routing.py`, `tests/model/test_multistate_state_vmap_scores.py`, `tests/streaming/` — 45 passed.

**Outcome auditor:** APPROVE — extended slice green; `uv run ruff check src/prxteinmpnn/model/mpnn_scoring_state_vmap_exact_ligand.py` clean.
