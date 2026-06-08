# Sprint `refactor-phase5-5e-cont-20260509` (Phase 5e-cont — Ligand `sample_autoregressive_state_vmap_exact` extract)

**Signed scope:** Move the body of `PrxteinLigandMPNN.sample_autoregressive_state_vmap_exact` into `model/mpnn_autoregressive_state_vmap_exact_ligand.py` as module-level `run_sample_autoregressive_state_vmap_exact_ligand(model, ...)`; keep the class method as a thin delegate (lazy import of the runner inside the method to avoid import cycles with `ligand_encode_stack_row`). `sample_autoregressive_state_vmap_exact_from_payload` unchanged apart from routing through the delegate.

**Non-goals:** SamplingDriver (5f); scoring sink unify; further mpnn split beyond this slice.

**Plan-auditor:** PASS (extract-only; parity-preserving move).

**Verification:** `.agents/verification_logs/sprint_phase5_5e_cont_ligand_20260509.filtered.txt` (full raw: same stem `.pytest.raw.txt`).

**Outcome auditor:** APPROVE — targeted pytest slice green (`28 passed`); `uv run ruff check` clean on `mpnn_autoregressive_state_vmap_exact_ligand.py`.
