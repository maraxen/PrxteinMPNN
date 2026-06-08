# Sprint `refactor-phase5-5e-cont-20260508` (Phase 5e-cont — `sample_autoregressive_state_vmap_exact` extract)

**Signed scope:** Move the body of `PrxteinMPNN.sample_autoregressive_state_vmap_exact` into `model/mpnn_autoregressive_state_vmap_exact.py` as module-level `run_sample_autoregressive_state_vmap_exact(model, ...)`; keep the class method as a thin delegate and preserve `sample_autoregressive_state_vmap_exact_from_payload` unchanged apart from routing through the delegate.

**Non-goals:** LigandMPNN parallel extraction; SamplingDriver (5f); scoring sink unify.

**Plan-auditor:** PASS (extract-only; parity-preserving move).

**Verification:** `.agents/verification_logs/sprint_phase5_5e_cont_20260508.filtered.txt` (full raw: same stem `.pytest.raw.txt`).

**Outcome auditor:** APPROVE — targeted pytest green; `PrxteinLigandMPNN` re-export at EOF preserved.
