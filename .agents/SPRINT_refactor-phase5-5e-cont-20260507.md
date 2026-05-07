# Sprint `refactor-phase5-5e-cont-20260507` (Phase 5e-cont — mpnn LoC split)

**Signed scope:** Extract ProteinMPNN autoregressive scan internals (`_run_tied_position_scan`, `_sample_and_broadcast_to_group`, `_run_autoregressive_scan` body) into `model/mpnn_autoregressive_scan.py` as module-level `run_*` helpers; keep `PrxteinMPNN._run_autoregressive_scan` as a thin delegating bound method for stable HLO naming and public ergonomics.

**Non-goals:** LigandMPNN parallel extraction; SamplingDriver (5f); scoring sink unify; `OUTPUT_SINKS`.

**Plan-auditor:** PASS (single vertical slice, parity-preserving move).

**Verification:** See `.agents/verification_logs/sprint_phase5_5e_cont_20260507.filtered.txt`.
