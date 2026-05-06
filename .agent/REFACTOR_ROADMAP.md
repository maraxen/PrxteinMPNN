# Refactor roadmap (pointer)

The maintained **prxteinmpnn refactor roadmap** (phases, DoD, resolution log) lives at:

**[`.agents/REFACTOR_ROADMAP.md`](../.agents/REFACTOR_ROADMAP.md)**

**Recent parity / spike:** §**13.1** parity gates (**13440956**, **13441413**). §**13.2** Phase 0a spike **GO** (numeric + dual HLO advisory) recorded **2026-05-07**. **Phase 4 slice:** `MULTISTATE_MODES` + `SAMPLERS` / `SamplingDriver` prep (`2026-05-07`); **`test_state_vmap_exact_routing.py`** + driver `sampler_factory_keys` (`2026-05-06`). **Phase 5a (`2026-05-06`):** `encoder.py` helpers + protein `state_vmap_exact` / AR wiring. **Phase 5e (`2026-05-06`, `refactor-phase5e-ligand-encoder-20260506`):** ligand AR scan + ligand `state_vmap_exact` use `pack_encoder_context` (verification `.agents/verification_logs/phase5e_ligand_encoder_20260506.filtered.txt`; §14). **Phase 5f narrow (`2026-05-06`):** `run/sampling.sample()` obtains JIT sampler via `SamplingDriver.build_sampler_fn`. **PR2b (`2026-05-06`):** host coercion for loose `state_vmap_exact` stacks before `jax.jit` — see `.agents/REFACTOR_ROADMAP.md` §14. **Phase 6** adds **track A:** bucketing, ragged vs padded, `safe_map` vs stacked `vmap` (roadmap §15).
