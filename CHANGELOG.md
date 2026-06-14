# Changelog

## 0.1.0a6 (2026-06-14)

### Added

- **`PlannerTopology` sub-config in `RunSpec`** (RS-2): New `eqx.Module` sub-config wrapping
  aminx kernel dispatch topology. `RunSpec.plan` carries a `PlannerTopology` with a single
  field `use_unified_driver: bool` (default `True`, consistent with RS-5 fix in 0.1.0a5).
  Module-level `topology_hash(plan)` produces a deterministic 16-char hex digest for
  cache-key derivation.
  ([`src/aminx/run/spec.py`](src/aminx/run/spec.py))

  *Note:* `PlannerTopology` will gain an `xtrax.ExecutionProfile` field once xtrax reaches
  multi-phase `BatchPlanner` parity (T2.5).

### Performance

- **`PoeModel.__call__`**: Replaced Python `for i in range(self.n_backbones)` loop with
  `eqx.filter_vmap`, matching the pattern already used in `infer_all_params`. Backbone
  inference is now fully vectorized via JAX rather than traced sequentially at Python level.
  ([`src/aminx/potts/poe.py`](src/aminx/potts/poe.py))

- **`PoeModel.joint_energy`**: Replaced Python `for h, j, w in params_list` loop with
  `jax.vmap(PottsModel.log_prob, in_axes=(None, 0, 0, 0))` + `jnp.sum` over stacked params.
  ([`src/aminx/potts/poe.py`](src/aminx/potts/poe.py))

- **`_parallel_tempering_exchange`**: Replaced Python `for parity / while i` loops with
  `jax.vmap` over non-overlapping replica-pair edges within each parity group. Even/odd
  parity passes remain sequential (odd uses seqs updated by even). Keys split once per
  parity group; results scattered back via `.at[...].set`.
  ([`src/aminx/potts/sampling.py`](src/aminx/potts/sampling.py))

### Gates

- **G1 training parity gate**: All three criteria pass — pytest suite (8/8), checkpoint
  round-trip smoke (`ResumableState` save → load, all leaves match to atol=1e-7), and
  50-step overfit smoke (loss 3.14 → 0.00 over 50 steps).

### Breaking Changes

- **Checkpoint format**: Adopted `xtrax.checkpoint.orbax` single-PyTree checkpointing via
  `PyTreeCheckpointHandler`, replacing the legacy `ocp.args.Composite` format. Checkpoints
  saved with the old format are **NOT compatible** with this version. Delete existing
  checkpoint directories before resuming training.
  ([`src/aminx/training/checkpoint.py`](src/aminx/training/checkpoint.py),
  [`src/aminx/training/trainer.py`](src/aminx/training/trainer.py))

  The `ResumableState` (params, optimizer state, step, RNG, and extras) is now written as
  a single flat PyTree, improving checkpoint composability and enabling future multi-device
  training. The metrics dict (if any) lives in `ResumableState.extras`, not as a top-level
  checkpoint key.

## 0.1.0a5 (2026-06-10)

### Breaking Changes

- **API rename**: `xyz_37` → `atom_37` and `xyz_37_m` → `atom_37_mask` across the
  side-chain atom-context API — the public `GeometryBundle` fields,
  `build_inference_bundle(...)` / kernel keyword arguments, and `model.features(...)`
  parameters. The new names describe the 37-atom representation and its validity mask
  clearly. No deprecated alias (pre-release clean break).

### Bug Fixes

- **`ProteinFeaturesLigand._make_angle_features`**: correct the residue-frame projection
  einsum ([`src/aminx/model/ligand_features.py`](src/aminx/model/ligand_features.py))

  The projection used `jnp.einsum("lqp, lym -> lyp", R_residue, diff)`. Because `q` and
  `m` each appear in only one operand and not in the output, einsum summed both
  independently — `(Σ_q R[l,q,p])·(Σ_m diff[l,y,m])`, an outer product of column-sums
  rather than the frame projection `e_p·diff`. Corrected to `"lqp, lyq -> lyp"`.

  This was the root cause of the side-chain-context cross-framework divergence (~0.85
  Pearson vs ~0.9998 baseline vs the LigandMPNN reference). It only surfaced with side
  chains ON: the no-side-chain baseline's dummy ligand is fully masked, so the corrupted
  node features never contributed.

### Tests

- Add side-chain-context logits parity test vs the PyTorch LigandMPNN reference, and
  migrate the tied-autoregressive / multistate side-chain tests to the current bundle
  API (tie groups via `build_inference_bundle(tie_group_map=...)`; side-chain context
  packaged onto `GeometryBundle` rather than loose kernel kwargs).
- Add `tests/model/test_unconditional_sidechain_bundle.py` covering the
  `build_inference_bundle` → `score_unconditional.kernel` side-chain path: shape
  normalization for 3-D and 4-D `atom_37` inputs, logits sensitivity to `atom_37` when
  `use_side_chains=True` (requires `fixed_mask=1` so fixed residues contribute context),
  logits invariance when `use_side_chains=False`, and `ValueError` guard for missing
  `atom_37` with a side-chain model.

## 0.1.0a4 (2026-06-09)

### Bug Fixes

- **`ProteinFeaturesLigand`**: fix `top_k` crash when ligand atom count < `atom_context_num`
  ([`src/aminx/model/ligand_features.py`](src/aminx/model/ligand_features.py))

  The 0.1.0a3 fix moved `top_k` (A → `atom_context_num=16`) outside the `use_side_chains`
  guard, but did not account for dummy ligand inputs (`with_ligand=False`) where A=1.
  `jax.lax.top_k` raises `ValueError: k argument to top_k must be no larger than size along
  axis` when k=16 > A=1.

  Fix: clamp k to `min(atom_context_num, A)`, matching the existing pattern for the protein
  graph at `k = min(self.k_neighbors, Ca.shape[0])` (line 303).

## 0.1.0a3 (2026-06-09)

### Bug Fixes

- **`ProteinFeaturesLigand`**: fix OOM on large ligand atom counts when `use_side_chains=False`
  ([`src/aminx/model/ligand_features.py`](src/aminx/model/ligand_features.py))

  The `top_k` atom selection (A → `atom_context_num=16`) was only applied inside the
  `use_side_chains` branch. Without sidechain mode the full `A=155` atoms flowed into
  `_y_edges_coords_to_embed`, whose output buffer is pre-allocated at
  `(L, A, A, node_features)`. With flat-multistate inputs (`L≈2048`, `A=155`) this
  allocates `≈20 GiB` before any other live buffers, causing
  `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 23.63 GiB` on all GPU
  tiers including H200.

  Fix: move the `top_k` selection outside the `use_side_chains` guard so it always runs.

- **`pyproject.toml`**: bump `proxide>=0.1.0a8`

## 0.1.0a2

- Initial public alpha — Sprint 2 inference API (`build_inference_bundle`,
  `score_unconditional`, `score_conditional`), flat-multistate support, ligand chunking.

## 0.1.0a1

- Initial release.
