# Changelog

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
