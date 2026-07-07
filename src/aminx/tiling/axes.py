"""Canonical AxisSpec registry for all BatchingConfig-mapped axes.

ALL_AXES ordering (innermost first, outermost last) is the demotion-priority
contract for xtrax's joint-budget BatchPlanner (EPIC #1541 T-PLANNER):
candidates are demoted from Vmap to SafeMap strictly in the order specs are
given. This list's order IS that contract -- there is no separate axis_index
field (xtrax.tiling.AxisSpec doesn't have one); reordering this list changes
demotion priority.

  0: n_residues       — residue dimension within a computation
  1: n_ligand_atoms   — per-residue atom count
  2: n_states         — multistate stack (heterogeneous)
  3: n_structures     — batch of proteins (heterogeneous)
  4: n_samples        — sample sweep
  5: n_temperatures   — temperature sweep
  6: n_noises         — backbone noise sweep
  7: n_jacobian_pairs — residue-pair products (deferred)
  8: n_combine        — multistate combine (deferred)
  9: n_apc_pairs      — all-pair contact scoring (deferred)
"""

from xtrax.tiling import AxisSpec

# Residue/position dimension within a single structure. Fixed after LENGTH_BUCKETS binning.
N_RESIDUES = AxisSpec(
  name="n_residues",
  cardinality=1200,
  default_batch_size=0,
  tile_granularity=128,
  heterogeneous=False,
)

# Ligand atom dimension (ligand_mpnn.py:437 triple-vmap). Fixed per structure.
N_LIGAND_ATOMS = AxisSpec(
  name="n_ligand_atoms",
  cardinality=64,
  default_batch_size=0,
  tile_granularity=1,
  heterogeneous=False,
)

# Multistate stack axis (ProteinBundle.n_states). Shapes vary across states.
N_STATES = AxisSpec(
  name="n_states",
  cardinality=64,
  default_batch_size=1,  # tile_granularity: iterate one element at a time
  tile_granularity=1,
  heterogeneous=True,
)

# Batch of protein structures (BatchingConfig.batch_size). Lengths vary before LENGTH_BUCKETS.
N_STRUCTURES = AxisSpec(
  name="n_structures",
  cardinality=32,
  default_batch_size=1,  # tile_granularity: iterate one element at a time
  tile_granularity=1,
  heterogeneous=True,
  dedup_eligible=True,  # repeated backbone structures can be deduplicated
)

# Sequence sample sweep (BatchingConfig.samples_batch_size, samples_chunk_size).
N_SAMPLES = AxisSpec(
  name="n_samples",
  cardinality=128,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
)

# Temperature sweep axis (BatchingConfig.temperature_batch_size).
N_TEMPERATURES = AxisSpec(
  name="n_temperatures",
  cardinality=8,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
)

# Backbone noise sweep axis (BatchingConfig.noise_batch_size).
N_NOISES = AxisSpec(
  name="n_noises",
  cardinality=8,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
)

# Residue-pair axis for Jacobian computation (BatchingConfig.jacobian_batch_size). DEFERRED.
N_JACOBIAN_PAIRS = AxisSpec(
  name="n_jacobian_pairs",
  cardinality=10000,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
)

# Multistate combine step (BatchingConfig.combine_batch_size). DEFERRED.
N_COMBINE = AxisSpec(
  name="n_combine",
  cardinality=64,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
)

# All-pair contact scoring (BatchingConfig.apc_batch_size, apc_residue_batch_size). DEFERRED.
N_APC_PAIRS = AxisSpec(
  name="n_apc_pairs",
  cardinality=10000,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
)

ALL_AXES: list[AxisSpec] = [
  N_RESIDUES,
  N_LIGAND_ATOMS,
  N_STATES,
  N_STRUCTURES,
  N_SAMPLES,
  N_TEMPERATURES,
  N_NOISES,
  N_JACOBIAN_PAIRS,
  N_COMBINE,
  N_APC_PAIRS,
]
