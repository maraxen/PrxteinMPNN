"""Canonical AxisSpec registry for all BatchingConfig-mapped axes.

axis_index ordering (innermost = 0, outermost = 9):
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

from aminx.tiling.planner import AxisSpec

N_RESIDUES = AxisSpec(
  name="n_residues",
  axis_index=0,
  cardinality=1200,
  default_batch_size=0,
  tile_granularity=128,
  heterogeneous=False,
  doc="Residue/position dimension within a single structure. Fixed after LENGTH_BUCKETS binning.",
)

N_LIGAND_ATOMS = AxisSpec(
  name="n_ligand_atoms",
  axis_index=1,
  cardinality=64,
  default_batch_size=0,
  tile_granularity=1,
  heterogeneous=False,
  doc="Ligand atom dimension (ligand_mpnn.py:437 triple-vmap). Fixed per structure.",
)

N_STATES = AxisSpec(
  name="n_states",
  axis_index=2,
  cardinality=64,
  default_batch_size=1,  # tile_granularity: iterate one element at a time
  tile_granularity=1,
  heterogeneous=True,
  doc="Multistate stack axis (ProteinBundle.n_states). Shapes vary across states.",
)

N_STRUCTURES = AxisSpec(
  name="n_structures",
  axis_index=3,
  cardinality=32,
  default_batch_size=1,  # tile_granularity: iterate one element at a time
  tile_granularity=1,
  heterogeneous=True,
  doc="Batch of protein structures (BatchingConfig.batch_size). Lengths vary before LENGTH_BUCKETS.",
  dedup_eligible=True,  # repeated backbone structures can be deduplicated
)

N_SAMPLES = AxisSpec(
  name="n_samples",
  axis_index=4,
  cardinality=128,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
  doc="Sequence sample sweep (BatchingConfig.samples_batch_size, samples_chunk_size).",
)

N_TEMPERATURES = AxisSpec(
  name="n_temperatures",
  axis_index=5,
  cardinality=8,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
  doc="Temperature sweep axis (BatchingConfig.temperature_batch_size).",
)

N_NOISES = AxisSpec(
  name="n_noises",
  axis_index=6,
  cardinality=8,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
  doc="Backbone noise sweep axis (BatchingConfig.noise_batch_size).",
)

N_JACOBIAN_PAIRS = AxisSpec(
  name="n_jacobian_pairs",
  axis_index=7,
  cardinality=10000,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
  doc="Residue-pair axis for Jacobian computation (BatchingConfig.jacobian_batch_size). DEFERRED.",
)

N_COMBINE = AxisSpec(
  name="n_combine",
  axis_index=8,
  cardinality=64,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
  doc="Multistate combine step (BatchingConfig.combine_batch_size). DEFERRED.",
)

N_APC_PAIRS = AxisSpec(
  name="n_apc_pairs",
  axis_index=9,
  cardinality=10000,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
  doc="All-pair contact scoring (BatchingConfig.apc_batch_size, apc_residue_batch_size). DEFERRED.",
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
