from prxteinmpnn.utils.batching_registry import (
    ALL_AXES,
    N_APC_PAIRS,
    N_COMBINE,
    N_JACOBIAN_PAIRS,
    N_LIGAND_ATOMS,
    N_NOISES,
    N_RESIDUES,
    N_SAMPLES,
    N_STATES,
    N_STRUCTURES,
    N_TEMPERATURES,
)


def test_all_axes_present():
    assert len(ALL_AXES) == 10

def test_axis_indices_unique():
    indices = [ax.axis_index for ax in ALL_AXES]
    assert len(indices) == len(set(indices))

def test_axis_indices_contiguous():
    indices = sorted(ax.axis_index for ax in ALL_AXES)
    assert indices == list(range(len(ALL_AXES)))

def test_heterogeneous_axes():
    assert N_STRUCTURES.heterogeneous is True
    assert N_STATES.heterogeneous is True

def test_homogeneous_axes():
    for ax in [N_RESIDUES, N_LIGAND_ATOMS, N_SAMPLES, N_TEMPERATURES, N_NOISES,
               N_JACOBIAN_PAIRS, N_COMBINE, N_APC_PAIRS]:
        assert ax.heterogeneous is False, f"{ax.name} should not be heterogeneous"

def test_residues_tile_granularity():
    assert N_RESIDUES.tile_granularity == 128

def test_vmap_defaults():
    for ax in [N_RESIDUES, N_LIGAND_ATOMS]:
        assert ax.default_batch_size == 0, f"{ax.name} should default to vmap"

def test_safe_map_defaults():
    for ax in [N_STRUCTURES, N_SAMPLES, N_TEMPERATURES, N_NOISES,
               N_STATES, N_JACOBIAN_PAIRS, N_COMBINE, N_APC_PAIRS]:
        assert ax.default_batch_size > 0, f"{ax.name} should default to safe_map"

def test_positive_cardinalities():
    for ax in ALL_AXES:
        assert ax.cardinality > 0, f"{ax.name}.cardinality must be positive"

def test_positive_tile_granularities():
    for ax in ALL_AXES:
        assert ax.tile_granularity >= 1, f"{ax.name}.tile_granularity must be >= 1"
