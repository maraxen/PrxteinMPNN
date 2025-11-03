
import jax.numpy as jnp
import numpy as np
import pytest
from prxteinmpnn.run.specs import RunSpecification
from prxteinmpnn.utils.autoregression import (
    get_decoding_step_map,
    make_autoregressive_mask,
    resolve_tie_groups,
)
from prxteinmpnn.utils.data_structures import Protein


def test_tied_autoregressive_mask():
    """Test the tied autoregressive mask generation."""
    tie_group_map = jnp.array([0, 1, 0, 2])
    group_decoding_order = jnp.array([1, 0, 2])
    decoding_step_map = get_decoding_step_map(tie_group_map, group_decoding_order)
    mask = make_autoregressive_mask(decoding_step_map)

    assert jnp.array_equal(decoding_step_map, jnp.array([1, 0, 1, 2]))
    assert mask[0, 2] == False
    assert mask[0, 0] == True
    assert mask[0, 1] == True
    assert mask[1, 0] == False
    assert mask[3, 0] == True


def make_mock_protein(L, chain_ids, res_indices=None):
    """Creates a mock Protein object for testing."""
    if res_indices is None:
        res_indices = np.arange(L)
    return Protein(
        coordinates=jnp.ones((L, 5, 3)),
        aatype=jnp.ones((L,)),
        one_hot_sequence=jnp.ones((L, 21)),
        mask=jnp.ones((L,)),
        residue_index=jnp.array(res_indices),
        chain_index=jnp.array(chain_ids),
    )

def test_resolve_tie_groups_none():
    """Test resolve_tie_groups with tied_positions=None."""
    p = make_mock_protein(10, [0] * 10)
    spec = RunSpecification(inputs=["p"], tied_positions=None)
    tie_group_map = resolve_tie_groups(spec, p)
    assert jnp.array_equal(tie_group_map, jnp.arange(10))

def test_resolve_tie_groups_direct():
    """Test resolve_tie_groups with tied_positions='direct'."""
    p1 = make_mock_protein(10, [0] * 10)
    p2 = make_mock_protein(10, [1] * 10)
    combined = Protein(
        coordinates=jnp.concatenate([p1.coordinates, p2.coordinates]),
        aatype=jnp.concatenate([p1.aatype, p2.aatype]),
        one_hot_sequence=jnp.concatenate([p1.one_hot_sequence, p2.one_hot_sequence]),
        mask=jnp.concatenate([p1.mask, p2.mask]),
        residue_index=jnp.concatenate([p1.residue_index, p2.residue_index]),
        chain_index=jnp.concatenate([p1.chain_index, p2.chain_index]),
    )
    spec = RunSpecification(inputs=["p1", "p2"], tied_positions="direct", pass_mode="inter")
    tie_group_map = resolve_tie_groups(spec, combined)
    assert jnp.array_equal(tie_group_map, jnp.tile(jnp.arange(10), 2))

def test_resolve_tie_groups_auto():
    """Test resolve_tie_groups with tied_positions='auto'."""
    p = make_mock_protein(10, [0] * 10)
    spec = RunSpecification(inputs=["p"], tied_positions="auto", pass_mode="inter")
    structure_mappings = {0: [0, 5], 1: [1, 6]}
    tie_group_map = resolve_tie_groups(spec, p, structure_mappings)
    assert tie_group_map[0] == tie_group_map[5]
    assert tie_group_map[1] == tie_group_map[6]
    assert tie_group_map[0] != tie_group_map[1]

def test_resolve_tie_groups_explicit():
    """Test resolve_tie_groups with explicit tied_positions."""
    p = make_mock_protein(20, [10000] * 10 + [10001] * 10, res_indices=list(range(10)) + list(range(10)))
    spec = RunSpecification(
        inputs=["p"],
        tied_positions=[((10000, 5), (10001, 5)), ((10000, 6), (10001, 6))],
        pass_mode="inter",
    )
    tie_group_map = resolve_tie_groups(spec, p)
    assert tie_group_map[5] == tie_group_map[15]
    assert tie_group_map[6] == tie_group_map[16]
    assert tie_group_map[5] != tie_group_map[6]

