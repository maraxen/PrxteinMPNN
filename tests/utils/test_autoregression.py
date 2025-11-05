from types import SimpleNamespace
from prxteinmpnn.run.specs import RunSpecification
from prxteinmpnn.utils.autoregression import resolve_tie_groups

def make_input(chain_id, residue_index):
    return SimpleNamespace(chain_id=jnp.array(chain_id), residue_index=jnp.array(residue_index))

def test_resolve_tie_groups_none():
    inp = make_input([0, 0, 1], [1, 2, 3])
    spec = RunSpecification(inputs=["dummy"], tied_positions=None)
    out = resolve_tie_groups(spec, inp)
    assert (out == jnp.arange(3)).all()

def test_resolve_tie_groups_direct():
    # N=6, L=3, K=2
    inp = make_input([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2])
    inp.num_inputs = 2
    spec = RunSpecification(inputs=["dummy"], tied_positions="direct", pass_mode="inter")
    out = resolve_tie_groups(spec, inp)
    assert (out == jnp.array([0, 1, 2, 0, 1, 2])).all()

def test_resolve_tie_groups_auto():
    inp = make_input([0, 0, 1, 1], [0, 1, 0, 1])
    spec = RunSpecification(inputs=["dummy"], tied_positions="auto", pass_mode="inter")
    # structure_mappings: seq_pos 0 -> [0,2], seq_pos 1 -> [1,3]
    structure_mappings = [[0,2], [1,3]]
    out = resolve_tie_groups(spec, inp, structure_mappings)
    # Should assign same group to 0,2 and to 1,3
    assert out[0] == out[2]
    assert out[1] == out[3]
    assert len(jnp.unique(out)) == 2

def test_resolve_tie_groups_explicit():
    inp = make_input([10000, 10001, 10000, 10001], [5, 10, 6, 11])
    spec = RunSpecification(inputs=["dummy"], tied_positions=[[(10000, 5), (10001, 10)]])
    out = resolve_tie_groups(spec, inp)
    idx0 = jnp.where((inp.chain_id == 10000) & (inp.residue_index == 5))[0][0]
    idx1 = jnp.where((inp.chain_id == 10001) & (inp.residue_index == 10))[0][0]
    assert out[idx0] == out[idx1]
    # The other indices should be in different groups
    assert len(jnp.unique(out)) == 3
"""Tests for autoregression utilities."""

import chex
import jax.numpy as jnp
import pytest
from prxteinmpnn.utils.autoregression import generate_ar_mask


@pytest.mark.parametrize(
    "decoding_order, expected_mask",
    [
        (
            jnp.array([0, 1, 2]),
            jnp.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]]),
        ),
        (
            jnp.array([2, 0, 1]),
            jnp.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]]),
        ),
        (
            jnp.array([1, 2, 0]),
            jnp.array([[1, 0, 1], [1, 1, 1], [0, 0, 1]]),
        ),
    ],
)
def test_generate_ar_mask(decoding_order, expected_mask):
    """Test the generation of the autoregressive mask.

    Args:
        decoding_order: The order in which atoms are decoded.
        expected_mask: The expected autoregressive mask.

    Raises:
        AssertionError: If the output does not match the expected value.
    """
    mask = generate_ar_mask(decoding_order)
    chex.assert_trees_all_equal(mask, expected_mask)
    chex.assert_shape(mask, (len(decoding_order), len(decoding_order)))
    chex.assert_type(mask, int)