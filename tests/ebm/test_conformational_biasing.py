"""Fast, synthetic regression tests for the E7 conformational-biasing wiring.

Covers ``aminx.ebm.conformational_biasing``: the pure-array residue-alignment
core (:func:`align_conformational_states`), the sequence-length alignment
guard (:func:`score_conformational_bias`), and the AF-alphabet sequence
conversion helper. Real-PDB-file loading (:func:`load_conformational_states`)
and the real ``eval_data/lpla.csv`` validation live in
``scripts/ebm/lpla_biasing_check.py`` instead -- deliberately not part of
this fast suite (network/large-checkpoint dependent, per the E7 task brief).
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from aminx.ebm.conformational_biasing import (
  ConformationalStates,
  align_conformational_states,
  score_conformational_bias,
  sequence_to_af_aatype,
)
from aminx.ebm.dispatch import score_state_difference
from aminx.ebm.model import ProteinEBMModel
from aminx.utils.aa_convert import AF_ALPHABET

TOKEN_S = 16
TOKEN_Z = 8
DEPTH = 2
HEADS = 2


def _make_model(key: jax.Array) -> ProteinEBMModel:
  return ProteinEBMModel(
    token_s=TOKEN_S,
    token_z=TOKEN_Z,
    dim_fourier=12,
    conditioning_transition_layers=1,
    transformer_depth=DEPTH,
    transformer_heads=HEADS,
    key=key,
  )


class TestAlignConformationalStates:
  def test_intersection_mask_excludes_residues_missing_in_either_state(self) -> None:
    # State 0 ("closed"): residues 1..5, all present.
    coords_a = np.arange(15, dtype=np.float32).reshape(5, 3)
    ridx_a = np.array([1, 2, 3, 4, 5])
    aatype_a = np.array([0, 1, 2, 3, 4])

    # State 1 ("open"): residue 3 missing (a disordered loop), matches on the rest.
    coords_b = np.arange(100, 100 + 12, dtype=np.float32).reshape(4, 3)
    ridx_b = np.array([1, 2, 4, 5])
    aatype_b = np.array([0, 1, 3, 4])  # identical aatype at every shared residue

    states = align_conformational_states(
      [coords_a, coords_b],
      [ridx_a, ridx_b],
      [aatype_a, aatype_b],
      coordinate_scaling=0.1,
    )

    assert isinstance(states, ConformationalStates)
    # Canonical numbering = union = {1,2,3,4,5} -> N=5.
    assert states.mask.shape == (5,)
    np.testing.assert_array_equal(np.asarray(states.residue_numbers), [1, 2, 3, 4, 5])
    # Residue 3 (canonical index 2) is missing from state 1 -> excluded from the mask.
    expected_mask = np.array([True, True, False, True, True])
    np.testing.assert_array_equal(np.asarray(states.mask), expected_mask)
    assert states.coords_states.shape == (2, 5, 3)

    # Centering: each state is centered at the centroid of the SHARED (mask=True) residues.
    shared = expected_mask
    centroid_a = coords_a[[0, 1, 3, 4]].mean(axis=0)  # canonical positions 0,1,3,4 map to rows 0,1,3,4 of state 0
    expected_state_a = (coords_a - centroid_a) * 0.1
    np.testing.assert_allclose(np.asarray(states.coords_states)[0][shared], expected_state_a[shared], atol=1e-5)

  def test_raises_on_amino_acid_identity_mismatch_between_states(self) -> None:
    coords_a = np.zeros((3, 3), dtype=np.float32)
    ridx_a = np.array([1, 2, 3])
    aatype_a = np.array([0, 1, 2])

    coords_b = np.zeros((3, 3), dtype=np.float32)
    ridx_b = np.array([1, 2, 3])
    aatype_b = np.array([0, 1, 5])  # residue 3 disagrees: 2 vs 5

    with pytest.raises(ValueError, match="residue 3"):
      align_conformational_states([coords_a, coords_b], [ridx_a, ridx_b], [aatype_a, aatype_b])

  def test_raises_on_fewer_than_two_states(self) -> None:
    coords_a = np.zeros((3, 3), dtype=np.float32)
    ridx_a = np.array([1, 2, 3])
    aatype_a = np.array([0, 1, 2])

    with pytest.raises(ValueError, match="need >=2 states"):
      align_conformational_states([coords_a], [ridx_a], [aatype_a])

  def test_raises_on_per_state_length_mismatch(self) -> None:
    coords_a = np.zeros((3, 3), dtype=np.float32)
    ridx_a = np.array([1, 2, 3])
    aatype_a = np.array([0, 1])  # wrong length: 2 vs 3

    coords_b = np.zeros((3, 3), dtype=np.float32)
    ridx_b = np.array([1, 2, 3])
    aatype_b = np.array([0, 1, 2])

    with pytest.raises(ValueError, match="mismatched per-residue lengths"):
      align_conformational_states([coords_a, coords_b], [ridx_a, ridx_b], [aatype_a, aatype_b])


class TestSequenceToAfAatype:
  def test_matches_af_alphabet_index_per_character(self) -> None:
    seq = "ARNDCQEGHILKMFPSTWYV"  # every standard residue, in AF_ALPHABET's own order
    aatype = sequence_to_af_aatype(seq)
    np.testing.assert_array_equal(np.asarray(aatype), np.arange(20))

  def test_unknown_character_maps_to_x_index(self) -> None:
    aatype = sequence_to_af_aatype("AZR")  # 'Z' is not a standard residue
    expected_x = AF_ALPHABET.index("X")
    assert int(aatype[1]) == expected_x

  def test_raises_on_empty_sequence(self) -> None:
    with pytest.raises(ValueError, match="non-empty"):
      sequence_to_af_aatype("")


class TestScoreConformationalBias:
  def test_matches_manual_score_state_difference_and_direct_energy_calls(self) -> None:
    n = 6
    model = _make_model(jax.random.PRNGKey(20))
    k_coords, k_aatype = jax.random.split(jax.random.PRNGKey(21))
    coords_states = jax.random.normal(k_coords, (2, n, 3)) * 0.1
    aatype = jax.random.randint(k_aatype, (n,), 0, 21)
    mask = jnp.ones((n,), dtype=bool)
    t = jnp.array(0.05)

    states = ConformationalStates(
      coords_states=coords_states,
      mask=mask,
      reference_aatype=aatype,
      residue_numbers=jnp.arange(1, n + 1),
    )

    gap = score_conformational_bias(model, states, aatype, t, default_batch_size=4)

    # Must agree with calling score_state_difference (E4) directly...
    expected_via_dispatch = score_state_difference(
      model, coords_states, aatype, t, mask, default_batch_size=4,
    )
    assert jnp.allclose(gap, expected_via_dispatch, atol=1e-6)

    # ...and with the default difference_fuse's [closed=0] - [open=1] convention.
    expected_manual = model.energy(coords_states[0], aatype, t, mask) - model.energy(
      coords_states[1], aatype, t, mask,
    )
    assert jnp.allclose(gap, expected_manual, atol=1e-5)

  def test_raises_on_aatype_structure_length_mismatch(self) -> None:
    n = 6
    model = _make_model(jax.random.PRNGKey(22))
    coords_states = jax.random.normal(jax.random.PRNGKey(23), (2, n, 3)) * 0.1
    mask = jnp.ones((n,), dtype=bool)
    states = ConformationalStates(
      coords_states=coords_states,
      mask=mask,
      reference_aatype=jnp.zeros((n,), dtype=jnp.int32),
      residue_numbers=jnp.arange(1, n + 1),
    )
    wrong_length_aatype = jnp.zeros((n - 1,), dtype=jnp.int32)  # deliberately misaligned

    with pytest.raises(ValueError, match="does not match"):
      score_conformational_bias(model, states, wrong_length_aatype, jnp.array(0.05))
