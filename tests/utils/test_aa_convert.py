"""Unit tests for amino acid conversion functions in the prxteinmpnn.utils.aa_convert module."""

import pytest
import jax.numpy as jnp
import numpy as np

from prxteinmpnn.utils import aa_convert
from prxteinmpnn.utils.residue_constants import (
    restype_order,
    restype_order_with_x,
    unk_restype_index,
)


def test_af_to_mpnn_with_integer_sequence():
    """Test af_to_mpnn with integer-encoded AF sequence."""
    seq = jnp.array([0, 1, 2, 3, 4])  # A, R, N, D, C in AF_ALPHABET
    af_seq = aa_convert.af_to_mpnn(seq)
    expected = jnp.array([aa_convert.MPNN_ALPHABET.index(k) for k in "ARNDC"])
    assert jnp.allclose(af_seq, expected), f"Expected {expected}, got {af_seq}"


def test_mpnn_to_af_and_back_roundtrip():
    """Test roundtrip conversion mpnn_to_af followed by af_to_mpnn returns original."""
    mpnn_seq = jnp.array([0, 1, 2, 3, 4])  # A, R, N, D, C in MPNN_ALPHABET
    af_seq = aa_convert.mpnn_to_af(mpnn_seq)
    mpnn_seq_back = aa_convert.af_to_mpnn(af_seq)
    assert jnp.allclose(
        mpnn_seq_back, mpnn_seq
    ), f"Expected {mpnn_seq}, got {mpnn_seq_back}"


def test_string_key_to_index():
    """Test string_key_to_index with known and unknown keys."""
    key_map = {"A": 0, "C": 1, "D": 2}
    string_keys = np.array(["A", "D", "Z", "C"])
    expected = jnp.array([0, 2, 3, 1])
    result = aa_convert.string_key_to_index(string_keys, key_map)
    assert jnp.allclose(result, expected)

    result_unk = aa_convert.string_key_to_index(string_keys, key_map, unk_index=10)
    expected_unk = jnp.array([0, 2, 10, 1])
    assert jnp.allclose(result_unk, expected_unk)


def test_string_to_protein_sequence():
    """Test string_to_protein_sequence with default and custom mappings."""
    sequence = "ARND"
    expected = jnp.array([0, 14, 11, 2])  # In MPNN order
    result = aa_convert.string_to_protein_sequence(sequence)
    assert jnp.allclose(result, expected)

    custom_map = {"A": 10, "R": 20}
    result_custom = aa_convert.string_to_protein_sequence(
        "ARX", aa_map=custom_map, unk_index=99
    )
    expected_custom = jnp.array([10, 20, 99])
    assert jnp.allclose(result_custom, expected_custom)


def test_protein_sequence_to_string():
    """Test protein_sequence_to_string with default and custom mappings."""
    sequence = jnp.array([0, 14, 11, 20])  # A, R, N, X in MPNN order
    expected = "ARNX"
    result = aa_convert.protein_sequence_to_string(sequence)
    assert result == expected

    # MPNN input sequence [0, 11, 5] is "A", "N", "G"
    # Corresponding AF sequence is [0, 2, 7]
    # We want to map AF 0 to "ALA", AF 2 to "ASN", and AF 7 should be unknown ("X")
    af_custom_map = {0: "ALA", 2: "ASN"}
    result_custom = aa_convert.protein_sequence_to_string(
        jnp.array([0, 11, 5]), aa_map=af_custom_map
    )
    expected_custom = "ALAASNX"
    assert result_custom == expected_custom
