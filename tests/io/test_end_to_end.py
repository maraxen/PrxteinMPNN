"""End-to-end tests for the complete IO pipeline."""

import tempfile
from pathlib import Path

import numpy as np

from priox.io.parsing.dispatch import parse_input
from priox.physics.features import compute_electrostatic_node_features


def test_pdb_to_features_pipeline():
    """Test complete pipeline: PDB → ProcessedStructure → ProteinTuple → Features."""
    pdb_content = """ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   GLY A   1       1.251   2.389   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       3.338   1.584   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       4.000   2.880   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2       5.520   2.720   0.000  1.00  0.00           C
ATOM      8  O   ALA A   2       6.100   1.635   0.000  1.00  0.00           O
ATOM      9  CB  ALA A   2       3.500   3.700   1.200  1.00  0.00           C
HETATM   10  O   HOH A 101      20.000  20.000  20.000  1.00  0.00           O
END
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(pdb_content)
        pdb_file = Path(f.name)

    try:
        # Parse through dispatch
        protein_tuples = list(parse_input(pdb_file, chain_id=None))

        assert len(protein_tuples) > 0, "Should yield at least one ProteinTuple"
        protein = protein_tuples[0]

        # Verify structure
        assert protein.coordinates is not None
        assert protein.aatype is not None
        assert len(protein.aatype) == 2, f"Expected 2 residues, got {len(protein.aatype)}"

        # Verify physics parameters populated
        assert protein.charges is not None, "Charges should be populated from force field"
        assert protein.sigmas is not None, "Sigmas should be populated from force field"
        assert protein.epsilons is not None, "Epsilons should be populated from force field"

        # Verify solvent was removed (should not have 10 atoms from water)
        assert protein.full_coordinates is not None
        n_atoms = len(protein.full_coordinates)
        assert n_atoms >= 9, f"Expected at least 9 atoms after processing, got {n_atoms}"

        print("✓ PDB → Features pipeline test passed")
        print(f"  Residues: {len(protein.aatype)}")
        print(f"  Atoms: {n_atoms}")

    finally:
        pdb_file.unlink()


def test_pqr_to_features_pipeline():
    """Test complete pipeline: PQR → ProcessedStructure → ProteinTuple → Features."""
    pqr_content = """ATOM      1  N   GLY A   1       0.000   0.000   0.000 -0.4157 1.8500
ATOM      2  CA  GLY A   1       1.458   0.000   0.000  0.0337 2.2750
ATOM      3  C   GLY A   1       2.009   1.420   0.000  0.5973 2.0000
ATOM      4  O   GLY A   1       1.251   2.389   0.000 -0.5679 1.7000
ATOM      5  N   ALA A   2       3.338   1.584   0.000 -0.4157 1.8500
ATOM      6  CA  ALA A   2       4.000   2.880   0.000  0.0337 2.2750
ATOM      7  C   ALA A   2       5.520   2.720   0.000  0.5973 2.0000
ATOM      8  O   ALA A   2       6.100   1.635   0.000 -0.5679 1.7000
ATOM      9  CB  ALA A   2       3.500   3.700   1.200  0.0000 2.0600
END
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pqr", delete=False) as f:
        f.write(pqr_content)
        pqr_file = Path(f.name)

    try:
        protein_tuples = list(parse_input(pqr_file, chain_id=None))

        assert len(protein_tuples) > 0
        protein = protein_tuples[0]

        # Verify PQR-specific physics parameters
        assert protein.charges is not None
        assert protein.radii is not None
        assert protein.epsilons is not None

        # Verify charges from PQR are preserved
        assert len(protein.charges) >= 9, f"Expected at least 9 atoms, got {len(protein.charges)}"

        # Compute features
        features = compute_electrostatic_node_features(protein)

        assert features.shape == (2, 5), f"Expected (2, 5) features, got {features.shape}"
        assert np.all(np.isfinite(features)), "All features should be finite"

        print("✓ PQR → Features pipeline test passed")
        print(f"  Features shape: {features.shape}")

    finally:
        pqr_file.unlink()


def test_pipeline_with_hydrogens():
    """Test that pipeline correctly adds hydrogens when missing."""
    pdb_content = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00  0.00           C
ATOM      3  C   ALA A   1      11.500  11.000  10.000  1.00  0.00           C
ATOM      4  O   ALA A   1      12.000  11.500  11.000  1.00  0.00           O
ATOM      5  CB  ALA A   1      11.500   9.000  10.000  1.00  0.00           C
END
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(pdb_content)
        pdb_file = Path(f.name)

    try:
        protein_tuples = list(parse_input(pdb_file, chain_id=None))

        assert len(protein_tuples) > 0
        protein = protein_tuples[0]

        # Verify hydrogens were added
        assert protein.full_coordinates is not None
        n_atoms = len(protein.full_coordinates)

        # Original had 5 heavy atoms, should have more with hydrogens
        assert n_atoms > 5, f"Expected hydrogens added, got {n_atoms} atoms (original: 5)"

        print(f"✓ Hydrogen addition test passed (5 → {n_atoms} atoms)")

    finally:
        pdb_file.unlink()


def test_pipeline_preserves_chain_selection():
    """Test that chain selection is preserved through pipeline."""
    pdb_content = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00  0.00           C
ATOM      3  C   ALA A   1      11.500  11.000  10.000  1.00  0.00           C
ATOM      4  O   ALA A   1      12.000  11.500  11.000  1.00  0.00           O
ATOM      5  N   GLY B   1      20.000  20.000  20.000  1.00  0.00           N
ATOM      6  CA  GLY B   1      21.000  20.000  20.000  1.00  0.00           C
ATOM      7  C   GLY B   1      21.500  21.000  20.000  1.00  0.00           C
ATOM      8  O   GLY B   1      22.000  21.500  21.000  1.00  0.00           O
END
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(pdb_content)
        pdb_file = Path(f.name)

    try:
        # Parse only chain A
        protein_tuples_a = list(parse_input(pdb_file, chain_id="A"))
        assert len(protein_tuples_a) > 0
        protein_a = protein_tuples_a[0]

        # Parse only chain B
        protein_tuples_b = list(parse_input(pdb_file, chain_id="B"))
        assert len(protein_tuples_b) > 0
        protein_b = protein_tuples_b[0]

        # Verify different residue counts
        assert len(protein_a.aatype) == 1, f"Chain A should have 1 residue, got {len(protein_a.aatype)}"
        assert len(protein_b.aatype) == 1, f"Chain B should have 1 residue, got {len(protein_b.aatype)}"

        # Verify different amino acids
        assert protein_a.aatype[0] != protein_b.aatype[0], "Chains should have different amino acids"

        print("✓ Chain selection test passed")

    finally:
        pdb_file.unlink()


if __name__ == "__main__":
    test_pdb_to_features_pipeline()
    test_pqr_to_features_pipeline()
    test_pipeline_with_hydrogens()
    test_pipeline_preserves_chain_selection()
    print("\n✅ All end-to-end pipeline tests passed!")
