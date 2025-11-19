
import pytest
import pathlib
import numpy as np
from prxteinmpnn.io.parsing.dispatch import parse_input
from prxteinmpnn.utils.data_structures import ProteinTuple

def test_pdb_loading_with_hydride(tmp_path):
    # Create a dummy PDB file (Glycine) without hydrogens
    pdb_content = """ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.009   1.362   0.000  1.00  0.00           C
ATOM      4  O   GLY A   1       1.362   2.405   0.000  1.00  0.00           O
ATOM      5  OXT GLY A   1       3.362   1.362   0.000  1.00  0.00           O
"""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(pdb_content)
    
    # Parse
    frames = list(parse_input(str(pdb_file)))
    assert len(frames) == 1
    pt = frames[0]
    
    assert isinstance(pt, ProteinTuple)
    # Check if hydrogens were added (Glycine should have H on N, CA)
    # We can check full_coordinates shape or atom_mask?
    # full_coordinates should contain all atoms including H.
    # But ProteinTuple.full_coordinates usually matches the input atoms?
    # If hydride added atoms, they should be in full_coordinates.
    
    # Let's check the number of atoms in full_coordinates
    # Original: 5 atoms.
    # Glycine zwitterion (standard in hydride?): NH3+, COO-
    # H on N: 3
    # H on CA: 2
    # Total H: 5
    # Total atoms: 10
    
    # Note: ProcessedStructure.atom_array has all atoms.
    # processed_structure_to_protein_tuples puts frame.coord into full_coordinates.
    
    n_atoms = pt.full_coordinates.shape[0]
    print(f"Number of atoms: {n_atoms}")
    assert n_atoms > 5, "Hydrogens should have been added"

def test_pqr_loading(tmp_path):
    # Create a dummy PQR file
    pqr_content = """ATOM      1  N   GLY A   1       0.000   0.000   0.000  -0.30 1.50
ATOM      2  CA  GLY A   1       1.458   0.000   0.000   0.10 1.70
"""
    pqr_file = tmp_path / "test.pqr"
    pqr_file.write_text(pqr_content)
    
    frames = list(parse_input(str(pqr_file)))
    assert len(frames) == 1
    pt = frames[0]
    
    assert pt.charges is not None
    assert pt.radii is not None
    assert pt.charges.shape[0] == 2
    assert np.allclose(pt.charges, [-0.30, 0.10])
    assert np.allclose(pt.radii, [1.50, 1.70])

if __name__ == "__main__":
    # Manually run if executed as script
    import sys
    from pathlib import Path
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_pdb_loading_with_hydride(Path(tmp_dir))
        test_pqr_loading(Path(tmp_dir))
        print("All tests passed!")
