
import h5py
import numpy as np
import pytest
import pathlib
from prxteinmpnn.io.parsing.mdcath import parse_mdcath_to_processed_structure

def create_mock_mdcath_file(path: pathlib.Path, with_chain: bool = False):
    with h5py.File(path, "w") as f:
        domain_grp = f.create_group("1oa4A00")
        
        # Residues: ALA, GLY, SER
        resnames = np.array(["ALA", "GLY", "SER"], dtype="S3")
        domain_grp.create_dataset("resname", data=resnames)
        
        if with_chain:
            # Chain: A, A, B
            chains = np.array(["A", "A", "B"], dtype="S1")
            domain_grp.create_dataset("chain", data=chains)
        
        temp_grp = domain_grp.create_group("320")
        replica_grp = temp_grp.create_group("0")
        
        # Coords: 3 residues * 5 atoms * 2 frames
        n_res = 3
        n_atoms = 15
        n_frames = 2
        coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        replica_grp.create_dataset("coords", data=coords)
        
        # DSSP for residue count check
        dssp = np.zeros((n_frames, n_res), dtype="S1")
        replica_grp.create_dataset("dssp", data=dssp)

def test_parse_mdcath_defaults(tmp_path):
    """Test parsing MDcath file without chain info (defaults to chain 0)."""
    h5_path = tmp_path / "test_mdcath_no_chain.h5"
    create_mock_mdcath_file(h5_path, with_chain=False)
    
    structures = list(parse_mdcath_to_processed_structure(h5_path, chain_id=None))
    assert len(structures) == 2 # 2 frames
    
    for struct in structures:
        # Check chain IDs are all 0
        assert np.all(struct.chain_ids == 0)
        # Check residue names (mapped to atom array)
        # 3 residues, 5 atoms each.
        # resnames: ALA, GLY, SER
        # atom_array.res_name should reflect this.
        # Note: parse_mdcath_to_processed_structure returns ProcessedStructure
        # struct.atom_array.res_name
        assert struct.atom_array.res_name[0] == "ALA"
        assert struct.atom_array.res_name[5] == "GLY"
        assert struct.atom_array.res_name[10] == "SER"

def test_parse_mdcath_with_chain(tmp_path):
    """Test parsing MDcath file with chain info."""
    h5_path = tmp_path / "test_mdcath_chain.h5"
    create_mock_mdcath_file(h5_path, with_chain=True)
    
    structures = list(parse_mdcath_to_processed_structure(h5_path, chain_id=None))
    assert len(structures) == 2
    
    for struct in structures:
        # Chain IDs: A, A, B -> 0, 0, 1
        # Expanded to atoms (5 atoms per res)
        # First 10 atoms -> chain 0
        # Last 5 atoms -> chain 1
        assert np.all(struct.chain_ids[:10] == 0)
        assert np.all(struct.chain_ids[10:] == 1)
        
        # Check atom_array.chain_id
        assert struct.atom_array.chain_id[0] == "A"
        assert struct.atom_array.chain_id[10] == "B"
