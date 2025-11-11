"""Unit tests for the prxteinmpnn.io.parsing submodule."""

import pathlib
import tempfile
from io import StringIO

import h5py
import mdtraj as md
import numpy as np
import pytest
from biotite.structure import Atom, AtomArray, AtomArrayStack, array as strucarray
from chex import assert_trees_all_close

from prxteinmpnn.io.parsing.dispatch import (
    _determine_h5_structure,
    parse_input,
)
from prxteinmpnn.utils.data_structures import ProteinTuple
from prxteinmpnn.utils.residue_constants import resname_to_idx, restype_order, unk_restype_index
from conftest import (
    PDB_1UBQ_STRING as PDB_STRING,
    pdb_file,
    cif_file,
    hdf5_file,
    mdcath_hdf5_file,
    single_model_hdf5_file,
)



def test_determine_h5_structure_mdcath(mdcath_hdf5_file):
    """Test HDF5 structure determination for mdCATH files."""
    structure = _determine_h5_structure(mdcath_hdf5_file)
    assert structure == "mdcath"


def test_determine_h5_structure_mdtraj(hdf5_file):
    """Test HDF5 structure determination for mdtraj files."""
    structure = _determine_h5_structure(hdf5_file)
    assert structure == "mdtraj"


def test_determine_h5_structure_unknown():
    """Test HDF5 structure determination for unknown files."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        filepath = tmp.name
    
    with h5py.File(filepath, "w") as f:
        f.create_dataset("random_data", data=[1, 2, 3])
    
    structure = _determine_h5_structure(filepath)
    assert structure == "mdcath"
    
    pathlib.Path(filepath).unlink()


class TestParseInput:
    def test_parse_pqr_file(self):
        """Test parsing a PQR file using parse_input (integration)."""
        import pathlib
        test_pqr_path = pathlib.Path(__file__).parent.parent.parent / "data" / "1a00.pqr"
        protein_stream = parse_input(test_pqr_path)
        protein_list = list(protein_stream)
        assert len(protein_list) == 1
        protein = protein_list[0]
        assert hasattr(protein, "charges")
        assert hasattr(protein, "radii")
        assert hasattr(protein, "estat_backbone_mask")
        assert hasattr(protein, "estat_resid")
        assert hasattr(protein, "estat_chain_index")
        # Check that charges and radii are numpy arrays and have the same length
        assert protein.charges is not None
        assert protein.radii is not None
        assert protein.charges.shape == protein.radii.shape
        assert protein.charges.dtype == np.float32
        assert protein.radii.dtype == np.float32
        assert protein.estat_backbone_mask is not None
        assert protein.estat_backbone_mask.dtype == bool
        assert protein.estat_resid is not None
        assert protein.estat_resid.dtype == np.int32
        assert protein.estat_chain_index is not None
        assert protein.estat_chain_index.dtype == np.int32
    """Tests for the main `parse_input` function."""

    def test_parse_pdb_string(self):
        """Test parsing a PDB file from a string."""
        protein_stream = parse_input(StringIO(PDB_STRING))
        protein_list = list(protein_stream)
        assert len(protein_list) == 1
        protein = protein_list[0]
        assert isinstance(protein, ProteinTuple)
        assert protein.aatype.shape == (2,)
        assert protein.atom_mask.shape == (2, 37)
        assert protein.coordinates.shape == (2, 37, 3)
        assert protein.residue_index.shape == (2,)
        assert protein.chain_index.shape == (2,)
        assert protein.dihedrals is None
        assert protein.full_coordinates is not None

    def test_parse_pdb_file(self, pdb_file):
        """Test parsing a PDB file from a file path."""
        protein_stream = parse_input(pdb_file)
        protein_list = list(protein_stream)
        assert len(protein_list) == 4
        assert isinstance(protein_list[0], ProteinTuple)

    def test_parse_cif_file(self, cif_file):
        """Test parsing a CIF file from a file path."""
        protein_stream = parse_input(cif_file)
        protein_list = list(protein_stream)
        assert len(protein_list) == 1
        assert isinstance(protein_list[0], ProteinTuple)
        assert protein_list[0].aatype.shape == (1,)

    def test_parse_with_chain_id(self, pdb_file):
        """Test parsing with a specific chain ID."""
        protein_stream = parse_input(pdb_file, chain_id="A")
        protein_list = list(protein_stream)
        assert len(protein_list) == 4
        assert np.all(protein_list[0].chain_index == 0)

    def test_parse_with_invalid_chain_id(self, pdb_file):
        """Test parsing with an invalid chain ID."""
        with pytest.raises(RuntimeError, match="Failed to parse structure from source"):
            list(parse_input(pdb_file, chain_id="Z"))

    def test_parse_empty_file(self):
        """Test parsing an empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=True) as tmp:
            with pytest.raises(RuntimeError):
                list(parse_input(tmp.name))

    def test_parse_empty_pdb_string(self):
        """Test parsing an empty PDB string."""
        with pytest.raises(RuntimeError, match="Failed to parse structure from source"):
            list(parse_input(""))

    def test_parse_invalid_file(self):
        """Test parsing an invalid file path."""
        with pytest.raises(RuntimeError):
            list(parse_input("non_existent_file.pdb"))

    def test_parse_unsupported_format(self):
        """Test parsing an unsupported file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("hello")
            filepath = tmp.name
        # Match any path ending with .txt and the error message
        regex = r"Failed to parse structure from source: .*\.txt\. ValueError: Unknown file format '.txt'"
        with pytest.raises(RuntimeError, match=regex):
            list(parse_input(filepath))
        pathlib.Path(filepath).unlink()

    def test_parse_mdtraj_trajectory(self, pdb_file):
        """Test parsing an mdtraj.Trajectory object."""
        traj = md.load_pdb(pdb_file)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".h5", delete=False) as tmp:
            traj.save_hdf5(tmp.name)
            filepath = tmp.name
        
        protein_stream = parse_input(filepath)
        protein_list = list(protein_stream)
        assert len(protein_list) == 4
        assert isinstance(protein_list[0], ProteinTuple)
        pathlib.Path(filepath).unlink()

    def test_parse_atom_array_stack(self):
        """Test parsing a biotite.structure.AtomArrayStack."""
        stack = AtomArrayStack(1, 4)
        stack.atom_name = np.array(["N", "CA", "C", "O"])
        stack.res_name = np.array(["GLY", "GLY", "GLY", "GLY"])
        stack.res_id = np.array([1, 1, 1, 1])
        stack.chain_id = np.array(["A", "A", "A", "A"])
        stack.coord = np.random.rand(1, 4, 3)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            from biotite.structure.io.pdb import PDBFile
            pdb_file = PDBFile()
            pdb_file.set_structure(stack)
            pdb_file.write(tmp)
            filepath = tmp.name

        protein_stream = parse_input(filepath)
        protein_list = list(protein_stream)
        assert len(protein_list) == 1
        assert isinstance(protein_list[0], ProteinTuple)
        pathlib.Path(filepath).unlink()

    def test_parse_atom_array(self):
        """Test parsing a biotite.structure.AtomArray."""
        arr = AtomArray(4)
        arr.atom_name = np.array(["N", "CA", "C", "O"])
        arr.res_name = np.array(["GLY", "GLY", "GLY", "GLY"])
        arr.res_id = np.array([1, 1, 1, 1])
        arr.chain_id = np.array(["A", "A", "A", "A"])
        arr.coord = np.random.rand(4, 3)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            from biotite.structure.io.pdb import PDBFile
            pdb_file = PDBFile()
            pdb_file.set_structure(arr)
            pdb_file.write(tmp)
            filepath = tmp.name

        protein_stream = parse_input(filepath)
        protein_list = list(protein_stream)
        assert len(protein_list) == 1
        assert isinstance(protein_list[0], ProteinTuple)
        pathlib.Path(filepath).unlink()

    def test_parse_with_dihedrals(self):
        """Test parsing with dihedral angle extraction."""
        protein_stream = parse_input(StringIO(PDB_STRING), extract_dihedrals=True)
        protein_list = list(protein_stream)
        assert len(protein_list) == 1
        protein = protein_list[0]
        assert protein.dihedrals is None

    def test_parse_hdf5(self, hdf5_file):
        """Test parsing an HDF5 file."""
        protein_stream = parse_input(hdf5_file)
        protein_list = list(protein_stream)
        assert len(protein_list) == 4
        protein = protein_list[0]
        assert isinstance(protein, ProteinTuple)
        assert protein.aatype.shape == (2,)
        assert protein.atom_mask.shape == (2, 37)
        assert protein.coordinates.shape == (2, 37, 3)


    def test_parse_mdcath_hdf5_chain_selection_not_supported(self, mdcath_hdf5_file):
        """Test that chain selection issues a warning for mdCATH files."""
        with pytest.warns(UserWarning, match="Chain selection is not supported for mdCATH files"):
            protein_list = list(parse_input(mdcath_hdf5_file, chain_id="A"))
        assert len(protein_list) == 0

    def test_parse_mdtraj_hdf5_with_chain_selection(self, single_model_hdf5_file):
        """Test parsing mdtraj HDF5 with chain selection."""
        protein_stream = parse_input(single_model_hdf5_file, chain_id="A")
        protein_list = list(protein_stream)
        assert len(protein_list) == 1
        assert protein_list[0].chain_index.shape[0] == 2

    def test_parse_hdf5_malformed_file(self):
        """Test parsing a malformed HDF5 file."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            filepath = tmp.name
        
        # Create an empty HDF5 file
        with h5py.File(filepath, "w") as f:
            pass
        
        protein_list = list(parse_input(filepath))
        assert len(protein_list) == 0
        
        pathlib.Path(filepath).unlink()

    def test_parse_hdf5_invalid_mdcath(self):
        """Test parsing an invalid mdCATH HDF5 file."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            filepath = tmp.name
        
        with h5py.File(filepath, "w") as f:
            f.attrs["layout"] = "mdcath_v1.0"
            # Missing required data
        
        protein_list = list(parse_input(filepath))
        assert len(protein_list) == 0
        
        pathlib.Path(filepath).unlink()

    def test_parse_hdf5_invalid_mdtraj(self):
        """Test parsing an invalid mdtraj HDF5 file."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            filepath = tmp.name
        
        # Create HDF5 with invalid structure for mdtraj
        with h5py.File(filepath, "w") as f:
            f.create_dataset("invalid", data=[1, 2, 3])
        
        protein_list = list(parse_input(filepath))
        assert len(protein_list) == 0
        
        pathlib.Path(filepath).unlink()



