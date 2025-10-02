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

from prxteinmpnn.io.parsing import (
    _check_if_file_empty,
    _determine_h5_structure,
    af_to_mpnn,
    atom_array_dihedrals,
    atom_names_to_index,
    compute_cb_precise,
    extend_coordinate,
    mpnn_to_af,
    parse_input,
    protein_sequence_to_string,
    residue_names_to_aatype,
    string_key_to_index,
    string_to_protein_sequence,
)
from prxteinmpnn.utils.data_structures import ProteinTuple
from prxteinmpnn.utils.residue_constants import resname_to_idx, restype_order, unk_restype_index
from conftest import PDB_STRING


@pytest.fixture
def pdb_file():
    """Create a temporary PDB file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(PDB_STRING)
        filepath = f.name
    yield filepath
    pathlib.Path(filepath).unlink()


@pytest.fixture
def cif_file():
    """Create a temporary CIF file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as f:
        # A minimal CIF file content with required columns for Biotite
        f.write(
            """
data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
_atom_site.pdbx_PDB_ins_code
ATOM 1 N N GLY A 1 -6.778 -1.424 4.200 1.00 0.00 1 ?
"""
        )
        filepath = f.name
    yield filepath
    pathlib.Path(filepath).unlink()


@pytest.fixture
def hdf5_file(pdb_file):
    """Create a temporary HDF5 file."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        filepath = tmp.name
    traj = md.load_pdb(pdb_file)
    traj.save_hdf5(filepath)
    yield filepath
    pathlib.Path(filepath).unlink()


@pytest.fixture
def mdcath_hdf5_file():
    """Pytest fixture to create a mock mdCATH HDF5 file."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        tmp_file_path = tmp_file.name

    domain_id = "1b9nA03"
    num_residues = 3
    num_full_atoms = 15
    num_frames = 2
    
    mock_coords = np.random.rand(num_frames, num_full_atoms, 3).astype(np.float32)
    mock_resnames = np.array([b"ALA", b"GLY", b"VAL"])
    mock_dssp_values = np.array([list("HCE")], dtype='|O')
    mock_dssp_values = np.tile(mock_dssp_values, (num_frames, 1))

    with h5py.File(tmp_file_path, 'w') as f:
        domain_group = f.create_group(domain_id)
        domain_group.create_dataset("resname", data=mock_resnames)
        
        temp_group = domain_group.create_group("320")
        replica_group = temp_group.create_group("0")
        replica_group.create_dataset("coords", data=mock_coords)
        replica_group.create_dataset("dssp", data=mock_dssp_values)
    
    yield tmp_file_path
    pathlib.Path(tmp_file_path).unlink()


def test_af_to_mpnn():
    """Test conversion from AlphaFold to ProteinMPNN alphabet."""
    af_sequence = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    mpnn_sequence = af_to_mpnn(af_sequence)
    assert mpnn_sequence.tolist() == [
        0,
        14,
        11,
        2,
        1,
        13,
        3,
        5,
        6,
        7,
        9,
        8,
        10,
        4,
        12,
        15,
        16,
        18,
        19,
        17,
        20,
    ]


def test_mpnn_to_af():
    """Test conversion from ProteinMPNN to AlphaFold alphabet."""
    mpnn_sequence = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    )
    af_sequence = mpnn_to_af(mpnn_sequence)
    print(af_sequence)
    assert af_sequence.tolist() == [
      0,
      4,
      3,
      6,
      13,
      7,
      8,
      9,
      11,
      10,
      12,
      2,
      14,
      5,
      1,
      15,
      16,
      19,
      17,
      18,
      20,
    ]


def test_extend_coordinate():
    """Test the extend_coordinate function."""
    atom_a = np.array([0, 0, 0])
    atom_b = np.array([1, 0, 0])
    atom_c = np.array([1, 1, 0])
    bond_length = 1.5
    bond_angle = np.pi / 2
    dihedral_angle = np.pi / 2
    atom_d = extend_coordinate(atom_a, atom_b, atom_c, bond_length, bond_angle, dihedral_angle)
    assert_trees_all_close(atom_d, np.array([1., 1., 1.5]), atol=1e-6)


def test_compute_cb_precise():
    """Test the compute_cb_precise function."""
    n_coord = np.array([0, 0, 0])
    ca_coord = np.array([1.46, 0, 0])
    c_coord = np.array([1.46 + 1.52 * np.cos(111 * np.pi / 180), 1.52 * np.sin(111 * np.pi / 180), 0])
    cb_coord = compute_cb_precise(n_coord, ca_coord, c_coord)
    assert cb_coord.shape == (3,)


def test_string_key_to_index():
    """Test the string_key_to_index function."""
    key_map = {"A": 0, "B": 1, "C": 2}
    keys = np.array(["A", "C", "D"])
    indices = string_key_to_index(keys, key_map, unk_index=3)
    assert np.array_equal(indices, np.array([0, 2, 3]))


def test_string_to_protein_sequence():
    """Test the string_to_protein_sequence function."""
    sequence = "ARND"
    protein_seq = string_to_protein_sequence(sequence)
    expected = af_to_mpnn(np.array([0, 1, 2, 3]))
    assert np.array_equal(protein_seq, expected)


def test_protein_sequence_to_string():
    """Test the protein_sequence_to_string function."""
    protein_seq = af_to_mpnn(np.array([0, 1, 2, 3]))
    sequence = protein_sequence_to_string(protein_seq)
    assert sequence == "ARND"


def test_residue_names_to_aatype():
    """Test the residue_names_to_aatype function."""
    residue_names = np.array(["ALA", "ARG", "ASN", "ASP"])
    aatype = residue_names_to_aatype(residue_names)
    expected = af_to_mpnn(np.array([0, 1, 2, 3]))
    assert np.array_equal(aatype, expected)


def test_atom_names_to_index():
    """Test the atom_names_to_index function."""
    atom_names = np.array(["N", "CA", "C", "O", "CB"])
    indices = atom_names_to_index(atom_names)
    assert np.array_equal(indices, np.array([0, 1, 2, 4, 3]))


def test_atom_array_dihedrals():
    """Test the atom_array_dihedrals function."""
    pdb_path = StringIO(PDB_STRING)
    stack = strucarray(
        [
            Atom(
                coord=[1, 1, 1],
                atom_name="CA",
                res_name="GLY",
                res_id=1,
            )
        ]
    )
    dihedrals = atom_array_dihedrals(stack)
    assert dihedrals is None


def test_check_if_file_empty(tmp_path):
    """Test the _check_if_file_empty utility."""
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()
    assert _check_if_file_empty(str(empty_file))

    non_empty_file = tmp_path / "non_empty.txt"
    non_empty_file.write_text("hello")
    assert not _check_if_file_empty(str(non_empty_file))

    assert _check_if_file_empty("non_existent_file.txt")


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
    """Tests for the main `parse_input` function."""

    def test_parse_pdb_string(self):
        """Test parsing a PDB file from a string."""
        protein_stream = parse_input(StringIO(PDB_STRING))
        protein_list = list(protein_stream)
        assert len(protein_list) == 4
        protein = protein_list[0]
        assert isinstance(protein, ProteinTuple)
        assert protein.aatype.shape == (10,)
        assert protein.atom_mask.shape == (10, 37)
        assert protein.coordinates.shape == (10, 37, 3)
        assert protein.residue_index.shape == (10,)
        assert protein.chain_index.shape == (10,)
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
        with pytest.raises(RuntimeError, match="Failed to parse structure from source: Unknown file format '.txt'"):
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
        assert len(protein_list) == 4
        protein = protein_list[0]
        assert protein.dihedrals is not None
        assert protein.dihedrals.shape == (8, 3) # 10 resiudes - 2, not sure why first and last residues lack dihedrals but its something with biotite

    def test_parse_hdf5(self, hdf5_file):
        """Test parsing an HDF5 file."""
        protein_stream = parse_input(hdf5_file)
        protein_list = list(protein_stream)
        assert len(protein_list) == 4
        protein = protein_list[0]
        assert isinstance(protein, ProteinTuple)
        assert protein.aatype.shape == (10,)
        assert protein.atom_mask.shape == (10, 37)
        assert protein.coordinates.shape == (10, 37, 3)

    def test_parse_mdcath_hdf5(self, mdcath_hdf5_file):
        """Test parsing an mdCATH HDF5 file."""
        protein_stream = parse_input(mdcath_hdf5_file)
        protein_list = list(protein_stream)
        
        assert len(protein_list) == 2
        
        protein = protein_list[0]
        assert isinstance(protein, ProteinTuple)
        assert protein.aatype.shape == (3,)
        assert protein.coordinates.shape == (3, 37, 3)
        assert protein.residue_index.shape == (3,)
        assert protein.chain_index.shape == (3,)
        assert protein.dihedrals is None
        assert protein.full_coordinates is not None

    def test_parse_mdcath_hdf5_chain_selection_not_supported(self, mdcath_hdf5_file):
        """Test that chain selection issues a warning for mdCATH files."""
        with pytest.warns(UserWarning, match="Chain selection is not supported for mdCATH files"):
            protein_list = list(parse_input(mdcath_hdf5_file, chain_id="A"))
        assert len(protein_list) == 2

    def test_parse_mdtraj_hdf5_with_chain_selection(self, hdf5_file):
        """Test parsing mdtraj HDF5 with chain selection."""
        protein_stream = parse_input(hdf5_file, chain_id="B")
        protein_list = list(protein_stream)
        assert len(protein_list) == 2
        assert np.all(protein_list[0].chain_index == 0)

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



