
import pathlib
import tempfile

import h5py
import mdtraj as md
import numpy as np
import pytest

from prxteinmpnn.utils import residue_constants as rc

PDB_1UBQ_STRING = """
HEADER    CHROMOSOMAL PROTEIN                     02-JAN-87   1UBQ
ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N
ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 10.38           C
ATOM      3  C   MET A   1      26.913  26.639   3.531  1.00  9.62           C
ATOM      4  O   MET A   1      27.886  26.463   4.263  1.00  9.62           O
ATOM      5  CB  MET A   1      25.112  24.880   3.649  1.00 13.77           C
ATOM      6  CG  MET A   1      25.353  24.860   5.134  1.00 16.29           C
ATOM      7  SD  MET A   1      23.930  23.959   5.904  1.00 17.17           S
ATOM      8  CE  MET A   1      24.447  23.984   7.620  1.00 16.11           C
ATOM      9  N   GLN A   2      26.335  27.770   3.258  1.00  9.27           N
ATOM     10  CA  GLN A   2      26.850  29.021   3.898  1.00  9.07           C
END
"""

@pytest.fixture
def pdb_file():
    """Create a temporary PDB file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write("MODEL        1\n")
        f.write(PDB_1UBQ_STRING)
        f.write("ENDMDL\n")
        f.write("MODEL        2\n")
        f.write(PDB_1UBQ_STRING)
        f.write("ENDMDL\n")
        f.write("MODEL        3\n")
        f.write(PDB_1UBQ_STRING)
        f.write("ENDMDL\n")
        f.write("MODEL        4\n")
        f.write(PDB_1UBQ_STRING)
        f.write("ENDMDL\n")
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
""",
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
def single_model_pdb_file():
    """Create a temporary PDB file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(PDB_1UBQ_STRING)
        filepath = f.name
    yield filepath
    pathlib.Path(filepath).unlink()


@pytest.fixture
def single_model_hdf5_file(single_model_pdb_file):
    """Create a temporary HDF5 file."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        filepath = tmp.name
    traj = md.load_pdb(single_model_pdb_file)
    traj.save_hdf5(filepath)
    yield filepath
    pathlib.Path(filepath).unlink()


@pytest.fixture
def mdcath_hdf5_file():
    """Pytest fixture to create a mock mdCATH HDF5 file.
    It creates a simplified structure with a single domain, one temperature,
    one replica, and mock datasets.
    """
    # Create a temporary file to store the HDF5 data
    # Using tempfile.NamedTemporaryFile ensures it's cleaned up automatically
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        tmp_file_path = tmp_file.name

    domain_id = "1b9nA03"
    num_residues = 71
    num_full_atoms = 1055 # As per your coords example
    num_frames = 10 # Number of frames in the trajectory

    # Mock data for datasets
    mock_box = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]], dtype=np.float32)
    mock_coords = np.random.rand(num_frames, num_full_atoms, 3).astype(np.float32) * 100

    # Mock dssp: (frames, num_residues), often string/object type.
    # Let's use characters 'H', 'E', 'C' for helix, strand, coil.
    mock_dssp_values = np.array([list("HHHHHEEECCCCEEEEHHHHHCCHHHHCCCCCHHHHHHHHHHHHHHHHCCEEEEECC") * 2], dtype="|O")
    mock_dssp_values = np.tile(mock_dssp_values, (num_frames, 1))[:, :num_residues] # Adjust length

    mock_forces = np.random.rand(num_frames, num_full_atoms, 3).astype(np.float32)
    mock_gyration_radius = np.random.rand(num_frames).astype(np.float64) * 10
    mock_rmsd = np.random.rand(num_frames).astype(np.float32) * 5
    mock_rmsf = np.random.rand(num_residues).astype(np.float32) * 2

    # Mock 'resid' for aatype (integer representation of amino acid types)
    # Let's create a sequence of 71 residues, e.g., 0=ALA, 1=ARG, 2=ASN...
    # Make sure it's an array of integer types
    mock_aatype_ints = np.arange(num_residues, dtype=np.int32) % 20 # Cycle through 20 AA types
    mock_resnames = np.array(
        [rc.restype_1to3.get(i, "UNK") for i in mock_aatype_ints], dtype="S3",
    )

    with h5py.File(tmp_file_path, "w") as f:
        domain_group = f.create_group(domain_id)

        # Add 'resid' dataset directly under the domain group
        # This is where we assume the 'resid' for aatype is stored
        domain_group.create_dataset("resid", data=mock_aatype_ints)
        domain_group.create_dataset("resname", data=mock_resnames)

        # Add a dummy 'numResidues' attribute for consistency, though we derive from resid
        domain_group.attrs["numResidues"] = num_residues

        # Create a single temperature group
        temp_id = "320"
        temp_group = domain_group.create_group(temp_id)

        # Create multiple replica groups (e.g., 0 to 2)
        for replica_id in range(3): # Let's create 3 replicas for this mock file
            replica_group = temp_group.create_group(str(replica_id))

            replica_group.create_dataset("box", data=mock_box)
            replica_group.create_dataset("coords", data=mock_coords)
            replica_group.create_dataset("dssp", data=mock_dssp_values)
            replica_group.create_dataset("forces", data=mock_forces)
            replica_group.create_dataset("gyrationRadius", data=mock_gyration_radius)
            replica_group.create_dataset("rmsd", data=mock_rmsd)
            replica_group.create_dataset("rmsf", data=mock_rmsf)

    # The fixture yields the path to the created mock HDF5 file
    yield tmp_file_path

    # Teardown: Clean up the temporary file after the tests are done
    pathlib.Path(tmp_file_path).unlink()

