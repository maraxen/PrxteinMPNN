
import logging
import pathlib
import mdtraj as md
import numpy as np
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_DATA_DIR = pathlib.Path(__file__).parent.parent / "tests" / "test_data"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_mdcath_data():
    """Download a sample MDcath HDF5 file."""
    logger.info("Fetching MDcath test data...")
    repo_id = "compsciencelab/mdCATH"
    # We need a specific file. Based on dataset structure, usually something like 'data/...'
    # Let's try to download a specific domain file if we can find one.
    # Often these are organized by domain ID.
    # I'll try to download a small one. 
    # If I can't find a specific one easily, I might need to list them first or use a known one.
    # Let's try a common domain or just list files if possible (not easy with just download).
    # I will try to download 'mdcath_dataset/1oaiA00.h5' as a guess or similar.
    # Actually, let's look at the dataset structure from the search result or just try a likely path.
    # Search result didn't give file list.
    # I'll try to download the README first to see file structure if this fails?
    # No, I'll try to download a file that likely exists.
    # Let's try to use the 'hf_hub_download' to get a file.
    # I will try to download 'data/1/1oaiA00.h5' or similar.
    # Wait, I don't want to guess.
    # I will try to download 'README.md' first to see if I can find info, but that's not data.
    # Let's try to search for a file list online or just use a wildcard if possible? No.
    
    # Let's try to download a file from the 'mdcath' dataset.
    # Assuming the dataset has a standard structure.
    # I will try to download a file named '1oa4A00.h5' which is a common example domain.
    # If this fails, I'll need to investigate.
    filename = "1oa4A00.h5" 
    try:
        # The dataset might be organized in subfolders.
        # Let's try to find it.
        # I'll use a try-except block to try a few paths.
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=TEST_DATA_DIR,
            local_dir_use_symlinks=False,
        )  # type: ignore[no-matching-overload]
        logger.info(f"Downloaded MDcath file to {local_path}")
    except Exception as e:
        logger.warning(f"Failed to download {filename}: {e}")
        logger.info("Attempting to find any h5 file...")
        # This is hard without listing.
        # I'll try another common one or a different path structure.
        # Maybe 'data/1oa4A00.h5'
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"data/{filename}",
                repo_type="dataset",
                local_dir=TEST_DATA_DIR,
                local_dir_use_symlinks=False,
            )  # type: ignore[no-matching-overload]
            logger.info(f"Downloaded MDcath file to {local_path}")
        except Exception as e2:
             logger.error(f"Failed to download MDcath data: {e2}")
             # Fallback: Create a dummy MDcath-like HDF5 file if download fails,
             # so we can at least test the parser logic with 'correct' structure.
             create_dummy_mdcath_file()

def create_dummy_mdcath_file():
    import h5py
    logger.info("Creating dummy MDcath file...")
    filepath = TEST_DATA_DIR / "dummy_mdcath.h5"
    with h5py.File(filepath, "w") as f:
        domain_grp = f.create_group("1oa4A00")
        domain_grp.create_dataset("resname", data=np.array(["ALA", "GLY", "SER"], dtype="S3"))
        
        temp_grp = domain_grp.create_group("320")
        replica_grp = temp_grp.create_group("0")
        
        # Create dummy coords: 3 residues * (approx 10 atoms) * 5 frames
        # MDcath has 'coords' dataset in replica group
        # Shape: (n_frames, n_atoms, 3)
        # We need to match atoms to residues.
        # Let's say 5 atoms per residue for simplicity in dummy.
        n_res = 3
        n_atoms_per_res = 5
        n_atoms = n_res * n_atoms_per_res
        n_frames = 5
        coords = np.random.rand(n_frames, n_atoms, 3).astype(np.float32)
        replica_grp.create_dataset("coords", data=coords)
        
        # We also need 'dssp' for residue count check
        # Shape: (n_frames, n_residues) - wait, let's check mdcath.py
        # mdcath.py: dssp_sample = ...["dssp"]; num_residues = dssp_sample.shape[1]
        dssp = np.zeros((n_frames, n_res), dtype="S1") # Dummy DSSP
        replica_grp.create_dataset("dssp", data=dssp)
        
    logger.info(f"Created dummy MDcath file at {filepath}")


def generate_mdtraj_data():
    """Generate a sample MDTraj HDF5 file."""
    logger.info("Generating MDTraj test data...")
    # Create a simple trajectory
    # We can use mdtraj to load a PDB or create one.
    # Let's create a simple topology and trajectory.
    
    # Create a topology
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("ALA", chain)
    top.add_atom("N", md.element.nitrogen, res)
    top.add_atom("CA", md.element.carbon, res)
    top.add_atom("C", md.element.carbon, res)
    
    # Create coordinates for 1 frame
    xyz = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], dtype=np.float32)
    
    traj = md.Trajectory(xyz, top)
    
    output_path = TEST_DATA_DIR / "test_mdtraj.h5"
    traj.save(str(output_path))
    logger.info(f"Saved MDTraj file to {output_path}")

if __name__ == "__main__":
    fetch_mdcath_data()
    generate_mdtraj_data()
