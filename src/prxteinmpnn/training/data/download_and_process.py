"""Script to download and process PDB sample data for PrxteinMPNN training."""

import argparse
import logging
import subprocess
import tarfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from array_record.python.array_record_module import ArrayRecordWriter
import msgpack
import msgpack_numpy as m

from prxteinmpnn.io.parsing.mappings import string_to_protein_sequence

m.patch()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDB_SAMPLE_URL = "https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02_sample.tar.gz"
PDB_SAMPLE_FILENAME = "pdb_2021aug02_sample.tar.gz"


def download_data(output_dir: Path) -> Path:
    """Download the PDB sample dataset."""
    output_path = output_dir / PDB_SAMPLE_FILENAME
    if output_path.exists():
        logger.info(f"File {output_path} already exists, skipping download.")
        return output_path

    logger.info(f"Downloading {PDB_SAMPLE_URL} to {output_dir}...")
    subprocess.run(["wget", PDB_SAMPLE_URL, "-P", str(output_dir)], check=True)
    return output_path


def extract_data(tar_path: Path, extract_dir: Path) -> Path:
    """Extract the tar.gz file."""
    logger.info(f"Extracting {tar_path} to {extract_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    
    # The tarball usually contains a subdirectory, let's find it or return extract_dir
    # Based on typical structure, it might dump files directly or in a folder.
    # Let's check what's inside.
    return extract_dir


def parse_pt_file(pt_path: Path) -> list[dict[str, Any]]:
    """Parse a ProteinMPNN .pt file and return a list of protein dictionaries."""
    try:
        data = torch.load(pt_path)
        # The .pt files usually contain a list of dictionaries
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            logger.warning(f"Unknown data structure in {pt_path}: {type(data)}")
            return []
    except Exception as e:
        logger.error(f"Failed to load {pt_path}: {e}")
        return []


def convert_to_array_record(
    pt_files: list[Path], output_path: Path, index_path: Path
) -> None:
    """Convert parsed data to ArrayRecord format."""
    logger.info(f"Converting {len(pt_files)} .pt files to ArrayRecord at {output_path}...")
    
    writer = ArrayRecordWriter(str(output_path), "zstd:9,group_size:1")
    protein_index = {}
    global_record_index = 0
    
    for pt_file in tqdm.tqdm(pt_files, desc="Processing files"):
        proteins = parse_pt_file(pt_file)
        
        for protein in proteins:
            # Extract relevant fields and map to our schema
            # Original keys often include: 'seq', 'coords', 'name', 'num_chains_per_cluster' etc.
            
            try:
                # Basic validation - we need at least coords (xyz) and sequence
                if 'seq' not in protein or 'xyz' not in protein:
                    continue
                
                name = protein.get('name', str(uuid.uuid4()))
                
                # Coords: [L, 4, 3] (N, CA, C, O)
                coords = protein['xyz']
                if isinstance(coords, torch.Tensor):
                    coords = coords.numpy()
                
                # Ensure coords are [L, 4, 3]
                # If input is [L, 14, 3] or similar, take first 4 (N, CA, C, O)
                if coords.shape[1] > 4:
                    coords = coords[:, :4, :]
                
                # Handle NaNs in coordinates
                # Create mask for valid atoms (not NaN)
                # Check if any coordinate in the atom is NaN
                atom_is_nan = np.isnan(coords).any(axis=-1) # [L, 4]
                backbone_mask = ~atom_is_nan
                
                # Replace NaNs with 0.0 to avoid numerical issues
                coords = np.nan_to_num(coords, nan=0.0)
                
                # Sequence
                seq = protein['seq']
                if isinstance(seq, torch.Tensor):
                    seq = seq.numpy()
                elif isinstance(seq, str):
                    seq = string_to_protein_sequence(seq)
                
                # If sequence is one-hot or something else, we might need to convert.
                # But usually in these .pt files it's integer indices.
                # Let's ensure it's integer.
                if seq.dtype == np.float32 or seq.dtype == np.float64:
                     seq = seq.astype(np.int32)

                # Mask (residue level)
                mask = protein.get('mask')
                if mask is None:
                    mask = np.ones(len(seq), dtype=bool)
                elif isinstance(mask, torch.Tensor):
                    mask = mask.numpy().astype(bool)
                    # If mask is 2D [L, 14] or similar, take CA (index 1) or reduce
                    if mask.ndim == 2:
                        # Assuming standard order N, CA, C, O...
                        # If shape is [L, 4] or [L, 14] or [L, 37]
                        if mask.shape[1] > 1:
                             mask = mask[:, 1] # Take CA
                        else:
                             mask = mask.flatten()
                
                # Update residue mask based on CA existence
                # If CA is missing (NaN), the residue should probably be masked out too
                # or at least we should know.
                # ProteinMPNN uses 'mask' for residue validity.
                # Let's AND it with CA validity.
                ca_valid = backbone_mask[:, 1] # CA is index 1
                mask = mask & ca_valid
                
                # Chain index
                chain_idx = protein.get('chain_idx')
                if chain_idx is None:
                    chain_idx = np.zeros(len(seq), dtype=np.int32)
                elif isinstance(chain_idx, torch.Tensor):
                    chain_idx = chain_idx.numpy().astype(np.int32)
                
                # Residue index
                res_idx = protein.get('residue_idx')
                if res_idx is None:
                    res_idx = np.arange(len(seq), dtype=np.int32)
                elif isinstance(res_idx, torch.Tensor):
                    res_idx = res_idx.numpy().astype(np.int32)

                # Physics features placeholder (zeros for now as we are just loading raw data)
                # In a real pipeline, we'd compute these.
                physics_features = np.zeros((len(seq), 5), dtype=np.float32)

                # Construct full atom mask (L, 37)
                # We only have backbone (4 atoms), so set first 4 cols from backbone_mask
                full_atom_mask = np.zeros((len(seq), 37), dtype=bool)
                full_atom_mask[:, :4] = backbone_mask

                record_data = {
                    "protein_id": name,
                    "source_file": str(pt_file),
                    "coordinates": coords.astype(np.float32),
                    "aatype": seq.astype(np.int8),
                    "atom_mask": full_atom_mask,
                    "residue_index": res_idx.astype(np.int32),
                    "chain_index": chain_idx.astype(np.int32),
                    "mask": mask.astype(bool),
                    "physics_features": physics_features,
                    # Dummy full atomic data to satisfy schema
                    "full_coordinates": np.zeros((len(seq) * 37, 3), dtype=np.float32),
                    "charges": np.zeros(len(seq) * 37, dtype=np.float32),
                    "radii": np.zeros(len(seq) * 37, dtype=np.float32),
                    "estat_backbone_mask": np.zeros(len(seq) * 37, dtype=bool),
                    "estat_resid": np.zeros(len(seq) * 37, dtype=np.int32),
                    "estat_chain_index": np.zeros(len(seq) * 37, dtype=np.int32),
                }
                
                writer.write(msgpack.packb(record_data, use_bin_type=True))
                protein_index[name] = global_record_index
                global_record_index += 1
                
            except Exception as e:
                logger.warning(f"Failed to process protein {name} in {pt_file}: {e}")
                continue

    writer.close()
    
    # Write index
    import json
    with open(index_path, "w") as f:
        json.dump(protein_index, f, indent=2)
    
    logger.info(f"Processed {global_record_index} proteins.")


def main():
    parser = argparse.ArgumentParser(description="Download and process PDB sample data.")
    parser.add_argument("--output_dir", type=Path, default=Path("src/prxteinmpnn/training/data"), help="Output directory")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download
    tar_path = download_data(args.output_dir)
    
    # 2. Extract
    extract_dir = args.output_dir / "pdb_sample"
    extract_dir.mkdir(exist_ok=True)
    extract_data(tar_path, extract_dir)
    
    # 3. Find .pt files
    # The sample usually has a specific structure. Let's search recursively.
    pt_files = list(extract_dir.rglob("*.pt"))
    logger.info(f"Found {len(pt_files)} .pt files.")
    
    if not pt_files:
        logger.warning("No .pt files found. Please check the extraction.")
        return

    # 4. Convert
    output_record = args.output_dir / "pdb_sample.array_record"
    index_record = args.output_dir / "pdb_sample.index.json"
    convert_to_array_record(pt_files, output_record, index_record)


if __name__ == "__main__":
    main()
