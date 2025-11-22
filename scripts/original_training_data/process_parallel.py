import argparse
import logging
import multiprocessing as mp
import os
import subprocess
import uuid
import time
from pathlib import Path
from typing import Any, List, Dict
import sys

import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import msgpack
import msgpack_numpy as m
import numpy as np
import torch
from array_record.python.array_record_module import ArrayRecordWriter

# Local Imports
from prxteinmpnn.physics.force_fields import load_force_field_from_hub
from prxteinmpnn.io.parsing.mappings import string_to_protein_sequence
from prxteinmpnn.utils.residue_constants import (
    atom_types,
    restype_1to3,
    restypes,
    van_der_waals_radius,
)

m.patch()

PDB_SAMPLE_URL = "https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz"
PDB_SAMPLE_FILENAME = "pdb_2021aug02.tar.gz"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def parse_pt_file(pt_path: Path) -> List[Dict[str, Any]]:
    """Parse a ProteinMPNN .pt file safely."""
    try:
        data = torch.load(pt_path, map_location=torch.device('cpu'))
        if isinstance(data, list): return data
        elif isinstance(data, dict): return [data]
        return []
    except Exception:
        return []

def process_shard(
    files: List[Path],
    shard_idx: int,
    total_shards: int,
    output_dir: Path,
    force_field_name: str = "ff14SB"
) -> int:
    """
    Worker function: Processes files and writes to a specific ArrayRecord shard.
    """
    m.patch() # Ensure msgpack_numpy is patched in worker
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # Explicitly re-force CPU in worker just in case
    jax.config.update("jax_platform_name", "cpu")
    
    # Naming convention compatible with standard sharding
    shard_name = f"data-{shard_idx:05d}-of-{total_shards:05d}"
    output_path = output_dir / f"{shard_name}.array_record"
    
    writer = ArrayRecordWriter(str(output_path), "zstd:9,group_size:1")
    
    # Suppress TF/JAX logs in workers
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
    force_field = load_force_field_from_hub(force_field_name)
    
    record_count = 0
    
    for pt_file in files:
        proteins = parse_pt_file(pt_file)
        
        for protein in proteins:
            try:
                # --- Validation ---
                if 'chains' in protein and 'xyz' not in protein: continue
                if 'seq' not in protein or 'xyz' not in protein: continue
                
                name = protein.get('name', str(uuid.uuid4()))
                
                # --- Extraction ---
                coords = protein['xyz']
                if isinstance(coords, torch.Tensor): coords = coords.numpy()
                if coords.shape[1] > 4: coords = coords[:, :4, :]
                
                atom_is_nan = np.isnan(coords).any(axis=-1)
                backbone_mask = ~atom_is_nan
                coords = np.nan_to_num(coords, nan=0.0)
                
                seq = protein['seq']
                if isinstance(seq, torch.Tensor): seq = seq.numpy()
                elif isinstance(seq, str): seq = string_to_protein_sequence(seq)
                if seq.dtype in [np.float32, np.float64]: seq = seq.astype(np.int32)

                mask = protein.get('mask')
                if mask is None: mask = np.ones(len(seq), dtype=bool)
                elif isinstance(mask, torch.Tensor):
                    mask = mask.numpy().astype(bool)
                    if mask.ndim == 2: mask = mask[:, 1] if mask.shape[1] > 1 else mask.flatten()
                
                mask = mask & backbone_mask[:, 1] 
                
                chain_idx = protein.get('chain_idx', np.zeros(len(seq), dtype=np.int32))
                if isinstance(chain_idx, torch.Tensor): chain_idx = chain_idx.numpy().astype(np.int32)
                
                res_idx = protein.get('residue_idx', np.arange(len(seq), dtype=np.int32))
                if isinstance(res_idx, torch.Tensor): res_idx = res_idx.numpy().astype(np.int32)

                # --- Physics Features ---
                seq_len = len(seq)
                full_atom_mask = np.zeros((seq_len, 37), dtype=bool)
                full_atom_mask[:, :4] = backbone_mask

                # Pre-allocate
                charges = np.zeros((seq_len, 37), dtype=np.float32)
                sigmas = np.zeros((seq_len, 37), dtype=np.float32)
                epsilons = np.zeros((seq_len, 37), dtype=np.float32)
                radii = np.zeros((seq_len, 37), dtype=np.float32)

                # Calculate params
                for i, r_idx in enumerate(seq):
                    res_letter = restypes[r_idx] if r_idx < len(restypes) else "X"
                    res_name = restype_1to3.get(res_letter, "UNK")
                    
                    for j, atom_name in enumerate(atom_types):
                        q = force_field.get_charge(res_name, atom_name)
                        sig, eps = force_field.get_lj_params(res_name, atom_name)
                        rad = van_der_waals_radius.get(atom_name[0], 1.5)
                        
                        charges[i, j] = q
                        sigmas[i, j] = sig
                        epsilons[i, j] = eps
                        radii[i, j] = rad

                # --- Serialize ---
                record_data = {
                    "protein_id": name,
                    "source_file": str(pt_file),
                    "coordinates": coords.astype(np.float32),
                    "aatype": seq.astype(np.int8),
                    "atom_mask": full_atom_mask,
                    "residue_index": res_idx.astype(np.int32),
                    "chain_index": chain_idx.astype(np.int32),
                    "mask": mask.astype(bool),
                    "physics_features": np.zeros((seq_len, 5), dtype=np.float32),
                    "charges": charges,
                    "sigmas": sigmas,
                    "epsilons": epsilons,
                    "radii": radii,
                }
                # logger.info("Writing to database...")
                
                writer.write(msgpack.packb(record_data, use_bin_type=True))
                record_count += 1
                
            except Exception as e:
                logger.exception(f"Error processing file: {pt_file}, protein: {name}")
                continue

    writer.close()
    return record_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("src/prxteinmpnn/training/data"))
    parser.add_argument("--num_workers", type=int, default=64)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download & Extract
    tar_path = args.output_dir / PDB_SAMPLE_FILENAME
    if not tar_path.exists():
        logger.info(f"Downloading {PDB_SAMPLE_URL}...")
        subprocess.run(["wget", PDB_SAMPLE_URL, "-P", str(args.output_dir)], check=True)
    
    extract_dir = args.output_dir / "pdb_2021aug02"
    extract_dir.mkdir(exist_ok=True)
    if not list(extract_dir.iterdir()):
        logger.info("Extracting tarball...")
        subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(extract_dir)], check=True)

    # 2. Scan Files
    logger.info("Scanning .pt files...")
    all_pt_files = list(extract_dir.rglob("*.pt"))
    if not all_pt_files:
        logger.error("No .pt files found.")
        return

    # 3. Pre-cache ForceField
    load_force_field_from_hub("ff14SB")
    
    # 4. Parallel Processing
    np.random.shuffle(all_pt_files)
    num_workers = min(args.num_workers, len(all_pt_files))
    chunks = np.array_split(all_pt_files, num_workers)
    
    logger.info(f"Launching {num_workers} workers...")
    mp.set_start_method("spawn", force=True)
    
    tasks = [(chunks[i].tolist(), i, num_workers, args.output_dir) for i in range(num_workers)]
    
    start_time = time.time()
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(process_shard, tasks)
        
    duration = time.time() - start_time
    total_records = sum(results)
    logger.info(f"Creation complete: {total_records} records in {duration:.2f}s")
    logger.info("Next: Run 'cat data-*.array_record > pdb_2021aug02.array_record' then run create_index.py")

if __name__ == "__main__":
    main()