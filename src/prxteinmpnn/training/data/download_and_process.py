"""Script to download and process PDB sample data for PrxteinMPNN training."""

import argparse
import json
import logging
import subprocess
import tarfile
import uuid
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy as m
import numpy as np
import torch
import tqdm
from array_record.python.array_record_module import ArrayRecordWriter

from prxteinmpnn.io.parsing.mappings import string_to_protein_sequence
from prxteinmpnn.physics.force_fields import load_force_field_from_hub
from prxteinmpnn.utils.residue_constants import (
    atom_types,
    restype_1to3,
    restypes,
    van_der_waals_radius,
)

m.patch()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDB_SAMPLE_URL = "https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02_sample.tar.gz"
PDB_SAMPLE_FILENAME = "pdb_2021aug02_sample.tar.gz"


def download_data(output_dir: Path) -> Path:
    """Download the PDB sample dataset."""
    output_path = output_dir / PDB_SAMPLE_FILENAME
    if output_path.exists():
        logger.info("File %s already exists, skipping download.", output_path)
        return output_path

    logger.info("Downloading %s to %s...", PDB_SAMPLE_URL, output_dir)
    subprocess.run(["wget", PDB_SAMPLE_URL, "-P", str(output_dir)], check=True)
    return output_path


def extract_data(tar_path: Path, extract_dir: Path) -> Path:
    """Extract the tar.gz file."""
    logger.info("Extracting %s to %s...", tar_path, extract_dir)
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
        if isinstance(data, dict):
            return [data]
        logger.warning("Unknown data structure in %s: %s", pt_path, type(data))
        return []
    except Exception:
        logger.exception("Failed to load %s", pt_path)
        return []


def convert_to_array_record(
    pt_files: list[Path], output_path: Path, index_path: Path,
) -> None:
    """Convert parsed data to ArrayRecord format."""
    logger.info(
        "Converting %d .pt files to ArrayRecord at %s...",
        len(pt_files),
        output_path,
    )

    writer = ArrayRecordWriter(str(output_path), "zstd:9,group_size:1")
    protein_index = {}
    global_record_index = 0

    # Load force field
    logger.info("Loading force field...")
    force_field = load_force_field_from_hub("ff14SB")

    for pt_file in tqdm.tqdm(pt_files, desc="Processing files"):
        proteins = parse_pt_file(pt_file)

        for protein in proteins:
            try:
                record_data = _process_protein_dict(protein, pt_file, force_field)
                if record_data:
                    writer.write(msgpack.packb(record_data, use_bin_type=True))
                    protein_index[record_data["protein_id"]] = global_record_index
                    global_record_index += 1

            except Exception:
                name = protein.get("name", "unknown")
                logger.warning("Failed to process protein %s in %s", name, pt_file, exc_info=True)
                continue

    writer.close()

    # Write index
    with index_path.open("w") as f:
        json.dump(protein_index, f, indent=2)

    logger.info("Processed %d proteins.", global_record_index)


def _extract_and_validate_data(
    protein: dict[str, Any],
    pt_file: Path,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract and validate data from protein dictionary."""
    if "chains" in protein and "xyz" not in protein:
        logger.debug("Skipping metadata file %s", pt_file)
        return None

    if "seq" not in protein or "xyz" not in protein:
        logger.warning(
            "Skipping protein %s in %s: missing 'seq' or 'xyz'",
            protein.get("name", "unknown"),
            pt_file,
        )
        return None

    name = protein.get("name", str(uuid.uuid4()))
    seq = protein["seq"]
    coords = protein["xyz"]
    mask = protein.get("mask")

    # Convert to numpy
    if isinstance(coords, torch.Tensor):
        coords = coords.numpy()

    max_atoms = 4
    if coords.shape[1] > max_atoms:
        coords = coords[:, :max_atoms, :]

    # Handle NaNs
    atom_is_nan = np.isnan(coords).any(axis=-1)
    backbone_mask = ~atom_is_nan
    coords = np.nan_to_num(coords, nan=0.0)

    # Process sequence
    if isinstance(seq, torch.Tensor):
        seq = seq.numpy()
    elif isinstance(seq, str):
        seq = string_to_protein_sequence(seq)

    if seq.dtype in (np.float32, np.float64):
        seq = seq.astype(np.int32)

    # Process mask
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy().astype(bool)
        if mask.ndim == 2:
            mask = mask[:, 1] if mask.shape[1] > 1 else mask.flatten()

    # Combine with coordinate validity
    if mask is None:
        mask = np.ones(len(coords), dtype=bool)
    elif len(mask) != len(coords):
        # Handle mismatch if necessary, or just skip
        pass

    ca_valid = backbone_mask[:, 1]
    mask = mask & ca_valid

    return name, seq, coords, mask, backbone_mask


def _compute_physics_features(
    seq: np.ndarray,
    force_field: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute physics parameters for the sequence."""
    charges = np.zeros((len(seq), 37), dtype=np.float32)
    sigmas = np.zeros((len(seq), 37), dtype=np.float32)
    epsilons = np.zeros((len(seq), 37), dtype=np.float32)
    radii = np.zeros((len(seq), 37), dtype=np.float32)

    for i, r_idx in enumerate(seq):
        if r_idx >= len(restypes):
            res_name = "UNK"
        else:
            res_letter = restypes[r_idx]
            res_name = restype_1to3.get(res_letter, "UNK")

        for j, atom_name in enumerate(atom_types):
            # Physics params from Force Field
            q = force_field.get_charge(res_name, atom_name)
            sig, eps = force_field.get_lj_params(res_name, atom_name)

            # Radius from constants (element based)
            element = atom_name[0]
            rad = van_der_waals_radius.get(element, 1.5)

            charges[i, j] = q
            sigmas[i, j] = sig
            epsilons[i, j] = eps
            radii[i, j] = rad

    return charges, sigmas, epsilons, radii


def _process_protein_dict(
    protein: dict[str, Any],
    pt_file: Path,
    force_field: Any,
) -> dict[str, Any] | None:
    """Process a single protein dictionary into an ArrayRecord-compatible dictionary."""
    extracted = _extract_and_validate_data(protein, pt_file)
    if extracted is None:
        return None

    name, seq, coords, mask, backbone_mask = extracted

    # Metadata
    chain_idx = protein.get("chain_idx")
    if chain_idx is None:
        chain_idx = np.zeros(len(seq), dtype=np.int32)
    elif isinstance(chain_idx, torch.Tensor):
        chain_idx = chain_idx.numpy().astype(np.int32)

    res_idx = protein.get("residue_idx")
    if res_idx is None:
        res_idx = np.arange(len(seq), dtype=np.int32)
    elif isinstance(res_idx, torch.Tensor):
        res_idx = res_idx.numpy().astype(np.int32)

    # Construct full atom mask from backbone
    full_atom_mask = np.zeros((len(seq), 37), dtype=bool)
    full_atom_mask[:, :4] = backbone_mask

    charges, sigmas, epsilons, radii = _compute_physics_features(seq, force_field)

    return {
        "protein_id": name,
        "source_file": str(pt_file),
        "coordinates": coords.astype(np.float32),
        "aatype": seq.astype(np.int8),
        "atom_mask": full_atom_mask,
        "residue_index": res_idx.astype(np.int32),
        "chain_index": chain_idx.astype(np.int32),
        "mask": mask.astype(bool),
        "physics_features": np.zeros((len(seq), 5), dtype=np.float32),
        "full_coordinates": np.zeros((len(seq) * 37, 3), dtype=np.float32),
        "charges": charges,
        "sigmas": sigmas,
        "epsilons": epsilons,
        "radii": radii,
        "estat_backbone_mask": np.zeros(len(seq) * 37, dtype=bool),
        "estat_resid": np.zeros(len(seq) * 37, dtype=np.int32),
        "estat_chain_index": np.zeros(len(seq) * 37, dtype=np.int32),
    }


def main() -> None:
    """Download and process PDB sample data."""
    parser = argparse.ArgumentParser(description="Download and process PDB sample data.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("src/prxteinmpnn/training/data"),
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download
    tar_path = download_data(output_dir)

    # 2. Extract
    extract_dir = output_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    extract_data(tar_path, extract_dir)

    # 3. Find .pt files
    # The sample usually has a specific structure. Let's search recursively.
    pt_files = list(extract_dir.rglob("*.pt"))
    logger.info("Found %d .pt files.", len(pt_files))

    if not pt_files:
        logger.warning("No .pt files found. Please check the extraction.")
        return

    # 4. Convert
    output_record = args.output_dir / "pdb_sample.array_record"
    index_record = args.output_dir / "pdb_sample.index.json"
    convert_to_array_record(pt_files, output_record, index_record)


if __name__ == "__main__":
    main()
