#!/usr/bin/env python3
"""Prepare protein structure fixtures for benchmark suite at multiple sequence lengths.

This script loads test fixtures (1ubq.pdb and 5awl.pdb) and creates normalized
protein structure arrays suitable for benchmarking across different sequence lengths.

For 1ubq.pdb (76 residues), we generate fixtures at lengths [76, 150, 300, 500]
by truncating or zero-padding the coordinate and metadata arrays.

Usage:
    uv run python scripts/benchmarks/prepare_fixtures.py \
        --fixture-dir outputs/benchmark_fixtures/ \
        --verbose

    uv run python scripts/benchmarks/prepare_fixtures.py --dry-run

Exit codes:
    0: SUCCESS
    1: FAILURE (missing fixture files, parsing error, write error)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


class StructureInfo(NamedTuple):
    """Information about a loaded protein structure."""

    num_residues: int
    has_nonstandard_residues: bool
    residue_types: set[str]


def _parse_first_structure(path: Path):
    """Parse a single structure and raise if parser backend is unavailable."""
    try:
        from prxteinmpnn.io.parsing import parse_input
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Cannot parse structure: {exc}") from exc
    try:
        return next(parse_input(str(path)))
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Cannot parse structure: {exc}") from exc
    except StopIteration:
        raise RuntimeError(f"No structures found in {path}") from None


def load_structure(pdb_path: Path) -> tuple:
    """Load a protein structure from a PDB file.

    Parameters
    ----------
    pdb_path : Path
        Path to the PDB file.

    Returns
    -------
    tuple
        (coords, mask, residue_index, chain_index, num_residues) where:
        - coords: (L, 4, 3) array of atom coordinates
        - mask: (L,) boolean array indicating valid residues
        - residue_index: (L,) array of residue indices
        - chain_index: (L,) array of chain indices
        - num_residues: integer number of residues in structure
    """
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    protein = _parse_first_structure(pdb_path)

    # Extract arrays from protein structure
    coords = protein.coordinates  # (L, 4, 3)
    mask = protein.mask  # (L,)
    residue_index = protein.residue_index  # (L,)
    chain_index = protein.chain_index  # (L,)

    num_residues = coords.shape[0]

    logger.info(f"Loaded {pdb_path.name}: {num_residues} residues")

    return coords, mask, residue_index, chain_index, num_residues


def inspect_structure(pdb_path: Path) -> StructureInfo:
    """Inspect a structure and report basic information.

    Parameters
    ----------
    pdb_path : Path
        Path to the PDB file.

    Returns
    -------
    StructureInfo
        Named tuple with num_residues, has_nonstandard_residues, residue_types.
    """
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    protein = _parse_first_structure(pdb_path)
    num_residues = protein.coordinates.shape[0]

    # Standard amino acid one-letter codes
    standard_aa = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "UNK",  # Unknown residue
    }

    # Extract residue names from the structure (if available)
    residue_types = set()
    has_nonstandard = False

    # Try to extract from aatype field if available
    if hasattr(protein, "aatype") and protein.aatype is not None:
        # aatype is typically numeric; we'd need a mapping to residue names
        # For now, just note that we can't easily inspect this way
        logger.debug("Protein has aatype field (numeric codes)")
    else:
        logger.debug("Cannot extract residue names from protein structure")

    return StructureInfo(
        num_residues=num_residues,
        has_nonstandard_residues=has_nonstandard,
        residue_types=residue_types,
    )


def pad_or_truncate_structure(
    coords, mask, residue_index, chain_index, target_length: int
) -> tuple:
    """Pad or truncate a structure to a target sequence length.

    Uses the padding logic from commit 132eca7:
    - If target_length < current length: truncate to first target_length residues
    - If target_length > current length: pad with zeros, marking padded
      residue_index with -1 and mask with 0.0

    Parameters
    ----------
    coords : array (L, 4, 3)
        Atomic coordinates.
    mask : array (L,)
        Residue mask.
    residue_index : array (L,)
        Residue index values.
    chain_index : array (L,)
        Chain index values.
    target_length : int
        Target sequence length.

    Returns
    -------
    tuple
        (coords, mask, residue_index, chain_index) padded/truncated to target_length.
    """
    L = coords.shape[0]

    if target_length < L:
        # Truncate
        coords = coords[:target_length]
        mask = mask[:target_length]
        residue_index = residue_index[:target_length]
        chain_index = chain_index[:target_length]
    elif target_length > L:
        # Pad with zeros
        pad_len = target_length - L
        coords = jnp.pad(coords, ((0, pad_len), (0, 0), (0, 0)))
        mask = jnp.pad(mask, (0, pad_len))
        residue_index = jnp.pad(
            residue_index, (0, pad_len), constant_values=-1
        )
        chain_index = jnp.pad(chain_index, (0, pad_len))

    return coords, mask, residue_index, chain_index


def save_fixture(
    output_dir: Path,
    fixture_name: str,
    coords,
    mask,
    residue_index,
    chain_index,
    dry_run: bool = False,
) -> bool:
    """Save a structure fixture as a .npz file.

    Parameters
    ----------
    output_dir : Path
        Directory to save the fixture.
    fixture_name : str
        Name of the fixture (e.g., "structure_L76").
    coords : array
        (L, 4, 3) coordinates.
    mask : array
        (L,) mask.
    residue_index : array
        (L,) residue indices.
    chain_index : array
        (L,) chain indices.
    dry_run : bool
        If True, print what would be done but don't write.

    Returns
    -------
    bool
        True if successful (or would succeed in dry_run), False otherwise.
    """
    output_path = output_dir / f"{fixture_name}.npz"

    if dry_run:
        logger.info(f"[DRY RUN] Would save: {output_path}")
        logger.info(
            f"  coords: {coords.shape}, mask: {mask.shape}, "
            f"residue_index: {residue_index.shape}, chain_index: {chain_index.shape}"
        )
        return True

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            coords=np.array(coords),
            mask=np.array(mask),
            residue_index=np.array(residue_index),
            chain_index=np.array(chain_index),
        )
        logger.info(f"Saved: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {output_path}: {e}")
        return False


def main():
    """Prepare benchmark fixtures from test data."""
    parser = argparse.ArgumentParser(
        description="Prepare protein structure fixtures for benchmark suite.",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("outputs/benchmark_fixtures"),
        help="Output directory for fixtures (default: outputs/benchmark_fixtures/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    # Locate test fixture files (relative to repo root)
    repo_root = Path(__file__).parent.parent.parent
    ubq_path = repo_root / "tests" / "data" / "1ubq.pdb"
    smd_path = repo_root / "tests" / "data" / "1SMD.pdb"
    awl_path = repo_root / "tests" / "data" / "5awl.pdb"

    if not ubq_path.exists():
        logger.error(f"Required fixture 1ubq.pdb not found at {ubq_path}")
        sys.exit(1)

    # Step 1: Load 1ubq and create fixtures at L=76, 150, 300
    try:
        coords, mask, residue_index, chain_index, num_residues = load_structure(
            ubq_path
        )
    except Exception as e:
        logger.error(f"Failed to load 1ubq.pdb: {e}")
        sys.exit(1)

    ubq_lengths = [76, 150, 300]
    logger.info(f"Creating fixtures for 1ubq at lengths: {ubq_lengths}")

    all_success = True
    for target_len in ubq_lengths:
        try:
            padded_coords, padded_mask, padded_residue_index, padded_chain_index = (
                pad_or_truncate_structure(
                    coords, mask, residue_index, chain_index, target_len
                )
            )
            success = save_fixture(
                args.fixture_dir,
                f"structure_L{target_len}",
                padded_coords,
                padded_mask,
                padded_residue_index,
                padded_chain_index,
                dry_run=args.dry_run,
            )
            if not success:
                all_success = False
        except Exception as e:
            logger.error(f"Failed to process length {target_len}: {e}")
            all_success = False

    # Step 2: Load 1SMD (salivary amylase, 496 residues) for L=500 fixture.
    # 1ubq is only 76 residues; padding it to 500 would mask 424/500 positions,
    # under-reporting latency by ~20-40% (spec §8). 1SMD needs only 4 pad residues.
    if smd_path.exists():
        try:
            smd_coords, smd_mask, smd_residue_index, smd_chain_index, smd_num = (
                load_structure(smd_path)
            )
            logger.info(f"Loaded 1SMD.pdb: {smd_num} residues for L=500 fixture")
            padded_coords, padded_mask, padded_residue_index, padded_chain_index = (
                pad_or_truncate_structure(
                    smd_coords, smd_mask, smd_residue_index, smd_chain_index, 500
                )
            )
            success = save_fixture(
                args.fixture_dir,
                "structure_L500",
                padded_coords,
                padded_mask,
                padded_residue_index,
                padded_chain_index,
                dry_run=args.dry_run,
            )
            if not success:
                all_success = False
        except Exception as e:
            logger.error(f"Failed to create L=500 fixture from 1SMD.pdb: {e}")
            all_success = False
    else:
        logger.warning(
            f"1SMD.pdb not found at {smd_path}; falling back to padded 1ubq for L=500 "
            f"(WARNING: 424/500 positions will be masked — latency will be under-reported)"
        )
        try:
            padded_coords, padded_mask, padded_residue_index, padded_chain_index = (
                pad_or_truncate_structure(
                    coords, mask, residue_index, chain_index, 500
                )
            )
            success = save_fixture(
                args.fixture_dir,
                "structure_L500",
                padded_coords,
                padded_mask,
                padded_residue_index,
                padded_chain_index,
                dry_run=args.dry_run,
            )
            if not success:
                all_success = False
        except Exception as e:
            logger.error(f"Failed to process L=500 fallback: {e}")
            all_success = False

    # Step 3: Inspect 5awl.pdb (report info, don't save fixture yet)
    if awl_path.exists():
        try:
            info = inspect_structure(awl_path)
            logger.info(f"Inspected 5awl.pdb: {info.num_residues} residues")
            if info.has_nonstandard_residues:
                logger.info(f"  Contains non-standard residues: {info.residue_types}")
            else:
                logger.info("  May contain ligand or heteroatom groups (inspect manually)")
        except Exception as e:
            logger.warning(f"Could not inspect 5awl.pdb: {e}")
    else:
        logger.warning(f"Optional fixture 5awl.pdb not found at {awl_path}")

    if all_success:
        logger.info("All fixtures prepared successfully")
        sys.exit(0)
    else:
        logger.error("Some fixtures failed to prepare")
        sys.exit(1)


if __name__ == "__main__":
    main()
