"""Verify that the processed data can be loaded."""

import logging
from pathlib import Path

import jax
import numpy as np
from prxteinmpnn.io.loaders import create_protein_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_loading():
    data_dir = Path("src/prxteinmpnn/training/data")
    array_record_path = data_dir / "pdb_sample.array_record"
    index_path = data_dir / "pdb_sample.index.json"
    
    if not array_record_path.exists():
        logger.error(f"File not found: {array_record_path}")
        return

    logger.info(f"Loading data from {array_record_path}...")
    
    # Create dataset
    # Note: We use use_preprocessed=True to use our ArrayRecordDataSource
    ds = create_protein_dataset(
        str(data_dir), # This might be interpreted as a directory of PDBs if not careful, 
                       # but create_protein_dataset logic usually handles preprocessed if flags are set.
                       # Let's check the signature of create_protein_dataset in loaders.py again if this fails.
                       # Actually, looking at trainer.py:
                       #     val_inputs = spec.validation_preprocessed_path
                       #     val_loader = create_protein_dataset(
                       #       val_inputs, ... use_preprocessed=True, preprocessed_index_path=...
                       #     )
        batch_size=4,
        use_preprocessed=True,
        preprocessed_index_path=str(index_path),
        # We need to pass the specific file path if use_preprocessed is True, usually.
        # Let's pass the array_record path as the first argument.
    )
    
    # However, create_protein_dataset first arg is 'data_path'.
    # If use_preprocessed is True, it likely expects the array_record path.
    # Let's try passing the array_record path.
    
    ds = create_protein_dataset(
        str(array_record_path),
        batch_size=4,
        use_preprocessed=True,
        preprocessed_index_path=str(index_path),
        use_electrostatics=False, # We didn't compute them yet
        use_vdw=False,
    )

    logger.info("Dataset created. Iterating...")
    
    for i, batch in enumerate(ds):
        logger.info(f"Batch {i}:")
        logger.info(f"  Coords shape: {batch.coordinates.shape}")
        logger.info(f"  Sequence shape: {batch.aatype.shape}")
        logger.info(f"  Mask shape: {batch.mask.shape}")
        
        if i >= 2:
            break
            
    logger.info("Verification successful!")

if __name__ == "__main__":
    verify_loading()
