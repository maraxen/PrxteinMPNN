# Data Processing Scripts

Scripts for preprocessing and managing training datasets.

## Scripts

### `preprocess_dataset_h5.py`

Preprocess proteins from PDB/PQR files into HDF5 format for fast random access.

### `process_parallel.py`

Parallel processing of PDB/Torch `.pt` files into ArrayRecord format.

### `combine_shards.py`

Combine multiple ArrayRecord shards into a single file.

### `create_index.py`

Create train/valid/test split index from cluster metadata.

### `upload_dataset.py`

Upload processed dataset to cloud storage.

### `debug_*.py`

Debugging utilities for data pipeline issues.

## Usage

See each script's `--help` for detailed usage instructions.
