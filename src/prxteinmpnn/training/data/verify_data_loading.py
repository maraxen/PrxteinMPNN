"""Verify data loading functionality."""

import logging
from pathlib import Path

from prxteinmpnn.io.loaders import create_protein_dataset

logger = logging.getLogger(__name__)


def verify_loading() -> None:
  """Verify that the dataset can be loaded correctly."""
  data_dir = Path("src/prxteinmpnn/training/data")
  array_record_path = data_dir / "pdb_sample.array_record"
  index_path = data_dir / "pdb_sample.index.json"

  if not array_record_path.exists():
    logger.error("File not found: %s", array_record_path)
    return

  logger.info("Loading data from %s...", array_record_path)

  # Create dataset
  # Note: We use use_preprocessed=True to use our ArrayRecordDataSource
  ds = create_protein_dataset(
    str(array_record_path),
    batch_size=4,
    use_preprocessed=True,
    preprocessed_index_path=str(index_path),
    use_electrostatics=False,  # We didn't compute them yet
    use_vdw=False,
  )

  logger.info("Dataset created. Iterating...")

  for i, batch in enumerate(ds):
    logger.info("Batch %d:", i)
    logger.info("  Coords shape: %s", batch.coordinates.shape)
    logger.info("  Sequence shape: %s", batch.aatype.shape)
    logger.info("  Mask shape: %s", batch.mask.shape)

    max_batches = 2
    if i >= max_batches:
      break

  logger.info("Verification successful!")


if __name__ == "__main__":
  verify_loading()
