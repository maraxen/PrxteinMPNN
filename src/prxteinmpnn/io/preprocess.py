"""Pre-process various input formats into a single HDF5 file for efficient loading."""

import logging  # Import logging
import pathlib
from collections.abc import Sequence
from io import StringIO

import h5py

from prxteinmpnn.io.parsers import parse_input

# Instantiate the logger
logger = logging.getLogger(__name__)


def preprocess_inputs_to_hdf5(
  inputs: Sequence[str | pathlib.Path | StringIO],
  output_path: str | pathlib.Path,
  parse_kwargs: dict | None = None,
) -> None:
  """Parse a mixed list of inputs (files, StringIO) and saves all frames to a single HDF5 file."""
  logger.info("Cache not found. Pre-processing inputs to %s...", output_path)
  parse_kwargs = parse_kwargs or {}

  frame_iterator = (frame for source in inputs for frame in parse_input(source, **parse_kwargs))

  try:
    first_frame = next(frame_iterator)
  except StopIteration:
    logger.warning("No frames found in any input sources. Creating an empty HDF5 file.")
    with h5py.File(output_path, "w") as f:
      f.attrs["format"] = "prxteinmpnn_preprocessed"
      f.attrs["status"] = "empty"
    return

  with h5py.File(output_path, "w") as f:
    f.attrs["format"] = "prxteinmpnn_preprocessed"

    datasets = {}
    for field, data in first_frame._asdict().items():
      if data is not None and hasattr(data, "shape"):
        datasets[field] = f.create_dataset(
          field,
          shape=(1, *data.shape),
          maxshape=(None, *data.shape),
          dtype=data.dtype,
          chunks=True,
        )

    for field, data in datasets.items():
      data[0] = getattr(first_frame, field)

    count = 1
    for frame in frame_iterator:
      for dset in datasets.values():
        dset.resize(count + 1, axis=0)

      for field, dset in datasets.items():
        dset[count] = getattr(frame, field)

      count += 1
      if count % 1000 == 0:
        logger.info("...processed %d frames...", count)

  logger.info("✅ Pre-processing complete. Saved %d frames.", count)
