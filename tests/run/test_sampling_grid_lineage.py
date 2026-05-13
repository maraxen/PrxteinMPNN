"""Determinism regression test for grid lineage and base sampling key.

This test pins the PRNGKey bytes generated for grid-based sampling to catch
unintended changes to the hash chain. If this test fails after refactoring,
the grid lineage hash chain changed and in-flight grid sampling jobs will
produce different sequences, requiring a GRID_SCHEMA_VERSION bump.
"""

import jax  # noqa: F401
import numpy as np
from prxteinmpnn.run.sampling import _base_sampling_key, _resolve_grid_lineage
from prxteinmpnn.run.specs import SamplingSpecification

# Captured during pre-task step C. Replace with actual bytes from assertion error.
PINNED_KEY_BYTES = b'\xd8\x91\xc7\xb5\x5f\x05\x4a\x7d'  # FIXER MUST REPLACE before committing


def test_base_sampling_key_determinism():
  """Pin PRNGKey bytes for a fixed spec + grid lineage.

  If this fails after refactor, hash chain changed and in-flight grid
  sampling jobs will produce different sequences.
  """
  spec = SamplingSpecification(
      inputs=["/tmp/dummy.pdb"],
      random_seed=42,
      num_samples=10,
      grid_mode=True,
      job_id="test_job",
      chunk_id=0,
      sample_start=0,
      sample_count=10,
  )
  lineage = _resolve_grid_lineage(spec)
  key = _base_sampling_key(spec, grid_lineage=lineage)
  key_bytes = np.asarray(jax.random.key_data(key)).tobytes()
  assert key_bytes == PINNED_KEY_BYTES, (
      f"PRNGKey bytes changed: {key_bytes.hex()} != {PINNED_KEY_BYTES.hex()}. "
      "Grid lineage hash chain was altered; bump GRID_SCHEMA_VERSION if intentional."
  )
