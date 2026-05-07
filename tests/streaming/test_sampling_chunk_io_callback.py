"""``_sample_batch`` chunk-marker ``io_callback`` (Phase 5g PR1 — ordered=False)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.run.sampling import SamplingSpecification, sample
from prxteinmpnn.utils.data_structures import Protein


def test_sample_non_streaming_chunk_markers_use_ordered_false_path(monkeypatch: pytest.MonkeyPatch) -> None:
  """Monkeypatch host hook and force multi-chunk batching; callbacks receive chunk index + count."""
  mock_protein = Protein(
    coordinates=jnp.ones((1, 10, 4, 3)),
    aatype=jnp.ones((1, 10), dtype=jnp.int8),
    one_hot_sequence=jax.nn.one_hot(jnp.ones((1, 10), dtype=jnp.int8), 21),
    mask=jnp.ones((1, 10)),
    residue_index=jnp.arange(10)[None, :],
    chain_index=jnp.zeros((1, 10)),
  )
  mock_model = MagicMock()
  mock_sampler_fn = MagicMock()
  mock_sampler_fn.return_value = (jnp.ones((10,), dtype=jnp.int8), jnp.ones((10, 21)), jnp.arange(10))

  calls: list[tuple[int, int]] = []

  def recorder(chunk_idx: object, chunk_count: object) -> None:
    calls.append((int(np.asarray(chunk_idx)), int(np.asarray(chunk_count))))

  monkeypatch.setattr(
    "prxteinmpnn.run.sampling._noop_sampling_chunk_io",
    recorder,
    raising=True,
  )

  with patch(
    "prxteinmpnn.run.sampling.prep_protein_stream_and_model",
    return_value=([mock_protein], mock_model),
  ):
    with patch(
      "prxteinmpnn.run.sampling.SamplingDriver.build_sampler_fn",
      return_value=mock_sampler_fn,
    ):
      spec = SamplingSpecification(
        inputs=["1ubq.pdb"],
        num_samples=4,
        backbone_noise=[0.1],
        samples_chunk_size=2,
      )
      sample(spec)

  assert sorted(calls, key=lambda t: t[0]) == [(0, 2), (1, 2)]
