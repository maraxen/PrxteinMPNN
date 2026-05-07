"""Sampling tensor-batch ``io_callback`` (Phase 5g PR4 — ``ordered=False``)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.run.sampling import SamplingSpecification, sample
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.testing import get_tolerances


def test_sample_non_streaming_tensor_hook_matches_sample_output(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Two iterator batches: recorded tensors (sorted by batch idx) match ``sample()`` concat."""
  recordings: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []

  def recorder(
    batch_idx: object,
    batch_count: object,
    chunk_start: object,
    chunk_count: object,
    sequences_host: object,
    logits_host: object,
  ) -> None:
    recordings.append(
      (
        int(np.asarray(batch_idx)),
        int(np.asarray(batch_count)),
        int(np.asarray(chunk_start)),
        int(np.asarray(chunk_count)),
        np.asarray(sequences_host),
        np.asarray(logits_host),
      ),
    )

  monkeypatch.setattr(
    "prxteinmpnn.run.sampling._noop_sampling_tensor_batch_io",
    recorder,
    raising=True,
  )

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

  with patch(
    "prxteinmpnn.run.sampling.prep_protein_stream_and_model",
    return_value=([mock_protein, mock_protein], mock_model),
  ):
    with patch(
      "prxteinmpnn.run.sampling.SamplingDriver.build_sampler_fn",
      return_value=mock_sampler_fn,
    ):
      spec = SamplingSpecification(
        inputs=["dummy_path", "dummy_path"],
        num_samples=2,
        backbone_noise=[0.1],
        temperature=[1.0],
        return_logits=True,
      )
      result = sample(spec)

  rtol, atol = get_tolerances(jnp.float32)
  sorted_recs = sorted(recordings, key=lambda t: (t[0], t[2], t[3]))
  sequences_from_hook = jnp.concatenate([jnp.asarray(r[4]) for r in sorted_recs], axis=0)
  logits_from_hook = jnp.concatenate([jnp.asarray(r[5]) for r in sorted_recs], axis=0)

  assert jnp.allclose(sequences_from_hook, result["sequences"], rtol=rtol, atol=atol)
  assert jnp.allclose(logits_from_hook, result["logits"], rtol=rtol, atol=atol)
