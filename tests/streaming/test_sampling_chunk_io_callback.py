"""``_sample_batch`` chunk + structure-batch ``io_callback`` markers (Phase 5g PR1 + PR3a).

Chunk markers: ``ordered=False`` per chunk inside ``_sample_batch``.
Structure-batch markers: one per protein-iterator batch at the tail of ``_sample_batch``
(campaign chunked paths emit only on the **last** chunk per batch).
"""

from __future__ import annotations

from pathlib import Path
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


def test_sample_non_streaming_structure_batch_markers_two_batches(monkeypatch: pytest.MonkeyPatch) -> None:
  """Two iterator batches → two structure-batch callbacks with batch_count=len(iterator)."""
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

  def recorder(batch_idx: object, batch_count: object) -> None:
    calls.append((int(np.asarray(batch_idx)), int(np.asarray(batch_count))))

  monkeypatch.setattr(
    "prxteinmpnn.run.sampling._noop_sampling_structure_batch_io",
    recorder,
    raising=True,
  )

  with patch(
    "prxteinmpnn.run.sampling.prep_protein_stream_and_model",
    return_value=([mock_protein, mock_protein], mock_model),
  ):
    with patch(
      "prxteinmpnn.run.sampling.SamplingDriver.build_sampler_fn",
      return_value=mock_sampler_fn,
    ):
      spec = SamplingSpecification(
        inputs=["a.pdb", "b.pdb"],
        num_samples=2,
        backbone_noise=[0.1],
        samples_chunk_size=2,
      )
      sample(spec)

  assert sorted(calls, key=lambda t: t[0]) == [(0, 2), (1, 2)]


def test_sample_streaming_campaign_structure_marker_only_on_last_chunk(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Campaign HDF5 chunks multiple ``_sample_batch`` calls; structure marker fires once per batch."""
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

  def recorder(batch_idx: object, batch_count: object) -> None:
    calls.append((int(np.asarray(batch_idx)), int(np.asarray(batch_count))))

  monkeypatch.setattr(
    "prxteinmpnn.run.sampling._noop_sampling_structure_batch_io",
    recorder,
    raising=True,
  )

  out_h5 = tmp_path / "campaign.h5"
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
        temperature=[1.0],
        samples_chunk_size=2,
        output_h5_path=str(out_h5),
        campaign_mode=True,
        return_logits=False,
      )
      with pytest.warns(DeprecationWarning, match="HDF5 output path is deprecated"):
        sample(spec)

  assert calls == [(0, 1)]
