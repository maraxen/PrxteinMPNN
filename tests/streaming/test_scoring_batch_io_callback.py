"""Scoring structure-batch ``io_callback`` (Phase 5g PR2b — ordered=False)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.run.scoring import score
from prxteinmpnn.run.specs import ScoringSpecification
from prxteinmpnn.utils.data_structures import Protein


@pytest.fixture(autouse=True)
def mock_inference_mode():
  with patch("equinox.nn.inference_mode", side_effect=lambda x, **kwargs: x):
    yield


@pytest.fixture
def mock_protein():
  aatype = jnp.zeros(10, dtype=jnp.int8)
  return Protein(
    coordinates=jnp.ones((10, 27, 3)),
    mask=jnp.ones(10),
    residue_index=jnp.arange(10),
    chain_index=jnp.zeros(10),
    aatype=aatype,
    one_hot_sequence=jnp.eye(21)[aatype],
  )


@pytest.fixture
def mock_model():
  model = MagicMock()
  model.return_value = (None, jnp.zeros((10, 21)))
  return model


def test_score_non_streaming_structure_batch_markers_use_ordered_false_path(
  monkeypatch: pytest.MonkeyPatch,
  mock_protein: Protein,
  mock_model: MagicMock,
) -> None:
  """Monkeypatch host hook; two iterator batches → two structure-batch markers."""
  calls: list[tuple[int, int]] = []

  def recorder(batch_idx: object, batch_count: object) -> None:
    calls.append((int(np.asarray(batch_idx)), int(np.asarray(batch_count))))

  monkeypatch.setattr(
    "prxteinmpnn.run.scoring._noop_scoring_structure_batch_io",
    recorder,
    raising=True,
  )

  with patch(
    "prxteinmpnn.run.scoring.prep_protein_stream_and_model",
    return_value=([mock_protein, mock_protein], mock_model),
  ):
    spec = ScoringSpecification(
      inputs=["dummy_path"],
      sequences_to_score=["G" * 10, "A" * 10],
    )
    score(spec)

  assert sorted(calls, key=lambda t: t[0]) == [(0, 2), (1, 2)]
