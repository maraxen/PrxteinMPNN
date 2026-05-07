"""Scoring tensor-batch ``io_callback`` (Phase 5g PR3b — ordered=False)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.run.scoring import score
from prxteinmpnn.run.specs import ScoringSpecification
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.testing import get_tolerances


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


@pytest.fixture
def mock_batched_protein():
  """Batch size 1, length 10 — matches ``tests/run/test_scoring_averaged``."""
  aatype = jnp.zeros((1, 10), dtype=jnp.int8)
  return Protein(
    coordinates=jnp.ones((1, 10, 4, 3)),
    mask=jnp.ones((1, 10)),
    residue_index=jnp.arange(10)[None, :],
    chain_index=jnp.zeros((1, 10)),
    aatype=aatype,
    one_hot_sequence=jnp.eye(21)[aatype],
  )


@pytest.fixture
def small_inference_model():
  key = jax.random.key(0)
  model = PrxteinMPNN(
    node_features=16,
    edge_features=16,
    hidden_features=16,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=5,
    key=key,
  )
  return eqx.tree_inference(model, value=True)


def test_score_standard_tensor_hook_matches_score_output(
  monkeypatch: pytest.MonkeyPatch,
  mock_protein: Protein,
  mock_model: MagicMock,
) -> None:
  """Two iterator batches: recorded tensors (sorted by batch idx) match ``score()`` concat."""
  recordings: list[tuple[int, int, np.ndarray, np.ndarray]] = []

  def recorder(
    batch_idx: object,
    batch_count: object,
    scores_host: object,
    logits_host: object,
  ) -> None:
    recordings.append(
      (
        int(np.asarray(batch_idx)),
        int(np.asarray(batch_count)),
        np.asarray(scores_host),
        np.asarray(logits_host),
      ),
    )

  monkeypatch.setattr(
    "prxteinmpnn.run.scoring._noop_scoring_tensor_batch_io",
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
    result = score(spec)

  rtol, atol = get_tolerances(jnp.float32)
  sorted_recs = sorted(recordings, key=lambda t: t[0])
  scores_from_hook = jnp.concatenate([jnp.asarray(r[2]) for r in sorted_recs], axis=0)
  logits_from_hook = jnp.concatenate([jnp.asarray(r[3]) for r in sorted_recs], axis=0)

  assert jnp.allclose(scores_from_hook, result["scores"], rtol=rtol, atol=atol)
  assert jnp.allclose(logits_from_hook, result["logits"], rtol=rtol, atol=atol)


def test_score_averaged_inputs_and_noise_tensor_hook_matches_score_output(
  monkeypatch: pytest.MonkeyPatch,
  mock_batched_protein: Protein,
  small_inference_model: PrxteinMPNN,
) -> None:
  """Averaged ``inputs_and_noise``: hook tensors match ``score()`` after concat."""
  recordings: list[tuple[int, int, np.ndarray, np.ndarray]] = []

  def recorder(
    batch_idx: object,
    batch_count: object,
    scores_host: object,
    logits_host: object,
  ) -> None:
    recordings.append(
      (
        int(np.asarray(batch_idx)),
        int(np.asarray(batch_count)),
        np.asarray(scores_host),
        np.asarray(logits_host),
      ),
    )

  monkeypatch.setattr(
    "prxteinmpnn.run.scoring._noop_scoring_tensor_batch_io",
    recorder,
    raising=True,
  )

  L = 10
  with patch(
    "prxteinmpnn.run.scoring.prep_protein_stream_and_model",
    return_value=([mock_batched_protein, mock_batched_protein], small_inference_model),
  ):
    spec = ScoringSpecification(
      inputs=["dummy_path"],
      sequences_to_score=["G" * L],
      average_node_features=True,
      average_encoding_mode="inputs_and_noise",
      backbone_noise=[0.1],
    )
    result = score(spec)

  rtol, atol = get_tolerances(jnp.float32)
  sorted_recs = sorted(recordings, key=lambda t: t[0])
  scores_from_hook = jnp.concatenate([jnp.asarray(r[2]) for r in sorted_recs], axis=0)
  logits_from_hook = jnp.concatenate([jnp.asarray(r[3]) for r in sorted_recs], axis=0)

  assert jnp.allclose(scores_from_hook, result["scores"], rtol=rtol, atol=atol)
  assert jnp.allclose(logits_from_hook, result["logits"], rtol=rtol, atol=atol)
