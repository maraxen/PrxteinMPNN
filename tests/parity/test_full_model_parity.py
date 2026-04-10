"""Core ProteinMPNN heavy parity checks against LigandMPNN reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import pearsonr

from prxteinmpnn.model.mpnn import PrxteinMPNN
from tests.parity.reference_utils import require_heavy_parity_prereqs


@dataclass(frozen=True)
class HeavyParityModels:
  """Container for loaded heavy parity models and reference modules."""

  torch: Any
  model_utils: Any
  pt_model: Any
  jax_model: PrxteinMPNN


@dataclass(frozen=True)
class ParityBatch:
  """Deterministic synthetic batch shared across heavy parity tests."""

  x_pytorch: np.ndarray
  x_jax_atom37: jax.Array
  sequence: np.ndarray
  mask: np.ndarray
  chain_mask: np.ndarray
  residue_index: np.ndarray
  chain_index: np.ndarray
  randn: np.ndarray
  decoding_order: np.ndarray
  ar_mask: np.ndarray
  bias: np.ndarray


def _pearson_correlation(lhs: np.ndarray, rhs: np.ndarray) -> float:
  """Compute Pearson correlation on flattened arrays."""
  return float(pearsonr(lhs.ravel(), rhs.ravel())[0])


def _build_parity_batch(*, seq_len: int = 20, seed: int = 4) -> ParityBatch:
  """Create deterministic synthetic inputs used by heavy parity tests."""
  rng = np.random.default_rng(seed)
  batch_size = 1

  x_pytorch = rng.normal(size=(batch_size, seq_len, 4, 3)).astype(np.float32)
  sequence = rng.integers(0, 21, size=(batch_size, seq_len), dtype=np.int64)
  mask = np.ones((batch_size, seq_len), dtype=np.float32)
  chain_mask = np.ones((batch_size, seq_len), dtype=np.float32)
  residue_index = np.tile(np.arange(seq_len), (batch_size, 1)).astype(np.int64)
  chain_index = np.zeros((batch_size, seq_len), dtype=np.int64)
  randn = rng.normal(size=(batch_size, seq_len)).astype(np.float32)
  decoding_order = np.argsort((chain_mask[0] + 0.0001) * np.abs(randn[0]))

  ar_mask = np.zeros((seq_len, seq_len), dtype=np.int32)
  order_position = {token: pos for pos, token in enumerate(decoding_order)}
  for i in range(seq_len):
    for j in range(seq_len):
      if order_position[j] < order_position[i]:
        ar_mask[i, j] = 1

  x_jax_atom37 = jnp.zeros((seq_len, 37, 3), dtype=jnp.float32)
  x_jax_atom37 = x_jax_atom37.at[:, 0, :].set(x_pytorch[0, :, 0, :])
  x_jax_atom37 = x_jax_atom37.at[:, 1, :].set(x_pytorch[0, :, 1, :])
  x_jax_atom37 = x_jax_atom37.at[:, 2, :].set(x_pytorch[0, :, 2, :])
  x_jax_atom37 = x_jax_atom37.at[:, 4, :].set(x_pytorch[0, :, 3, :])

  bias = np.zeros((batch_size, seq_len, 21), dtype=np.float32)
  bias[..., 0] = 100.0

  return ParityBatch(
    x_pytorch=x_pytorch,
    x_jax_atom37=x_jax_atom37,
    sequence=sequence,
    mask=mask,
    chain_mask=chain_mask,
    residue_index=residue_index,
    chain_index=chain_index,
    randn=randn,
    decoding_order=decoding_order,
    ar_mask=ar_mask,
    bias=bias,
  )


def _build_torch_feature_dict(
  torch: Any,
  batch: ParityBatch,
  *,
  include_sampling_args: bool = False,
  symmetry_residues: list[list[int]] | None = None,
  symmetry_weights: list[list[float]] | None = None,
) -> dict[str, Any]:
  """Build reference feature dictionary expected by LigandMPNN torch paths."""
  feature_dict: dict[str, Any] = {
    "X": torch.from_numpy(batch.x_pytorch),
    "S": torch.from_numpy(batch.sequence),
    "mask": torch.from_numpy(batch.mask),
    "chain_mask": torch.from_numpy(batch.chain_mask),
    "R_idx": torch.from_numpy(batch.residue_index),
    "chain_labels": torch.from_numpy(batch.chain_index),
    "randn": torch.from_numpy(batch.randn),
    "batch_size": 1,
    "symmetry_residues": symmetry_residues if symmetry_residues is not None else [[]],
    "symmetry_weights": symmetry_weights if symmetry_weights is not None else [[]],
  }
  if include_sampling_args:
    feature_dict["bias"] = torch.from_numpy(batch.bias)
    feature_dict["temperature"] = 1.0
  return feature_dict


def _build_tie_group_map(seq_len: int, tie_groups: list[list[int]]) -> np.ndarray:
  """Build JAX tie-group map from LigandMPNN symmetry group lists."""
  tie_group_map = np.arange(seq_len, dtype=np.int32)
  for group in tie_groups:
    anchor = group[0]
    for residue_idx in group[1:]:
      tie_group_map[residue_idx] = anchor
  return tie_group_map


def _row_log_softmax(logits: np.ndarray) -> np.ndarray:
  shifted = logits - np.max(logits, axis=-1, keepdims=True)
  return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _combine_reference_tied_log_probs(
  reference_log_probs: np.ndarray,
  *,
  tie_groups: list[list[int]],
  tie_weights: list[list[float]],
) -> np.ndarray:
  combined = np.asarray(reference_log_probs, dtype=np.float64).copy()
  for group, weights in zip(tie_groups, tie_weights, strict=True):
    indices = np.asarray(group, dtype=np.int32)
    weight_array = np.asarray(weights, dtype=np.float64)
    combined_logits = np.sum(reference_log_probs[indices] * weight_array[:, None], axis=0, keepdims=True)
    combined_log_probs = _row_log_softmax(combined_logits)
    combined[indices] = combined_log_probs
  return combined.astype(np.float32)


@pytest.fixture(scope="module")
def heavy_parity_models() -> HeavyParityModels:
  """Load reference torch and converted JAX models for heavy parity checks."""
  pytest.importorskip("torch")
  reference_root, repo_root = require_heavy_parity_prereqs(
    reference_rel_paths=["model_params/proteinmpnn_v_48_020.pt"],
    converted_rel_paths=["model_params/proteinmpnn_v_48_020_converted.eqx"],
  )
  import model_utils
  import torch

  pt_checkpoint_path = reference_root / "model_params/proteinmpnn_v_48_020.pt"
  jax_checkpoint_path = repo_root / "model_params/proteinmpnn_v_48_020_converted.eqx"
  checkpoint = torch.load(pt_checkpoint_path, map_location="cpu")

  pos_weight = checkpoint["model_state_dict"].get("features.embeddings.linear.weight")
  if pos_weight is None:
    pos_weight = checkpoint["model_state_dict"].get("features.positional_embeddings.linear.weight")
  num_positional_embeddings = int((pos_weight.shape[1] - 2) // 2) if pos_weight is not None else 16

  pt_model = model_utils.ProteinMPNN(
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
  )
  pt_model.load_state_dict(checkpoint["model_state_dict"])
  pt_model.eval()

  jax_model = PrxteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    num_positional_embeddings=num_positional_embeddings,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(0),
  )
  jax_model = eqx.tree_deserialise_leaves(jax_checkpoint_path, jax_model)

  return HeavyParityModels(
    torch=torch,
    model_utils=model_utils,
    pt_model=pt_model,
    jax_model=jax_model,
  )


@pytest.fixture(scope="module")
def parity_batch() -> ParityBatch:
  """Return deterministic synthetic parity inputs."""
  return _build_parity_batch()


@pytest.mark.parity_heavy
def test_protein_feature_extraction_parity(
  heavy_parity_models: HeavyParityModels,
  parity_batch: ParityBatch,
) -> None:
  """protein-feature-extraction: projected edge features and neighbors match reference."""
  feature_dict = _build_torch_feature_dict(heavy_parity_models.torch, parity_batch)
  with heavy_parity_models.torch.no_grad():
    pt_edges, pt_neighbor_indices = heavy_parity_models.pt_model.features(feature_dict)
    pt_projected_edges = heavy_parity_models.pt_model.W_e(pt_edges).numpy()[0]

  jax_edges, jax_neighbor_indices, _, _ = heavy_parity_models.jax_model.features(
    jax.random.PRNGKey(0),
    parity_batch.x_jax_atom37,
    jnp.array(parity_batch.mask[0]),
    jnp.array(parity_batch.residue_index[0]),
    jnp.array(parity_batch.chain_index[0]),
    jnp.array(0.0, dtype=jnp.float32),
  )
  jax_edges_np = np.asarray(jax_edges)

  assert np.array_equal(pt_neighbor_indices.numpy()[0], np.asarray(jax_neighbor_indices))
  np.testing.assert_allclose(pt_projected_edges, jax_edges_np, rtol=1e-5, atol=2e-5)
  max_abs_diff = float(np.max(np.abs(pt_projected_edges - jax_edges_np)))
  assert max_abs_diff <= 2e-5


@pytest.mark.parity_heavy
def test_protein_encoder_parity(
  heavy_parity_models: HeavyParityModels,
  parity_batch: ParityBatch,
) -> None:
  """protein-encoder: encoder node/edge activations maintain strong correlation."""
  feature_dict = _build_torch_feature_dict(heavy_parity_models.torch, parity_batch)
  with heavy_parity_models.torch.no_grad():
    pt_node_features, pt_edge_features, pt_neighbor_indices = heavy_parity_models.pt_model.encode(feature_dict)

  jax_edges, jax_neighbor_indices, jax_initial_node_features, _ = heavy_parity_models.jax_model.features(
    jax.random.PRNGKey(0),
    parity_batch.x_jax_atom37,
    jnp.array(parity_batch.mask[0]),
    jnp.array(parity_batch.residue_index[0]),
    jnp.array(parity_batch.chain_index[0]),
    jnp.array(0.0, dtype=jnp.float32),
  )
  jax_node_features, jax_edge_features = heavy_parity_models.jax_model.encoder(
    jax_edges,
    jax_neighbor_indices,
    jnp.array(parity_batch.mask[0]),
    initial_node_features=jax_initial_node_features,
    key=jax.random.PRNGKey(1),
  )

  assert np.array_equal(pt_neighbor_indices.numpy()[0], np.asarray(jax_neighbor_indices))
  node_corr = _pearson_correlation(pt_node_features.numpy()[0], np.asarray(jax_node_features))
  edge_corr = _pearson_correlation(pt_edge_features.numpy()[0], np.asarray(jax_edge_features))
  assert node_corr >= 0.95
  assert edge_corr >= 0.95


@pytest.mark.parity_heavy
def test_decoder_unconditional_parity(
  heavy_parity_models: HeavyParityModels,
  parity_batch: ParityBatch,
) -> None:
  """decoder-unconditional: manual reference branch and JAX branch stay correlated."""
  feature_dict = _build_torch_feature_dict(heavy_parity_models.torch, parity_batch)
  with heavy_parity_models.torch.no_grad():
    pt_node_features, pt_edge_features, pt_neighbor_indices = heavy_parity_models.pt_model.encode(feature_dict)
    pt_sequence_embedding = heavy_parity_models.torch.zeros_like(pt_node_features)
    pt_encoder_context = heavy_parity_models.model_utils.cat_neighbors_nodes(
      pt_sequence_embedding,
      pt_edge_features,
      pt_neighbor_indices,
    )
    pt_decoder_context = heavy_parity_models.model_utils.cat_neighbors_nodes(
      pt_node_features,
      pt_encoder_context,
      pt_neighbor_indices,
    )
    pt_decoded_nodes = pt_node_features
    for layer in heavy_parity_models.pt_model.decoder_layers:
      pt_decoded_nodes = layer(pt_decoded_nodes, pt_decoder_context, feature_dict["mask"])
    pt_log_probs = heavy_parity_models.torch.log_softmax(
      heavy_parity_models.pt_model.W_out(pt_decoded_nodes),
      dim=-1,
    ).numpy()[0]

  _, jax_logits = heavy_parity_models.jax_model(
    parity_batch.x_jax_atom37,
    jnp.array(parity_batch.mask[0]),
    jnp.array(parity_batch.residue_index[0]),
    jnp.array(parity_batch.chain_index[0]),
    "unconditional",
  )
  jax_log_probs = np.asarray(jax.nn.log_softmax(jax_logits, axis=-1))

  corr = _pearson_correlation(pt_log_probs, jax_log_probs)
  assert corr >= 0.95


@pytest.mark.parity_heavy
def test_decoder_conditional_scoring_parity(
  heavy_parity_models: HeavyParityModels,
  parity_batch: ParityBatch,
) -> None:
  """decoder-conditional-scoring: conditional score logits remain highly correlated."""
  feature_dict = _build_torch_feature_dict(heavy_parity_models.torch, parity_batch)
  with heavy_parity_models.torch.no_grad():
    pt_score = heavy_parity_models.pt_model.score(feature_dict, use_sequence=True)
  pt_log_probs = pt_score["log_probs"].numpy()[0]

  _, jax_logits = heavy_parity_models.jax_model(
    parity_batch.x_jax_atom37,
    jnp.array(parity_batch.mask[0]),
    jnp.array(parity_batch.residue_index[0]),
    jnp.array(parity_batch.chain_index[0]),
    "conditional",
    one_hot_sequence=jax.nn.one_hot(jnp.array(parity_batch.sequence[0]), 21),
    ar_mask=jnp.array(parity_batch.ar_mask),
  )
  jax_log_probs = np.asarray(jax.nn.log_softmax(jax_logits, axis=-1))

  assert np.array_equal(pt_score["decoding_order"].numpy()[0], parity_batch.decoding_order)
  corr = _pearson_correlation(pt_log_probs, jax_log_probs)
  assert corr >= 0.95


@pytest.mark.parity_heavy
def test_autoregressive_sampling_parity(
  heavy_parity_models: HeavyParityModels,
  parity_batch: ParityBatch,
) -> None:
  """autoregressive-sampling: deterministic token agreement and log-prob correlation."""
  feature_dict = _build_torch_feature_dict(
    heavy_parity_models.torch,
    parity_batch,
    include_sampling_args=True,
  )
  with heavy_parity_models.torch.no_grad():
    pt_sample = heavy_parity_models.pt_model.sample(feature_dict)

  jax_sequence, jax_logits = heavy_parity_models.jax_model(
    parity_batch.x_jax_atom37,
    jnp.array(parity_batch.mask[0]),
    jnp.array(parity_batch.residue_index[0]),
    jnp.array(parity_batch.chain_index[0]),
    "autoregressive",
    prng_key=jax.random.PRNGKey(7),
    ar_mask=jnp.array(parity_batch.ar_mask),
    temperature=jnp.array(1.0),
    bias=jnp.array(parity_batch.bias[0]),
  )

  pt_tokens = pt_sample["S"].numpy()[0]
  jax_tokens = np.asarray(jnp.argmax(jax_sequence, axis=-1))
  pt_log_probs = pt_sample["log_probs"].numpy()[0]
  jax_log_probs = np.asarray(jax.nn.log_softmax(jax_logits, axis=-1))

  assert np.array_equal(pt_sample["decoding_order"].numpy()[0], parity_batch.decoding_order)
  token_agreement = float(np.mean(pt_tokens == jax_tokens))
  corr = _pearson_correlation(pt_log_probs, jax_log_probs)
  assert token_agreement >= 0.95
  assert corr >= 0.95


@pytest.mark.parity_heavy
def test_tied_positions_and_multi_state_parity(
  heavy_parity_models: HeavyParityModels,
  parity_batch: ParityBatch,
) -> None:
  """tied-positions-and-multi-state: weighted-sum reference aligns with JAX product strategy."""
  tie_groups = [[0, 1, 2], [6, 7]]
  tie_weights = [[1.0, 1.0, 1.0], [1.0, 1.0]]
  tie_group_map = _build_tie_group_map(parity_batch.mask.shape[1], tie_groups)

  feature_dict = _build_torch_feature_dict(
    heavy_parity_models.torch,
    parity_batch,
    include_sampling_args=True,
    symmetry_residues=tie_groups,
    symmetry_weights=tie_weights,
  )
  with heavy_parity_models.torch.no_grad():
    pt_sample = heavy_parity_models.pt_model.sample(feature_dict)

  jax_sequence, jax_logits = heavy_parity_models.jax_model(
    parity_batch.x_jax_atom37,
    jnp.array(parity_batch.mask[0]),
    jnp.array(parity_batch.residue_index[0]),
    jnp.array(parity_batch.chain_index[0]),
    "autoregressive",
    prng_key=jax.random.PRNGKey(7),
    ar_mask=jnp.array(parity_batch.ar_mask),
    temperature=jnp.array(1.0),
    bias=jnp.array(parity_batch.bias[0]),
    tie_group_map=jnp.array(tie_group_map),
    multi_state_strategy="product",
  )

  pt_tokens = pt_sample["S"].numpy()[0]
  jax_tokens = np.asarray(jnp.argmax(jax_sequence, axis=-1))
  pt_log_probs = _combine_reference_tied_log_probs(
    pt_sample["log_probs"].numpy()[0],
    tie_groups=tie_groups,
    tie_weights=tie_weights,
  )
  jax_log_probs = np.asarray(jax.nn.log_softmax(jax_logits, axis=-1))

  token_agreement = float(np.mean(pt_tokens == jax_tokens))
  corr = _pearson_correlation(pt_log_probs, jax_log_probs)
  assert token_agreement >= 0.95
  assert corr >= 0.95
  for group in tie_groups:
    assert np.all(pt_tokens[group] == pt_tokens[group[0]])
    assert np.all(jax_tokens[group] == jax_tokens[group[0]])
