"""Reference-backed ligand parity checks for PrxteinLigandMPNN."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import pearsonr

from prxteinmpnn.model.mpnn import PrxteinLigandMPNN
from prxteinmpnn.parity.matrix import ligand_tied_multistate_rollout_outcome
from tests.parity.reference_utils import require_heavy_parity_prereqs

pytestmark = pytest.mark.parity_heavy
_LIGAND_TIED_PATH_ID = "ligand-tied-positions-and-multi-state"
_LIGAND_TIED_SAMPLING_LANE = "ligand_context_reference_weighted_sum__jax_product"
_LIGAND_TIED_SCORING_LANE = "ligand_context_reference_arithmetic_mean__jax_arithmetic_mean"


@dataclass(frozen=True)
class LigandBatch:
  """Deterministic ligand test payload shared across parity checks."""

  x: np.ndarray
  s: np.ndarray
  mask: np.ndarray
  chain_mask: np.ndarray
  residue_index: np.ndarray
  chain_index: np.ndarray
  randn: np.ndarray
  y: np.ndarray
  y_t: np.ndarray
  y_m: np.ndarray


def _pearson_correlation(lhs: np.ndarray, rhs: np.ndarray) -> float:
  """Return flattened Pearson correlation for two arrays."""
  return float(pearsonr(lhs.ravel(), rhs.ravel())[0])


def _enforce_ligand_tied_rollout(
  condition: bool,
  *,
  lane_condition: str,
  requirement: str,
) -> None:
  """Apply staged rollout policy for ligand tied/multistate parity checks."""
  if condition:
    return
  tier = os.environ.get("PRXTEIN_PARITY_TIER", "parity_heavy")
  outcome = ligand_tied_multistate_rollout_outcome(condition_passed=False, tier=tier)
  message = (
    f"{_LIGAND_TIED_PATH_ID}/{lane_condition}: {requirement}. "
    f"tier={tier} outcome={outcome}"
  )
  if outcome == "warn":
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return
  pytest.fail(message)


def _build_ar_mask(randn: np.ndarray) -> np.ndarray:
  """Build deterministic autoregressive mask from random-order logits."""
  decoding_order = np.argsort(np.abs(randn))
  order_pos = {int(token): pos for pos, token in enumerate(decoding_order)}
  ar_mask = np.zeros((randn.shape[0], randn.shape[0]), dtype=np.int32)
  for i in range(randn.shape[0]):
    for j in range(randn.shape[0]):
      if order_pos[j] < order_pos[i]:
        ar_mask[i, j] = 1
  return ar_mask


def _build_tie_group_map(seq_len: int, tie_groups: list[list[int]]) -> np.ndarray:
  """Build a dense residue-to-group mapping for tied-position runs."""
  tie_group_map = np.arange(seq_len, dtype=np.int32)
  for group in tie_groups:
    representative = group[0]
    for position in group[1:]:
      tie_group_map[position] = representative
  _, compact_tie_group_map = np.unique(tie_group_map, return_inverse=True)
  return compact_tie_group_map.astype(np.int32, copy=False)


def _row_log_softmax(logits: np.ndarray) -> np.ndarray:
  """Compute row-wise log-softmax in NumPy."""
  shifted = logits - np.max(logits, axis=-1, keepdims=True)
  logsumexp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
  return shifted - logsumexp


def _combine_reference_tied_log_probs(
  log_probs: np.ndarray,
  *,
  tie_groups: list[list[int]],
  tie_weights: list[list[float]],
) -> np.ndarray:
  """Match reference weighted-sum tied combiner on log-probabilities."""
  combined = np.array(log_probs, copy=True)
  for group, weights in zip(tie_groups, tie_weights, strict=True):
    group_probs = np.exp(log_probs[group, :])
    weight_arr = np.asarray(weights, dtype=np.float64)
    weight_arr = weight_arr / max(float(np.sum(weight_arr)), 1e-8)
    combined_probs = np.sum(group_probs * weight_arr[:, None], axis=0)
    combined[group, :] = np.log(np.clip(combined_probs, a_min=1e-8, a_max=None))
  return combined


def _combine_reference_tied_logits(
  logits: np.ndarray,
  *,
  tie_groups: list[list[int]],
  tie_weights: list[list[float]],
  combiner: str,
) -> np.ndarray:
  """Combine reference tied logits to match configured lane semantics."""
  combined = np.array(logits, copy=True)
  for group, weights in zip(tie_groups, tie_weights, strict=True):
    group_logits = logits[group, :]
    if combiner == "weighted_sum":
      weight_arr = np.asarray(weights, dtype=np.float64)
      weight_arr = weight_arr / max(float(np.sum(weight_arr)), 1e-8)
      combined_logits = np.sum(group_logits * weight_arr[:, None], axis=0)
    elif combiner == "arithmetic_mean":
      combined_logits = np.mean(group_logits, axis=0)
    else:
      msg = f"Unsupported combiner {combiner!r}."
      raise ValueError(msg)
    combined[group, :] = combined_logits
  return combined


def _to_torch_feature_dict(
  batch: LigandBatch,
  torch_module: Any,
  *,
  symmetry_residues: list[list[int]] | None = None,
  symmetry_weights: list[list[float]] | None = None,
) -> dict[str, Any]:
  """Convert deterministic payload to reference-model feature dictionary."""
  resolved_symmetry_residues = symmetry_residues if symmetry_residues is not None else [[]]
  resolved_symmetry_weights = symmetry_weights if symmetry_weights is not None else [[]]
  return {
    "X": torch_module.from_numpy(batch.x),
    "S": torch_module.from_numpy(batch.s),
    "mask": torch_module.from_numpy(batch.mask),
    "chain_mask": torch_module.from_numpy(batch.chain_mask),
    "R_idx": torch_module.from_numpy(batch.residue_index),
    "chain_labels": torch_module.from_numpy(batch.chain_index),
    "randn": torch_module.from_numpy(batch.randn),
    "batch_size": int(batch.x.shape[0]),
    "symmetry_residues": resolved_symmetry_residues,
    "symmetry_weights": resolved_symmetry_weights,
    "Y": torch_module.from_numpy(batch.y),
    "Y_t": torch_module.from_numpy(batch.y_t),
    "Y_m": torch_module.from_numpy(batch.y_m),
  }


def _jax_ligand_context_state(
  model: PrxteinLigandMPNN,
  batch: LigandBatch,
  *,
  prng_key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Mirror ligand context path to expose hidden-state tensors for parity checks."""
  keys = jax.random.split(prng_key, 2)
  x = jnp.array(batch.x[0])
  mask = jnp.array(batch.mask[0])
  residue_index = jnp.array(batch.residue_index[0])
  chain_index = jnp.array(batch.chain_index[0])
  y = jnp.array(batch.y[0])
  y_t = jnp.array(batch.y_t[0])
  y_m = jnp.array(batch.y_m[0])

  V, E, E_idx, Y_nodes, Y_edges, Y_m = model.features(
    keys[0],
    x,
    mask,
    residue_index,
    chain_index,
    y,
    y_t,
    y_m,
  )

  h_V = jnp.zeros((E.shape[0], model.node_features_dim))
  h_E = E

  mask_2d = mask[:, None] * mask[None, :]
  mask_attend = jnp.take_along_axis(mask_2d, E_idx.astype(jnp.int32), axis=1)
  for layer in model.encoder.layers:
    h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend, inference=True)

  h_V_C = jax.vmap(model.w_c)(h_V)
  h_E_context = jax.vmap(jax.vmap(model.w_v))(V)

  Y_nodes = jax.vmap(jax.vmap(model.w_nodes_y))(Y_nodes)
  Y_edges = jax.vmap(jax.vmap(jax.vmap(model.w_edges_y)))(Y_edges)
  y_m_edges = Y_m[..., None] * Y_m[..., None, :]

  for i in range(len(model.context_encoder)):
    Y_nodes = jax.vmap(
      lambda node, edge, mask_l, mask_e: model.y_context_encoder[i](
        node,
        edge,
        mask_l,
        attention_mask=mask_e,
        inference=True,
      ),
    )(Y_nodes, Y_edges, Y_m, y_m_edges)
    h_E_context_cat = jnp.concatenate([h_E_context, Y_nodes], axis=-1)
    h_V_C = model.context_encoder[i](
      h_V_C,
      h_E_context_cat,
      mask,
      attention_mask=Y_m,
      inference=True,
    )

  h_V_C = jax.vmap(model.v_c)(h_V_C)
  h_V = h_V + jax.vmap(model.v_c_norm)(model.dropout(h_V_C, key=keys[1], inference=True))
  return h_V, h_E, E_idx


@pytest.fixture(scope="module")
def ligand_models() -> tuple[Any, PrxteinLigandMPNN]:
  """Load reference + JAX ligand models from the same checkpoint for parity checks."""
  pytest.importorskip("torch")
  reference_root, _ = require_heavy_parity_prereqs(
    reference_rel_paths=["model_params/ligandmpnn_v_32_020_25.pt"],
  )
  import model_utils
  import torch

  from scripts.convert_weights import convert_full_model

  checkpoint = torch.load(
    reference_root / "model_params/ligandmpnn_v_32_020_25.pt",
    map_location="cpu",
  )
  state_dict = checkpoint["model_state_dict"]
  state_dict_np = {name: value.detach().cpu().numpy() for name, value in state_dict.items()}

  pos_weight = state_dict.get("features.embeddings.linear.weight")
  num_positional_embeddings = int((pos_weight.shape[1] - 2) // 2) if pos_weight is not None else 16

  pt_model = model_utils.ProteinMPNN(
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=32,
    atom_context_num=16,
    model_type="ligand_mpnn",
    ligand_mpnn_use_side_chain_context=False,
    dropout=0.0,
  )
  pt_model.load_state_dict(state_dict)
  pt_model.eval()

  jax_model = PrxteinLigandMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=32,
    num_context_layers=2,
    num_positional_embeddings=num_positional_embeddings,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(0),
  )
  jax_model = convert_full_model(state_dict_np, jax_model)
  return pt_model, jax_model


@pytest.fixture(scope="module")
def ligand_batch() -> LigandBatch:
  """Create deterministic ligand test payload."""
  rng = np.random.default_rng(1)
  seq_len = 12
  atom_context_num = 16

  return LigandBatch(
    x=rng.normal(size=(1, seq_len, 4, 3)).astype(np.float32),
    s=rng.integers(0, 20, size=(1, seq_len), dtype=np.int64),
    mask=np.ones((1, seq_len), dtype=np.float32),
    chain_mask=np.ones((1, seq_len), dtype=np.float32),
    residue_index=np.tile(np.arange(seq_len), (1, 1)).astype(np.int64),
    chain_index=np.zeros((1, seq_len), dtype=np.int64),
    randn=rng.normal(size=(1, seq_len)).astype(np.float32),
    y=rng.normal(size=(1, seq_len, atom_context_num, 3)).astype(np.float32),
    y_t=rng.integers(1, 30, size=(1, seq_len, atom_context_num), dtype=np.int64),
    y_m=(rng.random(size=(1, seq_len, atom_context_num)) > 0.2).astype(np.float32),
  )


def test_ligand_feature_extraction_reference_parity(
  ligand_models: tuple[Any, PrxteinLigandMPNN],
  ligand_batch: LigandBatch,
) -> None:
  """Check ligand feature tensors against reference extraction."""
  import torch

  pt_model, jax_model = ligand_models
  feature_dict = _to_torch_feature_dict(ligand_batch, torch)

  with torch.no_grad():
    v_pt, e_pt, e_idx_pt, y_nodes_pt, y_edges_pt, y_m_pt = pt_model.features(feature_dict)
    e_proj_pt = pt_model.W_e(e_pt)

  v_jax, e_jax, e_idx_jax, y_nodes_jax, y_edges_jax, y_m_jax = jax_model.features(
    jax.random.PRNGKey(17),
    jnp.array(ligand_batch.x[0]),
    jnp.array(ligand_batch.mask[0]),
    jnp.array(ligand_batch.residue_index[0]),
    jnp.array(ligand_batch.chain_index[0]),
    jnp.array(ligand_batch.y[0]),
    jnp.array(ligand_batch.y_t[0]),
    jnp.array(ligand_batch.y_m[0]),
  )

  assert np.array_equal(e_idx_pt.numpy()[0], np.asarray(e_idx_jax))
  assert _pearson_correlation(v_pt.numpy()[0], np.asarray(v_jax)) > 0.85
  assert _pearson_correlation(e_proj_pt.numpy()[0], np.asarray(e_jax)) > 0.995
  np.testing.assert_allclose(y_nodes_pt.numpy()[0], np.asarray(y_nodes_jax), rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(y_edges_pt.numpy()[0], np.asarray(y_edges_jax), rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(y_m_pt.numpy()[0], np.asarray(y_m_jax), rtol=1e-6, atol=1e-6)


def test_ligand_conditioning_context_reference_correlation(
  ligand_models: tuple[Any, PrxteinLigandMPNN],
  ligand_batch: LigandBatch,
) -> None:
  """Validate ligand context integration and conditional logits correlation."""
  import torch

  pt_model, jax_model = ligand_models
  feature_dict = _to_torch_feature_dict(ligand_batch, torch)
  ar_mask = _build_ar_mask(ligand_batch.randn[0])

  with torch.no_grad():
    h_v_pt, h_e_pt, e_idx_pt = pt_model.encode(feature_dict)
    log_probs_pt = pt_model.score(feature_dict, use_sequence=True)["log_probs"].numpy()[0]

  h_v_jax, h_e_jax, e_idx_jax = _jax_ligand_context_state(
    jax_model,
    ligand_batch,
    prng_key=jax.random.PRNGKey(29),
  )

  assert np.array_equal(e_idx_pt.numpy()[0], np.asarray(e_idx_jax))
  assert _pearson_correlation(h_v_pt.numpy()[0], np.asarray(h_v_jax)) > 0.85
  assert _pearson_correlation(h_e_pt.numpy()[0], np.asarray(h_e_jax)) > 0.99

  _, logits_jax = jax_model(
    jnp.array(ligand_batch.x[0]),
    jnp.array(ligand_batch.mask[0]),
    jnp.array(ligand_batch.residue_index[0]),
    jnp.array(ligand_batch.chain_index[0]),
    jnp.array(ligand_batch.y[0]),
    jnp.array(ligand_batch.y_t[0]),
    jnp.array(ligand_batch.y_m[0]),
    "conditional",
    prng_key=jax.random.PRNGKey(31),
    ar_mask=jnp.array(ar_mask),
    one_hot_sequence=jax.nn.one_hot(jnp.array(ligand_batch.s[0]), 21),
    inference=True,
  )
  log_probs_jax = np.asarray(jax.nn.log_softmax(logits_jax, axis=-1))

  assert _pearson_correlation(log_probs_pt, log_probs_jax) > 0.9


def test_ligand_autoregressive_reference_alignment(
  ligand_models: tuple[Any, PrxteinLigandMPNN],
  ligand_batch: LigandBatch,
) -> None:
  """Check deterministic autoregressive ligand sampling parity under forced-token bias."""
  import torch

  pt_model, jax_model = ligand_models
  feature_dict = _to_torch_feature_dict(ligand_batch, torch)
  ar_mask = _build_ar_mask(ligand_batch.randn[0])

  forcing_rng = np.random.default_rng(23)
  forced_tokens = forcing_rng.integers(0, 20, size=(ligand_batch.x.shape[1],), dtype=np.int64)
  bias = np.zeros((1, ligand_batch.x.shape[1], 21), dtype=np.float32)
  bias[0, np.arange(ligand_batch.x.shape[1]), forced_tokens] = 50.0

  feature_dict["bias"] = torch.from_numpy(bias)
  feature_dict["temperature"] = 1.0
  feature_dict["chain_mask"] = torch.ones_like(feature_dict["chain_mask"])
  torch.manual_seed(0)

  with torch.no_grad():
    sampled_pt = pt_model.sample(feature_dict)

  sampled_seq_pt = sampled_pt["S"].numpy()[0]
  sampled_log_probs_pt = sampled_pt["log_probs"].numpy()[0]

  sampled_seq_jax, logits_jax = jax_model(
    jnp.array(ligand_batch.x[0]),
    jnp.array(ligand_batch.mask[0]),
    jnp.array(ligand_batch.residue_index[0]),
    jnp.array(ligand_batch.chain_index[0]),
    jnp.array(ligand_batch.y[0]),
    jnp.array(ligand_batch.y_t[0]),
    jnp.array(ligand_batch.y_m[0]),
    "autoregressive",
    prng_key=jax.random.PRNGKey(37),
    ar_mask=jnp.array(ar_mask),
    temperature=1.0,
    bias=jnp.array(bias[0]),
    inference=True,
  )
  sampled_tokens_jax = np.asarray(sampled_seq_jax).argmax(axis=-1)
  sampled_log_probs_jax = np.asarray(jax.nn.log_softmax(logits_jax, axis=-1))

  assert np.array_equal(sampled_tokens_jax, forced_tokens)
  assert np.array_equal(sampled_seq_pt, sampled_tokens_jax)
  assert _pearson_correlation(sampled_log_probs_pt, sampled_log_probs_jax) > 0.9


@pytest.mark.parity_audit
def test_ligand_tied_sampling_weighted_sum_product_alignment(
  ligand_models: tuple[Any, PrxteinLigandMPNN],
  ligand_batch: LigandBatch,
) -> None:
  """Validate ligand tied sampling lane parity for weighted-sum vs product mapping."""
  import torch

  pt_model, jax_model = ligand_models
  tie_groups = [[0, 1, 2], [6, 7]]
  tie_weights = [[1.0, 1.0, 1.0], [1.0, 1.0]]
  tie_group_map = _build_tie_group_map(ligand_batch.x.shape[1], tie_groups)
  feature_dict = _to_torch_feature_dict(
    ligand_batch,
    torch,
    symmetry_residues=tie_groups,
    symmetry_weights=tie_weights,
  )
  ar_mask = _build_ar_mask(ligand_batch.randn[0])

  forced_tokens = np.arange(ligand_batch.x.shape[1], dtype=np.int64) % 20
  forced_tokens[tie_groups[0]] = 3
  forced_tokens[tie_groups[1]] = 11
  bias = np.zeros((1, ligand_batch.x.shape[1], 21), dtype=np.float32)
  bias[0, np.arange(ligand_batch.x.shape[1]), forced_tokens] = 50.0

  feature_dict["bias"] = torch.from_numpy(bias)
  feature_dict["temperature"] = 1.0
  feature_dict["chain_mask"] = torch.ones_like(feature_dict["chain_mask"])
  torch.manual_seed(0)
  with torch.no_grad():
    sampled_pt = pt_model.sample(feature_dict)

  sampled_seq_jax, logits_jax = jax_model(
    jnp.array(ligand_batch.x[0]),
    jnp.array(ligand_batch.mask[0]),
    jnp.array(ligand_batch.residue_index[0]),
    jnp.array(ligand_batch.chain_index[0]),
    jnp.array(ligand_batch.y[0]),
    jnp.array(ligand_batch.y_t[0]),
    jnp.array(ligand_batch.y_m[0]),
    "autoregressive",
    prng_key=jax.random.PRNGKey(43),
    ar_mask=jnp.array(ar_mask),
    temperature=1.0,
    bias=jnp.array(bias[0]),
    tie_group_map=jnp.array(tie_group_map),
    multi_state_strategy="product",
    inference=True,
  )

  sampled_tokens_pt = sampled_pt["S"].numpy()[0]
  sampled_tokens_jax = np.asarray(sampled_seq_jax).argmax(axis=-1)
  sampled_log_probs_pt = _combine_reference_tied_log_probs(
    sampled_pt["log_probs"].numpy()[0],
    tie_groups=tie_groups,
    tie_weights=tie_weights,
  )
  sampled_log_probs_jax = np.asarray(jax.nn.log_softmax(logits_jax, axis=-1))

  tokens_match = np.array_equal(sampled_tokens_pt, sampled_tokens_jax)
  _enforce_ligand_tied_rollout(
    tokens_match,
    lane_condition=_LIGAND_TIED_SAMPLING_LANE,
    requirement="sampled tokens diverged between reference and JAX",
  )

  correlation = _pearson_correlation(sampled_log_probs_pt, sampled_log_probs_jax)
  _enforce_ligand_tied_rollout(
    correlation > 0.6,
    lane_condition=_LIGAND_TIED_SAMPLING_LANE,
    requirement=f"pearson correlation {correlation:.6f} was below 0.6",
  )

  for group in tie_groups:
    reference_group_consistent = bool(np.all(sampled_tokens_pt[group] == sampled_tokens_pt[group[0]]))
    jax_group_consistent = bool(np.all(sampled_tokens_jax[group] == sampled_tokens_jax[group[0]]))
    _enforce_ligand_tied_rollout(
      reference_group_consistent,
      lane_condition=_LIGAND_TIED_SAMPLING_LANE,
      requirement=f"reference tie group {group} broke tied-position invariants",
    )
    _enforce_ligand_tied_rollout(
      jax_group_consistent,
      lane_condition=_LIGAND_TIED_SAMPLING_LANE,
      requirement=f"JAX tie group {group} broke tied-position invariants",
    )


@pytest.mark.parity_audit
def test_ligand_tied_scoring_arithmetic_mean_alignment(
  ligand_models: tuple[Any, PrxteinLigandMPNN],
  ligand_batch: LigandBatch,
) -> None:
  """Validate ligand tied scoring lane parity for arithmetic-mean mapping."""
  import torch

  pt_model, jax_model = ligand_models
  tie_groups = [[0, 1, 2], [6, 7]]
  tie_weights = [[1.0, 1.0, 1.0], [1.0, 1.0]]
  tie_group_map = _build_tie_group_map(ligand_batch.x.shape[1], tie_groups)
  feature_dict = _to_torch_feature_dict(
    ligand_batch,
    torch,
    symmetry_residues=tie_groups,
    symmetry_weights=tie_weights,
  )
  ar_mask = _build_ar_mask(ligand_batch.randn[0])

  with torch.no_grad():
    scored_pt = pt_model.score(feature_dict, use_sequence=True)

  combined_logits_pt = _combine_reference_tied_logits(
    scored_pt["logits"].numpy()[0],
    tie_groups=tie_groups,
    tie_weights=tie_weights,
    combiner="arithmetic_mean",
  )
  scored_log_probs_pt = _row_log_softmax(combined_logits_pt)
  scored_tokens_pt = np.asarray(np.argmax(scored_log_probs_pt, axis=-1), dtype=np.int32)

  _, logits_jax = jax_model(
    jnp.array(ligand_batch.x[0]),
    jnp.array(ligand_batch.mask[0]),
    jnp.array(ligand_batch.residue_index[0]),
    jnp.array(ligand_batch.chain_index[0]),
    jnp.array(ligand_batch.y[0]),
    jnp.array(ligand_batch.y_t[0]),
    jnp.array(ligand_batch.y_m[0]),
    "conditional",
    prng_key=jax.random.PRNGKey(47),
    ar_mask=jnp.array(ar_mask),
    one_hot_sequence=jax.nn.one_hot(jnp.array(ligand_batch.s[0]), 21),
    tie_group_map=jnp.array(tie_group_map),
    multi_state_strategy="arithmetic_mean",
    inference=True,
  )
  scored_log_probs_jax = np.asarray(jax.nn.log_softmax(logits_jax, axis=-1))
  scored_tokens_jax = np.asarray(np.argmax(scored_log_probs_jax, axis=-1), dtype=np.int32)

  scoring_correlation = _pearson_correlation(scored_log_probs_pt, scored_log_probs_jax)
  _enforce_ligand_tied_rollout(
    scoring_correlation > 0.9,
    lane_condition=_LIGAND_TIED_SCORING_LANE,
    requirement=f"pearson correlation {scoring_correlation:.6f} was below 0.9",
  )

  for group in tie_groups:
    reference_group_consistent = bool(np.all(scored_tokens_pt[group] == scored_tokens_pt[group[0]]))
    jax_group_consistent = bool(np.all(scored_tokens_jax[group] == scored_tokens_jax[group[0]]))
    _enforce_ligand_tied_rollout(
      reference_group_consistent,
      lane_condition=_LIGAND_TIED_SCORING_LANE,
      requirement=f"reference tie group {group} broke tied-position invariants",
    )
    _enforce_ligand_tied_rollout(
      jax_group_consistent,
      lane_condition=_LIGAND_TIED_SCORING_LANE,
      requirement=f"JAX tie group {group} broke tied-position invariants",
    )
