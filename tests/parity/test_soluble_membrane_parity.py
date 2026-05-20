"""Reference-backed parity checks for SolubleeMPNN and MembraneMPNN variants."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import pearsonr

from prxteinmpnn.inference.bundle_builder import build_inference_bundle
from prxteinmpnn.inference import score_unconditional
from prxteinmpnn.io.weights import load_weights
from prxteinmpnn.model.mpnn import PrxteinMPNN
from tests.parity.reference_utils import require_heavy_parity_prereqs


JaxHeavyWeightSource = Literal["eqx", "pt_convert"]


def _jax_model_for_source(
    models: SolubleMembraneParityModels, source: JaxHeavyWeightSource, checkpoint_kind: str
) -> PrxteinMPNN:
  """Select JAX model for the given source and checkpoint kind."""
  if checkpoint_kind == "soluble":
    return models.jax_soluble_eqx if source == "eqx" else models.jax_soluble_pt_convert
  elif checkpoint_kind == "membrane_per_residue":
    return models.jax_membrane_per_residue_eqx if source == "eqx" else models.jax_membrane_per_residue_pt_convert
  elif checkpoint_kind == "membrane_global":
    return models.jax_membrane_global_eqx if source == "eqx" else models.jax_membrane_global_pt_convert
  else:
    raise ValueError(f"Unknown checkpoint kind: {checkpoint_kind}")


@dataclass(frozen=True)
class SolubleMembraneParityModels:
  """Container for loaded heavy parity models for soluble and membrane variants."""

  torch: Any
  model_utils: Any

  # Soluble MPNN
  pt_soluble: Any
  jax_soluble_eqx: PrxteinMPNN
  jax_soluble_pt_convert: PrxteinMPNN

  # Membrane per-residue label
  pt_membrane_per_residue: Any
  jax_membrane_per_residue_eqx: PrxteinMPNN
  jax_membrane_per_residue_pt_convert: PrxteinMPNN

  # Membrane global label
  pt_membrane_global: Any
  jax_membrane_global_eqx: PrxteinMPNN
  jax_membrane_global_pt_convert: PrxteinMPNN


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
  physics_features: np.ndarray | None


def _pearson_correlation(lhs: np.ndarray, rhs: np.ndarray) -> float:
  """Compute Pearson correlation on flattened arrays."""
  return float(pearsonr(lhs.ravel(), rhs.ravel())[0])


def _build_parity_batch(*, seq_len: int = 12, seed: int = 4) -> ParityBatch:
  """Create deterministic synthetic inputs for unconditional scoring."""
  rng = np.random.default_rng(seed)
  batch_size = 1

  x_pytorch = rng.normal(size=(batch_size, seq_len, 4, 3)).astype(np.float32)
  sequence = rng.integers(0, 21, size=(batch_size, seq_len), dtype=np.int64)
  mask = np.ones((batch_size, seq_len), dtype=np.float32)
  chain_mask = np.ones((batch_size, seq_len), dtype=np.float32)
  residue_index = np.tile(np.arange(seq_len), (batch_size, 1)).astype(np.int64)
  chain_index = np.zeros((batch_size, seq_len), dtype=np.int64)
  randn = rng.normal(size=(batch_size, seq_len)).astype(np.float32)

  # Membrane models require physics features (3D)
  physics_features = rng.normal(size=(batch_size, seq_len, 3)).astype(np.float32)

  x_jax_atom37 = jnp.zeros((seq_len, 37, 3), dtype=jnp.float32)
  x_jax_atom37 = x_jax_atom37.at[:, 0, :].set(x_pytorch[0, :, 0, :])
  x_jax_atom37 = x_jax_atom37.at[:, 1, :].set(x_pytorch[0, :, 1, :])
  x_jax_atom37 = x_jax_atom37.at[:, 2, :].set(x_pytorch[0, :, 2, :])
  x_jax_atom37 = x_jax_atom37.at[:, 4, :].set(x_pytorch[0, :, 3, :])

  return ParityBatch(
    x_pytorch=x_pytorch,
    x_jax_atom37=x_jax_atom37,
    sequence=sequence,
    mask=mask,
    chain_mask=chain_mask,
    residue_index=residue_index,
    chain_index=chain_index,
    randn=randn,
    physics_features=physics_features,
  )


def _build_torch_feature_dict(
  torch: Any,
  batch: ParityBatch,
  *,
  physics_features: np.ndarray | None = None,
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
    "symmetry_residues": [[]],
    "symmetry_weights": [[]],
  }
  if physics_features is not None:
    feature_dict["phi"] = torch.from_numpy(physics_features)
  return feature_dict


def _load_soluble_membrane_models_impl() -> SolubleMembraneParityModels:
  """Load reference torch + JAX parity models for soluble and membrane variants."""
  pytest.importorskip("torch")
  reference_root, repo_root = require_heavy_parity_prereqs(
    reference_rel_paths=[
      "model_params/solublempnn_v_48_020.pt",
      "model_params/per_residue_label_membrane_mpnn_v_48_020.pt",
      "model_params/global_label_membrane_mpnn_v_48_020.pt",
    ],
    converted_rel_paths=[
      "src/prxteinmpnn/model_params/solublempnn_v_48_020.eqx.zst",
      "src/prxteinmpnn/model_params/per_residue_label_membrane_mpnn_v_48_020.eqx.zst",
      "src/prxteinmpnn/model_params/global_label_membrane_mpnn_v_48_020.eqx.zst",
    ],
  )
  import model_utils
  import torch

  jax_key = jax.random.PRNGKey(0)

  # Load Soluble MPNN
  pt_soluble_checkpoint_path = reference_root / "model_params/solublempnn_v_48_020.pt"
  jax_soluble_checkpoint_path = repo_root / "src/prxteinmpnn/model_params/solublempnn_v_48_020.eqx.zst"
  soluble_checkpoint = torch.load(pt_soluble_checkpoint_path, map_location="cpu")

  pos_weight = soluble_checkpoint["model_state_dict"].get("features.embeddings.linear.weight")
  if pos_weight is None:
    pos_weight = soluble_checkpoint["model_state_dict"].get("features.positional_embeddings.linear.weight")
  num_positional_embeddings = int((pos_weight.shape[1] - 2) // 2) if pos_weight is not None else 16

  pt_soluble = model_utils.ProteinMPNN(
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
  )
  pt_soluble.load_state_dict(soluble_checkpoint["model_state_dict"])
  pt_soluble.eval()

  jax_soluble_skeleton = PrxteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    num_positional_embeddings=num_positional_embeddings,
    dropout_rate=0.0,
    key=jax_key,
  )
  jax_soluble_eqx = load_weights(local_path=str(jax_soluble_checkpoint_path), skeleton=jax_soluble_skeleton)

  soluble_state_dict_numpy: dict[str, np.ndarray] = {}
  for k, v in soluble_checkpoint["model_state_dict"].items():
    if hasattr(v, "detach"):
      v = v.detach()
    arr = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    soluble_state_dict_numpy[str(k)] = arr

  repo_rt_str = str(repo_root.resolve())
  if repo_rt_str not in sys.path:
    sys.path.insert(0, repo_rt_str)

  from scripts.convert_weights import convert_full_model

  jax_soluble_pt_skeleton = PrxteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    num_positional_embeddings=num_positional_embeddings,
    dropout_rate=0.0,
    key=jax_key,
  )
  jax_soluble_pt_convert = convert_full_model(soluble_state_dict_numpy, jax_soluble_pt_skeleton)

  # Load Membrane Per-Residue MPNN
  pt_membrane_per_residue_checkpoint_path = reference_root / "model_params/per_residue_label_membrane_mpnn_v_48_020.pt"
  jax_membrane_per_residue_checkpoint_path = repo_root / "src/prxteinmpnn/model_params/per_residue_label_membrane_mpnn_v_48_020.eqx.zst"
  membrane_per_residue_checkpoint = torch.load(pt_membrane_per_residue_checkpoint_path, map_location="cpu")

  pos_weight_membrane = membrane_per_residue_checkpoint["model_state_dict"].get("features.embeddings.linear.weight")
  if pos_weight_membrane is None:
    pos_weight_membrane = membrane_per_residue_checkpoint["model_state_dict"].get(
      "features.positional_embeddings.linear.weight"
    )
  num_positional_embeddings_membrane = int((pos_weight_membrane.shape[1] - 2) // 2) if pos_weight_membrane is not None else 16

  pt_membrane_per_residue = model_utils.ProteinMPNN(
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    model_type="per_residue_label",
  )
  pt_membrane_per_residue.load_state_dict(membrane_per_residue_checkpoint["model_state_dict"])
  pt_membrane_per_residue.eval()

  jax_membrane_per_residue_skeleton = PrxteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    num_positional_embeddings=num_positional_embeddings_membrane,
    physics_feature_dim=3,
    dropout_rate=0.0,
    key=jax_key,
  )
  jax_membrane_per_residue_eqx = load_weights(
    local_path=str(jax_membrane_per_residue_checkpoint_path), skeleton=jax_membrane_per_residue_skeleton
  )

  membrane_per_residue_state_dict_numpy: dict[str, np.ndarray] = {}
  for k, v in membrane_per_residue_checkpoint["model_state_dict"].items():
    if hasattr(v, "detach"):
      v = v.detach()
    arr = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    membrane_per_residue_state_dict_numpy[str(k)] = arr

  jax_membrane_per_residue_pt_skeleton = PrxteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    num_positional_embeddings=num_positional_embeddings_membrane,
    physics_feature_dim=3,
    dropout_rate=0.0,
    key=jax_key,
  )
  jax_membrane_per_residue_pt_convert = convert_full_model(
    membrane_per_residue_state_dict_numpy, jax_membrane_per_residue_pt_skeleton
  )

  # Load Membrane Global MPNN
  pt_membrane_global_checkpoint_path = reference_root / "model_params/global_label_membrane_mpnn_v_48_020.pt"
  jax_membrane_global_checkpoint_path = repo_root / "src/prxteinmpnn/model_params/global_label_membrane_mpnn_v_48_020.eqx.zst"
  membrane_global_checkpoint = torch.load(pt_membrane_global_checkpoint_path, map_location="cpu")

  pos_weight_global = membrane_global_checkpoint["model_state_dict"].get("features.embeddings.linear.weight")
  if pos_weight_global is None:
    pos_weight_global = membrane_global_checkpoint["model_state_dict"].get(
      "features.positional_embeddings.linear.weight"
    )
  num_positional_embeddings_global = int((pos_weight_global.shape[1] - 2) // 2) if pos_weight_global is not None else 16

  pt_membrane_global = model_utils.ProteinMPNN(
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    model_type="global_label",
  )
  pt_membrane_global.load_state_dict(membrane_global_checkpoint["model_state_dict"])
  pt_membrane_global.eval()

  jax_membrane_global_skeleton = PrxteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    num_positional_embeddings=num_positional_embeddings_global,
    physics_feature_dim=3,
    dropout_rate=0.0,
    key=jax_key,
  )
  jax_membrane_global_eqx = load_weights(
    local_path=str(jax_membrane_global_checkpoint_path), skeleton=jax_membrane_global_skeleton
  )

  membrane_global_state_dict_numpy: dict[str, np.ndarray] = {}
  for k, v in membrane_global_checkpoint["model_state_dict"].items():
    if hasattr(v, "detach"):
      v = v.detach()
    arr = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    membrane_global_state_dict_numpy[str(k)] = arr

  jax_membrane_global_pt_skeleton = PrxteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    num_positional_embeddings=num_positional_embeddings_global,
    physics_feature_dim=3,
    dropout_rate=0.0,
    key=jax_key,
  )
  jax_membrane_global_pt_convert = convert_full_model(membrane_global_state_dict_numpy, jax_membrane_global_pt_skeleton)

  return SolubleMembraneParityModels(
    torch=torch,
    model_utils=model_utils,
    pt_soluble=pt_soluble,
    jax_soluble_eqx=jax_soluble_eqx,
    jax_soluble_pt_convert=jax_soluble_pt_convert,
    pt_membrane_per_residue=pt_membrane_per_residue,
    jax_membrane_per_residue_eqx=jax_membrane_per_residue_eqx,
    jax_membrane_per_residue_pt_convert=jax_membrane_per_residue_pt_convert,
    pt_membrane_global=pt_membrane_global,
    jax_membrane_global_eqx=jax_membrane_global_eqx,
    jax_membrane_global_pt_convert=jax_membrane_global_pt_convert,
  )


@pytest.fixture(scope="module")
def soluble_membrane_parity_models() -> SolubleMembraneParityModels:
  """Load reference torch and converted JAX models for soluble/membrane parity checks."""
  return _load_soluble_membrane_models_impl()


@pytest.fixture(scope="module")
def parity_batch() -> ParityBatch:
  """Return deterministic synthetic parity inputs."""
  return _build_parity_batch()


@pytest.mark.parity_heavy
@pytest.mark.parametrize("weight_source", ("eqx", "pt_convert"))
@pytest.mark.parametrize(
  "checkpoint_kind",
  ("soluble", "membrane_per_residue", "membrane_global"),
)
def test_unconditional_parity(
  soluble_membrane_parity_models: SolubleMembraneParityModels,
  parity_batch: ParityBatch,
  weight_source: JaxHeavyWeightSource,
  checkpoint_kind: str,
) -> None:
  """unconditional-scoring: manual reference branch and JAX branch stay correlated across variant families."""
  jax.config.update("jax_default_matmul_precision", "highest")

  # Select checkpoint and model paths
  if checkpoint_kind == "soluble":
    pt_model = soluble_membrane_parity_models.pt_soluble
    physics_features = None
  elif checkpoint_kind == "membrane_per_residue":
    pt_model = soluble_membrane_parity_models.pt_membrane_per_residue
    physics_features = parity_batch.physics_features
  elif checkpoint_kind == "membrane_global":
    pt_model = soluble_membrane_parity_models.pt_membrane_global
    physics_features = parity_batch.physics_features
  else:
    raise ValueError(f"Unknown checkpoint kind: {checkpoint_kind}")

  feature_dict = _build_torch_feature_dict(
    soluble_membrane_parity_models.torch,
    parity_batch,
    physics_features=physics_features,
  )
  with soluble_membrane_parity_models.torch.no_grad():
    pt_sequence_embedding = soluble_membrane_parity_models.torch.zeros_like(
      soluble_membrane_parity_models.torch.zeros(pt_model.node_features, dtype=soluble_membrane_parity_models.torch.float32)
    )
    pt_node_features, pt_edge_features, pt_neighbor_indices = pt_model.encode(feature_dict)
    pt_encoder_context = soluble_membrane_parity_models.model_utils.cat_neighbors_nodes(
      pt_sequence_embedding[None, None, :].expand(1, parity_batch.sequence.shape[1], -1),
      pt_edge_features,
      pt_neighbor_indices,
    )
    pt_decoder_context = soluble_membrane_parity_models.model_utils.cat_neighbors_nodes(
      pt_node_features,
      pt_encoder_context,
      pt_neighbor_indices,
    )
    pt_decoded_nodes = pt_node_features
    for layer in pt_model.decoder_layers:
      pt_decoded_nodes = layer(pt_decoded_nodes, pt_decoder_context, feature_dict["mask"])
    pt_log_probs = soluble_membrane_parity_models.torch.log_softmax(
      pt_model.W_out(pt_decoded_nodes),
      dim=-1,
    ).numpy()[0]

  jax_model = _jax_model_for_source(soluble_membrane_parity_models, weight_source, checkpoint_kind)

  # Extract first 4 atoms to match (L, 4, 3) format
  coords_batch = parity_batch.x_jax_atom37[:, :4, :][None, ...]  # (1, L, 4, 3)
  mask_batch = jnp.array(parity_batch.mask[0])[None, ...]  # (1, L)
  residue_index_batch = jnp.array(parity_batch.residue_index[0])[None, ...]  # (1, L)
  chain_index_batch = jnp.array(parity_batch.chain_index[0])[None, ...]  # (1, L)

  bundle, config, stage_set = build_inference_bundle(
    coords=coords_batch,
    mask=mask_batch,
    residue_index=residue_index_batch,
    chain_index=chain_index_batch,
    mode="score_unconditional",
  )
  jax_logits = score_unconditional.kernel(jax_model, jax.random.PRNGKey(0), bundle, config, stage_set)
  jax_log_probs = np.asarray(jax.nn.log_softmax(jax_logits, axis=-1))

  corr = _pearson_correlation(pt_log_probs, jax_log_probs)
  assert corr >= 0.90, f"Expected correlation >= 0.90 for {checkpoint_kind}, got {corr}"
