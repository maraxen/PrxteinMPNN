"""Unified loader for PrxteinMPNN weights from internal package resources."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.nn.initializers as init
import zstandard as zstd
from importlib.resources import files

from prxteinmpnn.model import PrxteinMPNN, PrxteinLigandMPNN
from prxteinmpnn.model.packer import Packer

if TYPE_CHECKING:
  from jaxtyping import PyTree

MODEL_WEIGHTS = Literal["original", "soluble", "ligand", "sc", "membrane"]
MODEL_VERSION = Literal[
  "v_48_002",
  "v_48_010",
  "v_48_020",
  "v_48_030",
  "v_48_v2",
  "v_32_010_25",
  "v_32_002_16",
]
HF_REPO_ID = "maraxen/prxteinmpnn"  # Legacy reference

NODE_FEATURES = 128
EDGE_FEATURES = 128
HIDDEN_FEATURES = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
VOCAB_SIZE = 21
LIGAND_DEFAULT_CHECKPOINT = "ligandmpnn_v_32_020_25"

LEGACY_ALIAS_MAP = {
  ("original", "v_48_002"): "proteinmpnn_v_48_002.eqx.zst",
  ("original", "v_48_010"): "proteinmpnn_v_48_010.eqx.zst",
  ("original", "v_48_020"): "proteinmpnn_v_48_020.eqx.zst",
  ("original", "v_48_030"): "proteinmpnn_v_48_030.eqx.zst",
  ("soluble", "v_48_002"): "solublempnn_v_48_002.eqx.zst",
  ("soluble", "v_48_010"): "solublempnn_v_48_010.eqx.zst",
  ("soluble", "v_48_020"): "solublempnn_v_48_020.eqx.zst",
  ("soluble", "v_48_030"): "solublempnn_v_48_030.eqx.zst",
}

def get_topology_for_checkpoint(checkpoint_id: str) -> dict[str, int | bool | str]:
  """Infer model topology and type from the checkpoint filename/id."""
  topology = {
    "k_neighbors": 48,
    "num_positional_embeddings": 32,
    "atom_context_num": 16,
    "physics_feature_dim": 0,
    "model_type": "protein",
  }

  name = checkpoint_id.lower()

  if "v_32" in name:
    topology["k_neighbors"] = 32
    topology["num_positional_embeddings"] = 16
  elif "v_30" in name:
    topology["k_neighbors"] = 30
    topology["num_positional_embeddings"] = 16
  elif "v_48" in name:
    topology["k_neighbors"] = 48
    topology["num_positional_embeddings"] = 32

  if "ligandmpnn_sc" in name or "_sc_" in name:
    topology["model_type"] = "packer"
    topology["num_positional_embeddings"] = 16
  elif "ligandmpnn" in name:
    topology["model_type"] = "ligand"
    topology["num_positional_embeddings"] = 32
  elif "membrane" in name:
    topology["physics_feature_dim"] = 3

  parts = name.split("_")
  if len(parts) >= 2:
    try:
      last_part = parts[-1].replace(".eqx", "").replace(".zst", "")
      if last_part.isdigit() and int(last_part) not in [32, 48, 30]:
        topology["atom_context_num"] = int(last_part)
    except ValueError:
      pass

  return topology


def load_weights(
  checkpoint_id: str | None = None,
  skeleton: eqx.Module | None = None,
  key: jax.Array | None = None,
  local_path: str | None = None,
) -> eqx.Module:
  """Load weights into a skeleton from internal resources or local path."""
  if checkpoint_id is None and local_path is None:
    if skeleton is None:
      msg = "skeleton is required for reinitialization"
      raise ValueError(msg)
    if key is None:
      key = jax.random.PRNGKey(0)

    params, static = eqx.partition(skeleton, eqx.is_inexact_array)
    param_leaves = jax.tree_util.tree_leaves(params)
    keys = jax.random.split(key, len(param_leaves))

    def initialize_param(param: jax.Array, key: jax.Array) -> jax.Array:
      shape = param.shape
      if len(shape) >= 2:
        return init.glorot_normal()(key, shape, param.dtype)
      return init.normal(stddev=0.01)(key, shape, param.dtype)

    initialized_leaves = [initialize_param(p, k) for p, k in zip(param_leaves, keys, strict=True)]
    new_params = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params), initialized_leaves)
    return eqx.combine(new_params, static)

  if local_path:
    if local_path.endswith(".zst"):
      with open(local_path, "rb") as f:
        data = f.read()
        dctx = zstd.ZstdDecompressor()
        stream = io.BytesIO(dctx.decompress(data))
        return eqx.tree_deserialise_leaves(stream, skeleton)
    return eqx.tree_deserialise_leaves(local_path, skeleton)
  else:
    filename = checkpoint_id
    if not filename.endswith(".zst"):
      filename = f"{filename}.eqx.zst"
    
    resource_path = files("prxteinmpnn.model_params").joinpath(filename)
    if not resource_path.exists():
      msg = f"Weight file {filename} not found in package resources."
      raise FileNotFoundError(msg)
    
    data = resource_path.read_bytes()
    dctx = zstd.ZstdDecompressor()
    stream = io.BytesIO(dctx.decompress(data))
    return eqx.tree_deserialise_leaves(stream, skeleton)


def load_model(
  checkpoint_id: str | None = None,
  model_weights: str | None = "original",
  local_path: str | None = None,
  key: jax.Array | None = None,
  *,
  use_electrostatics: bool = False,
  use_vdw: bool = False,
  dropout_rate: float = 0.1,
  training_mode: Literal["autoregressive", "diffusion"] = "autoregressive",
  # legacy parameter for backwards compatibility
  model_version: str | None = None,
) -> eqx.Module:
  """Load a fully instantiated PrxteinMPNN model with pre-trained weights."""
  if key is None:
    key = jax.random.PRNGKey(0)

  # Support legacy API usage where checkpoint_id was named model_version
  if checkpoint_id and ("_v_" in checkpoint_id or "mpnn" in checkpoint_id):
    # Valid checkpoint id
    pass
  elif checkpoint_id and not model_version:
    # They passed 'v_48_020' into checkpoint_id
    model_version = checkpoint_id
    checkpoint_id = None
    
  if not checkpoint_id:
    if model_weights and model_version:
      filename = LEGACY_ALIAS_MAP.get((model_weights, model_version))
      if filename:
        checkpoint_id = filename
      else:
        checkpoint_id = f"{model_weights}_{model_version}.eqx.zst"
    else:
      checkpoint_id = "proteinmpnn_v_48_020.eqx.zst"

  if not checkpoint_id.endswith(".zst") and not local_path:
    checkpoint_id = f"{checkpoint_id}.eqx.zst"

  topo = get_topology_for_checkpoint(checkpoint_id if not local_path else local_path)

  physics_feature_dim = topo["physics_feature_dim"]
  if use_electrostatics or use_vdw:
    physics_feature_dim = (5 if use_electrostatics else 0) + (5 if use_vdw else 0)

  model_type = topo["model_type"]
  if model_type == "ligand":
    skeleton = PrxteinLigandMPNN(
      node_features=NODE_FEATURES,
      edge_features=EDGE_FEATURES,
      hidden_features=HIDDEN_FEATURES,
      num_encoder_layers=NUM_ENCODER_LAYERS,
      num_decoder_layers=NUM_DECODER_LAYERS,
      k_neighbors=topo["k_neighbors"],
      num_positional_embeddings=topo["num_positional_embeddings"],
      key=key,
    )
  elif model_type == "packer":
    skeleton = Packer(
      node_features=NODE_FEATURES,
      edge_features=EDGE_FEATURES,
      hidden_dim=HIDDEN_FEATURES,
      num_encoder_layers=NUM_ENCODER_LAYERS,
      num_decoder_layers=NUM_DECODER_LAYERS,
      top_k=topo["k_neighbors"],
      atom_context_num=topo["atom_context_num"],
      num_positional_embeddings=topo["num_positional_embeddings"],
      key=key,
    )
  else:
    if training_mode == "diffusion":
      from prxteinmpnn.model.diffusion_mpnn import DiffusionPrxteinMPNN
      skeleton = DiffusionPrxteinMPNN(
        node_features=NODE_FEATURES,
        edge_features=EDGE_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        physics_feature_dim=physics_feature_dim if physics_feature_dim > 0 else None,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        vocab_size=VOCAB_SIZE,
        k_neighbors=topo["k_neighbors"],
        num_positional_embeddings=topo["num_positional_embeddings"],
        key=key,
      )
    else:
      skeleton = PrxteinMPNN(
        node_features=NODE_FEATURES,
        edge_features=EDGE_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        physics_feature_dim=physics_feature_dim if physics_feature_dim > 0 else None,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        vocab_size=VOCAB_SIZE,
        k_neighbors=topo["k_neighbors"],
        num_positional_embeddings=topo["num_positional_embeddings"],
        dropout_rate=dropout_rate,
        key=key,
      )

  loaded = load_weights(
    checkpoint_id=checkpoint_id,
    local_path=local_path,
    skeleton=skeleton,
  )

  return loaded
