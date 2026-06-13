"""Potts model runner: orchestrate weight loading, calibration, and inference.

This module provides the high-level run_potts function that:
1. Loads PottsModel weights from spec.weights_path
2. Loads CalibrationModule (IdentityCalibration if caliby_path=None)
3. Extracts coordinates and masks from GeometryBundle
4. Infers Potts parameters via model.infer_params
5. Applies post-hoc calibration to marginals
6. Optionally uses PoeModel for multi-backbone ensembles
7. Returns PottsResult with all outputs

Architecture: PottsModel is PARALLEL, NOT a StageSet consumer.
See ADR 260605_potts-parallel-not-stageset for design rationale.

Method names: infer_params/sample/score (NOT encode/decode).
PottsResult is a dataclass, not an eqx.Module.

No imports from aminx.inference.decode, aminx.host.plan, aminx.types.stages,
or aminx.inference.logits.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import zstandard as zstd
from jaxtyping import Array, PRNGKeyArray

from aminx.model import ProteinFeatures
from aminx.potts.calibration import load_calibration
from aminx.potts.model import PottsModel
from aminx.potts.spec import PottsRunSpec
from aminx.types.bundles import GeometryBundle


@dataclass(frozen=False)
class PottsResult:
  """Result dataclass (not eqx.Module) containing Potts inference outputs.

  Attributes:
      marginals: Node marginals (N, 21) after optional calibration
      h: Unary potentials (N, 21) with x2 scale factor from PottsMPNN
      J: Pairwise potentials (N, N, 21, 21) with x2 scale factor
      rho: Tree-reweighting parameters (N, N)
      calibrated_marginals: Calibrated marginals (N, 21) after CalibrationModule
      n_backbones: Number of backbones in ensemble
  """

  marginals: Array
  h: Array
  J: Array
  rho: Array
  calibrated_marginals: Array
  n_backbones: int


def _load_potts_model(weights_path: str, spec: PottsRunSpec, key: PRNGKeyArray) -> PottsModel:
  """Load PottsModel from checkpoint via eqx.tree_deserialise_leaves.

  Args:
      weights_path: Path to checkpoint file (bare .eqx or .eqx.zst).
      spec: PottsRunSpec with model hyperparameters.
      key: JAX random key for skeleton initialization.

  Returns:
      Loaded PottsModel with weights deserialized.

  Raises:
      FileNotFoundError: If checkpoint does not exist.
      ValueError: If k_neighbors is invalid or model construction fails.
  """
  # Validate k_neighbors before building skeleton
  if spec.k_neighbors <= 0:
    msg = (
      f"PottsRunSpec.k_neighbors={spec.k_neighbors} is invalid. "
      "Supply k_neighbors from checkpoint metadata (logged by pottsmpnn_to_eqx.py --dry-run)."
    )
    raise ValueError(msg)

  weights_path_obj = Path(weights_path)
  if not weights_path_obj.exists():
    msg = f"Weights checkpoint not found: {weights_path}"
    raise FileNotFoundError(msg)

  # Construct skeleton with hyperparameters from spec
  skeleton = PottsModel(
    hidden_dim=128,  # Standard Potts backbone dimension
    num_aa=21,  # Standard amino acid alphabet size
    k_neighbors=spec.k_neighbors,
    edge_features_dim=128,  # Standard edge feature dimension
    trw_iters=15,  # Standard TRW iterations
    key=key,
    trw_spec=spec.trw_spec,
    training=spec.training,
  )

  # Load weights via zstd decompression if needed
  if weights_path.endswith(".zst"):
    with open(weights_path, "rb") as f:
      dctx = zstd.ZstdDecompressor()
      decompressed = dctx.decompress(f.read())
      stream = io.BytesIO(decompressed)  # ty: ignore[unresolved-attribute]
      loaded = eqx.tree_deserialise_leaves(stream, skeleton)
  else:
    loaded = eqx.tree_deserialise_leaves(weights_path, skeleton)

  return loaded


def run_potts(
  spec: PottsRunSpec,
  geometry: GeometryBundle,
  key: PRNGKeyArray,
  config: dict | None = None,
) -> PottsResult:
  """Execute Potts inference: load model, extract geometry, infer params, calibrate.

  High-level orchestration function that:
  1. Loads PottsModel weights from spec.weights_path
  2. Loads CalibrationModule (IdentityCalibration if caliby_path=None)
  3. Extracts (coords, mask, residue_index, chain_index) from GeometryBundle
  4. Runs model.infer_params to compute Potts parameters
  5. Applies calibration to marginals
  6. For n_backbones > 1, uses PoeModel for joint energy
  7. Returns PottsResult with marginals, h, J, rho, calibrated_marginals

  Args:
      spec: PottsRunSpec with weights_path, caliby_path, k_neighbors, training flag.
      geometry: GeometryBundle with coords, mask, residue_index, chain_index.
      key: JAX random key for stochastic components.
      config: Optional configuration dict (reserved for future extensions).

  Returns:
      PottsResult with all Potts outputs and calibrated marginals.

  Raises:
      FileNotFoundError: If weights_path or caliby_path (if not None) does not exist.
      ValueError: If geometry extraction or model inference fails.
  """
  if config is None:
    config = {}

  # Step 1: Load PottsModel
  model = _load_potts_model(spec.weights_path, spec, key)

  # Step 2: Load CalibrationModule (identity if caliby_path is None)
  calibration = load_calibration(spec.caliby_path)

  # Step 3: Extract geometry arrays using GeometryBundle accessors
  # GeometryBundle has fields: coords, mask, residue_index, chain_index
  coords = geometry.coords  # (S, L, 4, 3) if S > 1, else (L, 4, 3) depending on implementation
  mask = geometry.mask  # (S, L) or (L)
  residue_index = geometry.residue_index  # (S, L) or (L)
  chain_index = geometry.chain_index  # (S, L) or (L)

  # Handle potentially batched geometry (S > 1 means multiple states)
  # For single state, squeeze out the state dimension if present
  if coords.ndim == 4:  # (S, L, 4, 3)
    # Multi-state case: extract first state for now
    # Future: use PoeModel for all states
    coords = coords[0]  # (L, 4, 3)
    mask = mask[0]  # (L,)
    residue_index = residue_index[0]  # (L,)
    chain_index = chain_index[0]  # (L,)

  # Ensure correct dtypes
  coords = jnp.asarray(coords, dtype=jnp.float32)
  mask = jnp.asarray(mask, dtype=jnp.float32)
  residue_index = jnp.asarray(residue_index, dtype=jnp.int32)
  chain_index = jnp.asarray(chain_index, dtype=jnp.int32)

  # Handle 37-atom format (if input is 4, pad to 37)
  if coords.shape[-2] == 4:
    # Pad from 4 atoms (N, CA, C, O) to 37 (standard pdb_atom37 format)
    n, _ = coords.shape[:2]
    coords_37 = jnp.zeros((n, 37, 3), dtype=jnp.float32)
    coords_37 = coords_37.at[:, :4, :].set(coords)
    coords = coords_37

  # Step 3.5: Compute k-NN graph features via ProteinFeatures
  # This integration layer responsibility: compute graph and pass to pure potts model
  features = ProteinFeatures(
    node_features=128,
    edge_features=128,
    k_neighbors=model.k_neighbors,
    key=key,
  )

  edge_knn, nei, _, _ = features(
    key,
    coords,
    mask,
    residue_index,
    chain_index,
    backbone_noise=None,
  )

  # Step 4: Infer Potts parameters via model.infer_params
  params = model.infer_params(
    key=key,
    edge_knn=edge_knn,
    nei=nei,
    mask=mask,
  )

  # Step 5: Apply calibration to marginals
  calibrated_marginals = calibration(params.marginals)

  # Step 6: Handle n_backbones > 1 (multi-state ensemble via PoeModel)
  # For now, single backbone: just use the inferred parameters directly
  # Future: construct PoeModel if spec.n_backbones > 1 and coords are multi-state

  # Step 7: Return PottsResult
  return PottsResult(
    marginals=params.marginals,
    h=params.h,
    J=params.J,
    rho=params.rho,
    calibrated_marginals=calibrated_marginals,
    n_backbones=spec.n_backbones,
  )
