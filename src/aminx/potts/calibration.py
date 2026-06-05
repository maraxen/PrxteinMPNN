"""Calibration module for Potts marginal post-hoc corrections.

Caliby (calibration y) is a learned correction applied post-inference to TRW marginals.
It captures systematic deviations from target distributions (e.g., training distribution,
experimental folding data) that are not correctable by re-tuning the Potts energy directly.

The CalibrationModule protocol defines a pure JAX interface:
  __call__(marginals: Float[Array, 'N num_aa']) -> Float[Array, 'N num_aa']

Implementations:
- IdentityCalibration: returns marginals unchanged (default if caliby_path=None)
- LearnedCalibration: applies learned scalar/vector corrections (loaded from checkpoint)

Both compile under eqx.filter_jit. No Python runtime dispatch of None branches inside
jit scope; type selection happens at load time.

References:
  - PottsRunSpec.caliby_path: path to caliby_<id>.eqx.zst checkpoint (None → IdentityCalibration)
  - ADR 260605_potts-parallel-not-stageset.md: Potts as parallel model family

"""

import pickle
from pathlib import Path
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import zstandard as zstd
from jaxtyping import Array, Float


class CalibrationModule(Protocol):
  """Protocol for marginal calibration modules.

  Calibration is a pure function mapping marginals to corrected marginals.
  No state updates, no side effects, JAX-compatible (eqx.filter_jit safe).
  """

  def __call__(
      self, marginals: Float[Array, "N num_aa"],
  ) -> Float[Array, "N num_aa"]:
    """Apply calibration to TRW marginals.

    Args:
      marginals: Per-residue posterior marginals, shape (N, num_aa).
        Row sums equal 1 (probability distributions).

    Returns:
      Calibrated marginals, same shape. Caller responsible for re-normalizing
      if needed (calibration may not preserve sum-to-1).

    """
    ...


class IdentityCalibration(eqx.Module):
  """No-op calibration: returns marginals unchanged.

  Used as default when caliby_path=None in PottsRunSpec.
  Enables stateless inference pipeline without None-branches inside jit.
  """

  def __call__(
      self, marginals: Float[Array, "N num_aa"],
  ) -> Float[Array, "N num_aa"]:
    """Return marginals unchanged.

    Args:
      marginals: Per-residue posterior marginals, shape (N, num_aa).

    Returns:
      Same array (no-op).

    """
    return marginals


class LearnedCalibration(eqx.Module):
  """Learned scalar or vector corrections to TRW marginals.

  Caliby checkpoint format (eqx.zst):
    - correction: Float[Array, "N num_aa"] or Float[Array, "num_aa"]
      Additive or multiplicative correction to apply.
    - correction_type: str in {"additive", "multiplicative", "learned_scale"}
      How to apply the correction.
    - metadata: dict (optional)
      Dataset name, training configuration, etc.

  Correction modes:
    - "additive": corrected = marginals + correction
    - "multiplicative": corrected = marginals * correction
    - "learned_scale": corrected = sigmoid(logit(marginals) + correction)
  """

  correction: Float[Array, "..."]
  correction_type: str = eqx.field(static=True)
  metadata: dict = eqx.field(static=True)

  def __init__(
      self,
      correction: Float[Array, "..."],
      correction_type: str = "additive",
      metadata: dict | None = None,
  ) -> None:
    """Initialize LearnedCalibration with pre-learned corrections.

    Args:
      correction: Array of correction values. Typically (N, num_aa) shaped
        to match marginals, but may be (num_aa,) for broadcasted row correction.
      correction_type: One of "additive", "multiplicative", "learned_scale".
      metadata: Optional metadata dict (e.g., dataset name, training config).

    """
    self.correction = correction
    self.correction_type = correction_type
    self.metadata = metadata or {}

  def __call__(
      self, marginals: Float[Array, "N num_aa"],
  ) -> Float[Array, "N num_aa"]:
    """Apply learned correction to marginals.

    Args:
      marginals: Per-residue posterior marginals, shape (N, num_aa).

    Returns:
      Corrected marginals, same shape.
      Caller responsible for normalizing if correction breaks probability semantics.

    """
    if self.correction_type == "additive":
      return marginals + self.correction
    if self.correction_type == "multiplicative":
      return marginals * self.correction
    if self.correction_type == "learned_scale":
      # Logit(marginals) + correction, then sigmoid back.
      # Avoids 0/1 boundary issues by clamping.
      eps = 1e-7
      clamped = jnp.clip(marginals, eps, 1 - eps)
      logits = jnp.log(clamped) - jnp.log(1 - clamped)
      corrected_logits = logits + self.correction
      return jax.nn.sigmoid(corrected_logits)
    raise ValueError(
        f"Unknown correction_type: {self.correction_type}. "
        "Must be one of: additive, multiplicative, learned_scale",
    )


def load_calibration(caliby_path: str | None) -> CalibrationModule:
  """Load calibration module from checkpoint, or return IdentityCalibration.

  Args:
    caliby_path: Path to caliby_<id>.eqx.zst checkpoint. If None, returns IdentityCalibration.

  Returns:
    CalibrationModule instance (eqx.Module, JAX-compatible).

  Raises:
    FileNotFoundError: If caliby_path is provided but does not exist.

  """
  if caliby_path is None:
    return IdentityCalibration()

  path = Path(caliby_path)
  if not path.exists():
    raise FileNotFoundError(
        f"Calibration checkpoint not found: {caliby_path}",
    )

  # Load via pickle after decompressing zstandard checkpoint.
  # Checkpoint is expected to contain a serialized LearnedCalibration instance.

  with path.open("rb") as f:
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.stream_reader(f).read()

  return pickle.loads(decompressed)  # noqa: S301 — checkpoints are produced by trusted in-house training pipeline, not user-supplied
