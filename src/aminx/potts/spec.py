"""Frozen specification for Potts model inference run (weights, calibration, TRW config).

PottsRunSpec couples inference hyperparameters (TRW backend, loop strategy) with data paths
(weights checkpoint, optional calibration model). Frozen dataclass (not eqx.Module) to avoid
PyTree registration; deserialized from JSON or constructed programmatically.

Guards enforce:
  1. trw_loop='fori' + training=True raises ValueError (OOM risk)
  2. n_backbones >= 1
  3. caliby_path=None is valid (identity default)
  4. k_neighbors read-only from checkpoint metadata
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from aminx.potts._trw_spec import PottsTRWRunSpec


def _get_potts_trw_run_spec_class() -> type[PottsTRWRunSpec]:
  """Return PottsTRWRunSpec class (already imported above).

  Returns:
      PottsTRWRunSpec class from aminx.potts._trw_spec.

  Note:
    Type checkers cannot resolve attributes on the returned `type` object,
    so callers must use # type: ignore for .default_dense(), .from_json_dict(), etc.
  """
  return PottsTRWRunSpec


@dataclass(frozen=True)
class PottsRunSpec:
  """Frozen config for Potts model inference run.

  Couples model checkpoint (weights_path), optional post-hoc calibration (caliby_path),
  TRW numerics config (trw_spec), and metadata (n_backbones, k_neighbors, training mode).

  Attributes:
      n_backbones: Number of backbones in the ensemble (default 1). Must be >= 1.
      weights_path: Path to Potts model checkpoint (weights file).
      caliby_path: Optional path to learned calibration model. None = identity (valid).
      trw_spec: TRW numerics config (default: dense_pinv rho, dense messages, fori loop).
      k_neighbors: Number of nearest neighbours; must be > 0. Read from checkpoint metadata.
      training: If True, use training-safe TRW config (scan loop + checkpoint). Default False.

  Raises:
      ValueError: If training=True and trw_spec.trw_loop='fori' (OOM risk in reverse-mode AD).
      ValueError: If n_backbones < 1.
  """

  n_backbones: int = 1
  weights_path: str = ""
  caliby_path: str | None = None
  trw_spec: PottsTRWRunSpec | None = None
  k_neighbors: int = 0
  training: bool = False

  def __post_init__(self) -> None:
    """Validate spec fields and enforce safety constraints."""
    if self.trw_spec is None:
      potts_trw_run_spec_cls = _get_potts_trw_run_spec_class()
      object.__setattr__(self, "trw_spec", potts_trw_run_spec_cls.default_dense())  # type: ignore[unresolved-attribute, no-untyped-call]

    if self.k_neighbors <= 0:
      msg = (
        f"k_neighbors must be > 0; got {self.k_neighbors}. "
        "Supply k_neighbors from checkpoint metadata (logged by pottsmpnn_to_eqx.py --dry-run)."
      )
      raise ValueError(msg)

    if self.trw_spec is not None and self.training and self.trw_spec.trw_loop == "fori":
      msg = (
        "trw_loop=fori is unsafe for training: materialises all TRW intermediate states "
        "under reverse-mode autodiff causing OOM. Use trw_loop=scan with checkpoint_trw_step=True."
      )
      raise ValueError(msg)

    if self.n_backbones < 1:
      msg = f"n_backbones must be >= 1, got {self.n_backbones}"
      raise ValueError(msg)

  def to_json(self) -> str:
    """Serialize to JSON string for checkpointing and config storage."""
    data = asdict(self)
    if self.trw_spec is not None:
      data["trw_spec"] = self.trw_spec.to_json_dict()  # type: ignore[attr-defined]
    return json.dumps(data)

  @classmethod
  def from_json(cls, s: str) -> PottsRunSpec:
    """Deserialize from JSON string."""
    potts_trw_run_spec_cls = _get_potts_trw_run_spec_class()
    data = json.loads(s)
    trw_dict = data.pop("trw_spec", {})
    trw_spec = potts_trw_run_spec_cls.from_json_dict(trw_dict) if trw_dict else None  # type: ignore[unresolved-attribute, no-untyped-call]
    return cls(trw_spec=trw_spec, **data)

  @classmethod
  def inference_default(cls, weights_path: str, k_neighbors: int) -> PottsRunSpec:
    """Construct spec for inference with default TRW config.

    Args:
        weights_path: Path to weights checkpoint.
        k_neighbors: Graph connectivity (from checkpoint metadata).

    Returns:
        PottsRunSpec with default TRW (dense pinv, dense messages, fori loop),
        no calibration, training=False.
    """
    potts_trw_run_spec_cls = _get_potts_trw_run_spec_class()
    return cls(
      n_backbones=1,
      weights_path=weights_path,
      caliby_path=None,
      trw_spec=potts_trw_run_spec_cls.default_dense(),  # type: ignore[unresolved-attribute, no-untyped-call]
      k_neighbors=k_neighbors,
      training=False,
    )

  @classmethod
  def training_default(
    cls,
    weights_path: str,
    k_neighbors: int,
    caliby_path: str | None = None,
  ) -> PottsRunSpec:
    """Construct spec for training with safe TRW config.

    Uses trw_loop='scan' with checkpoint_trw_step=True to avoid OOM in reverse-mode AD.

    Args:
        weights_path: Path to weights checkpoint.
        k_neighbors: Graph connectivity (from checkpoint metadata).
        caliby_path: Optional path to learned calibration model. Default None (identity).

    Returns:
        PottsRunSpec with training-safe TRW (scan loop + checkpoint).
    """
    potts_trw_run_spec_cls = _get_potts_trw_run_spec_class()
    return cls(
      n_backbones=1,
      weights_path=weights_path,
      caliby_path=caliby_path,
      trw_spec=potts_trw_run_spec_cls(  # type: ignore[call-arg]
        rho_backend="dense_pinv",
        message_backend="dense",
        tile_size=8,
        lanczos_rank=32,
        slq_num_samples=32,
        checkpoint_trw_step=True,
        trw_loop="scan",
        uniform_rho_value=0.5,
      ),
      k_neighbors=k_neighbors,
      training=True,
    )
