"""Static routing spec for Potts TRW inference (rho + messages + loop; JAX-static friendly)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

RhoBackend = Literal["dense_pinv", "edge_lanczos", "uniform"]
MessageBackend = Literal["dense", "tiled_scan"]
TrwLoop = Literal["fori", "scan"]


@dataclass(frozen=True, slots=True)
class PottsTRWRunSpec:
  """Frozen config for TRW numerics; must be ``eqx.field(static=True)`` on modules.

  Branch on these fields with **Python** ``if`` / ``match`` outside traced values so each
  backend gets its own compile cache entry (avoid nested ``@jit`` switching on tracers).
  """

  rho_backend: RhoBackend = "dense_pinv"
  message_backend: MessageBackend = "dense"
  tile_size: int = 8
  lanczos_rank: int = 32
  slq_num_samples: int = 32
  checkpoint_trw_step: bool = False
  trw_loop: TrwLoop = "fori"
  uniform_rho_value: float = 0.5

  @staticmethod
  def default_dense() -> PottsTRWRunSpec:
    """Match legacy ``DifferentiableTRW`` (dense pinv rho, dense messages, ``fori_loop``)."""
    return PottsTRWRunSpec()

  @staticmethod
  def default_bf3t() -> PottsTRWRunSpec:
    """Conservative defaults for BF3t-style sweeps (same numerics as legacy)."""
    return PottsTRWRunSpec()

  @staticmethod
  def default_debug() -> PottsTRWRunSpec:
    """Smaller iterative budget for fast local iteration."""
    return PottsTRWRunSpec(
      rho_backend="edge_lanczos",
      lanczos_rank=16,
      trw_loop="scan",
      checkpoint_trw_step=True,
    )

  def to_json_dict(self) -> dict[str, Any]:
    """JSON-serializable snapshot for sweep configs."""
    return dict(asdict(self))

  @classmethod
  def from_json_dict(cls, d: dict[str, Any]) -> PottsTRWRunSpec:
    rb = str(d.get("rho_backend", "dense_pinv"))
    mb = str(d.get("message_backend", "dense"))
    tl = str(d.get("trw_loop", "fori"))
    valid_rho = frozenset(("dense_pinv", "edge_lanczos", "uniform"))
    valid_msg = frozenset(("dense", "tiled_scan"))
    valid_loop = frozenset(("fori", "scan"))
    if rb not in valid_rho:
      raise ValueError(f"rho_backend must be one of {sorted(valid_rho)}, got {rb!r}")
    if mb not in valid_msg:
      raise ValueError(f"message_backend must be one of {sorted(valid_msg)}, got {mb!r}")
    if tl not in valid_loop:
      raise ValueError(f"trw_loop must be one of {sorted(valid_loop)}, got {tl!r}")
    return cls(
      rho_backend=rb,  # type: ignore[arg-type]
      message_backend=mb,  # type: ignore[arg-type]
      tile_size=int(d.get("tile_size", 8)),
      lanczos_rank=int(d.get("lanczos_rank", 32)),
      slq_num_samples=int(d.get("slq_num_samples", 32)),
      checkpoint_trw_step=bool(d.get("checkpoint_trw_step", False)),
      trw_loop=tl,  # type: ignore[arg-type]
      uniform_rho_value=float(d.get("uniform_rho_value", 0.5)),
    )
