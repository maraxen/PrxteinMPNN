"""Parity path matrix helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

_LIGAND_TIED_PATH_ID = "ligand-tied-positions-and-multi-state"
_ROLLOUT_POLICY_KEY = "rollout_policy"
_VALID_ROLLOUT_ACTIONS = {"warn", "fail"}
_REQUIRED_ROLLOUT_TIERS = {"parity_heavy", "parity_audit"}

RolloutAction = Literal["warn", "fail"]
RolloutOutcome = Literal["pass", "warn", "fail"]


@dataclass(frozen=True, slots=True)
class ParityPath:
  """Describe a single parity path and its acceptance policy."""

  id: str
  tier: str
  code_paths: tuple[str, ...]
  reference: str
  method: str
  metrics: tuple[str, ...]
  acceptance: dict[str, object]


def project_root() -> Path:
  """Return repository root."""
  return Path(__file__).resolve().parents[3]


def manifest_path() -> Path:
  """Return the parity path manifest path."""
  return project_root() / "tests/parity/parity_matrix.json"


def load_parity_matrix(path: Path | None = None) -> tuple[ParityPath, ...]:
  """Load the parity matrix manifest."""
  payload = json.loads((path or manifest_path()).read_text(encoding="utf-8"))
  paths = payload.get("paths")
  if not isinstance(paths, list):
    msg = "Parity matrix manifest must contain a 'paths' list."
    raise TypeError(msg)

  parsed: list[ParityPath] = []
  for item in paths:
    if not isinstance(item, dict):
      msg = "Parity matrix entries must be objects."
      raise TypeError(msg)

    code_paths = item.get("code_paths")
    metrics = item.get("metrics")
    acceptance = item.get("acceptance")
    if not isinstance(code_paths, list) or not all(isinstance(v, str) for v in code_paths):
      msg = f"Parity path {item.get('id', '<unknown>')} has invalid code_paths."
      raise TypeError(msg)
    if not isinstance(metrics, list) or not all(isinstance(v, str) for v in metrics):
      msg = f"Parity path {item.get('id', '<unknown>')} has invalid metrics."
      raise TypeError(msg)
    if not isinstance(acceptance, dict):
      msg = f"Parity path {item.get('id', '<unknown>')} has invalid acceptance."
      raise TypeError(msg)

    parsed.append(
      ParityPath(
        id=str(item.get("id")),
        tier=str(item.get("tier")),
        code_paths=tuple(code_paths),
        reference=str(item.get("reference", "")),
        method=str(item.get("method", "")),
        metrics=tuple(metrics),
        acceptance=acceptance,
      ),
    )

  return tuple(parsed)


def ligand_tied_multistate_rollout_policy(
  matrix: tuple[ParityPath, ...] | None = None,
) -> dict[str, RolloutAction]:
  """Return staged rollout policy for the ligand tied/multistate parity path."""
  paths = matrix or load_parity_matrix()
  ligand_path = next((path for path in paths if path.id == _LIGAND_TIED_PATH_ID), None)
  if ligand_path is None:
    msg = f"{_LIGAND_TIED_PATH_ID} is missing from the parity matrix."
    raise RuntimeError(msg)

  payload = ligand_path.acceptance.get(_ROLLOUT_POLICY_KEY)
  if not isinstance(payload, dict):
    msg = f"{_LIGAND_TIED_PATH_ID}.acceptance.{_ROLLOUT_POLICY_KEY} must be a mapping."
    raise TypeError(msg)

  parsed: dict[str, RolloutAction] = {}
  for tier, action in payload.items():
    if not isinstance(tier, str):
      msg = f"{_LIGAND_TIED_PATH_ID}.acceptance.{_ROLLOUT_POLICY_KEY} tier keys must be strings."
      raise TypeError(msg)
    if action not in _VALID_ROLLOUT_ACTIONS:
      msg = (
        f"{_LIGAND_TIED_PATH_ID}.acceptance.{_ROLLOUT_POLICY_KEY}.{tier} must be one of "
        f"{sorted(_VALID_ROLLOUT_ACTIONS)}."
      )
      raise TypeError(msg)
    parsed[tier] = cast("RolloutAction", action)

  missing_tiers = _REQUIRED_ROLLOUT_TIERS - set(parsed)
  if missing_tiers:
    msg = (
      f"{_LIGAND_TIED_PATH_ID}.acceptance.{_ROLLOUT_POLICY_KEY} is missing required tier(s): "
      f"{sorted(missing_tiers)}."
    )
    raise ValueError(msg)
  return parsed


def ligand_tied_multistate_enforcement_for_tier(
  tier: str,
  *,
  matrix: tuple[ParityPath, ...] | None = None,
) -> RolloutAction:
  """Resolve rollout enforcement for a tier; unknown tiers default to fail."""
  policy = ligand_tied_multistate_rollout_policy(matrix)
  return cast("RolloutAction", policy.get(tier, "fail"))


def ligand_tied_multistate_rollout_outcome(
  *,
  condition_passed: bool,
  tier: str,
  matrix: tuple[ParityPath, ...] | None = None,
) -> RolloutOutcome:
  """Classify tied/multistate rollout outcome for a single check under tier policy."""
  if condition_passed:
    return "pass"
  return cast("RolloutOutcome", ligand_tied_multistate_enforcement_for_tier(tier, matrix=matrix))
