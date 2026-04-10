"""Checkpoint-family parity audit for converted model assets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import pytest

from prxteinmpnn.model.mpnn import PrxteinLigandMPNN, PrxteinMPNN
from prxteinmpnn.model.packer import Packer
from tests.parity.reference_utils import project_root

PARITY_AUDIT_TIER = "parity_audit"
CHECKPOINT_AUDIT_STRATEGY = "checkpoint-family-load"
EXPECTED_FAMILIES = {"protein", "soluble", "ligand", "membrane", "sc"}
REPO_ROOT = project_root()
MODEL_PARAMS_DIR = REPO_ROOT / "model_params"
ASSET_MANIFEST_PATH = REPO_ROOT / "tests/parity/parity_assets.json"
pytestmark = pytest.mark.parity_audit


@dataclass(frozen=True, slots=True)
class FamilyCheckpointCase:
  """A converted checkpoint declared for family-level parity audit."""

  asset_id: str
  family: str
  checkpoint_path: Path
  validation: dict[str, object]


def _require_string(*, value: object, field: str, asset_id: str) -> str:
  if isinstance(value, str) and value:
    return value
  msg = f"{asset_id}: expected non-empty string for {field}."
  raise TypeError(msg)


def _require_int(*, value: object, field: str, asset_id: str) -> int:
  if isinstance(value, int):
    return value
  msg = f"{asset_id}: expected integer for {field}."
  raise TypeError(msg)


def _load_family_checkpoint_cases() -> tuple[FamilyCheckpointCase, ...]:
  payload = json.loads(ASSET_MANIFEST_PATH.read_text(encoding="utf-8"))
  assets = payload.get("assets")
  if not isinstance(assets, list):
    msg = "Parity asset manifest must contain an 'assets' list."
    raise TypeError(msg)

  cases: list[FamilyCheckpointCase] = []
  for asset in assets:
    if not isinstance(asset, dict):
      msg = "Parity asset entries must be objects."
      raise TypeError(msg)

    required_for = asset.get("required_for")
    if not isinstance(required_for, list) or not all(isinstance(item, str) for item in required_for):
      msg = f"Parity asset {asset.get('id', '<unknown>')} has invalid required_for."
      raise TypeError(msg)

    if PARITY_AUDIT_TIER not in required_for:
      continue
    if asset.get("asset_kind") != "converted_checkpoint":
      continue

    asset_id = _require_string(value=asset.get("id"), field="id", asset_id="<unknown>")
    family = _require_string(value=asset.get("family"), field="family", asset_id=asset_id)
    relative_path = _require_string(value=asset.get("path"), field="path", asset_id=asset_id)
    validation = asset.get("validation")
    if not isinstance(validation, dict):
      msg = f"{asset_id}: expected validation object for converted checkpoint assets."
      raise TypeError(msg)

    strategy = _require_string(
      value=validation.get("strategy"),
      field="validation.strategy",
      asset_id=asset_id,
    )
    if strategy != CHECKPOINT_AUDIT_STRATEGY:
      msg = f"{asset_id}: unsupported validation strategy {strategy!r}."
      raise TypeError(msg)

    cases.append(
      FamilyCheckpointCase(
        asset_id=asset_id,
        family=family,
        checkpoint_path=REPO_ROOT / relative_path,
        validation=validation,
      ),
    )

  return tuple(cases)


FAMILY_CHECKPOINT_CASES = _load_family_checkpoint_cases()


def _get_validation_int(case: FamilyCheckpointCase, field: str) -> int:
  return _require_int(
    value=case.validation.get(field),
    field=f"validation.{field}",
    asset_id=case.asset_id,
  )


def _get_num_positional_candidates(case: FamilyCheckpointCase) -> tuple[int, ...]:
  raw = case.validation.get("num_positional_embeddings_candidates", [32, 16])
  if not isinstance(raw, list) or not raw:
    msg = (
      f"{case.asset_id}: expected non-empty list for "
      "validation.num_positional_embeddings_candidates."
    )
    raise TypeError(msg)

  values: list[int] = []
  for item in raw:
    value = _require_int(
      value=item,
      field="validation.num_positional_embeddings_candidates[]",
      asset_id=case.asset_id,
    )
    if value not in values:
      values.append(value)
  return tuple(values)


def _build_template(
  case: FamilyCheckpointCase,
  *,
  key: jax.Array,
  num_positional_embeddings: int,
) -> eqx.Module:
  model_class = _require_string(
    value=case.validation.get("model_class"),
    field="validation.model_class",
    asset_id=case.asset_id,
  )

  if model_class == "PrxteinMPNN":
    physics_feature_dim = case.validation.get("physics_feature_dim")
    if physics_feature_dim is not None and not isinstance(physics_feature_dim, int):
      msg = f"{case.asset_id}: validation.physics_feature_dim must be null or integer."
      raise TypeError(msg)

    return PrxteinMPNN(
      node_features=_get_validation_int(case, "node_features"),
      edge_features=_get_validation_int(case, "edge_features"),
      hidden_features=_get_validation_int(case, "hidden_features"),
      num_encoder_layers=_get_validation_int(case, "num_encoder_layers"),
      num_decoder_layers=_get_validation_int(case, "num_decoder_layers"),
      k_neighbors=_get_validation_int(case, "k_neighbors"),
      num_positional_embeddings=num_positional_embeddings,
      physics_feature_dim=physics_feature_dim,
      key=key,
    )

  if model_class == "PrxteinLigandMPNN":
    return PrxteinLigandMPNN(
      node_features=_get_validation_int(case, "node_features"),
      edge_features=_get_validation_int(case, "edge_features"),
      hidden_features=_get_validation_int(case, "hidden_features"),
      num_encoder_layers=_get_validation_int(case, "num_encoder_layers"),
      num_decoder_layers=_get_validation_int(case, "num_decoder_layers"),
      k_neighbors=_get_validation_int(case, "k_neighbors"),
      num_positional_embeddings=num_positional_embeddings,
      key=key,
    )

  if model_class == "Packer":
    return Packer(
      edge_features=_get_validation_int(case, "edge_features"),
      node_features=_get_validation_int(case, "node_features"),
      num_positional_embeddings=num_positional_embeddings,
      num_rbf=_get_validation_int(case, "num_rbf"),
      top_k=_get_validation_int(case, "top_k"),
      atom_context_num=_get_validation_int(case, "atom_context_num"),
      hidden_dim=_get_validation_int(case, "hidden_dim"),
      num_encoder_layers=_get_validation_int(case, "num_encoder_layers"),
      num_decoder_layers=_get_validation_int(case, "num_decoder_layers"),
      num_mix=_get_validation_int(case, "num_mix"),
      key=key,
    )

  msg = f"{case.asset_id}: unsupported validation.model_class {model_class!r}."
  raise ValueError(msg)


def _load_with_positional_fallback(case: FamilyCheckpointCase, *, key: jax.Array) -> eqx.Module:
  failures: list[str] = []

  for num_positional_embeddings in _get_num_positional_candidates(case):
    template = _build_template(
      case,
      key=key,
      num_positional_embeddings=num_positional_embeddings,
    )
    try:
      loaded = eqx.tree_deserialise_leaves(case.checkpoint_path, template)
    except RuntimeError as error:
      failures.append(f"num_positional_embeddings={num_positional_embeddings}: {error}")
      continue

    if type(loaded) is not type(template):
      msg = (
        f"{case.asset_id}: deserialized checkpoint type {type(loaded)} "
        f"did not match template type {type(template)}"
      )
      raise TypeError(msg)

    return loaded

  details = " | ".join(failures[-2:])
  msg = f"{case.asset_id}: failed to load checkpoint with positional embedding fallbacks. {details}"
  raise RuntimeError(msg)


def _family_cases(family: str) -> tuple[FamilyCheckpointCase, ...]:
  return tuple(case for case in FAMILY_CHECKPOINT_CASES if case.family == family)


def test_checkpoint_family_manifest_declares_all_families() -> None:
  """Ensure the parity manifest declares all checkpoint families for audit."""
  families = {case.family for case in FAMILY_CHECKPOINT_CASES}
  if families != EXPECTED_FAMILIES:
    msg = f"Expected families {EXPECTED_FAMILIES}, got {families}"
    pytest.fail(msg)
  if not all(case.checkpoint_path.parent == MODEL_PARAMS_DIR for case in FAMILY_CHECKPOINT_CASES):
    pytest.fail("All converted checkpoint paths must be under model_params/")


@pytest.mark.parametrize("family", sorted(EXPECTED_FAMILIES))
def test_load_available_converted_family_checkpoints(family: str) -> None:
  """Load available converted checkpoints for each parity-audit family."""
  family_cases = _family_cases(family)
  if not family_cases:
    pytest.fail(f"Manifest declared no converted parity-audit cases for family '{family}'.")

  available_cases = [case for case in family_cases if case.checkpoint_path.exists()]
  if not available_cases:
    expected_paths = ", ".join(case.checkpoint_path.name for case in family_cases)
    pytest.skip(
      f"No converted checkpoints available for family '{family}'. "
      f"Expected one of: {expected_paths}",
    )

  base_key = jax.random.PRNGKey(0)
  for index, case in enumerate(available_cases):
    checkpoint_key = jax.random.fold_in(base_key, index)
    loaded = _load_with_positional_fallback(case, key=checkpoint_key)
    if type(loaded).__name__ not in {"PrxteinMPNN", "PrxteinLigandMPNN", "Packer"}:
      msg = f"{case.asset_id}: loaded unexpected model type {type(loaded).__name__}"
      pytest.fail(msg)
