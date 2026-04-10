"""Parity asset manifest checks."""

from __future__ import annotations

import pytest

from prxteinmpnn.parity.assets import load_parity_assets, verify_parity_assets
from tests.parity.reference_utils import require_reference_path


def test_fast_parity_assets_are_present_and_verified() -> None:
  """Validate the committed fast parity fixtures and checksums."""
  failures = verify_parity_assets(tiers={"parity_fast"})
  assert failures == []


def test_heavy_parity_assets_verify_when_reference_available() -> None:
  """Validate heavy parity assets when the upstream checkout is present."""
  reference_root = require_reference_path()
  failures = verify_parity_assets(
    tiers={"parity_heavy"},
    reference_root_path=reference_root,
  )
  # The converted checkpoint is committed locally; the reference checkpoint is
  # only checked when the upstream checkout is available.
  assert failures == []


def test_asset_manifest_has_project_and_reference_bases() -> None:
  """Ensure the manifest distinguishes cached project assets from reference assets."""
  assets = load_parity_assets()
  bases = {asset.base for asset in assets}
  assert bases == {"project", "reference"}
  assert any(asset.sha256 is not None for asset in assets)
  assert any(asset.sha256 is None for asset in assets)


@pytest.mark.parametrize(
  "asset_id",
  [
    "golden-linear-layer-fixture",
    "golden-linear-layer-metadata",
    "converted-proteinmpnn-020",
    "reference-proteinmpnn-020",
  ],
)
def test_asset_manifest_includes_expected_ids(asset_id: str) -> None:
  """Ensure the manifest tracks the assets required by the parity plan."""
  assert any(asset.id == asset_id for asset in load_parity_assets())
