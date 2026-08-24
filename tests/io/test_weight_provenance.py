"""Weight resolution must be identifiable, pinned, and never silently change source.

The built wheel ships no ``model_params/*.eqx.zst``, so the aminx version pin does not
determine which checkpoint bytes execute. These tests hold the three properties that make that
survivable: the provenance record describes the file the loader actually reads, an explicit
weights directory fails closed instead of falling through, and the Hub fallback is pinned.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aminx.io import weights as weights_mod
from aminx.io.weights import (
  HF_REVISION,
  REVISION_ENV,
  WEIGHTS_DIR_ENV,
  WeightProvenance,
  _load_weight_bytes,
  _resolve_weight_path,
  weight_provenance,
)

CHECKPOINT = "proteinmpnn_v_48_020.eqx.zst"


def test_provenance_describes_the_bytes_the_loader_returns() -> None:
  """The record must not be able to describe a different file from the one loaded.

  This is the property that makes a consumer-side mirror of the resolution order unnecessary
  -- and a mirror is exactly what would drift.
  """
  record = weight_provenance(CHECKPOINT)
  loaded = _load_weight_bytes(CHECKPOINT)

  assert record.sha256 == hashlib.sha256(loaded).hexdigest()
  assert record.filename == CHECKPOINT
  assert record.source in {"explicit_dir", "packaged", "hub"}
  assert Path(record.path).is_file()


def test_explicit_dir_is_authoritative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """When the env var names a directory, weights come from there and nowhere else."""
  payload = b"not a real checkpoint, but the resolver should not care"
  (tmp_path / CHECKPOINT).write_bytes(payload)
  monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path))

  source, path = _resolve_weight_path(CHECKPOINT)

  assert source == "explicit_dir"
  assert Path(path) == tmp_path / CHECKPOINT
  assert _load_weight_bytes(CHECKPOINT) == payload


def test_explicit_dir_fails_closed_rather_than_falling_through(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A missing file under an authoritative dir must RAISE, not silently reach the Hub.

  Falling through would reproduce the defect this setting exists to prevent: the source
  changing with no signal, which is what happened between a24 and the first a25 build.
  """
  monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path))

  with pytest.raises(FileNotFoundError, match="authoritative"):
    _resolve_weight_path(CHECKPOINT)


def test_hub_fallback_passes_the_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
  """The Hub call must be pinned, so a code pin implies a weights pin."""
  monkeypatch.delenv(WEIGHTS_DIR_ENV, raising=False)
  monkeypatch.delenv(REVISION_ENV, raising=False)
  # Force the packaged branch to miss so resolution reaches the Hub.
  monkeypatch.setattr(weights_mod, "files", lambda _pkg: Path("/nonexistent-aminx-pkg"))
  seen: dict[str, object] = {}

  def fake_download(*, repo_id: str, filename: str, revision: str) -> str:
    seen.update(repo_id=repo_id, filename=filename, revision=revision)
    return f"/cache/models--x/snapshots/{revision}/{filename}"

  monkeypatch.setattr(weights_mod, "hf_hub_download", fake_download)

  source, path = _resolve_weight_path(CHECKPOINT)

  assert source == "hub"
  assert seen["revision"] == HF_REVISION, "an unpinned revision serves whatever is on main"
  assert HF_REVISION in path


def test_revision_env_overrides_the_pin(monkeypatch: pytest.MonkeyPatch) -> None:
  """The escape hatch exists so a checkpoint newer than the pin is still reachable."""
  monkeypatch.delenv(WEIGHTS_DIR_ENV, raising=False)
  monkeypatch.setenv(REVISION_ENV, "deadbeef")
  monkeypatch.setattr(weights_mod, "files", lambda _pkg: Path("/nonexistent-aminx-pkg"))
  seen: dict[str, object] = {}

  def fake_download(*, repo_id: str, filename: str, revision: str) -> str:
    seen["revision"] = revision
    return f"/cache/models--x/snapshots/{revision}/{filename}"

  monkeypatch.setattr(weights_mod, "hf_hub_download", fake_download)
  _resolve_weight_path(CHECKPOINT)

  assert seen["revision"] == "deadbeef"


def test_hub_revision_is_read_back_from_the_cache_path(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A hub-sourced record carries the resolved commit, so results can stamp it."""
  commit = "0123456789abcdef0123456789abcdef01234567"
  snapshot = tmp_path / "models--maraxen--aminx" / "snapshots" / commit
  snapshot.mkdir(parents=True)
  (snapshot / CHECKPOINT).write_bytes(b"payload")

  monkeypatch.delenv(WEIGHTS_DIR_ENV, raising=False)
  monkeypatch.setattr(weights_mod, "files", lambda _pkg: Path("/nonexistent-aminx-pkg"))
  monkeypatch.setattr(
    weights_mod,
    "hf_hub_download",
    lambda **_kw: str(snapshot / CHECKPOINT),
  )

  record = weight_provenance(CHECKPOINT)

  assert isinstance(record, WeightProvenance)
  assert record.source == "hub"
  assert record.hub_revision == commit
  assert record.hub_repo_id == "maraxen/aminx"
  assert record.sha256 == hashlib.sha256(b"payload").hexdigest()


def test_malformed_cache_path_yields_none_revision_rather_than_raising(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Recording provenance must never be able to break weight loading."""
  flat = tmp_path / CHECKPOINT
  flat.write_bytes(b"payload")

  monkeypatch.delenv(WEIGHTS_DIR_ENV, raising=False)
  monkeypatch.setattr(weights_mod, "files", lambda _pkg: Path("/nonexistent-aminx-pkg"))
  monkeypatch.setattr(weights_mod, "hf_hub_download", lambda **_kw: str(flat))

  record = weight_provenance(CHECKPOINT)

  assert record.source == "hub"
  assert record.hub_revision is None
