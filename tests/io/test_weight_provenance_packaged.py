"""Coverage for the branches the first provenance suite could not discriminate.

Two mutations survived that suite (executed 2026-08-24): hardcoding ``source="hub"`` in the
packaged branch, and making the packaged branch ignore ``filename`` and resolve a different
real checkpoint. Both passed 7/7 because the only test reaching that branch asserted
``source in {...}`` -- a membership check -- and re-derived its expected file through the same
resolver it was testing. These tests kill both mutations, and cover the env-var and
filename-safety rules added after the jury review.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aminx.io import weights as weights_mod
from aminx.io.weights import (
  REVISION_ENV,
  WEIGHTS_DIR_ENV,
  _resolve_weight_path,
  weight_provenance,
)

CHECKPOINT = "proteinmpnn_v_48_020.eqx.zst"
OTHER_CHECKPOINT = "proteinmpnn_v_48_002.eqx.zst"


@pytest.fixture(autouse=True)
def _clear_weight_env(monkeypatch: pytest.MonkeyPatch) -> None:
  """These tests describe default resolution; a stray env var would silently redirect it."""
  monkeypatch.delenv(WEIGHTS_DIR_ENV, raising=False)
  monkeypatch.delenv(REVISION_ENV, raising=False)


def test_packaged_source_is_labelled_packaged_not_merely_a_known_value() -> None:
  """Kills the ``source="hub"``-hardcode mutation, which a membership check let through."""
  source, _path = _resolve_weight_path(CHECKPOINT)
  assert source == "packaged"


def test_packaged_branch_resolves_the_requested_filename() -> None:
  """Kills the ignore-``filename`` mutation.

  Asserting on the resolved PATH is what makes this work. Re-deriving the expectation through
  the resolver would agree with a consistently-wrong resolution, which is exactly how the
  original suite missed it.
  """
  _source, path_one = _resolve_weight_path(CHECKPOINT)
  _source, path_two = _resolve_weight_path(OTHER_CHECKPOINT)

  assert Path(path_one).name == CHECKPOINT
  assert Path(path_two).name == OTHER_CHECKPOINT
  assert path_one != path_two


def test_non_hub_provenance_leaves_the_hub_fields_empty() -> None:
  """The dataclass contract: hub fields are populated only for hub-sourced weights."""
  record = weight_provenance(CHECKPOINT)

  assert record.source == "packaged"
  assert record.hub_repo_id is None
  assert record.hub_revision is None


def test_packaged_lookup_failure_falls_through_to_the_hub(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The ``except`` around the packaged lookup was previously unexercised."""

  def boom(_pkg: str) -> object:
    raise ModuleNotFoundError("no such package")

  monkeypatch.setattr(weights_mod, "files", boom)
  monkeypatch.setattr(
    weights_mod, "hf_hub_download", lambda **_kw: "/cache/snapshots/abc123/x.eqx.zst",
  )

  source, _path = _resolve_weight_path(CHECKPOINT)

  assert source == "hub"


def test_traversable_without_a_real_filesystem_path_falls_through(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A zip/egg import yields ``is_file()`` True with an unopenable string form.

  Returning that path would hand the caller something ``Path(...).read_bytes()`` cannot open.
  Falling through to the Hub is recoverable; an unhandled read error is not.
  """

  class ZipTraversable:
    def joinpath(self, _name: str) -> "ZipTraversable":
      return self

    def is_file(self) -> bool:
      return True

    def __str__(self) -> str:
      return "/nonexistent.zip/aminx/model_params/proteinmpnn_v_48_020.eqx.zst"

  monkeypatch.setattr(weights_mod, "files", lambda _pkg: ZipTraversable())
  monkeypatch.setattr(
    weights_mod, "hf_hub_download", lambda **_kw: "/cache/snapshots/abc123/x.eqx.zst",
  )

  source, _path = _resolve_weight_path(CHECKPOINT)

  assert source == "hub", "an unopenable packaged path must not be returned to the caller"


@pytest.mark.parametrize("env_name", [WEIGHTS_DIR_ENV, REVISION_ENV])
@pytest.mark.parametrize("blank", ["", "   "])
def test_set_but_blank_env_var_fails_closed(
  env_name: str, blank: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """``FOO=`` is an unset variable interpolated somewhere, not a request for defaults.

  Silently ignoring it would change the weight source with no signal -- the failure these
  settings exist to prevent.
  """
  monkeypatch.setenv(env_name, blank)

  with pytest.raises(ValueError, match="set but blank"):
    _resolve_weight_path(CHECKPOINT)


def test_blank_revision_raises_even_when_resolution_would_not_reach_the_hub(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The blank check runs before the branch is chosen, and that is the point.

  The parametrized test above sets one variable at a time, so it never covers this: a run
  that would resolve entirely from the authoritative directory still rejects a blank
  revision. A value that only fails once something happens to reach the Hub is the failure
  mode this replaces.
  """
  (tmp_path / CHECKPOINT).write_bytes(b"payload")
  monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path))
  monkeypatch.setenv(REVISION_ENV, "")

  with pytest.raises(ValueError, match="set but blank"):
    _resolve_weight_path(CHECKPOINT)


def test_packaged_lookup_failure_is_logged_not_silently_swallowed(
  monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
  """A local permission error must leave a breadcrumb, not look like a Hub outage.

  On an air-gapped compute node the fall-through fails at the Hub, and without this log the
  operator sees a connectivity error for what is actually a fixable local file mode.
  """

  def denied(_pkg: str) -> object:
    raise PermissionError(13, "Permission denied")

  monkeypatch.setattr(weights_mod, "files", denied)
  monkeypatch.setattr(
    weights_mod, "hf_hub_download", lambda **_kw: "/cache/snapshots/abc123/x.eqx.zst",
  )

  with caplog.at_level("DEBUG", logger="aminx.io.weights"):
    source, _path = _resolve_weight_path(CHECKPOINT)

  assert source == "hub"
  assert any("packaged lookup failed" in record.getMessage() for record in caplog.records), (
    "an OSError swallowed with no trace makes a local misconfiguration look like a Hub outage"
  )


@pytest.mark.parametrize(
  "unsafe",
  ["/etc/passwd", "../../../etc/passwd", "/tmp/evil.eqx.zst"],
)
def test_filename_escaping_the_weights_directory_is_rejected(
  unsafe: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """``Path("/pinned") / "/etc/passwd"`` is ``/etc/passwd`` under pathlib join semantics."""
  monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path))

  with pytest.raises(ValueError, match="escape"):
    _resolve_weight_path(unsafe)


def test_local_path_override_warns_when_the_authoritative_dir_is_set(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
  """``local_path`` bypasses resolution by design -- it must not do so silently.

  Documents the real scope of the env var: authoritative for checkpoint-id resolution, not for
  every route by which weights enter the process.
  """
  monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path))
  stray = tmp_path / "elsewhere.eqx.zst"
  stray.write_bytes(b"not a real checkpoint")

  with caplog.at_level("WARNING"):
    # Whether deserialising the junk payload raises is not the point and is not asserted --
    # the warning must be emitted before that either way.
    try:
      weights_mod.load_weights(local_path=str(stray), skeleton=None)
    except Exception:  # noqa: BLE001, S110
      pass

  assert any(WEIGHTS_DIR_ENV in record.getMessage() for record in caplog.records), (
    "an explicit local_path silently overriding the authoritative dir is the failure mode"
  )
