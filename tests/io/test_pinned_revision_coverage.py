"""The pinned Hub revision must serve every checkpoint aminx can request.

A pin that omits a checkpoint converts a working download into a hard failure, and it fails
only for the consumer who requests that one file -- never in this repo's own suite, because
the packaged-resource branch short-circuits before the Hub is ever reached. That asymmetry is
the point: the pinned path is load-bearing exactly where it is least exercised.

This is the only test that touches the network. It SKIPS when the Hub is unreachable (offline
dev, an air-gapped runner) and FAILS when the Hub is reachable but the pin is incomplete --
those are different situations and must not collapse into the same outcome.
"""

from __future__ import annotations

from importlib.resources import files

import pytest

from aminx.io.weights import HF_REPO_ID, HF_REVISION

pytestmark = pytest.mark.requires_weights


def _packaged_checkpoint_names() -> set[str]:
  """Every ``*.eqx.zst`` shipped in the source tree -- the set a consumer may ask for."""
  root = files("aminx.model_params")
  return {entry.name for entry in root.iterdir() if entry.name.endswith(".eqx.zst")}


def _repo_files_at(revision: str) -> set[str]:
  """List ``*.eqx.zst`` at a revision, skipping the test if the Hub is unreachable."""
  from huggingface_hub import list_repo_files
  from huggingface_hub.errors import HfHubHTTPError

  try:
    listing = list_repo_files(HF_REPO_ID, revision=revision)
  except HfHubHTTPError as exc:  # pragma: no cover - depends on remote state
    if "404" in str(exc) or "Revision Not Found" in str(exc):
      pytest.fail(f"revision {revision!r} does not exist on {HF_REPO_ID}: {exc}")
    pytest.skip(f"Hub unreachable or refused: {exc}")
  except OSError as exc:  # pragma: no cover - offline
    pytest.skip(f"Hub unreachable: {exc}")
  return {name for name in listing if name.endswith(".eqx.zst")}


def test_pinned_revision_serves_every_packaged_checkpoint() -> None:
  """A checkpoint aminx ships must be fetchable at the pin, or the pin breaks that consumer."""
  packaged = _packaged_checkpoint_names()
  assert packaged, "no packaged checkpoints found; this test would be vacuous"

  at_pin = _repo_files_at(HF_REVISION)
  missing = sorted(packaged - at_pin)

  assert not missing, (
    f"{len(missing)} checkpoint(s) shipped in-tree are absent at the pinned revision "
    f"{HF_REVISION}: {missing}. Requesting one would hard-fail. Bump HF_REVISION to a commit "
    f"that contains them, together with any consumer-side weights manifest."
  )


def test_pin_is_not_behind_the_default_branch_in_content() -> None:
  """Warn-level guard: the pin should not omit checkpoints that ``main`` already has.

  Not a correctness failure on its own -- a deliberately older pin is legitimate -- but a pin
  that lags the branch is how "checkpoint X is missing" arrives later, for someone else.
  """
  at_pin = _repo_files_at(HF_REVISION)
  at_main = _repo_files_at("main")
  behind = sorted(at_main - at_pin)

  assert not behind, (
    f"the pinned revision omits {len(behind)} checkpoint(s) present on the default branch: "
    f"{behind}. If that is deliberate, note why next to HF_REVISION; otherwise re-pin."
  )
