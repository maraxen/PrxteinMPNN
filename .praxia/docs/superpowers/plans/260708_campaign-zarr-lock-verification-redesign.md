# Campaign.py Zarr Lock/Verification Redesign Implementation Plan

> **For Claude:** Execute task-by-task in this session, TDD discipline, commit after each task.

**Goal:** Replace `host/campaign.py`'s HDF5-specific content-verification (`_h5_content_digest`,
`_sha256_file`) with a Zarr-tree equivalent, so campaign manifest rows' lock/done-marker/retry-resume
machinery works against the Zarr stores that `streaming.py`/`runner.py` now produce, with the same
correctness guarantees as today (detect partial/corrupted output before trusting a "done" row).

**Architecture:** The lock layer (`_lock_path`, `_acquire_local_fs_lock`, etc.), path-naming helpers
(`_done_marker_path`, `_partial_output_path`), and atomic promotion (`partial_path.replace(final_path)`)
are already format-agnostic — verified empirically that `Path.replace()` is atomic for directory-to-directory
rename on POSIX, same as for files. Only three things are HDF5-specific and need replacing:
1. `_h5_content_digest` (tree-walk) → `_zarr_content_digest` (same hashing primitives, walks
   `zarr.Group`/`zarr.Array` instead of `h5py.Group`/`h5py.Dataset`)
2. `_sha256_file(partial_path)` (whole-file hash) → **dropped entirely**, not replaced. It's
   redundant with the semantic content digest (which already reads and hashes every array's raw
   bytes) and has no clean analog for a directory of many chunk files. `DONE_MARKER_SCHEMA_VERSION`
   bumps so old H5-era done markers are cleanly rejected (schema mismatch), not silently
   misinterpreted.
3. `_fsync_file(partial_path)` (single-file durability) → `_fsync_tree(partial_path)` (recursive:
   every file's data, then every directory's entries bottom-up) — a Zarr store's chunk data lives
   in many files, not one.

`partial_path.unlink(missing_ok=True)` in the `finally` cleanup also needs to become directory-aware
(`shutil.rmtree` when the path still exists and is a directory).

**Tech Stack:** Python stdlib (`hashlib`, `shutil`, `pathlib`), `zarr` (already a dependency via
`xtrax[io]`), pytest, `uv run pytest`.

**Non-goals (explicitly deferred):** No spot-check/sampling optimization for resume-time verification
speed (backlog #3182's Option C) — the semantic digest already does a full walk today for HDF5, so a
full walk for Zarr is behavior-preserving, not a regression; sampling-based verification would be a
genuine behavior change (weaker guarantee) and isn't justified without a demonstrated performance
problem. No migration path for existing H5-era done markers/outputs — matches this project's existing
clean-break convention (see CHANGELOG's other Breaking Changes entries this session).

---

### Task 1: `_zarr_content_digest` — Zarr tree-walk digest

**Files:**
- Modify: `src/aminx/host/campaign.py` (add functions near `_h5_content_digest`, ~line 340)
- Test: `tests/host/test_campaign_zarr_digest.py` (new file)

**Step 1: Write the failing tests**

```python
"""Tests for host/campaign.py's Zarr content-digest verification primitives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from aminx.host.campaign import _zarr_content_digest


def _make_store(tmp_path: Path, name: str = "test.zarr") -> Path:
  store_path = tmp_path / name
  root = zarr.open_group(str(store_path), mode="a")
  arr = root.create_array(name="data", shape=(3,), dtype="int32")
  arr[...] = np.array([1, 2, 3], dtype=np.int32)
  arr.attrs["label"] = "alpha"
  return store_path


def test_digest_is_deterministic(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  assert _zarr_content_digest(store_path) == _zarr_content_digest(store_path)


def test_digest_changes_when_array_data_changes(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  original = _zarr_content_digest(store_path)

  root = zarr.open_group(str(store_path), mode="a")
  root["data"][...] = np.array([9, 9, 9], dtype=np.int32)

  assert _zarr_content_digest(store_path) != original


def test_digest_changes_when_attrs_change(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  original = _zarr_content_digest(store_path)

  root = zarr.open_group(str(store_path), mode="a")
  root["data"].attrs["label"] = "beta"

  assert _zarr_content_digest(store_path) != original


def test_digest_covers_nested_groups(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  original = _zarr_content_digest(store_path)

  root = zarr.open_group(str(store_path), mode="a")
  sub = root.require_group("nested")
  arr = sub.create_array(name="extra", shape=(1,), dtype="int32")
  arr[...] = np.array([42], dtype=np.int32)

  assert _zarr_content_digest(store_path) != original


def test_digest_stable_across_reopen(tmp_path: Path) -> None:
  """Digest computed by a fresh process/session (new zarr.open_group call) matches."""
  store_path = _make_store(tmp_path)
  first = _zarr_content_digest(store_path)
  second = _zarr_content_digest(store_path)
  assert first == second
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/host/test_campaign_zarr_digest.py -v --no-cov`
Expected: FAIL with `ImportError: cannot import name '_zarr_content_digest'`

**Step 3: Write the implementation**

In `src/aminx/host/campaign.py`, add `import zarr` to the imports (near `import h5py` for now —
removed in Task 3), and add these functions immediately after `_h5_content_digest` (~line 344):

```python
def _update_zarr_node_digest(
  digest: hashlib._Hash,
  node: zarr.Group | zarr.Array,
  path: str,
) -> None:
  digest.update(path.encode("utf-8"))
  digest.update(b"\n")
  attrs_payload = {
    str(key): _normalize_json_value(value) for key, value in sorted(node.attrs.items())
  }
  digest.update(_canonical_json_bytes(attrs_payload))
  digest.update(b"\n")
  if isinstance(node, zarr.Array):
    _update_array_digest(digest, np.asarray(node[...]))
    digest.update(b"\n")
    return
  for key in sorted(node.keys()):
    child = node[key]
    _update_zarr_node_digest(digest, child, f"{path}/{key}")


def _zarr_content_digest(path: Path) -> str:
  digest = hashlib.sha256()
  root = zarr.open_group(str(path), mode="r")
  _update_zarr_node_digest(digest, root, "/")
  return digest.hexdigest()
```

Note: `_update_array_digest` and `_canonical_json_bytes` already exist and are format-agnostic
(they operate on `np.ndarray`/plain dicts) — reused unchanged.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/host/test_campaign_zarr_digest.py -v --no-cov`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/aminx/host/campaign.py tests/host/test_campaign_zarr_digest.py
git commit -m "feat(host): add _zarr_content_digest for campaign.py Zarr verification"
```

---

### Task 2: `_fsync_tree` — recursive directory-tree durability

**Files:**
- Modify: `src/aminx/host/campaign.py` (add function near `_fsync_directory`, ~line 357)
- Test: `tests/host/test_campaign_zarr_digest.py` (extend from Task 1)

**Step 1: Write the failing test**

Add to `tests/host/test_campaign_zarr_digest.py`:

```python
from unittest.mock import patch

from aminx.host.campaign import _fsync_tree


def test_fsync_tree_does_not_raise_on_real_store(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  _fsync_tree(store_path)  # should not raise


def test_fsync_tree_syncs_every_file_and_directory(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  root = zarr.open_group(str(store_path), mode="a")
  sub = root.require_group("nested")
  arr = sub.create_array(name="extra", shape=(1,), dtype="int32")
  arr[...] = np.array([1], dtype=np.int32)

  all_files = [p for p in store_path.rglob("*") if p.is_file()]
  all_dirs = [p for p in store_path.rglob("*") if p.is_dir()]
  assert all_files, "fixture should have produced at least one chunk/metadata file"

  with (
    patch("aminx.host.campaign._fsync_file") as mock_fsync_file,
    patch("aminx.host.campaign._fsync_directory") as mock_fsync_dir,
  ):
    _fsync_tree(store_path)
    synced_files = {call.args[0] for call in mock_fsync_file.call_args_list}
    synced_dirs = {call.args[0] for call in mock_fsync_dir.call_args_list}
    assert synced_files == set(all_files)
    assert synced_dirs == {*all_dirs, store_path}
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/host/test_campaign_zarr_digest.py -k fsync_tree -v --no-cov`
Expected: FAIL with `ImportError: cannot import name '_fsync_tree'`

**Step 3: Write the implementation**

In `src/aminx/host/campaign.py`, add immediately after `_fsync_directory` (~line 357):

```python
def _fsync_tree(path: Path) -> None:
  """Recursively durabilize a directory tree: every file's data, then every directory's
  entries bottom-up (deepest first), ending with `path` itself.

  Zarr stores are directories of many chunk/metadata files, unlike a single HDF5 file --
  content-digest verification is only meaningful if every file's bytes are actually on disk
  first.
  """
  for child in path.rglob("*"):
    if child.is_file():
      _fsync_file(child)
  dirs = sorted(
    (p for p in path.rglob("*") if p.is_dir()),
    key=lambda p: len(p.parts),
    reverse=True,
  )
  for d in dirs:
    _fsync_directory(d)
  _fsync_directory(path)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/host/test_campaign_zarr_digest.py -v --no-cov`
Expected: PASS (7 tests total)

**Step 5: Commit**

```bash
git add src/aminx/host/campaign.py tests/host/test_campaign_zarr_digest.py
git commit -m "feat(host): add _fsync_tree for recursive Zarr store durability"
```

---

### Task 3: Retarget `_write_done_marker`/`_validate_done_marker` to Zarr, drop whole-file hash

**Files:**
- Modify: `src/aminx/host/campaign.py`:
  - `DONE_MARKER_SCHEMA_VERSION` constant (~line 36)
  - `_validate_done_marker` (~line 585)
  - `_write_done_marker` (~line 625)
  - Remove `import h5py` (~line 22), remove `_update_h5_node_digest`/`_h5_content_digest`
    (~lines 318-344), remove `_sha256_file` (~line 284) — grep first to confirm no other callers
    before deleting (Task 1's own tests import `_zarr_content_digest` only; the only other
    `_sha256_file`/`_h5_content_digest` call site is Task 4's `run_manifest_row`, updated there)
- Test: `tests/host/test_campaign_done_marker.py` (new file)

**Step 1: Write the failing tests**

```python
"""Tests for host/campaign.py's done-marker write/validate round-trip against Zarr stores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from aminx.host.campaign import (
  DONE_MARKER_SCHEMA_VERSION,
  _done_marker_path,
  _read_done_marker,
  _validate_done_marker,
  _write_done_marker,
  _zarr_content_digest,
)


def _make_store(tmp_path: Path) -> Path:
  store_path = tmp_path / "output.zarr"
  root = zarr.open_group(str(store_path), mode="a")
  arr = root.create_array(name="data", shape=(2,), dtype="int32")
  arr[...] = np.array([1, 2], dtype=np.int32)
  return store_path


def test_write_then_validate_succeeds(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  marker_path = _done_marker_path(store_path)
  digest = _zarr_content_digest(store_path)

  _write_done_marker(
    marker_path=marker_path,
    output_h5_path=store_path,
    manifest_row_hash="hash123",
    attempt_id="attempt1",
    content_digest_sha256=digest,
    lock_backend="local_fs",
  )
  marker = _read_done_marker(marker_path)
  assert marker is not None
  assert "artifact_sha256" not in marker  # dropped -- redundant with content digest

  _validate_done_marker(
    marker=marker,
    marker_path=marker_path,
    output_h5_path=store_path,
    manifest_row_hash="hash123",
  )  # should not raise


def test_validate_rejects_schema_mismatch(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  marker_path = _done_marker_path(store_path)
  bad_marker = {
    "schema_version": "some_old_h5_era_schema",
    "manifest_row_hash": "hash123",
    "content_digest_sha256": _zarr_content_digest(store_path),
  }
  with pytest.raises(ValueError, match="schema mismatch"):
    _validate_done_marker(
      marker=bad_marker,
      marker_path=marker_path,
      output_h5_path=store_path,
      manifest_row_hash="hash123",
    )


def test_validate_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  marker_path = _done_marker_path(store_path)
  marker = {
    "schema_version": DONE_MARKER_SCHEMA_VERSION,
    "manifest_row_hash": "different_hash",
    "content_digest_sha256": _zarr_content_digest(store_path),
  }
  with pytest.raises(ValueError, match="manifest hash mismatch"):
    _validate_done_marker(
      marker=marker,
      marker_path=marker_path,
      output_h5_path=store_path,
      manifest_row_hash="hash123",
    )


def test_validate_rejects_missing_output(tmp_path: Path) -> None:
  store_path = tmp_path / "missing.zarr"
  marker_path = _done_marker_path(store_path)
  marker = {
    "schema_version": DONE_MARKER_SCHEMA_VERSION,
    "manifest_row_hash": "hash123",
    "content_digest_sha256": "irrelevant",
  }
  with pytest.raises(ValueError, match="output.*missing"):
    _validate_done_marker(
      marker=marker,
      marker_path=marker_path,
      output_h5_path=store_path,
      manifest_row_hash="hash123",
    )


def test_validate_detects_corrupted_content(tmp_path: Path) -> None:
  """Simulates post-completion corruption: array mutated after the marker was written."""
  store_path = _make_store(tmp_path)
  marker_path = _done_marker_path(store_path)
  digest = _zarr_content_digest(store_path)
  _write_done_marker(
    marker_path=marker_path,
    output_h5_path=store_path,
    manifest_row_hash="hash123",
    attempt_id="attempt1",
    content_digest_sha256=digest,
    lock_backend="local_fs",
  )
  marker = _read_done_marker(marker_path)
  assert marker is not None

  root = zarr.open_group(str(store_path), mode="a")
  root["data"][...] = np.array([999, 999], dtype=np.int32)

  with pytest.raises(ValueError, match="content digest mismatch"):
    _validate_done_marker(
      marker=marker,
      marker_path=marker_path,
      output_h5_path=store_path,
      manifest_row_hash="hash123",
    )
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/host/test_campaign_done_marker.py -v --no-cov`
Expected: FAIL — `_write_done_marker`'s current signature requires `artifact_sha256`, so the
`test_write_then_validate_succeeds` call (which omits it) fails with `TypeError`.

**Step 3: Write the implementation**

In `src/aminx/host/campaign.py`:

1. Bump the schema version (~line 36):
```python
DONE_MARKER_SCHEMA_VERSION = "campaign_done_marker_v2"
```

2. Replace `_validate_done_marker` (~line 585-622):
```python
def _validate_done_marker(
  *,
  marker: dict[str, Any],
  marker_path: Path,
  output_h5_path: Path,
  manifest_row_hash: str,
) -> None:
  if marker.get("schema_version") != DONE_MARKER_SCHEMA_VERSION:
    msg = (
      f"Done marker schema mismatch at {marker_path}: "
      f"expected {DONE_MARKER_SCHEMA_VERSION!r}, got {marker.get('schema_version')!r}."
    )
    raise ValueError(msg)
  if marker.get("manifest_row_hash") != manifest_row_hash:
    msg = (
      f"Done marker manifest hash mismatch at {marker_path}: "
      f"expected {manifest_row_hash!r}, got {marker.get('manifest_row_hash')!r}."
    )
    raise ValueError(msg)
  if not output_h5_path.exists():
    msg = f"Done marker exists at {marker_path} but output store is missing: {output_h5_path}."
    raise ValueError(msg)
  observed_content_digest = _zarr_content_digest(output_h5_path)
  expected_content_digest = marker.get("content_digest_sha256")
  if observed_content_digest != expected_content_digest:
    msg = (
      f"Done marker content digest mismatch at {marker_path}: "
      f"expected {expected_content_digest!r}, observed {observed_content_digest!r}."
    )
    raise ValueError(msg)
```

3. Replace `_write_done_marker` (~line 625-649) — drop `artifact_sha256` param and payload key:
```python
def _write_done_marker(
  *,
  marker_path: Path,
  output_h5_path: Path,
  manifest_row_hash: str,
  attempt_id: str,
  content_digest_sha256: str,
  lock_backend: str,
) -> None:
  marker_payload = {
    "schema_version": DONE_MARKER_SCHEMA_VERSION,
    "manifest_row_hash": manifest_row_hash,
    "attempt_id": attempt_id,
    "output_h5_path": str(output_h5_path.resolve()),
    "content_digest_sha256": content_digest_sha256,
    "lock_backend": lock_backend,
    "completed_at_unix_s": time.time(),
  }
  tmp_marker_path = marker_path.with_name(f"{marker_path.name}.tmp.{attempt_id}")
  tmp_marker_path.write_bytes(_canonical_json_bytes(marker_payload))
  _fsync_file(tmp_marker_path)
  tmp_marker_path.replace(marker_path)
  _fsync_directory(marker_path.parent)
```

4. Remove `import h5py` (~line 22) -- confirm first: `grep -n "h5py" src/aminx/host/campaign.py`
   should show zero remaining hits after this task's other deletions.
5. Remove `_sha256_file` (~line 284-289).
6. Remove `_update_h5_node_digest` and `_h5_content_digest` (~lines 318-344) -- Task 1's
   `_zarr_content_digest` replaces both.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/host/test_campaign_done_marker.py tests/host/test_campaign_zarr_digest.py -v --no-cov`
Expected: PASS (12 tests total)

Also run: `grep -n "h5py\|_sha256_file\|_h5_content_digest\|_update_h5_node_digest" src/aminx/host/campaign.py`
Expected: no output (all removed) except the one remaining call site fixed in Task 4.

**Step 5: Commit**

```bash
git add src/aminx/host/campaign.py tests/host/test_campaign_done_marker.py
git commit -m "refactor(host): retarget campaign.py done-marker verification to Zarr, drop whole-file hash"
```

---

### Task 4: Fix `run_manifest_row`'s write/promote/cleanup call sites

**Files:**
- Modify: `src/aminx/host/campaign.py`, inside `run_manifest_row` (~lines 914-932)
- Add `import shutil` near the top imports

**Step 1: Write the failing test**

Add to `tests/host/test_comp536_campaign_manifest.py` (matches its existing `TestIntegration`
class style) -- this is the one integration-level test for this task, mocking `sample()` since a
real model-inference call is out of scope:

```python
from unittest.mock import patch

import numpy as np
import zarr

from aminx.host.campaign import run_manifest_row


class TestRunManifestRowZarrLifecycle:
  def test_happy_path_creates_zarr_store_and_marker(self, tmp_path):
    output_path = tmp_path / "row_output.zarr"
    manifest_path = tmp_path / "manifest.json"
    row_hash = "test_row_hash_zarr_lifecycle"
    manifest = {
      "schema_version": "campaign_manifest_v1",
      "rows": [
        {
          "manifest_row_hash": row_hash,
          "job_id": "job0",
          "job_index": 0,
          "sampling_spec": {"output_h5_path": str(output_path)},
        },
      ],
    }
    manifest_path.write_text(__import__("json").dumps(manifest))

    def _fake_sample(spec):
      # Simulate the real worker: write a Zarr store at spec's (partial) output path.
      root = zarr.open_group(str(spec.output_h5_path), mode="a")
      arr = root.create_array(name="sequences", shape=(1,), dtype="int32")
      arr[...] = np.array([7], dtype=np.int32)
      return {"status": "completed"}

    with patch("aminx.host.campaign.sample", side_effect=_fake_sample):
      result = run_manifest_row(
        manifest_path=str(manifest_path),
        row_hash=row_hash,
        lock_backend="local_fs",
      )

    assert result["status"] == "completed"
    assert output_path.exists()
    assert not any(tmp_path.glob("*.partial.*"))  # cleaned up

    # Second call should short-circuit via the done marker, not re-run sample().
    with patch("aminx.host.campaign.sample", side_effect=_fake_sample) as mock_sample:
      result2 = run_manifest_row(
        manifest_path=str(manifest_path),
        row_hash=row_hash,
        lock_backend="local_fs",
      )
    assert result2["status"] == "already_done"
    mock_sample.assert_not_called()
```

Adjust the `run_manifest_row(...)` call's exact keyword arguments to match its real signature --
read the function's signature first (`grep -n "def run_manifest_row" -A 20 src/aminx/host/campaign.py`)
since this plan's summary may not have every parameter.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/host/test_comp536_campaign_manifest.py -k ZarrLifecycle -v --no-cov`
Expected: FAIL at `_fsync_file(partial_path)` with `IsADirectoryError` (can't open a directory
for reading in binary mode).

**Step 3: Write the implementation**

In `src/aminx/host/campaign.py`, inside `run_manifest_row` (~lines 914-932), replace:

```python
      if not partial_path.exists():
        msg = f"Worker did not produce expected partial output file: {partial_path}"
        raise RuntimeError(msg)
      _fsync_file(partial_path)
      artifact_sha256 = _sha256_file(partial_path)
      content_digest_sha256 = _h5_content_digest(partial_path)
      partial_path.replace(output_h5_path)
      _fsync_directory(output_h5_path.parent)
      _write_done_marker(
        marker_path=done_marker_path,
        output_h5_path=output_h5_path,
        manifest_row_hash=manifest_hash,
        attempt_id=attempt_id,
        artifact_sha256=artifact_sha256,
        content_digest_sha256=content_digest_sha256,
        lock_backend=lock_backend,
      )
    finally:
      partial_path.unlink(missing_ok=True)
```

with:

```python
      if not partial_path.exists():
        msg = f"Worker did not produce expected partial output store: {partial_path}"
        raise RuntimeError(msg)
      _fsync_tree(partial_path)
      content_digest_sha256 = _zarr_content_digest(partial_path)
      partial_path.replace(output_h5_path)
      _fsync_directory(output_h5_path.parent)
      _write_done_marker(
        marker_path=done_marker_path,
        output_h5_path=output_h5_path,
        manifest_row_hash=manifest_hash,
        attempt_id=attempt_id,
        content_digest_sha256=content_digest_sha256,
        lock_backend=lock_backend,
      )
    finally:
      if partial_path.exists():
        shutil.rmtree(partial_path)
```

Add `import shutil` to the top-level imports (alphabetical order with the existing `import sys`,
`import threading` etc.).

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/host/test_comp536_campaign_manifest.py tests/host/test_campaign_done_marker.py tests/host/test_campaign_zarr_digest.py -v --no-cov`
Expected: PASS (all tests, including the new lifecycle test)

**Step 5: Commit**

```bash
git add src/aminx/host/campaign.py tests/host/test_comp536_campaign_manifest.py
git commit -m "fix(host): fix run_manifest_row's write/promote/cleanup for Zarr directory stores"
```

---

### Task 5: Full verification sweep

**Step 1:** Run the complete new test surface:
```bash
uv run pytest tests/host/test_campaign_zarr_digest.py tests/host/test_campaign_done_marker.py tests/host/test_comp536_campaign_manifest.py tests/cli/test_campaign.py -v --no-cov
```
Expected: all pass.

**Step 2:** Static checks:
```bash
uv run ruff check src/aminx/host/campaign.py
uv run ty check src/aminx/host/campaign.py
```
Expected: no NEW findings vs. the pre-existing baseline (check via `git show HEAD:src/aminx/host/campaign.py | uv run ruff check -` for comparison if the diff isn't obviously clean).

**Step 3:** Full suite:
```bash
uv run pytest -q --no-cov
```
Expected: baseline (1413 passed as of the last full run) + this task's new tests, 0 failed.

**Step 4:** Update `CHANGELOG.md`'s `### Breaking Changes` section (the existing Zarr-migration entry) to
note campaign.py is now included, or add a new entry — campaign rows now produce Zarr stores with a
`campaign_done_marker_v2` schema; old `campaign_done_marker_v1` markers (and their HDF5 outputs) are
not migrated and will correctly fail with a schema-mismatch error if encountered.

**Step 5:** Update backlog #3182 to `completed`, describing what actually shipped vs. the original
3-option scoping (which option was chosen and why: a direct semantic-digest port, not the spot-check
optimization -- see this plan's Architecture section for the reasoning).

**Step 6:** Final commit for the CHANGELOG update, then push the branch (do not merge -- report
back to the user for the merge decision, matching this session's established pattern with PR #91).
