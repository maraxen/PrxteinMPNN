"""Atomic replace writes used by campaign manifests."""

from __future__ import annotations

from pathlib import Path

from prxteinmpnn.utils.atomic_write import atomic_write_text


def test_atomic_write_text_roundtrip(tmp_path: Path) -> None:
  path = tmp_path / "nested" / "out.txt"
  atomic_write_text(path, "hello\n")
  assert path.read_text(encoding="utf-8") == "hello\n"
