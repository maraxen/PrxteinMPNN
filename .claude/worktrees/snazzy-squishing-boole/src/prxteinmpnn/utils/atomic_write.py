"""Atomic replace-on-disk writes (roadmap §3.6 ``jaxbeans`` ``utils/io`` analogue).

Used where torn reads/writes would corrupt manifests or lineage metadata under concurrent jobs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
  """Write ``text`` to ``path`` via temp file + ``rename`` (same-filesystem atomic replace)."""
  dest = Path(path)
  dest.parent.mkdir(parents=True, exist_ok=True)
  data = text.encode(encoding)
  fd, tmppath = tempfile.mkstemp(
    dir=str(dest.parent),
    prefix=f".{dest.name}.",
    suffix=".tmp",
  )
  try:
    os.write(fd, data)
    os.fsync(fd)
  finally:
    os.close(fd)
  tmp_path = Path(tmppath)
  tmp_path.replace(dest)
  return dest
