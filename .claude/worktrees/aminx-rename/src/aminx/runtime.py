"""Runtime helpers for host process configuration."""

from __future__ import annotations

import multiprocessing


def configure_multiprocessing(method: str = "spawn", *, force: bool = True) -> None:
  """Configure the multiprocessing start method (opt-in).

  Must be called once per interpreter process before spawning workers—for example
  from notebooks, standalone scripts, or CLI entrypoints that use subprocess pools.

  Importing ``aminx`` does **not** call this; invoke it explicitly when needed.
  """
  multiprocessing.set_start_method(method, force=force)
