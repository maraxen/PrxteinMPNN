"""Structural protocols for host-side campaign orchestration.

Kept separate from ``types.protocols`` (model/inference-pipeline protocols)
since these describe host infrastructure interfaces (e.g. distributed
locking for scheduler-agnostic campaign manifests) rather than callable
single-``__call__`` shapes.
"""

from __future__ import annotations

from typing import Protocol


class DistributedLockBackend(Protocol):
  """Distributed lock backend interface used by campaign workers."""

  def acquire(self, *, lock_key: str, owner_token: str, lease_seconds: int) -> None:
    """Acquire a distributed lock key for an owner token."""

  def heartbeat(self, *, lock_key: str, owner_token: str, lease_seconds: int) -> None:
    """Refresh lease ownership for an active lock."""

  def release(self, *, lock_key: str, owner_token: str) -> None:
    """Release an owned lock key."""


__all__ = ["DistributedLockBackend"]
