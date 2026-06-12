"""Checkpointing utilities for training.

Thin re-exports of the xtrax checkpoint API.  All checkpoint I/O now uses
``xtrax.checkpoint.orbax``, which saves the full :class:`ResumableState` as a
single PyTree via ``ocp.PyTreeCheckpointHandler``.

Breaking change (T-R2 clean-break): checkpoints saved with the old
``ocp.args.Composite`` format are NOT compatible with this API.  Remove any
existing checkpoint directories before resuming training with this version.
"""

from __future__ import annotations

from xtrax.checkpoint.orbax import (
    get_checkpoint_manager,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "get_checkpoint_manager",
    "load_checkpoint",
    "save_checkpoint",
]
