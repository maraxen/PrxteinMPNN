"""Wave-selection guards for the port tier (debts #2494 and #2444).

Used by ``tests/conftest.py`` and ``tests/port/conftest.py`` and exercised directly by
``tests/test_port_guards_controls.py``. Pure functions only: no environment reads here.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

NONPORT_WAVE = "__nonport__"
EMPTY_PARAMSET_PREFIX = "got empty parameter set"


def declared_wave(item: pytest.Item) -> str | None:
  """Return the wave named by the item's ``port_wave`` marker, or None."""
  mark = item.get_closest_marker("port_wave")
  if mark is None or not mark.args:
    return None
  return str(mark.args[0])


def _is_empty_paramset(item: pytest.Item) -> bool:
  for mark in item.iter_markers("skip"):
    reason = str(mark.kwargs.get("reason", ""))
    if reason.startswith(EMPTY_PARAMSET_PREFIX):
      return True
  return False


def check_selected_wave(selected: Sequence[pytest.Item], wave: str) -> None:
  """Fail the session when an explicitly selected wave collected nothing real.

  Called only when ``AMINX_PORT_WAVE`` is set. Raises ``pytest.UsageError``
  (exit code 4) naming the wave when:
  - no selected item declares ``port_wave(wave)``, or
  - an item of that wave was generated from an empty parameter set (pytest turns
    that into a skip, which would otherwise read as green).
  """
  if wave == NONPORT_WAVE:
    return
  wave_items = [item for item in selected if declared_wave(item) == wave]
  if not wave_items:
    msg = (
      f"AMINX_PORT_WAVE={wave!r} collected zero test items for that wave; "
      "refusing a silent green run (check the wave's test module and marker)"
    )
    raise pytest.UsageError(msg)
  empty = [item.nodeid for item in wave_items if _is_empty_paramset(item)]
  if empty:
    preview = ", ".join(empty[:5])
    msg = (
      f"AMINX_PORT_WAVE={wave!r}: {len(empty)} parametrised test(s) were generated "
      f"from an empty parameter set and would pass vacuously: {preview}"
    )
    raise pytest.UsageError(msg)


def unset_wave_skip_reason(item_wave: str | None, default_wave: str) -> str | None:
  """Skip reason for a port item when ``AMINX_PORT_WAVE`` is unset.

  Only the pyproject default wave (``[tool.port] target``) may run without the
  variable. Any other wave is skipped with a reason, never run against the
  default oracle.
  """
  if item_wave == default_wave:
    return None
  return (
    f"set AMINX_PORT_WAVE={item_wave} to run this wave "
    f"(AMINX_PORT_WAVE unset only runs the default wave {default_wave!r})"
  )
