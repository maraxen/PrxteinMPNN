"""Negative controls for the port-wave guards (#2494, #2444) and the precision guard (#2428).

Each guard must both pass when it should and fail loudly when it should. The inner
sessions run in-process via ``pytester``; they load their own conftest, which calls the
same ``port_guards`` / ``matmul_precision`` functions the real conftests use.
"""

from __future__ import annotations

import jax
import pytest

from port_guards import check_selected_wave, unset_wave_skip_reason
from tests.matmul_precision import highest_matmul_precision

pytest_plugins = ["pytester"]

_INNER_WAVE_CONFTEST = """
import os
from port_guards import check_selected_wave

def pytest_collection_modifyitems(config, items):
    wave = os.environ.get("AMINX_PORT_WAVE")
    if wave:
        check_selected_wave(items, wave)
"""

_INNER_PRECISION_CONFTEST = """
import pytest
from tests.matmul_precision import matmul_precision_guard

@pytest.fixture(autouse=True)
def _guard():
    yield from matmul_precision_guard()
"""

_WAVE_OK = """
import pytest
pytestmark = pytest.mark.port_wave("w")

def test_real():
    assert True
"""

_WAVE_NO_ITEMS = """
import pytest
pytestmark = pytest.mark.port_wave("other")

def test_other_wave():
    assert True
"""

_WAVE_EMPTY_PARAMSET = """
import pytest
pytestmark = pytest.mark.port_wave("w")

@pytest.mark.parametrize("x", [])
def test_empty(x):
    assert x
"""


def test_selected_wave_with_real_items_passes(pytester: pytest.Pytester, monkeypatch) -> None:
  monkeypatch.setenv("AMINX_PORT_WAVE", "w")
  pytester.makeconftest(_INNER_WAVE_CONFTEST)
  pytester.makepyfile(test_wave_ok=_WAVE_OK)
  result = pytester.runpytest_inprocess()
  result.assert_outcomes(passed=1)
  assert result.ret == 0


def test_selected_empty_wave_fails(pytester: pytest.Pytester, monkeypatch) -> None:
  monkeypatch.setenv("AMINX_PORT_WAVE", "w")
  pytester.makeconftest(_INNER_WAVE_CONFTEST)
  pytester.makepyfile(test_wave_other=_WAVE_NO_ITEMS)
  result = pytester.runpytest_inprocess()
  assert result.ret == pytest.ExitCode.USAGE_ERROR
  output = result.stdout.str() + result.stderr.str()
  assert "AMINX_PORT_WAVE='w'" in output
  assert "zero test items" in output


def test_empty_parameter_set_in_selected_wave_fails(
  pytester: pytest.Pytester, monkeypatch
) -> None:
  monkeypatch.setenv("AMINX_PORT_WAVE", "w")
  pytester.makeconftest(_INNER_WAVE_CONFTEST)
  pytester.makepyfile(test_wave_empty=_WAVE_EMPTY_PARAMSET)
  result = pytester.runpytest_inprocess()
  assert result.ret == pytest.ExitCode.USAGE_ERROR
  output = result.stdout.str() + result.stderr.str()
  assert "empty parameter set" in output
  assert "AMINX_PORT_WAVE='w'" in output


def test_unselected_wave_guard_is_inert(pytester: pytest.Pytester, monkeypatch) -> None:
  # With no wave selected, a file whose items declare another wave is not an error.
  monkeypatch.delenv("AMINX_PORT_WAVE", raising=False)
  pytester.makeconftest(_INNER_WAVE_CONFTEST)
  pytester.makepyfile(test_wave_other=_WAVE_NO_ITEMS)
  result = pytester.runpytest_inprocess()
  result.assert_outcomes(passed=1)


def test_check_selected_wave_nonport_is_exempt() -> None:
  check_selected_wave([], "__nonport__")


def test_unset_wave_skips_non_default_wave() -> None:
  reason = unset_wave_skip_reason("potts_head", "port_selftest")
  assert reason is not None
  assert "set AMINX_PORT_WAVE=potts_head" in reason


def test_unset_wave_runs_default_wave() -> None:
  assert unset_wave_skip_reason("port_selftest", "port_selftest") is None


def test_precision_scoped_decorator_restores_value() -> None:
  previous = jax.config.jax_default_matmul_precision

  @highest_matmul_precision
  def body() -> str:
    return str(jax.config.jax_default_matmul_precision)

  assert body() == "highest"
  assert jax.config.jax_default_matmul_precision == previous


def test_precision_leak_is_caught_by_autouse_guard(
  pytester: pytest.Pytester, monkeypatch
) -> None:
  previous = jax.config.jax_default_matmul_precision
  pytester.makeconftest(_INNER_PRECISION_CONFTEST)
  pytester.makepyfile(
    test_leak="""
import jax

def test_leaks():
    jax.config.update("jax_default_matmul_precision", "highest")
""",
    test_clean="""
import jax
from tests.matmul_precision import highest_matmul_precision

@highest_matmul_precision
def test_scoped():
    assert jax.config.jax_default_matmul_precision == "highest"
""",
  )
  try:
    result = pytester.runpytest_inprocess()
  finally:
    jax.config.update("jax_default_matmul_precision", previous)
  result.assert_outcomes(passed=2, errors=1)
  result.stdout.fnmatch_lines(["*leaked jax_default_matmul_precision*"])
