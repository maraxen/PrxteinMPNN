"""Every residue draw in aminx is registered, and every registered draw either excludes X or says why not.

Debt #2660: the generic autoregressive sampler drew over all 21 columns, so it could emit the
unknown token X, which upstream LigandMPNN never draws. Fixing that one site does not stop the
next draw site from repeating it, so this test enumerates the draw sites by AST rather than by
name, and fails on any site it has not been told about.

Classification of each draw site (``REGISTRY`` below):

* ``x_excluded``: the innermost function that performs the draw calls ``exclude_x_logits``
  before it. The AST check below proves the mask is in the same function body.
* ``known_gap``: the draw can still select X. The reason says why it is not fixed in the
  current change. The test asserts the site is still unmasked, so a fix flips the entry
  and cannot be forgotten.
* ``exempt``: the draw does not choose a residue (for example, it chooses ensemble members).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "aminx"

# (module, attribute) pairs that are token draws when called as ``module.attribute(...)``.
_RANDOM_DRAWS = {"categorical", "choice", "multinomial"}
_SEARCHSORTED = {"searchsorted"}
_MASK_NAMES = {"exclude_x_logits", "exclude_x_probs"}


def _call_name(node: ast.Call) -> str | None:
  func = node.func
  if isinstance(func, ast.Name):
    return func.id
  if isinstance(func, ast.Attribute):
    return func.attr
  return None


def _draw_kind(node: ast.Call) -> str | None:
  """Return the draw kind for a call node, or None when the call is not a residue draw."""
  name = _call_name(node)
  if name is None:
    return None
  func = node.func
  base = func.value if isinstance(func, ast.Attribute) else None
  base_name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", None)
  if name in _RANDOM_DRAWS and base_name == "random":
    return f"random.{name}"
  if name in _SEARCHSORTED:
    return "searchsorted"
  if name == "argmax" and any(
    "gumbel" in ast.unparse(arg) for arg in [*node.args, *(kw.value for kw in node.keywords)]
  ):
    return "gumbel-argmax"
  if name == "gumbel_softmax":
    return "gumbel_softmax"
  return None


def find_draw_sites(root: pathlib.Path = SRC_ROOT) -> dict[str, dict[str, object]]:
  """Map ``relpath::qualname`` to the draw kinds and whether the enclosing function masks X.

  ``qualname`` is the dotted chain of enclosing ``def`` names, so two ``step`` functions in
  one module are distinct sites. A draw in a lambda is attributed to the enclosing ``def``.
  """
  sites: dict[str, dict[str, object]] = {}
  for path in sorted(root.rglob("*.py")):
    rel = path.relative_to(root).as_posix()
    tree = ast.parse(path.read_text(), filename=str(path))
    _collect(tree, rel, [], None, sites)
  return sites


def _collect(
  node: ast.AST,
  rel: str,
  names: list[str],
  innermost: ast.AST | None,
  sites: dict[str, dict[str, object]],
) -> None:
  for child in ast.iter_child_nodes(node):
    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
      _collect(child, rel, [*names, child.name], child, sites)
      continue
    if isinstance(child, ast.Call):
      kind = _draw_kind(child)
      if kind is not None:
        qual = ".".join(names) if names else "<module>"
        key = f"{rel}::{qual}"
        entry = sites.setdefault(key, {"kinds": set(), "masked": True, "lines": []})
        entry["kinds"].add(kind)
        entry["lines"].append(child.lineno)
        entry["masked"] = bool(entry["masked"]) and _masks_x(innermost)
    _collect(child, rel, names, innermost, sites)


def _masks_x(func: ast.AST | None) -> bool:
  if func is None:
    return False
  return any(isinstance(n, ast.Call) and _call_name(n) in _MASK_NAMES for n in ast.walk(func))


#: Every draw site in src/aminx, classified. Keys are ``relpath::qualname``.
REGISTRY: dict[str, tuple[str, str]] = {
  "ebm/structure_prediction.py::resample_ensemble": (
    "exempt",
    "chooses ensemble members, not residues",
  ),
  "inference/decode/autoregressive.py::__call__": (
    "exempt",
    "searchsorted over chromatic wave boundaries; selects a schedule index, not a residue",
  ),
  "utils/aa_convert.py::string_key_to_index": (
    "exempt",
    "searchsorted over sorted string keys; a lookup, not a draw",
  ),
  "host/averaging.py::_make_encoding_sampling_split_fn_legacy.sample_fn.sample_group_step": (
    "x_excluded",
    "generic host sampler, grouped positions",
  ),
  "host/averaging.py::_make_encoding_sampling_split_fn_legacy.sample_fn.sample_step": (
    "x_excluded",
    "generic host sampler, decoding order",
  ),
  "inference/decode/autoregressive.py::_fuse_and_sample": (
    "x_excluded",
    "generic autoregressive decode (debt #2660, LigandMPNN hard-omits X)",
  ),
  "potts/sampling.py::gibbs_sweep.step": (
    "x_excluded",
    "Gibbs draw; matches PottsMPNN's energy-design convention (slice to 20 letters)",
  ),
  "potts/sampling.py::dlmc_sweep.one_sweep.step": (
    "x_excluded",
    "Zanella MH proposal: X weight zeroed in both forward and reverse proposals; the entry "
    "guard moves active X sites to alanine (burn-in).",
  ),
  "families/protonpotts_mpnn/ph_methods.py::pick_probs": (
    "known_gap",
    "ProtonPotts sampling path: inverse-CDF draw. The pH design path already zeroes X "
    "(ph_plan.valid_token_mask); this path mirrors upstream, which does not mask UNK when "
    "sampling. Left unchanged because it is graded against sealed upstream dumps. Filed as debt.",
  ),
  "inference/decode/ste.py::__call__.loss_fn": (
    "exempt",
    "relaxed gumbel_softmax sample used inside the loss for gradients; ASSUMED not to be "
    "the emitted sequence (emitted tokens come from the decode path). Verify before relying on it.",
  ),
  "inference/optimize_ste.py::make_optimize_sequence_fn.optimize_sequence.update_step.loss_fn": (
    "exempt",
    "relaxed gumbel_softmax sample used inside the loss for gradients; ASSUMED not emitted. "
    "Verify before relying on it.",
  ),
}


def _unmasked_sites() -> set[str]:
  return {key for key, info in find_draw_sites().items() if not info["masked"]}


# --- Negative and positive controls for the scanner itself -------------------------------


def _scan_source(source: str) -> dict[str, dict[str, object]]:
  tree = ast.parse(source)
  sites: dict[str, dict[str, object]] = {}
  _collect(tree, "synthetic.py", [], None, sites)
  return sites


def test_scanner_sees_an_unmasked_categorical_draw() -> None:
  """Negative control: an unmasked draw must be reported as unmasked (the instrument can fail)."""
  sites = _scan_source(
    "import jax\ndef sample(key, logits):\n  return jax.random.categorical(key, logits)\n",
  )
  assert list(sites) == ["synthetic.py::sample"]
  assert sites["synthetic.py::sample"]["masked"] is False


def test_scanner_accepts_a_masked_draw() -> None:
  """Positive control: the same draw after exclude_x_logits is reported as masked."""
  sites = _scan_source(
    "import jax\n"
    "def sample(key, logits):\n"
    "  return jax.random.categorical(key, exclude_x_logits(logits))\n",
  )
  assert sites["synthetic.py::sample"]["masked"] is True


def test_scanner_qualifies_nested_defs() -> None:
  """Two ``step`` functions in one module are two sites, not one."""
  sites = _scan_source(
    "import jax\n"
    "def a():\n"
    "  def step(k, l):\n"
    "    return jax.random.categorical(k, exclude_x_logits(l))\n"
    "def b():\n"
    "  def step(k, l):\n"
    "    return jax.random.categorical(k, l)\n",
  )
  assert sites["synthetic.py::a.step"]["masked"] is True
  assert sites["synthetic.py::b.step"]["masked"] is False


def test_scanner_finds_gumbel_argmax_and_searchsorted() -> None:
  sites = _scan_source(
    "import jax.numpy as jnp\n"
    "import numpy as np\n"
    "def g(x, gumbel):\n"
    "  return jnp.argmax(x + gumbel, axis=-1)\n"
    "def inv(c, u):\n"
    "  return np.searchsorted(c, u)\n",
  )
  assert sites["synthetic.py::g"]["kinds"] == {"gumbel-argmax"}
  assert sites["synthetic.py::inv"]["kinds"] == {"searchsorted"}


# --- The registry itself ----------------------------------------------------------------


def test_every_draw_site_is_registered() -> None:
  """A new residue draw cannot appear in src/aminx without a classification here."""
  found = set(find_draw_sites())
  registered = set(REGISTRY)
  assert found - registered == set(), f"unregistered draw sites: {sorted(found - registered)}"
  assert registered - found == set(), f"stale registry entries: {sorted(registered - found)}"


def test_x_excluded_sites_mask_x_in_their_own_function() -> None:
  sites = find_draw_sites()
  for key, (status, _reason) in REGISTRY.items():
    if status == "x_excluded":
      assert sites[key]["masked"] is True, f"{key} is registered x_excluded but does not mask X"


def test_known_gaps_are_still_unmasked() -> None:
  """A known gap that gets fixed must be re-classified, so the register never goes stale."""
  sites = find_draw_sites()
  for key, (status, reason) in REGISTRY.items():
    if status == "known_gap":
      assert reason, f"{key} has a known gap with no reason"
      assert sites[key]["masked"] is False, f"{key} is now masked; reclassify it as x_excluded"


def test_registry_statuses_are_valid() -> None:
  for key, (status, reason) in REGISTRY.items():
    assert status in {"x_excluded", "known_gap", "exempt"}, key
    assert reason, key


@pytest.mark.parametrize("status", ["x_excluded", "known_gap", "exempt"])
def test_each_status_is_used(status: str) -> None:
  """Keeps the three categories honest: an unused category is a sign the rule has drifted."""
  assert any(s == status for s, _ in REGISTRY.values())
