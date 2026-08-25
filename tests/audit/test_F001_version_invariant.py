"""Tier B contract assertion for F001 (task_id: 260826_aminx-invariant-audit).

CLOSURE GATE (a′):
  This test MUST FAIL on the pinned wheel (aminx-0.1.0a26.dist-info, __version__="0.1.0")
  and MUST PASS on the rebuilt wheel (audit/260826-invariant-fixes branch, __version__
  dynamically derived from importlib.metadata).

Finding summary: aminx.__version__ returned the hardcoded literal "0.1.0" while
importlib.metadata.version("aminx") reported the correct dist-info version (e.g. "0.1.0a26").
The fix makes aminx.__version__ derive from importlib.metadata at import time.
"""

import importlib.metadata

import aminx


def test_version_attribute_matches_distribution_metadata() -> None:
    """aminx.__version__ must equal importlib.metadata.version('aminx').

    Tier B contract assertion.  Fails pre-fix (literal '0.1.0' != '0.1.0a26');
    passes post-fix (importlib.metadata derivation).
    """
    dist_version = importlib.metadata.version("aminx")
    # Record for closure evidence collection.
    print(f"aminx.__version__   = {aminx.__version__!r}")
    print(f"dist-info version   = {dist_version!r}")
    assert aminx.__version__ == dist_version, (
        f"aminx.__version__ ({aminx.__version__!r}) does not match "
        f"importlib.metadata.version('aminx') ({dist_version!r}). "
        "This indicates a stale hardcoded literal in aminx/__init__.py."
    )
