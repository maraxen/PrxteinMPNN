"""FA2/FA3 (audit 260826_chain-selection-vendor-superset-audit): score()/jacobian() must not
silently ignore spec.fixed_mask.

A differential probe (.praxia/audit_chain_vendor/evidence/F_A2_fixed_mask_score_jacobian_probe.py)
confirmed runner.score() and runner.jacobian() produced bit-identical output regardless of
fixed_mask -- the field was silently accepted and had zero effect. Neither surface has an
established semantics for what fixed_mask should mean there (score() already scores every
position under full context by design; jacobian's kernels have no fixed_mask parameter at all),
so the fix makes the gap loud (NotImplementedError naming the field) rather than guessing at
correctness-sensitive behavior in either kernel.

The guard fires before any model/weights are loaded (spec construction only), so these tests
need no real PDB/model fixture -- a placeholder input path is enough since the raise happens
first.
"""

from __future__ import annotations

import numpy as np
import pytest

from aminx.host.runner import jacobian, score
from aminx.run.batch_mapping import MappedBy
from aminx.run.specs import JacobianSpecification, ScoringSpecification

_PLACEHOLDER_PDB = "does/not/need/to/exist.pdb"
_N_RESIDUES = 8


def test_score_raises_when_fixed_mask_has_fixed_positions():
    fixed_mask = np.zeros((_N_RESIDUES,), dtype=np.float32)
    fixed_mask[0] = 1.0
    spec = ScoringSpecification(
        inputs=[_PLACEHOLDER_PDB],
        sequences_to_score=["A" * _N_RESIDUES],
        max_length=_N_RESIDUES,
        fixed_mask=fixed_mask,
    )
    with pytest.raises(NotImplementedError, match="fixed_mask"):
        score(spec)


def _assert_does_not_hit_fixed_mask_guard(callable_) -> None:
    """No real PDB/model in this unit test, so the call may raise something else (missing
    input file) past the guard, or -- with a placeholder path proxide can't categorize --
    may simply return an empty result without raising at all. Either outcome is fine; only
    our fixed_mask NotImplementedError is a failure here.
    """
    try:
        callable_()
    except NotImplementedError as exc:
        assert "fixed_mask" not in str(exc)
    except Exception:  # noqa: BLE001 - any other failure mode is out of scope for this guard test
        pass


def test_score_does_not_raise_the_fixed_mask_guard_when_fixed_mask_is_none():
    spec = ScoringSpecification(
        inputs=[_PLACEHOLDER_PDB],
        sequences_to_score=["A" * _N_RESIDUES],
        max_length=_N_RESIDUES,
    )
    _assert_does_not_hit_fixed_mask_guard(lambda: score(spec))


def test_score_does_not_raise_the_fixed_mask_guard_when_fixed_mask_is_all_zero():
    spec = ScoringSpecification(
        inputs=[_PLACEHOLDER_PDB],
        sequences_to_score=["A" * _N_RESIDUES],
        max_length=_N_RESIDUES,
        fixed_mask=np.zeros((_N_RESIDUES,), dtype=np.float32),
    )
    _assert_does_not_hit_fixed_mask_guard(lambda: score(spec))


def test_jacobian_raises_when_fixed_mask_has_fixed_positions():
    fixed_mask = np.zeros((_N_RESIDUES,), dtype=np.float32)
    fixed_mask[0] = 1.0
    spec = JacobianSpecification(
        inputs=[_PLACEHOLDER_PDB],
        max_length=_N_RESIDUES,
        fixed_mask=fixed_mask,
    )
    with pytest.raises(NotImplementedError, match="fixed_mask"):
        jacobian(spec)


def test_jacobian_does_not_raise_the_fixed_mask_guard_when_fixed_mask_is_none():
    spec = JacobianSpecification(
        inputs=[_PLACEHOLDER_PDB],
        max_length=_N_RESIDUES,
    )
    _assert_does_not_hit_fixed_mask_guard(lambda: jacobian(spec))


def test_score_raises_when_fixed_mask_is_a_mapped_by():
    """A MappedBy fixed_mask must still hit the FA2 guard, not silently bypass it (G1 spec's
    explicit interaction requirement between G1 and FA2/FA3).
    """
    spec = ScoringSpecification(
        inputs=[_PLACEHOLDER_PDB],
        sequences_to_score=["A" * _N_RESIDUES],
        max_length=_N_RESIDUES,
        fixed_mask=MappedBy(by="path", mapping={"exist": np.ones((_N_RESIDUES,))}),
    )
    with pytest.raises(NotImplementedError, match="fixed_mask"):
        score(spec)


def test_jacobian_raises_when_fixed_mask_is_a_mapped_by():
    spec = JacobianSpecification(
        inputs=[_PLACEHOLDER_PDB],
        max_length=_N_RESIDUES,
        fixed_mask=MappedBy(by="path", mapping={"exist": np.ones((_N_RESIDUES,))}),
    )
    with pytest.raises(NotImplementedError, match="fixed_mask"):
        jacobian(spec)
