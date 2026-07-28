"""Tests that sidechain_conditioning implies use_side_chain_context in prep.py.

Regression guard for a silent no-op: `sidechain_conditioning` makes
_sampling_helper build and pass the atom_37 sidechain context, but
`ligand_mpnn_use_side_chain_context` is what BUILDS the model branch that
consumes it. Requesting the former while leaving the latter unset (None ->
False) fed real sidechain context to a model that structurally ignored it.
Confirmed empirically 2026-07-28: AR-decode logits were bit-identical to
sidechain_conditioning=False (max|diff|=0.0) until the ctx flag was set, at
which point the same context moved them (max|diff|=2.47).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aminx.host.prep import prep_protein_stream_and_model
from aminx.run.specs import SamplingSpecification


def _use_side_chain_context_passed_to_load_model(spec: SamplingSpecification) -> bool:
  """Run prep with load_model mocked; return the use_side_chain_context kwarg it got."""
  with patch("aminx.host.prep.create_protein_dataset") as mock_dataset, \
       patch("aminx.host.prep.load_model") as mock_load_model:
    mock_dataset.return_value = MagicMock()
    mock_load_model.return_value = MagicMock()
    try:
      prep_protein_stream_and_model(spec)
    except Exception:  # noqa: BLE001 - downstream mocking raises; we only need the call
      pass
    assert mock_load_model.called, "load_model was never called"
    return bool(mock_load_model.call_args.kwargs["use_side_chain_context"])


class TestSidechainConditioningImpliesContext:
  """sidechain_conditioning=True must build the sidechain branch."""

  def test_sidechain_conditioning_alone_implies_context(self) -> None:
    """THE REGRESSION: sc=True with the ctx flag unset must still build the branch."""
    spec = SamplingSpecification(
      inputs="dummy.pdb",
      sidechain_conditioning=True,
      # ligand_mpnn_use_side_chain_context left unset (None) -- the campaign's config
    )
    assert spec.ligand_mpnn_use_side_chain_context is None
    assert _use_side_chain_context_passed_to_load_model(spec) is True

  def test_explicit_context_flag_still_honored(self) -> None:
    """Setting the ctx flag explicitly (without sc) keeps working as before."""
    spec = SamplingSpecification(
      inputs="dummy.pdb",
      ligand_mpnn_use_side_chain_context=True,
    )
    assert _use_side_chain_context_passed_to_load_model(spec) is True

  def test_both_flags_set_is_fine(self) -> None:
    spec = SamplingSpecification(
      inputs="dummy.pdb",
      sidechain_conditioning=True,
      ligand_mpnn_use_side_chain_context=True,
    )
    assert _use_side_chain_context_passed_to_load_model(spec) is True

  def test_neither_flag_leaves_branch_off(self) -> None:
    """No sidechain request -> model built without the branch (unchanged default)."""
    spec = SamplingSpecification(inputs="dummy.pdb")
    assert _use_side_chain_context_passed_to_load_model(spec) is False

  def test_contradictory_flags_raise(self) -> None:
    """sc=True with ctx explicitly False is contradictory -- raise, don't guess."""
    spec = SamplingSpecification(
      inputs="dummy.pdb",
      sidechain_conditioning=True,
      ligand_mpnn_use_side_chain_context=False,
    )
    with patch("aminx.host.prep.create_protein_dataset") as mock_dataset, \
         patch("aminx.host.prep.load_model") as mock_load_model:
      mock_dataset.return_value = MagicMock()
      mock_load_model.return_value = MagicMock()
      with pytest.raises(ValueError, match="contradictory"):
        prep_protein_stream_and_model(spec)
      assert not mock_load_model.called, "must raise before building the model"
