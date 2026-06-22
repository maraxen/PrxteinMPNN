"""Tests for portable JSON serialization and guards (RS-8)."""

from __future__ import annotations

import pytest

from aminx.run import InspectionSpecification
import aminx.run
from aminx.run.run_spec_portable_json import (
    run_spec_portable_from_dict,
    run_spec_portable_to_dict,
)
from aminx.run.spec import (
    GridLineageConfig,
    LigandConfig,
)
import equinox as eqx


class TestPortableJsonRoundtrip:
    """Test basic portable JSON serialization roundtrip."""

    @pytest.fixture
    def baseline_spec(self) -> dict:
        """Minimal valid v2 portable JSON for a plain proteinmpnn non-grid spec."""
        return {
            "version": 2,
            "io": {
                "sink_kind": "none",
                "output_dir": None,
                "manifest_path": None,
            },
            "multistate": {
                "mode": "single",
                "n_states": 1,
                "combine_strategy": "arithmetic_mean",
            },
            "resource": {
                "n_devices": 1,
                "sample_batch_size": 32,
                "structure_batch_size": 32,
                "max_buffer_size": None,
            },
            "precision": {"compute": "fp32"},
        }

    def test_portable_to_dict_roundtrips_plain_spec(
        self, baseline_spec: dict
    ) -> None:
        """Plain proteinmpnn/non-grid spec should roundtrip through portable JSON."""
        # Deserialize baseline to RunSpec
        baseline = run_spec_portable_from_dict(baseline_spec)

        # Serialize back to dict
        d = run_spec_portable_to_dict(baseline)

        # Verify version
        assert d["version"] == 2

        # Verify roundtrip: deserialize again and should succeed
        roundtripped = run_spec_portable_from_dict(d)
        assert roundtripped is not None
        assert roundtripped.ligand.model_family == "proteinmpnn"
        assert roundtripped.grid.grid_mode is False


class TestPortableJsonGuards:
    """Test RS-8 guards that prevent silent data loss."""

    @pytest.fixture
    def baseline_spec(self) -> dict:
        """Minimal valid v2 portable JSON for a plain proteinmpnn non-grid spec."""
        return {
            "version": 2,
            "io": {
                "sink_kind": "none",
                "output_dir": None,
                "manifest_path": None,
            },
            "multistate": {
                "mode": "single",
                "n_states": 1,
                "combine_strategy": "arithmetic_mean",
            },
            "resource": {
                "n_devices": 1,
                "sample_batch_size": 32,
                "structure_batch_size": 32,
                "max_buffer_size": None,
            },
            "precision": {"compute": "fp32"},
        }

    def test_portable_to_dict_raises_on_grid_mode(
        self, baseline_spec: dict
    ) -> None:
        """to_dict should raise ValueError when grid_mode=True (check first)."""
        baseline = run_spec_portable_from_dict(baseline_spec)

        # Mutate grid to grid_mode=True
        grid_mutated = eqx.tree_at(
            lambda s: s.grid,
            baseline,
            GridLineageConfig(
                grid_mode=True,
                campaign_mode=False,
                job_id=None,
                chunk_id=None,
                sample_start=None,
                sample_count=None,
            ),
        )

        # Should raise ValueError mentioning grid_mode
        with pytest.raises(ValueError, match="grid_mode"):
            run_spec_portable_to_dict(grid_mutated)

    def test_portable_to_dict_raises_on_ligandmpnn(
        self, baseline_spec: dict
    ) -> None:
        """to_dict should raise ValueError when model_family='ligandmpnn'."""
        baseline = run_spec_portable_from_dict(baseline_spec)

        # Mutate ligand to model_family='ligandmpnn'
        ligand_mutated = eqx.tree_at(
            lambda s: s.ligand,
            baseline,
            LigandConfig(
                model_family="ligandmpnn",
                use_side_chain_context=None,
                ligand_conditioning=False,
                sidechain_conditioning=False,
                context_path=None,
            ),
        )

        # Should raise ValueError mentioning ligandmpnn
        with pytest.raises(ValueError, match="ligandmpnn"):
            run_spec_portable_to_dict(ligand_mutated)


class TestInspectionSpecificationExport:
    """Test that InspectionSpecification is properly exported."""

    def test_inspection_specification_importable(self) -> None:
        """InspectionSpecification should be importable from aminx.run."""
        # Already imported at module top, but explicit check
        assert InspectionSpecification is not None

    def test_inspection_specification_in_all(self) -> None:
        """InspectionSpecification should be in aminx.run.__all__."""
        assert "InspectionSpecification" in aminx.run.__all__
