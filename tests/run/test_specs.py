"""Comprehensive tests for RunSpecification and related spec classes."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from aminx.run.specs import (
    InspectionSpecification,
    JacobianSpecification,
    RunSpecification,
    SamplingSpecification,
    ScoringSpecification,
    _DEPRECATED_SPEC_KWARGS,
    pop_deprecated_spec_kwargs,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_pdb_path(tmp_path: Path) -> Path:
    """Create a temporary PDB file path."""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text("DUMMY PDB")
    return str(pdb_file)


@pytest.fixture
def minimal_run_spec_kwargs() -> dict:
    """Minimal kwargs to construct a RunSpecification."""
    return {
        "inputs": "test.pdb",
    }


@pytest.fixture
def minimal_scoring_kwargs(minimal_run_spec_kwargs: dict) -> dict:
    """Minimal kwargs to construct a ScoringSpecification."""
    return {
        **minimal_run_spec_kwargs,
        "sequences_to_score": ["ACDEFGHIKLMNPQRSTVWY"],
    }


@pytest.fixture
def minimal_sampling_kwargs(minimal_run_spec_kwargs: dict) -> dict:
    """Minimal kwargs to construct a SamplingSpecification."""
    return minimal_run_spec_kwargs.copy()


@pytest.fixture
def minimal_jacobian_kwargs(minimal_run_spec_kwargs: dict) -> dict:
    """Minimal kwargs to construct a JacobianSpecification."""
    return minimal_run_spec_kwargs.copy()


@pytest.fixture
def minimal_inspection_kwargs(minimal_run_spec_kwargs: dict) -> dict:
    """Minimal kwargs to construct an InspectionSpecification."""
    return minimal_run_spec_kwargs.copy()


# ============================================================================
# RunSpecification Tests: Float → Tuple Coercions
# ============================================================================


class TestRunSpecificationCoercions:
    """Tests for float-to-tuple coercions in RunSpecification.__post_init__."""

    def test_backbone_noise_float_coercion(self, minimal_run_spec_kwargs: dict) -> None:
        """Float backbone_noise should coerce to 1-tuple."""
        spec = RunSpecification(**minimal_run_spec_kwargs, backbone_noise=0.5)
        assert isinstance(spec.backbone_noise, tuple)
        assert spec.backbone_noise == (0.5,)

    def test_backbone_noise_tuple_unchanged(self, minimal_run_spec_kwargs: dict) -> None:
        """Tuple backbone_noise should remain unchanged."""
        spec = RunSpecification(**minimal_run_spec_kwargs, backbone_noise=(0.1, 0.2, 0.5))
        assert spec.backbone_noise == (0.1, 0.2, 0.5)

    def test_backbone_noise_default(self, minimal_run_spec_kwargs: dict) -> None:
        """Default backbone_noise should be (0.0,)."""
        spec = RunSpecification(**minimal_run_spec_kwargs)
        assert spec.backbone_noise == (0.0,)

    def test_estat_noise_float_coercion(self, minimal_run_spec_kwargs: dict) -> None:
        """Float estat_noise should coerce to 1-tuple."""
        spec = RunSpecification(**minimal_run_spec_kwargs, estat_noise=0.3)
        assert isinstance(spec.estat_noise, tuple)
        assert spec.estat_noise == (0.3,)

    def test_estat_noise_enables_use_electrostatics(self, minimal_run_spec_kwargs: dict) -> None:
        """Setting estat_noise should auto-enable use_electrostatics."""
        spec = RunSpecification(**minimal_run_spec_kwargs, estat_noise=0.2)
        assert spec.use_electrostatics is True

    def test_estat_noise_none_does_not_enable_use_electrostatics(
        self, minimal_run_spec_kwargs: dict
    ) -> None:
        """Setting estat_noise=None should not enable use_electrostatics."""
        spec = RunSpecification(**minimal_run_spec_kwargs, estat_noise=None)
        assert spec.use_electrostatics is False

    def test_vdw_noise_float_coercion(self, minimal_run_spec_kwargs: dict) -> None:
        """Float vdw_noise should coerce to 1-tuple."""
        spec = RunSpecification(**minimal_run_spec_kwargs, vdw_noise=0.25)
        assert isinstance(spec.vdw_noise, tuple)
        assert spec.vdw_noise == (0.25,)

    def test_vdw_noise_enables_use_vdw(self, minimal_run_spec_kwargs: dict) -> None:
        """Setting vdw_noise should auto-enable use_vdw."""
        spec = RunSpecification(**minimal_run_spec_kwargs, vdw_noise=0.1)
        assert spec.use_vdw is True

    def test_vdw_noise_none_does_not_enable_use_vdw(self, minimal_run_spec_kwargs: dict) -> None:
        """Setting vdw_noise=None should not enable use_vdw."""
        spec = RunSpecification(**minimal_run_spec_kwargs, vdw_noise=None)
        assert spec.use_vdw is False


# ============================================================================
# RunSpecification Tests: Path Coercions
# ============================================================================


class TestRunSpecificationPathCoercions:
    """Tests for str→Path coercions in RunSpecification.__post_init__."""

    def test_cache_path_str_to_path(self, minimal_run_spec_kwargs: dict) -> None:
        """String cache_path should coerce to Path."""
        spec = RunSpecification(**minimal_run_spec_kwargs, cache_path="/tmp/cache")
        assert isinstance(spec.cache_path, Path)
        assert spec.cache_path == Path("/tmp/cache")

    def test_cache_path_path_unchanged(self, minimal_run_spec_kwargs: dict) -> None:
        """Path cache_path should remain unchanged."""
        path_obj = Path("/tmp/cache")
        spec = RunSpecification(**minimal_run_spec_kwargs, cache_path=path_obj)
        assert isinstance(spec.cache_path, Path)
        assert spec.cache_path == path_obj

    def test_cache_path_none(self, minimal_run_spec_kwargs: dict) -> None:
        """None cache_path should remain None."""
        spec = RunSpecification(**minimal_run_spec_kwargs, cache_path=None)
        assert spec.cache_path is None

    def test_output_dir_str_to_path(self, minimal_run_spec_kwargs: dict) -> None:
        """String output_dir should coerce to Path."""
        spec = RunSpecification(**minimal_run_spec_kwargs, output_dir="/tmp/out")
        assert isinstance(spec.output_dir, Path)
        assert spec.output_dir == Path("/tmp/out")

    def test_model_local_path_str_to_path(self, minimal_run_spec_kwargs: dict) -> None:
        """String model_local_path should coerce to Path."""
        spec = RunSpecification(**minimal_run_spec_kwargs, model_local_path="/tmp/model")
        assert isinstance(spec.model_local_path, Path)
        assert spec.model_local_path == Path("/tmp/model")

    def test_checkpoint_registry_path_str_to_path(self, minimal_run_spec_kwargs: dict) -> None:
        """String checkpoint_registry_path should coerce to Path."""
        spec = RunSpecification(
            **minimal_run_spec_kwargs, checkpoint_registry_path="/tmp/registry"
        )
        assert isinstance(spec.checkpoint_registry_path, Path)
        assert spec.checkpoint_registry_path == Path("/tmp/registry")


# ============================================================================
# RunSpecification Tests: Validation Errors
# ============================================================================


class TestRunSpecificationValidation:
    """Tests for ValueError branches in RunSpecification.__post_init__."""

    def test_tied_positions_auto_requires_inter_pass_mode(
        self, minimal_run_spec_kwargs: dict
    ) -> None:
        """tied_positions='auto' requires pass_mode='inter'."""
        with pytest.raises(ValueError, match="If tied_positions is 'auto'"):
            RunSpecification(
                **minimal_run_spec_kwargs,
                tied_positions="auto",
                pass_mode="intra",
            )

    def test_tied_positions_direct_requires_inter_pass_mode(
        self, minimal_run_spec_kwargs: dict
    ) -> None:
        """tied_positions='direct' requires pass_mode='inter'."""
        with pytest.raises(ValueError, match="If tied_positions is 'direct'"):
            RunSpecification(
                **minimal_run_spec_kwargs,
                tied_positions="direct",
                pass_mode="intra",
            )

    def test_tied_positions_auto_with_inter_passes(self, minimal_run_spec_kwargs: dict) -> None:
        """tied_positions='auto' with pass_mode='inter' should succeed."""
        spec = RunSpecification(
            **minimal_run_spec_kwargs,
            tied_positions="auto",
            pass_mode="inter",
        )
        assert spec.tied_positions == "auto"
        assert spec.pass_mode == "inter"

    def test_tied_positions_none_with_intra_passes(self, minimal_run_spec_kwargs: dict) -> None:
        """tied_positions=None (default) with any pass_mode should succeed."""
        spec = RunSpecification(
            **minimal_run_spec_kwargs,
            tied_positions=None,
            pass_mode="intra",
        )
        assert spec.tied_positions is None
        assert spec.pass_mode == "intra"


# ============================================================================
# ScoringSpecification Tests
# ============================================================================


class TestScoringSpecification:
    """Tests for ScoringSpecification.__post_init__."""

    def test_empty_sequences_to_score_raises(self, minimal_run_spec_kwargs: dict) -> None:
        """Empty sequences_to_score should raise ValueError."""
        with pytest.raises(ValueError, match="No sequences provided for scoring"):
            ScoringSpecification(**minimal_run_spec_kwargs, sequences_to_score=[])

    def test_empty_tuple_sequences_to_score_raises(self, minimal_run_spec_kwargs: dict) -> None:
        """Empty tuple sequences_to_score should raise ValueError."""
        with pytest.raises(ValueError, match="No sequences provided for scoring"):
            ScoringSpecification(**minimal_run_spec_kwargs, sequences_to_score=())

    def test_valid_sequences_to_score(self, minimal_scoring_kwargs: dict) -> None:
        """Valid sequences_to_score should succeed."""
        spec = ScoringSpecification(**minimal_scoring_kwargs)
        assert len(spec.sequences_to_score) == 1
        assert "ACDEFGHIKLMNPQRSTVWY" in spec.sequences_to_score

    def test_output_h5_path_str_coercion(self, minimal_scoring_kwargs: dict) -> None:
        """String output_h5_path should coerce to Path."""
        spec = ScoringSpecification(**minimal_scoring_kwargs, output_h5_path="/tmp/out.h5")
        assert isinstance(spec.output_h5_path, Path)
        assert spec.output_h5_path == Path("/tmp/out.h5")

    def test_multiple_sequences(self, minimal_run_spec_kwargs: dict) -> None:
        """Multiple sequences should succeed."""
        sequences = ["ACDEFGHIKLMNPQRSTVWY", "MKVLIVTGAGSDEDGLIAA"]
        spec = ScoringSpecification(**minimal_run_spec_kwargs, sequences_to_score=sequences)
        assert len(spec.sequences_to_score) == 2


# ============================================================================
# SamplingSpecification Tests: Validation
# ============================================================================


class TestSamplingSpecificationValidation:
    """Tests for ValueError branches in SamplingSpecification.__post_init__."""

    def test_temperature_float_coercion(self, minimal_sampling_kwargs: dict) -> None:
        """Float temperature should coerce to 1-tuple."""
        spec = SamplingSpecification(**minimal_sampling_kwargs, temperature=0.5)
        assert isinstance(spec.temperature, tuple)
        assert spec.temperature == (0.5,)

    def test_temperature_tuple_unchanged(self, minimal_sampling_kwargs: dict) -> None:
        """Tuple temperature should remain unchanged."""
        spec = SamplingSpecification(**minimal_sampling_kwargs, temperature=(0.1, 0.5, 1.0))
        assert spec.temperature == (0.1, 0.5, 1.0)

    def test_straight_through_requires_iterations(self, minimal_sampling_kwargs: dict) -> None:
        """straight_through sampling requires iterations."""
        with pytest.raises(ValueError, match="straight_through.*iterations.*learning_rate"):
            SamplingSpecification(
                **minimal_sampling_kwargs,
                sampling_strategy="straight_through",
                iterations=None,
                learning_rate=0.01,
            )

    def test_straight_through_requires_learning_rate(self, minimal_sampling_kwargs: dict) -> None:
        """straight_through sampling requires learning_rate."""
        with pytest.raises(ValueError, match="straight_through.*iterations.*learning_rate"):
            SamplingSpecification(
                **minimal_sampling_kwargs,
                sampling_strategy="straight_through",
                iterations=100,
                learning_rate=None,
            )

    def test_straight_through_with_both_params_passes(self, minimal_sampling_kwargs: dict) -> None:
        """straight_through with iterations and learning_rate should succeed."""
        spec = SamplingSpecification(
            **minimal_sampling_kwargs,
            sampling_strategy="straight_through",
            iterations=100,
            learning_rate=0.01,
        )
        assert spec.sampling_strategy == "straight_through"

    def test_grid_mode_requires_temperature_sampling(self, minimal_sampling_kwargs: dict) -> None:
        """grid_mode only supports temperature sampling."""
        with pytest.raises(ValueError, match="Grid mode only supports.*temperature"):
            SamplingSpecification(
                **minimal_sampling_kwargs,
                grid_mode=True,
                sampling_strategy="straight_through",
                iterations=10,
                learning_rate=0.01,
            )

    def test_grid_mode_forbids_average_node_features(self, minimal_sampling_kwargs: dict) -> None:
        """grid_mode does not support average_node_features=True."""
        with pytest.raises(ValueError, match="Grid mode does not support average_node_features"):
            SamplingSpecification(
                **minimal_sampling_kwargs,
                grid_mode=True,
                average_node_features=True,
            )

    def test_pseudo_perplexity_requires_return_logits(self, minimal_sampling_kwargs: dict) -> None:
        """compute_pseudo_perplexity requires return_logits=True."""
        with pytest.raises(ValueError, match="compute_pseudo_perplexity requires return_logits"):
            SamplingSpecification(
                **minimal_sampling_kwargs,
                return_logits=False,
                compute_pseudo_perplexity=True,
            )

    def test_logits_memory_budget_must_be_positive(self, minimal_sampling_kwargs: dict) -> None:
        """logits_memory_budget_mb must be positive when provided."""
        with pytest.raises(ValueError, match="logits_memory_budget_mb must be positive"):
            SamplingSpecification(
                **minimal_sampling_kwargs,
                logits_memory_budget_mb=0,
            )

    def test_logits_memory_budget_negative_raises(self, minimal_sampling_kwargs: dict) -> None:
        """Negative logits_memory_budget_mb should raise ValueError."""
        with pytest.raises(ValueError, match="logits_memory_budget_mb must be positive"):
            SamplingSpecification(
                **minimal_sampling_kwargs,
                logits_memory_budget_mb=-100,
            )

    def test_logits_memory_budget_coercion(self, minimal_sampling_kwargs: dict) -> None:
        """logits_memory_budget_mb should be coerced to int."""
        spec = SamplingSpecification(
            **minimal_sampling_kwargs,
            logits_memory_budget_mb=500.7,
        )
        assert isinstance(spec.logits_memory_budget_mb, int)
        assert spec.logits_memory_budget_mb == 500

    def test_campaign_mode_forbids_return_logits_unless_allowed(
        self, minimal_sampling_kwargs: dict
    ) -> None:
        """campaign_mode requires return_logits=False unless allow_logits_in_campaign=True."""
        with pytest.raises(ValueError, match="campaign_mode requires return_logits=False"):
            SamplingSpecification(
                **minimal_sampling_kwargs,
                campaign_mode=True,
                return_logits=True,
                allow_logits_in_campaign=False,
            )

    def test_campaign_mode_with_logits_requires_memory_budget(
        self, minimal_sampling_kwargs: dict
    ) -> None:
        """campaign_mode with return_logits requires logits_memory_budget_mb."""
        with pytest.raises(
            ValueError, match="campaign_mode with return_logits=True requires logits_memory_budget_mb"
        ):
            SamplingSpecification(
                **minimal_sampling_kwargs,
                campaign_mode=True,
                return_logits=True,
                allow_logits_in_campaign=True,
                logits_memory_budget_mb=None,
            )

    def test_campaign_mode_without_logits_succeeds(self, minimal_sampling_kwargs: dict) -> None:
        """campaign_mode with return_logits=False should succeed."""
        spec = SamplingSpecification(
            **minimal_sampling_kwargs,
            campaign_mode=True,
            return_logits=False,
        )
        assert spec.campaign_mode is True

    def test_campaign_mode_with_logits_and_budget_succeeds(
        self, minimal_sampling_kwargs: dict
    ) -> None:
        """campaign_mode with return_logits=True and budget should succeed."""
        spec = SamplingSpecification(
            **minimal_sampling_kwargs,
            campaign_mode=True,
            return_logits=True,
            allow_logits_in_campaign=True,
            logits_memory_budget_mb=1000,
        )
        assert spec.campaign_mode is True
        assert spec.return_logits is True


# ============================================================================
# SamplingSpecification Tests: grid_mode, job_id, chunk_id, sample_*
# ============================================================================


class TestSamplingGridModeValidation:
    """Tests for grid_mode and related field validation."""

    def test_job_id_whitespace_stripping(self, minimal_sampling_kwargs: dict) -> None:
        """job_id with whitespace should be stripped."""
        spec = SamplingSpecification(
            **minimal_sampling_kwargs,
            job_id="  test_job  ",
        )
        assert spec.job_id == "test_job"

    def test_job_id_empty_after_stripping_becomes_none(self, minimal_sampling_kwargs: dict) -> None:
        """job_id that's only whitespace should become None."""
        spec = SamplingSpecification(
            **minimal_sampling_kwargs,
            job_id="   ",
        )
        assert spec.job_id is None

    def test_grid_mode_requires_non_empty_job_id(self, minimal_sampling_kwargs: dict) -> None:
        """grid_mode requires non-empty job_id."""
        with pytest.raises(ValueError, match="job_id cannot be empty when grid_mode=True"):
            SamplingSpecification(
                **minimal_sampling_kwargs,
                grid_mode=True,
                job_id="",
            )

    def test_chunk_id_coercion_to_int(self, minimal_sampling_kwargs: dict) -> None:
        """chunk_id should be coerced to int."""
        spec = SamplingSpecification(
            **minimal_sampling_kwargs,
            chunk_id=5.9,
        )
        assert isinstance(spec.chunk_id, int)
        assert spec.chunk_id == 5

    def test_chunk_id_negative_with_grid_mode_raises(self, minimal_sampling_kwargs: dict) -> None:
        """Negative chunk_id with grid_mode should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_id must be non-negative when grid_mode=True"):
            SamplingSpecification(
                **minimal_sampling_kwargs,
                grid_mode=True,
                chunk_id=-1,
            )

    def test_sample_start_coercion_to_int(self, minimal_sampling_kwargs: dict) -> None:
        """sample_start should be coerced to int."""
        spec = SamplingSpecification(
            **minimal_sampling_kwargs,
            sample_start=42.7,
        )
        assert isinstance(spec.sample_start, int)
        assert spec.sample_start == 42

    def test_sample_start_negative_with_grid_mode_raises(self, minimal_sampling_kwargs: dict) -> None:
        """Negative sample_start with grid_mode should raise ValueError."""
        with pytest.raises(ValueError, match="sample_start must be non-negative when grid_mode=True"):
            SamplingSpecification(
                **minimal_sampling_kwargs,
                grid_mode=True,
                sample_start=-1,
            )

    def test_sample_count_coercion_to_int(self, minimal_sampling_kwargs: dict) -> None:
        """sample_count should be coerced to int."""
        spec = SamplingSpecification(
            **minimal_sampling_kwargs,
            sample_count=100.3,
        )
        assert isinstance(spec.sample_count, int)
        assert spec.sample_count == 100

    def test_sample_count_zero_with_grid_mode_raises(self, minimal_sampling_kwargs: dict) -> None:
        """Zero sample_count with grid_mode should raise ValueError."""
        with pytest.raises(ValueError, match="sample_count must be positive when grid_mode=True"):
            SamplingSpecification(
                **minimal_sampling_kwargs,
                grid_mode=True,
                sample_count=0,
            )

    def test_sample_count_negative_with_grid_mode_raises(self, minimal_sampling_kwargs: dict) -> None:
        """Negative sample_count with grid_mode should raise ValueError."""
        with pytest.raises(ValueError, match="sample_count must be positive when grid_mode=True"):
            SamplingSpecification(
                **minimal_sampling_kwargs,
                grid_mode=True,
                sample_count=-10,
            )

    def test_output_h5_path_str_coercion(self, minimal_sampling_kwargs: dict) -> None:
        """String output_h5_path should coerce to Path."""
        spec = SamplingSpecification(
            **minimal_sampling_kwargs,
            output_h5_path="/tmp/samples.h5",
        )
        assert isinstance(spec.output_h5_path, Path)
        assert spec.output_h5_path == Path("/tmp/samples.h5")

    def test_ligand_context_path_str_coercion(self, minimal_sampling_kwargs: dict) -> None:
        """String ligand_context_path should coerce to Path."""
        spec = SamplingSpecification(
            **minimal_sampling_kwargs,
            ligand_context_path="/tmp/ligand.pdb",
        )
        assert isinstance(spec.ligand_context_path, Path)
        assert spec.ligand_context_path == Path("/tmp/ligand.pdb")


# ============================================================================
# JacobianSpecification Tests
# ============================================================================


class TestJacobianSpecification:
    """Tests for JacobianSpecification.__post_init__."""

    def test_output_h5_path_str_coercion(self, minimal_jacobian_kwargs: dict) -> None:
        """String output_h5_path should coerce to Path."""
        spec = JacobianSpecification(
            **minimal_jacobian_kwargs,
            output_h5_path="/tmp/jacobian.h5",
        )
        assert isinstance(spec.output_h5_path, Path)
        assert spec.output_h5_path == Path("/tmp/jacobian.h5")

    def test_jacobian_spec_basic_construction(self, minimal_jacobian_kwargs: dict) -> None:
        """Basic JacobianSpecification should construct successfully."""
        spec = JacobianSpecification(**minimal_jacobian_kwargs)
        assert spec.inputs == minimal_jacobian_kwargs["inputs"]

    def test_jacobian_mode_default_categorical(self, minimal_jacobian_kwargs: dict) -> None:
        spec = JacobianSpecification(**minimal_jacobian_kwargs)
        assert spec.jacobian_mode == "categorical"

    def test_jacobian_mode_reverse(self, minimal_jacobian_kwargs: dict) -> None:
        spec = JacobianSpecification(**minimal_jacobian_kwargs, jacobian_mode="reverse")
        assert spec.jacobian_mode == "reverse"


# ============================================================================
# InspectionSpecification Tests
# ============================================================================


class TestInspectionSpecification:
    """Tests for InspectionSpecification.__post_init__."""

    def test_output_h5_path_str_coercion(self, minimal_inspection_kwargs: dict) -> None:
        """String output_h5_path should coerce to Path."""
        spec = InspectionSpecification(
            **minimal_inspection_kwargs,
            output_h5_path="/tmp/inspection.h5",
        )
        assert isinstance(spec.output_h5_path, Path)
        assert spec.output_h5_path == Path("/tmp/inspection.h5")

    def test_cross_input_similarity_list_single_raises(self) -> None:
        """cross_input_similarity with list containing 1 input should raise ValueError."""
        with pytest.raises(
            ValueError, match="Cross-input similarity requires at least 2 input structures"
        ):
            InspectionSpecification(
                inputs=["test1.pdb"],
                cross_input_similarity=True,
            )

    def test_cross_input_similarity_multiple_inputs_succeeds(self) -> None:
        """cross_input_similarity with 2+ inputs should succeed."""
        spec = InspectionSpecification(
            inputs=["test1.pdb", "test2.pdb"],
            cross_input_similarity=True,
        )
        assert spec.cross_input_similarity is True

    def test_inspection_spec_basic_construction(self, minimal_inspection_kwargs: dict) -> None:
        """Basic InspectionSpecification should construct successfully."""
        spec = InspectionSpecification(**minimal_inspection_kwargs)
        assert spec.inputs == minimal_inspection_kwargs["inputs"]


# ============================================================================
# Deprecated Kwargs Tests
# ============================================================================


class TestDeprecatedKwargs:
    """Tests for deprecated kwarg handling via register_spec decorator."""

    def test_output_path_deprecated_warning(self, minimal_run_spec_kwargs: dict) -> None:
        """output_path kwarg should emit DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="output_path.*deprecated and ignored"):
            RunSpecification(**minimal_run_spec_kwargs, output_path="/tmp/out")

    def test_score_batch_size_deprecated_warning(self, minimal_run_spec_kwargs: dict) -> None:
        """score_batch_size kwarg should emit DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="score_batch_size.*deprecated and ignored"):
            RunSpecification(**minimal_run_spec_kwargs, score_batch_size=32)

    def test_average_logits_deprecated_warning(self, minimal_run_spec_kwargs: dict) -> None:
        """average_logits kwarg should emit DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="average_logits.*deprecated and ignored"):
            RunSpecification(**minimal_run_spec_kwargs, average_logits=True)

    def test_combine_noise_batch_size_deprecated_warning(
        self, minimal_run_spec_kwargs: dict
    ) -> None:
        """combine_noise_batch_size kwarg should emit DeprecationWarning."""
        with pytest.warns(
            DeprecationWarning, match="combine_noise_batch_size.*deprecated and ignored"
        ):
            RunSpecification(**minimal_run_spec_kwargs, combine_noise_batch_size=16)

    def test_gmm_min_iters_deprecated_warning(self, minimal_run_spec_kwargs: dict) -> None:
        """gmm_min_iters kwarg should emit DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="gmm_min_iters.*deprecated and ignored"):
            RunSpecification(**minimal_run_spec_kwargs, gmm_min_iters=10)

    def test_all_deprecated_kwargs_together(self, minimal_run_spec_kwargs: dict) -> None:
        """All deprecated kwargs should emit warnings."""
        with pytest.warns(DeprecationWarning) as record:
            RunSpecification(
                **minimal_run_spec_kwargs,
                output_path="/tmp/out",
                score_batch_size=32,
                average_logits=True,
                combine_noise_batch_size=16,
                gmm_min_iters=10,
            )
        assert len(record) == 5

    def test_deprecated_kwargs_ignored_not_stored(self, minimal_run_spec_kwargs: dict) -> None:
        """Deprecated kwargs should be silently ignored and not stored."""
        with pytest.warns(DeprecationWarning):
            spec = RunSpecification(
                **minimal_run_spec_kwargs,
                output_path="/tmp/out",
                score_batch_size=999,
            )
        # Verify deprecated values don't overwrite existing attributes
        assert not hasattr(spec, "output_path")
        assert not hasattr(spec, "score_batch_size")

    def test_pop_deprecated_spec_kwargs_function(self) -> None:
        """pop_deprecated_spec_kwargs should remove all deprecated keys."""
        kwargs: MutableMapping[str, object] = {
            "output_path": "/tmp/out",
            "batch_size": 32,
            "other_field": "value",
        }
        with pytest.warns(DeprecationWarning, match="output_path"):
            pop_deprecated_spec_kwargs(kwargs)
        assert "output_path" not in kwargs
        assert "batch_size" in kwargs
        assert "other_field" in kwargs


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_multiple_noise_types_with_defaults(self, minimal_run_spec_kwargs: dict) -> None:
        """Setting multiple noise types should work independently."""
        spec = RunSpecification(
            **minimal_run_spec_kwargs,
            backbone_noise=0.1,
            estat_noise=0.2,
            vdw_noise=0.3,
        )
        assert spec.backbone_noise == (0.1,)
        assert spec.estat_noise == (0.2,)
        assert spec.vdw_noise == (0.3,)
        assert spec.use_electrostatics is True
        assert spec.use_vdw is True

    def test_chain_id_as_string(self, minimal_run_spec_kwargs: dict) -> None:
        """chain_id as string should be accepted."""
        spec = RunSpecification(**minimal_run_spec_kwargs, chain_id="A")
        assert spec.chain_id == "A"

    def test_chain_id_as_sequence(self, minimal_run_spec_kwargs: dict) -> None:
        """chain_id as sequence should be accepted."""
        spec = RunSpecification(**minimal_run_spec_kwargs, chain_id=["A", "B"])
        assert spec.chain_id == ["A", "B"]

    def test_run_spec_sync_called(self, minimal_run_spec_kwargs: dict) -> None:
        """__post_init__ should sync run_spec field."""
        spec = RunSpecification(**minimal_run_spec_kwargs)
        assert hasattr(spec, "run_spec")
        assert spec.run_spec is not None

    def test_sampling_spec_inherits_run_spec_features(self, minimal_sampling_kwargs: dict) -> None:
        """SamplingSpecification should inherit RunSpecification coercions."""
        spec = SamplingSpecification(
            **minimal_sampling_kwargs,
            backbone_noise=0.5,
            cache_path="/tmp/cache",
        )
        assert spec.backbone_noise == (0.5,)
        assert isinstance(spec.cache_path, Path)

    def test_scoring_spec_inherits_run_spec_features(self, minimal_scoring_kwargs: dict) -> None:
        """ScoringSpecification should inherit RunSpecification coercions."""
        spec = ScoringSpecification(
            **minimal_scoring_kwargs,
            vdw_noise=0.2,
            output_dir="/tmp/out",
        )
        assert spec.vdw_noise == (0.2,)
        assert isinstance(spec.output_dir, Path)
        assert spec.use_vdw is True
