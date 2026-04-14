from pathlib import Path

import pytest

from prxteinmpnn.run.specs import RunSpecification, SamplingSpecification, ScoringSpecification


def test_run_spec_instantiation_tied_modes():
    # Valid combinations
    dummy_inputs = ["dummy.pdb"]
    RunSpecification(inputs=dummy_inputs, tied_positions=None, pass_mode="intra")
    RunSpecification(inputs=dummy_inputs, tied_positions=None, pass_mode="inter")
    RunSpecification(inputs=dummy_inputs, tied_positions="auto", pass_mode="inter")
    RunSpecification(inputs=dummy_inputs, tied_positions="direct", pass_mode="inter")
    RunSpecification(inputs=dummy_inputs, tied_positions=[(0, 1)], pass_mode="inter")
    RunSpecification(inputs=dummy_inputs, tied_positions=[(0, 1)], pass_mode="intra")

def test_run_spec_validation_error():
    dummy_inputs = ["dummy.pdb"]
    # Invalid: tied_positions="auto" and pass_mode="intra"
    with pytest.raises(ValueError) as e1:
        RunSpecification(inputs=dummy_inputs, tied_positions="auto", pass_mode="intra")
    assert "pass_mode must be 'inter'" in str(e1.value)
    # Invalid: tied_positions="direct" and pass_mode="intra"
    with pytest.raises(ValueError) as e2:
        RunSpecification(inputs=dummy_inputs, tied_positions="direct", pass_mode="intra")
    assert "pass_mode must be 'inter'" in str(e2.value)


def test_run_spec_noise_normalization():
    """Test that noise fields are normalized to tuples."""
    dummy_inputs = ["dummy.pdb"]

    # Test backbone noise normalization
    spec = RunSpecification(inputs=dummy_inputs, backbone_noise=0.5)
    assert isinstance(spec.backbone_noise, tuple)
    assert spec.backbone_noise == (0.5,)

    # Test estat noise normalization
    spec = RunSpecification(inputs=dummy_inputs, estat_noise=0.5)
    assert isinstance(spec.estat_noise, tuple)
    assert spec.estat_noise == (0.5,)

    # Test vdw noise normalization
    spec = RunSpecification(inputs=dummy_inputs, vdw_noise=0.5)
    assert isinstance(spec.vdw_noise, tuple)
    assert spec.vdw_noise == (0.5,)


def test_run_spec_default_noise_modes():
    """Test default noise modes."""
    dummy_inputs = ["dummy.pdb"]
    spec = RunSpecification(inputs=dummy_inputs)
    assert spec.backbone_noise_mode == "direct"
    assert spec.estat_noise_mode == "direct"
    assert spec.vdw_noise_mode == "direct"


def test_run_spec_multistate_controls_defaults():
    """Test default multistate controls on run specs."""
    spec = RunSpecification(inputs=["dummy.pdb"])
    assert spec.tie_group_map is None
    assert spec.structure_mapping is None
    assert spec.multi_state_temperature == 1.0


def test_scoring_spec_multistate_strategy_default():
    """Test scoring spec exposes multistate strategy with safe default."""
    spec = ScoringSpecification(inputs=["dummy.pdb"], sequences_to_score=["A"])
    assert spec.multi_state_strategy == "arithmetic_mean"


def test_run_spec_model_family_fields():
    """Run specs should expose model-family selection and normalize local paths."""
    spec = RunSpecification(
        inputs=["dummy.pdb"],
        model_family="ligandmpnn",
        checkpoint_id="ligandmpnn_v_32_020_25",
        model_local_path="weights/test.eqx",
        checkpoint_registry_path="weights/registry.json",
    )
    assert spec.model_family == "ligandmpnn"
    assert spec.checkpoint_id == "ligandmpnn_v_32_020_25"
    assert spec.model_local_path == Path("weights/test.eqx")
    assert spec.checkpoint_registry_path == Path("weights/registry.json")


def test_sampling_spec_grid_mode_constraints():
    """Grid mode should reject unsupported execution combinations."""
    with pytest.raises(ValueError, match="Grid mode only supports 'temperature' sampling"):
        SamplingSpecification(
            inputs=["dummy.pdb"],
            grid_mode=True,
            sampling_strategy="straight_through",
            iterations=1,
            learning_rate=0.1,
        )

    with pytest.raises(ValueError, match="Grid mode does not support average_node_features=True"):
        SamplingSpecification(
            inputs=["dummy.pdb"],
            grid_mode=True,
            average_node_features=True,
        )


def test_sampling_spec_grid_lineage_validation():
    """Grid lineage controls should be validated and normalized."""
    with pytest.raises(ValueError, match="chunk_id must be non-negative"):
        SamplingSpecification(
            inputs=["dummy.pdb"],
            grid_mode=True,
            chunk_id=-1,
        )

    with pytest.raises(ValueError, match="sample_start must be non-negative"):
        SamplingSpecification(
            inputs=["dummy.pdb"],
            grid_mode=True,
            sample_start=-1,
        )

    with pytest.raises(ValueError, match="sample_count must be positive"):
        SamplingSpecification(
            inputs=["dummy.pdb"],
            grid_mode=True,
            sample_count=0,
        )

    spec = SamplingSpecification(
        inputs=["dummy.pdb"],
        grid_mode=True,
        job_id="  grid-job-42 ",
        chunk_id=3,
        sample_start=9,
        sample_count=4,
    )
    assert spec.job_id == "grid-job-42"
    assert spec.chunk_id == 3
    assert spec.sample_start == 9
    assert spec.sample_count == 4


def test_sampling_spec_pseudo_perplexity_requires_logits():
    """Pseudo-perplexity cannot be computed when logits are disabled."""
    with pytest.raises(ValueError, match="compute_pseudo_perplexity requires return_logits=True"):
        SamplingSpecification(
            inputs=["dummy.pdb"],
            return_logits=False,
            compute_pseudo_perplexity=True,
        )


def test_sampling_spec_campaign_mode_logits_guardrails():
    """Campaign mode should enforce explicit logits opt-in and memory budget."""
    with pytest.raises(
        ValueError,
        match="campaign_mode requires return_logits=False unless allow_logits_in_campaign=True",
    ):
        SamplingSpecification(
            inputs=["dummy.pdb"],
            campaign_mode=True,
        )

    with pytest.raises(
        ValueError,
        match="campaign_mode with return_logits=True requires logits_memory_budget_mb",
    ):
        SamplingSpecification(
            inputs=["dummy.pdb"],
            campaign_mode=True,
            return_logits=True,
            allow_logits_in_campaign=True,
        )

    spec = SamplingSpecification(
        inputs=["dummy.pdb"],
        campaign_mode=True,
        return_logits=False,
    )
    assert spec.campaign_mode is True
    assert spec.return_logits is False


def test_sampling_spec_logits_budget_must_be_positive():
    """Logits budget must be positive when provided."""
    with pytest.raises(ValueError, match="logits_memory_budget_mb must be positive"):
        SamplingSpecification(
            inputs=["dummy.pdb"],
            return_logits=True,
            allow_logits_in_campaign=True,
            logits_memory_budget_mb=0,
        )
