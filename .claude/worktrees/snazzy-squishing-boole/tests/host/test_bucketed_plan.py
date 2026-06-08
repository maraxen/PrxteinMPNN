"""Tests for host.plan.plan_bucketed function."""

import pytest

from prxteinmpnn.host.plan import plan_bucketed
from prxteinmpnn.run.specs import SamplingSpecification
from prxteinmpnn.tiling.axes import (
    N_LIGAND_ATOMS,
    N_NOISES,
    N_RESIDUES,
    N_SAMPLES,
    N_STRUCTURES,
    N_TEMPERATURES,
)
from prxteinmpnn.tiling.bucketing import BucketingConfig
from prxteinmpnn.tiling.planner import BatchPlanner, estimate_memory_theoretical


class TestPlanBucketed:
    """Test plan_bucketed function."""

    @pytest.fixture
    def default_spec(self) -> SamplingSpecification:
        """Default SamplingSpecification for testing."""
        return SamplingSpecification(inputs=[])

    @pytest.fixture
    def default_planner(self) -> BatchPlanner:
        """Default BatchPlanner for testing."""
        return BatchPlanner(
            axes=[N_RESIDUES, N_LIGAND_ATOMS, N_STRUCTURES, N_SAMPLES, N_TEMPERATURES, N_NOISES],
            budget_bytes=8e9,  # 8 GB
            estimate_memory=lambda ds: estimate_memory_theoretical(ds, 1.0, 2.5),
        )

    def test_three_sequences_two_buckets(
        self, default_spec: SamplingSpecification, default_planner: BatchPlanner
    ) -> None:
        """Three sequences in two buckets -> two entries in per_bucket_plans."""
        seq_lens = [50, 100, 110]  # 50 -> 64, 100 -> 128, 110 -> 128
        config = BucketingConfig(buckets=(64, 128, 256, 512))

        result = plan_bucketed(default_spec, seq_lens, default_planner, config)

        # Should have two buckets used
        assert set(result.bucket_groups.keys()) == {64, 128}
        assert len(result.per_bucket_plans) == 2
        assert set(result.bucket_boundaries) == {64, 128}

    def test_bucket_boundaries_sorted(
        self, default_spec: SamplingSpecification, default_planner: BatchPlanner
    ) -> None:
        """bucket_boundaries is sorted and matches used buckets."""
        seq_lens = [50, 100, 150]  # 50 -> 64, 100 -> 128, 150 -> 256
        config = BucketingConfig(buckets=(64, 128, 256, 512))

        result = plan_bucketed(default_spec, seq_lens, default_planner, config)

        assert result.bucket_boundaries == (64, 128, 256)

    def test_indices_match_input_positions(
        self, default_spec: SamplingSpecification, default_planner: BatchPlanner
    ) -> None:
        """Indices in bucket_groups match original input positions."""
        seq_lens = [50, 100, 75]  # 50 -> 64, 100 -> 128, 75 -> 128
        config = BucketingConfig(buckets=(64, 128, 256, 512))

        result = plan_bucketed(default_spec, seq_lens, default_planner, config)

        assert result.bucket_groups[64] == [0]  # seq_lens[0]
        assert result.bucket_groups[128] == [1, 2]  # seq_lens[1] and seq_lens[2]

    def test_empty_sequence_lengths_raises(
        self, default_spec: SamplingSpecification, default_planner: BatchPlanner
    ) -> None:
        """Empty sequence_lengths raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            plan_bucketed(default_spec, [], default_planner)

    def test_exceeding_bucket_raises(
        self, default_spec: SamplingSpecification, default_planner: BatchPlanner
    ) -> None:
        """Sequence length exceeding all buckets propagates ValueError."""
        seq_lens = [50, 2000]  # 2000 exceeds all buckets
        config = BucketingConfig(buckets=(64, 128, 256, 512))

        with pytest.raises(ValueError, match="exceeds all buckets"):
            plan_bucketed(default_spec, seq_lens, default_planner, config)

    def test_default_bucketing_config(
        self, default_spec: SamplingSpecification, default_planner: BatchPlanner
    ) -> None:
        """Default BucketingConfig is used when not provided."""
        seq_lens = [50, 100]

        result = plan_bucketed(default_spec, seq_lens, default_planner)

        # Default config is (64, 128, 256, 512)
        assert set(result.bucket_groups.keys()) == {64, 128}
