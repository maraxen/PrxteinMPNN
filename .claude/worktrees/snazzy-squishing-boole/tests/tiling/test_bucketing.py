"""Tests for tiling.bucketing module."""

import pytest

from prxteinmpnn.tiling.bucketing import (
    BucketAssignment,
    BucketingConfig,
    group_by_bucket,
    select_bucket,
)


class TestSelectBucket:
    """Test bucket selection logic."""

    def test_exact_match(self) -> None:
        """Sequence length matching bucket boundary exactly."""
        config = BucketingConfig(buckets=(64, 128, 256, 512))
        assert select_bucket(64, config) == 64
        assert select_bucket(128, config) == 128
        assert select_bucket(256, config) == 256
        assert select_bucket(512, config) == 512

    def test_below_ceiling(self) -> None:
        """Sequence length below bucket boundary."""
        config = BucketingConfig(buckets=(64, 128, 256, 512))
        assert select_bucket(1, config) == 64
        assert select_bucket(50, config) == 64
        assert select_bucket(100, config) == 128
        assert select_bucket(200, config) == 256

    def test_exceeds_all_buckets(self) -> None:
        """Sequence length exceeds all configured buckets."""
        config = BucketingConfig(buckets=(64, 128, 256, 512))
        with pytest.raises(ValueError, match="exceeds all buckets"):
            select_bucket(1000, config)


class TestBucketingConfig:
    """Test BucketingConfig validation."""

    def test_valid_config(self) -> None:
        """Valid configuration with sorted buckets."""
        config = BucketingConfig(buckets=(64, 128, 256, 512))
        assert config.buckets == (64, 128, 256, 512)

    def test_unsorted_buckets(self) -> None:
        """Configuration with unsorted buckets raises ValueError."""
        with pytest.raises(ValueError, match="must be sorted"):
            BucketingConfig(buckets=(256, 64, 128, 512))

    def test_empty_buckets(self) -> None:
        """Configuration with empty bucket list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            BucketingConfig(buckets=())

    def test_default_config(self) -> None:
        """Default configuration is valid."""
        config = BucketingConfig()
        assert config.buckets == (64, 128, 256, 512)


class TestGroupByBucket:
    """Test batch grouping by bucket."""

    def test_mixed_lengths_two_buckets(self) -> None:
        """Mixed sequence lengths partitioned into two buckets."""
        config = BucketingConfig(buckets=(64, 128, 256, 512))
        seq_lens = [30, 50, 100, 120]
        result = group_by_bucket(seq_lens, config)

        assert set(result.keys()) == {64, 128}
        assert set(result[64]) == {0, 1}  # indices for lengths 30, 50
        assert set(result[128]) == {2, 3}  # indices for lengths 100, 120

    def test_indices_preserved(self) -> None:
        """Returned indices match original sequence positions."""
        config = BucketingConfig(buckets=(64, 128, 256, 512))
        seq_lens = [50, 100, 75, 110, 200]
        result = group_by_bucket(seq_lens, config)

        assert result[64] == [0]  # 50 -> 64
        assert result[128] == [1, 2, 3]  # 100, 75, 110 -> 128
        assert result[256] == [4]  # 200 -> 256

    def test_single_bucket(self) -> None:
        """All sequences fit in one bucket."""
        config = BucketingConfig(buckets=(64, 128, 256, 512))
        seq_lens = [10, 20, 30, 40]
        result = group_by_bucket(seq_lens, config)

        assert len(result) == 1
        assert result[64] == [0, 1, 2, 3]

    def test_exceeds_all_buckets_propagates(self) -> None:
        """Length exceeding all buckets propagates ValueError."""
        config = BucketingConfig(buckets=(64, 128, 256, 512))
        seq_lens = [50, 100, 2000]  # 2000 exceeds all
        with pytest.raises(ValueError, match="exceeds all buckets"):
            group_by_bucket(seq_lens, config)


class TestBucketAssignment:
    """Test BucketAssignment construction and properties."""

    def test_bucket_boundaries_sorted(self) -> None:
        """bucket_boundaries is a sorted tuple."""
        from prxteinmpnn.tiling.planner import BatchPlan, AxisSpec

        # Create minimal mock BatchPlans
        ax = AxisSpec(
            name="n_structures",
            axis_index=3,
            cardinality=32,
            default_batch_size=1,
            tile_granularity=1,
            heterogeneous=True,
            doc="test",
        )
        plan1 = BatchPlan(
            decisions=[],
            total_memory_estimate=1000.0,
            axes_by_index={3: ax},
            budget_exceeded=False,
        )
        plan2 = BatchPlan(
            decisions=[],
            total_memory_estimate=2000.0,
            axes_by_index={3: ax},
            budget_exceeded=False,
        )

        # Intentionally pass unsorted buckets to constructor
        assignment = BucketAssignment(
            bucket_boundaries=(256, 64, 128),  # unsorted input
            bucket_groups={64: [0, 1], 128: [2], 256: [3]},
            per_bucket_plans={64: plan1, 128: plan1, 256: plan2},
        )

        # bucket_boundaries should be sorted in output
        assert assignment.bucket_boundaries == (64, 128, 256)

    def test_subset_of_buckets(self) -> None:
        """bucket_boundaries is sorted subset of used bucket keys."""
        from prxteinmpnn.tiling.planner import BatchPlan, AxisSpec

        ax = AxisSpec(
            name="n_structures",
            axis_index=3,
            cardinality=32,
            default_batch_size=1,
            tile_granularity=1,
            heterogeneous=True,
            doc="test",
        )
        plan = BatchPlan(
            decisions=[],
            total_memory_estimate=1000.0,
            axes_by_index={3: ax},
            budget_exceeded=False,
        )

        # Only 64 and 256 are used (no 128)
        assignment = BucketAssignment(
            bucket_boundaries=(64, 256),
            bucket_groups={64: [0, 1], 256: [2]},
            per_bucket_plans={64: plan, 256: plan},
        )

        assert assignment.bucket_boundaries == (64, 256)
        assert set(assignment.bucket_groups.keys()) == {64, 256}
