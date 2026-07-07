"""Tests for host.plan.plan_bucketed function."""

import pytest

from aminx.host.plan import plan_bucketed
from aminx.run.specs import SamplingSpecification
from aminx.tiling.axes import (
    N_LIGAND_ATOMS,
    N_NOISES,
    N_RESIDUES,
    N_SAMPLES,
    N_STRUCTURES,
    N_TEMPERATURES,
)
from aminx.tiling.bucketing import BucketingConfig
from aminx.tiling.planner import estimate_memory_theoretical
from xtrax.tiling import AxisSpec


class TestPlanBucketed:
    """Test plan_bucketed function.

    plan_bucketed takes axes/budget_bytes/estimate_fn directly (EPIC #1541
    T-PLANNER.3) rather than a pre-built BatchPlanner -- xtrax.tiling.
    BatchPlanner isn't a dataclass and doesn't hold axes as an attribute, so
    the old "pass a pre-built planner, mutate its .axes" design has no
    equivalent.
    """

    @pytest.fixture
    def default_spec(self) -> SamplingSpecification:
        """Default SamplingSpecification for testing."""
        return SamplingSpecification(inputs=[])

    @pytest.fixture
    def default_axes(self) -> list[AxisSpec]:
        """Default axes for testing."""
        return [N_RESIDUES, N_LIGAND_ATOMS, N_STRUCTURES, N_SAMPLES, N_TEMPERATURES, N_NOISES]

    @pytest.fixture
    def default_kwargs(self, default_axes: list[AxisSpec]) -> dict:
        """Shared budget_bytes/estimate_fn kwargs for plan_bucketed."""
        return {
            "budget_bytes": int(8e9),  # 8 GB
            "estimate_fn": lambda ds: estimate_memory_theoretical(ds, 1.0, 2.5),
        }

    def test_three_sequences_two_buckets(
        self, default_spec: SamplingSpecification, default_axes: list[AxisSpec], default_kwargs: dict
    ) -> None:
        """Three sequences in two buckets -> two entries in per_bucket_plans."""
        seq_lens = [50, 100, 110]  # 50 -> 64, 100 -> 128, 110 -> 128
        config = BucketingConfig(buckets=(64, 128, 256, 512))

        result = plan_bucketed(
            default_spec, seq_lens, default_axes, bucketing_config=config, **default_kwargs
        )

        # Should have two buckets used
        assert set(result.bucket_groups.keys()) == {64, 128}
        assert len(result.per_bucket_plans) == 2
        assert set(result.bucket_boundaries) == {64, 128}

    def test_bucket_boundaries_sorted(
        self, default_spec: SamplingSpecification, default_axes: list[AxisSpec], default_kwargs: dict
    ) -> None:
        """bucket_boundaries is sorted and matches used buckets."""
        seq_lens = [50, 100, 150]  # 50 -> 64, 100 -> 128, 150 -> 256
        config = BucketingConfig(buckets=(64, 128, 256, 512))

        result = plan_bucketed(
            default_spec, seq_lens, default_axes, bucketing_config=config, **default_kwargs
        )

        assert result.bucket_boundaries == (64, 128, 256)

    def test_indices_match_input_positions(
        self, default_spec: SamplingSpecification, default_axes: list[AxisSpec], default_kwargs: dict
    ) -> None:
        """Indices in bucket_groups match original input positions."""
        seq_lens = [50, 100, 75]  # 50 -> 64, 100 -> 128, 75 -> 128
        config = BucketingConfig(buckets=(64, 128, 256, 512))

        result = plan_bucketed(
            default_spec, seq_lens, default_axes, bucketing_config=config, **default_kwargs
        )

        assert result.bucket_groups[64] == [0]  # seq_lens[0]
        assert result.bucket_groups[128] == [1, 2]  # seq_lens[1] and seq_lens[2]

    def test_empty_sequence_lengths_raises(
        self, default_spec: SamplingSpecification, default_axes: list[AxisSpec], default_kwargs: dict
    ) -> None:
        """Empty sequence_lengths raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            plan_bucketed(default_spec, [], default_axes, **default_kwargs)

    def test_exceeding_bucket_raises(
        self, default_spec: SamplingSpecification, default_axes: list[AxisSpec], default_kwargs: dict
    ) -> None:
        """Sequence length exceeding all buckets propagates ValueError."""
        seq_lens = [50, 2000]  # 2000 exceeds all buckets
        config = BucketingConfig(buckets=(64, 128, 256, 512))

        with pytest.raises(ValueError, match="exceeds all buckets"):
            plan_bucketed(
                default_spec, seq_lens, default_axes, bucketing_config=config, **default_kwargs
            )

    def test_default_bucketing_config(
        self, default_spec: SamplingSpecification, default_axes: list[AxisSpec], default_kwargs: dict
    ) -> None:
        """Default BucketingConfig is used when not provided."""
        seq_lens = [50, 100]

        result = plan_bucketed(default_spec, seq_lens, default_axes, **default_kwargs)

        # Default config is (64, 128, 256, 512)
        assert set(result.bucket_groups.keys()) == {64, 128}
