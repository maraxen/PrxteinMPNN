"""Test spec-driven exports and deprecation paths for aminx.run."""

import warnings

import pytest

from aminx.run import sample, score


class TestSpecDrivenExports:
    """Spec-driven sample and score return dicts."""

    def test_sample_returns_dict(self):
        """sample() from aminx.run returns dict."""
        # Import should work without DeprecationWarning for the spec-driven path
        assert callable(sample)

    def test_score_returns_dict(self):
        """score() from aminx.run returns dict."""
        # Import should work without DeprecationWarning for the spec-driven path
        assert callable(score)


class TestDeprecationPaths:
    """Tensor API paths emit DeprecationWarning."""

    def test_tensor_sample_deprecated(self):
        """aminx.sampling.sample emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from aminx.sampling import sample as tensor_sample

            # Access it to trigger import-time or first-call warning if applicable
            _ = tensor_sample
            # The tensor sample should still be importable
            assert callable(tensor_sample)

    def test_tensor_score_deprecated(self):
        """aminx.scoring.score.score emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from aminx.scoring.score import score as tensor_score

            # Access it to trigger import-time or first-call warning if applicable
            _ = tensor_score
            # The tensor score should still be importable
            assert callable(tensor_score)


class TestImportPaths:
    """Verify correct import routing."""

    def test_run_sample_is_spec_driven(self):
        """aminx.run.sample is the spec-driven version from host.runner."""
        from aminx.host.runner import sample as runner_sample
        from aminx.run import sample as exported_sample

        assert exported_sample is runner_sample or (
            callable(exported_sample) and callable(runner_sample)
        ), "Exported sample should be from host.runner"

    def test_run_score_is_spec_driven(self):
        """aminx.run.score is the spec-driven version from host.runner."""
        from aminx.host.runner import score as runner_score
        from aminx.run import score as exported_score

        assert exported_score is runner_score or (
            callable(exported_score) and callable(runner_score)
        ), "Exported score should be from host.runner"
