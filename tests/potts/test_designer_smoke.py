"""Smoke test for MpnnPottsDesigner — collected-only gate."""
import pytest


@pytest.mark.slow
@pytest.mark.potts
class TestMpnnPottsDesignerSmoke:
    """Collected under 'potts and slow'; full run requires real Aminx + PottsModel weights."""

    def test_designer_attributes(self):
        """Verify MpnnPottsDesigner and DesignResult are importable with expected attrs."""
        from aminx.potts.designer import DesignResult, MpnnPottsDesigner

        assert hasattr(MpnnPottsDesigner, "run_design")
        assert hasattr(DesignResult, "__dataclass_fields__")

    def test_designer_run_requires_weights(self):
        """Full run_design requires real Aminx + PottsModel checkpoints."""
        pytest.skip("Integration gate: requires real model weights (Track C + Track K)")
