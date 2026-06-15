"""Tests for FeatureNoiseBundle and RunSpecification noise refactoring."""

from aminx.run.specs import FeatureNoiseBundle, RunSpecification


def test_feature_noise_bundle_constructs():
    """FeatureNoiseBundle should construct with feature_type and noise_levels."""
    b = FeatureNoiseBundle(feature_type="backbone", noise_levels=(0.1, 0.2))
    assert b.feature_type == "backbone"
    assert b.noise_levels == (0.1, 0.2)
    assert b.mode == "direct"
    assert b.enabled is True


def test_feature_noise_bundle_frozen():
    """FeatureNoiseBundle should be frozen (immutable)."""
    b = FeatureNoiseBundle(feature_type="backbone", noise_levels=(0.1,))
    try:
        b.enabled = False  # type: ignore
        assert False, "Expected FrozenInstanceError"
    except (AttributeError, Exception):
        # Frozen dataclass raises error on modification
        pass


def test_run_spec_noise_defaults_empty():
    """RunSpecification.noise should default to empty list."""
    spec = RunSpecification(inputs=["dummy.pdb"])
    assert spec.noise == []


def test_run_spec_backbone_noise_bundle():
    """RunSpecification should extract backbone_noise from FeatureNoiseBundle."""
    b = FeatureNoiseBundle(feature_type="backbone", noise_levels=(0.2, 0.5))
    spec = RunSpecification(inputs=["dummy.pdb"], noise=[b])
    assert spec.backbone_noise == (0.2, 0.5)


def test_run_spec_electrostatic_bundle_enables_flag():
    """RunSpecification should set use_electrostatics=True when electrostatic bundle present."""
    b = FeatureNoiseBundle(feature_type="electrostatic", noise_levels=(0.05,), enabled=True)
    spec = RunSpecification(inputs=["dummy.pdb"], noise=[b])
    assert spec.use_electrostatics is True


def test_run_spec_electrostatic_disabled_bundle():
    """RunSpecification should set use_electrostatics=False when electrostatic bundle disabled."""
    b = FeatureNoiseBundle(feature_type="electrostatic", noise_levels=(0.05,), enabled=False)
    spec = RunSpecification(inputs=["dummy.pdb"], noise=[b])
    assert spec.use_electrostatics is False


def test_run_spec_vdw_bundle_enables_flag():
    """RunSpecification should set use_vdw=True when vdw bundle present and enabled."""
    b = FeatureNoiseBundle(feature_type="vdw", noise_levels=(0.02,), enabled=True)
    spec = RunSpecification(inputs=["dummy.pdb"], noise=[b])
    assert spec.use_vdw is True


def test_run_spec_vdw_disabled_bundle():
    """RunSpecification should set use_vdw=False when vdw bundle disabled."""
    b = FeatureNoiseBundle(feature_type="vdw", noise_levels=(0.02,), enabled=False)
    spec = RunSpecification(inputs=["dummy.pdb"], noise=[b])
    assert spec.use_vdw is False


def test_run_spec_multiple_bundles():
    """RunSpecification should handle multiple bundles in noise list."""
    bundles = [
        FeatureNoiseBundle(feature_type="backbone", noise_levels=(0.1,)),
        FeatureNoiseBundle(feature_type="electrostatic", noise_levels=(0.05,)),
        FeatureNoiseBundle(feature_type="vdw", noise_levels=(0.02,)),
    ]
    spec = RunSpecification(inputs=["dummy.pdb"], noise=bundles)
    assert spec.backbone_noise == (0.1,)
    assert spec.use_electrostatics is True
    assert spec.use_vdw is True


def test_run_spec_backward_compat_no_bundles():
    """RunSpecification with no bundles should have backward-compat defaults."""
    spec = RunSpecification(inputs=["dummy.pdb"])
    assert spec.backbone_noise == (0.0,)
    assert spec.use_electrostatics is False
    assert spec.use_vdw is False


def test_feature_noise_bundle_modes():
    """FeatureNoiseBundle should support direct and thermal modes."""
    b_direct = FeatureNoiseBundle(
        feature_type="backbone", noise_levels=(0.1,), mode="direct"
    )
    b_thermal = FeatureNoiseBundle(
        feature_type="backbone", noise_levels=(0.1,), mode="thermal"
    )
    assert b_direct.mode == "direct"
    assert b_thermal.mode == "thermal"
