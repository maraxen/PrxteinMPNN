"""COMP-536: Tests for campaign manifest functions."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from aminx.host.campaign import (
    MANIFEST_ROW_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    build_manifest_row,
    load_manifest,
    validate_manifest_rows,
    write_manifest,
)


@dataclass
class DummySamplingConfig:
    """Minimal sampling config stub."""
    temperature: list[float] = field(default_factory=lambda: [1.0])
    backbone_noise: list[float] = field(default_factory=lambda: [0.0])
    return_logits: bool = False


@dataclass
class DummyRunSpec:
    """Minimal run_spec stub."""
    sampling: DummySamplingConfig = field(default_factory=DummySamplingConfig)


@dataclass
class DummySpec:
    """Minimal spec stub for testing manifest functions."""

    inputs: list[str] = field(default_factory=lambda: ["/tmp/test.pdb"])
    model_family: str = "ligandmpnn_v1"
    model_weights: str | None = None
    model_version: str | None = None
    checkpoint_id: str = "ckpt_001"
    model_local_path: str | None = None
    checkpoint_registry_path: str | None = None
    ligand_conditioning: bool = False
    sidechain_conditioning: bool = False
    temperature: list[float] = field(default_factory=lambda: [1.0])
    backbone_noise: list[float] = field(default_factory=lambda: [0.0])
    batch_size: int = 8
    multi_state_strategy: str = "arithmetic_mean"
    campaign_mode: bool = False
    return_logits: bool = False
    run_spec: DummyRunSpec = None

    def __post_init__(self):
        if self.run_spec is None:
            object.__setattr__(
                self,
                "run_spec",
                DummyRunSpec(
                    sampling=DummySamplingConfig(
                        temperature=self.temperature,
                        backbone_noise=self.backbone_noise,
                        return_logits=self.return_logits,
                    )
                ),
            )


def _make_row(**kwargs):
    """Helper: build a minimal valid manifest row dict."""
    defaults = {
        "manifest_row_hash": "hash_abc123",
        "checkpoint_id": "ckpt_001",
        "fixed_policy": "catalytic_triad",
        "state_weight_profile": "equal",
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# build_manifest_row
# ---------------------------------------------------------------------------


class TestBuildManifestRow:
    def test_returns_dict_with_hash(self):
        spec = DummySpec()
        row = build_manifest_row(
            spec,
            campaign_id="campaign_001",
            job_index=0,
            chunk_id=0,
            sample_start=0,
            sample_count=10,
            fixed_policy="catalytic_triad",
            state_weight_profile="equal",
            planner_version="planner_v1",
            dataset_fingerprint="fp_001",
            environment_image="img_001",
            git_sha="sha123",
            config_hash="ch123",
            job_id="campaign_001-job-0",
        )
        assert isinstance(row, dict)
        assert "manifest_row_hash" in row
        assert len(row["manifest_row_hash"]) == 64

    def test_required_fields_present(self):
        spec = DummySpec()
        row = build_manifest_row(
            spec,
            campaign_id="campaign_001",
            job_index=1,
            chunk_id=2,
            sample_start=100,
            sample_count=50,
            fixed_policy="active_site",
            state_weight_profile="weighted",
            planner_version="planner_v1",
            dataset_fingerprint="fp_001",
            environment_image="img_001",
            git_sha="sha123",
            config_hash="ch123",
            job_id="campaign_001-job-1",
        )
        required = {
            "manifest_row_hash",
            "job_id",
            "job_index",
            "chunk_id",
            "sample_start",
            "sample_count",
            "fixed_policy",
            "state_weight_profile",
            "multi_state_strategy",
            "temperature",
            "backbone_noise",
            "ligand_conditioning",
            "sidechain_conditioning",
            "checkpoint_id",
            "planner_version",
            "dataset_fingerprint",
            "environment_image",
            "git_sha",
            "config_hash",
        }
        for f in required:
            assert f in row, f"Missing: {f}"

    def test_deterministic_hash(self):
        def _make(**kwargs):
            return build_manifest_row(
                DummySpec(),
                campaign_id="campaign_001",
                job_index=0,
                chunk_id=0,
                sample_start=0,
                sample_count=10,
                fixed_policy="catalytic_triad",
                state_weight_profile="equal",
                planner_version="planner_v1",
                dataset_fingerprint="fp_001",
                environment_image="img_001",
                git_sha="sha123",
                config_hash="ch123",
                job_id="campaign_001-job-0",
                **kwargs,
            )

        assert _make()["manifest_row_hash"] == _make()["manifest_row_hash"]

    def test_different_args_different_hash(self):
        def _make(job_id, chunk_id):
            return build_manifest_row(
                DummySpec(),
                campaign_id="campaign_001",
                job_index=0,
                chunk_id=chunk_id,
                sample_start=0,
                sample_count=10,
                fixed_policy="catalytic_triad",
                state_weight_profile="equal",
                planner_version="planner_v1",
                dataset_fingerprint="fp_001",
                environment_image="img_001",
                git_sha="sha123",
                config_hash="ch123",
                job_id=job_id,
            )

        assert (
            _make("campaign_001-job-0", 0)["manifest_row_hash"]
            != _make("campaign_001-job-0", 1)["manifest_row_hash"]
        )

    def test_extracts_spec_fields(self):
        spec = DummySpec(
            temperature=[0.1, 0.5, 1.0],
            backbone_noise=[0.01, 0.05],
            ligand_conditioning=True,
            sidechain_conditioning=False,
            checkpoint_id="ckpt_special",
            multi_state_strategy="geometric_mean",
        )
        row = build_manifest_row(
            spec,
            campaign_id="c",
            job_index=0,
            chunk_id=0,
            sample_start=0,
            sample_count=5,
            fixed_policy="fp",
            state_weight_profile="eq",
            planner_version="v1",
            dataset_fingerprint="fp",
            environment_image="img",
            git_sha="sha",
            config_hash="ch",
            job_id="c-job-0",
        )
        assert row["temperature"] == [0.1, 0.5, 1.0]
        assert row["backbone_noise"] == [0.01, 0.05]
        assert row["ligand_conditioning"] is True
        assert row["sidechain_conditioning"] is False
        assert row["checkpoint_id"] == "ckpt_special"
        assert row["multi_state_strategy"] == "geometric_mean"


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


class TestLoadManifest:
    def test_reads_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            payload = {"schema_version": MANIFEST_SCHEMA_VERSION, "rows": [_make_row()]}
            path.write_text(json.dumps(payload))
            loaded = load_manifest(path)
            assert loaded["rows"] == payload["rows"]

    def test_requires_rows_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            path.write_text(json.dumps({"schema_version": MANIFEST_SCHEMA_VERSION}))
            with pytest.raises((ValueError, TypeError, KeyError)):
                load_manifest(path)

    def test_rows_must_be_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            path.write_text(json.dumps({"rows": {"hash": {}}}))
            with pytest.raises((ValueError, TypeError)):
                load_manifest(path)

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            path.write_text("{invalid}")
            with pytest.raises((ValueError, json.JSONDecodeError)):
                load_manifest(path)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_manifest(Path("/tmp/nonexistent_manifest_xyzabc.json"))

    def test_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            meta = {"campaign_id": "c001", "planner_version": "v1"}
            payload = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "metadata": meta,
                "rows": [],
            }
            path.write_text(json.dumps(payload))
            loaded = load_manifest(path)
            assert loaded["metadata"] == meta

    def test_empty_rows_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            path.write_text(
                json.dumps(
                    {"schema_version": MANIFEST_SCHEMA_VERSION, "rows": []}
                )
            )
            assert load_manifest(path)["rows"] == []


# ---------------------------------------------------------------------------
# validate_manifest_rows
# ---------------------------------------------------------------------------


class TestValidateManifestRows:
    def test_passes_valid(self):
        rows = [
            _make_row(manifest_row_hash="h1"),
            _make_row(
                manifest_row_hash="h2", fixed_policy="active_site"
            ),
        ]
        validate_manifest_rows(
            rows, required_fixed_policies=("catalytic_triad", "active_site")
        )

    def test_missing_hash_raises(self):
        with pytest.raises(ValueError, match="manifest_row_hash"):
            validate_manifest_rows(
                [
                    {
                        "checkpoint_id": "c",
                        "fixed_policy": "f",
                        "state_weight_profile": "e",
                    }
                ]
            )

    def test_duplicate_hash_raises(self):
        rows = [
            _make_row(manifest_row_hash="same"),
            _make_row(
                manifest_row_hash="same", fixed_policy="active_site"
            ),
        ]
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            validate_manifest_rows(rows)

    def test_empty_checkpoint_id_raises(self):
        with pytest.raises(ValueError, match="checkpoint_id"):
            validate_manifest_rows([_make_row(checkpoint_id="")])

    def test_empty_fixed_policy_raises(self):
        with pytest.raises(ValueError, match="fixed_policy"):
            validate_manifest_rows([_make_row(fixed_policy="")])

    def test_empty_state_weight_profile_raises(self):
        with pytest.raises(ValueError, match="state_weight_profile"):
            validate_manifest_rows([_make_row(state_weight_profile="")])

    def test_missing_required_policy_raises(self):
        rows = [_make_row(manifest_row_hash="h1")]  # only catalytic_triad
        with pytest.raises(ValueError):
            validate_manifest_rows(
                rows, required_fixed_policies=("catalytic_triad", "active_site")
            )

    def test_all_required_policies_present(self):
        rows = [
            _make_row(manifest_row_hash="h1"),
            _make_row(
                manifest_row_hash="h2", fixed_policy="active_site"
            ),
        ]
        validate_manifest_rows(
            rows, required_fixed_policies=("catalytic_triad", "active_site")
        )

    def test_empty_rows_valid(self):
        validate_manifest_rows([])

    def test_no_required_policies_no_constraint(self):
        validate_manifest_rows([_make_row()])


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------


class TestWriteManifest:
    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            result = write_manifest(path, [_make_row()])
            assert result == path.resolve()
            assert path.exists()

    def test_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            rows = [_make_row()]
            write_manifest(path, rows)
            loaded = json.loads(path.read_text())
            assert isinstance(loaded, dict)
            assert loaded["rows"] == rows

    def test_includes_schema_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            write_manifest(path, [])
            loaded = json.loads(path.read_text())
            assert loaded["schema_version"] == MANIFEST_SCHEMA_VERSION

    def test_includes_metadata_if_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            meta = {"campaign_id": "c001"}
            write_manifest(path, [], metadata=meta)
            loaded = json.loads(path.read_text())
            assert loaded["metadata"] == meta

    def test_without_metadata_no_metadata_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            write_manifest(path, [])
            loaded = json.loads(path.read_text())
            assert path.exists()
            assert loaded["schema_version"] == MANIFEST_SCHEMA_VERSION
            assert "metadata" not in loaded

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            rows = [
                _make_row(manifest_row_hash="h1"),
                _make_row(
                    manifest_row_hash="h2", fixed_policy="active_site"
                ),
            ]
            meta = {"campaign_id": "c001"}
            write_manifest(path, rows, metadata=meta)
            loaded = load_manifest(path)
            assert loaded["rows"] == rows
            assert loaded["metadata"] == meta

    def test_returns_resolved_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            result = write_manifest(path, [])
            assert result.is_absolute()
            assert result == path.resolve()

    def test_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            write_manifest(path, [_make_row(manifest_row_hash="old")])
            write_manifest(path, [_make_row(manifest_row_hash="new")])
            loaded = load_manifest(path)
            assert loaded["rows"][0]["manifest_row_hash"] == "new"

    def test_output_is_indented(self):
        """Manifests must be human-readable (not compact single-line JSON)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            write_manifest(path, [_make_row()])
            raw = path.read_text()
            assert "\n" in raw, "Manifest must be indented (multi-line JSON)"


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_plan_campaign_manifest_succeeds(self):
        from aminx.host.campaign import plan_campaign_manifest
        from aminx.run.specs import SamplingSpecification

        spec = SamplingSpecification(
            inputs=["/tmp/test.pdb"],
            checkpoint_id="ckpt_001",
            model_family="ligandmpnn_v1",
            temperature=[1.0],
            backbone_noise=[0.0],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = plan_campaign_manifest(
                base_spec=spec,
                campaign_id="test_campaign",
                designs_per_library_type=100,
                samples_chunk_size=50,
                output_root=tmpdir,
                fixed_policies=("catalytic_triad", "active_site"),
                state_weight_profiles=("equal",),
            )
            # 2 policies × 1 profile × 2 ligand × 2 sidechain × ceil(100/50) chunks = 16
            assert len(rows) == 16
            for row in rows:
                assert "manifest_row_hash" in row
                assert len(row["manifest_row_hash"]) == 64

    def test_write_campaign_manifest_roundtrip(self):
        from aminx.host.campaign import write_campaign_manifest
        from aminx.run.specs import SamplingSpecification

        spec = SamplingSpecification(
            inputs=["/tmp/test.pdb"],
            checkpoint_id="ckpt_001",
            model_family="ligandmpnn_v1",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            result = write_campaign_manifest(
                base_spec=spec,
                campaign_id="test_write",
                manifest_path=manifest_path,
                designs_per_library_type=50,
                samples_chunk_size=25,
                output_root=tmpdir,
                fixed_policies=("catalytic_triad",),
                state_weight_profiles=("equal",),
            )
            assert result == manifest_path.resolve()
            loaded = load_manifest(manifest_path)
            assert len(loaded["rows"]) > 0

    def test_plan_manifest_raises_when_checkpoint_id_missing(self):
        """validate_manifest_rows must catch empty checkpoint_id at plan time."""
        from aminx.host.campaign import plan_campaign_manifest
        from aminx.run.specs import SamplingSpecification

        spec = SamplingSpecification(
            inputs=["/tmp/test.pdb"],
            checkpoint_id=None,  # intentionally empty
            model_family="ligandmpnn_v1",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="checkpoint_id"):
                plan_campaign_manifest(
                    base_spec=spec,
                    campaign_id="test_no_ckpt",
                    designs_per_library_type=10,
                    samples_chunk_size=10,
                    output_root=tmpdir,
                    fixed_policies=("catalytic_triad",),
                    state_weight_profiles=("equal",),
                )


class TestRunManifestRowZarrLifecycle:
    """run_manifest_row's lock/write/promote/done-marker lifecycle against a Zarr store.

    Mocks aminx.host.campaign.sample -- a real model-inference call is out of scope for
    this test; what's under test is the orchestration (partial write, fsync, digest,
    atomic promote, done-marker write, and the already_done short-circuit on retry).
    """

    def test_happy_path_creates_zarr_store_and_marker(self, tmp_path: Path) -> None:
        import numpy as np
        import zarr

        from aminx.host.campaign import run_manifest_row

        output_path = tmp_path / "row_output.zarr"
        manifest_path = tmp_path / "manifest.json"
        row_hash = "test_row_hash_zarr_lifecycle"
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "rows": [
                {
                    "manifest_row_hash": row_hash,
                    "job_id": "job0",
                    "job_index": 0,
                    "sampling_spec": {
                        "inputs": ["/tmp/test.pdb"],
                        "output_h5_path": str(output_path),
                    },
                },
            ],
        }
        manifest_path.write_text(json.dumps(manifest))

        def _fake_sample(spec):
            # Simulate the real worker: write a Zarr store at spec's (partial) output path.
            root = zarr.open_group(str(spec.output_h5_path), mode="a")
            arr = root.create_array(name="sequences", shape=(1,), dtype="int32")
            arr[...] = np.array([7], dtype=np.int32)
            return {"status": "completed"}

        with patch("aminx.host.campaign.sample", side_effect=_fake_sample):
            result = run_manifest_row(
                manifest_path=str(manifest_path),
                row_hash=row_hash,
                lock_backend="local_fs",
            )

        assert result["status"] == "completed"
        assert output_path.exists()
        assert not any(tmp_path.glob("*.partial.*"))  # cleaned up

        # Second call should short-circuit via the done marker, not re-run sample().
        with patch("aminx.host.campaign.sample", side_effect=_fake_sample) as mock_sample:
            result2 = run_manifest_row(
                manifest_path=str(manifest_path),
                row_hash=row_hash,
                lock_backend="local_fs",
            )
        assert result2["status"] == "already_done"
        mock_sample.assert_not_called()
