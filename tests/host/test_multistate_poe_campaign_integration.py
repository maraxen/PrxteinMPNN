"""Real end-to-end test of the multistate PoE campaign-pipeline wiring.

Regression coverage for the 2026-07-14 integration (tev_design task
260709_multistate-fusion-strategy-comparison, Phase 3): host/campaign.py::run_manifest_row must
route genuine multi-state PoE rows (len(inputs) > 1) to
aminx.sampling.multistate_poe.sample_multistate_poe_campaign_row instead of
host/runner.py::sample(), while single-structure ("spike-in") rows keep using sample() unchanged.
See .praxia/docs/decisions/260713_no-real-multistate-sampling-path-exists.md.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import zarr

from aminx.host.campaign import MANIFEST_SCHEMA_VERSION, run_manifest_row
from aminx.host.prep import prep_protein_stream_and_model
from aminx.model import Aminx
from aminx.run.specs import SamplingSpecification
from aminx.utils.align import build_state_position_map

REPO_TESTS_DATA = __file__.rsplit("/tests/", 1)[0] + "/tests/data"


def _synthetic_model() -> Aminx:
  model = Aminx(
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=8,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(11),
  )
  return eqx.tree_inference(model, value=True)


def _real_state_position_map(inputs: list[str], synthetic_model: Aminx) -> list[list[int]]:
  """Mirrors tev_design's compute_state_position_map: real alignment, then -1-padded to L."""
  spec = SamplingSpecification(inputs=inputs, batch_size=len(inputs), return_logits=True)
  with patch("aminx.host.prep.load_model", return_value=synthetic_model):
    protein_iterator, _ = prep_protein_stream_and_model(spec)
    batched_ensemble = list(protein_iterator)[0]

  seq_len = batched_ensemble.coordinates.shape[1]
  n_states = batched_ensemble.coordinates.shape[0]
  native_lens = [int(m.sum()) for m in batched_ensemble.mask]
  max_native_len = max(native_lens)
  native_seqs = jnp.full((n_states, max_native_len), -1, dtype=jnp.int32)
  for s in range(n_states):
    native_seqs = native_seqs.at[s, : native_lens[s]].set(
      batched_ensemble.aatype[s, : native_lens[s]].astype(jnp.int32),
    )
  aligned = build_state_position_map(native_seqs, reference_state_index=0)
  padded = jnp.full((n_states, seq_len), -1, dtype=jnp.int32)
  padded = padded.at[:, :max_native_len].set(aligned)
  return np.asarray(padded).tolist()


class TestMultistatePoeCampaignRowRouting:
  """run_manifest_row must route multi-state rows to the new function, spike-in rows to sample()."""

  def test_multistate_poe_row_writes_real_fused_output(self, tmp_path):
    """Real end-to-end: a genuine multi-state PoE manifest row, run through
    run_manifest_row exactly as `aminx campaign run` would invoke it, produces real
    fused output at the row's output_h5_path -- not a per-structure independent result.
    Only the model weights are mocked; manifest loading, locking, done-marker writing,
    structure parsing, bundle construction, fusion, sampling, and Zarr writing are all
    real, unmocked code paths.
    """
    inputs = [f"{REPO_TESTS_DATA}/1ubq.pdb", f"{REPO_TESTS_DATA}/1mbn.pdb"]
    synthetic_model = _synthetic_model()
    state_position_map = _real_state_position_map(inputs, synthetic_model)

    output_path = tmp_path / "poe_row.h5"
    manifest_path = tmp_path / "manifest.json"
    row_hash = "test_multistate_poe_row"
    manifest = {
      "schema_version": MANIFEST_SCHEMA_VERSION,
      "rows": [
        {
          "manifest_row_hash": row_hash,
          "job_id": "job0",
          "job_index": 0,
          "sampling_spec": {
            "inputs": inputs,
            "batch_size": len(inputs),
            "multi_state_strategy": "product",
            "state_position_map": state_position_map,
            "return_logits": True,
            "num_samples": 2,
            "output_h5_path": str(output_path),
          },
        },
      ],
    }
    manifest_path.write_text(json.dumps(manifest))

    with patch("aminx.host.prep.load_model", return_value=synthetic_model):
      result = run_manifest_row(
        manifest_path=str(manifest_path),
        row_hash=row_hash,
        lock_backend="local_fs",
      )

    assert result["status"] == "completed"
    assert output_path.exists()
    assert not any(tmp_path.glob("*.partial.*"))

    root = zarr.open_group(str(output_path), mode="r")
    assert "poe_fused" in root
    assert "structure_0" not in root, (
      "a genuine PoE row must not produce per-structure groups -- that's exactly the "
      "broken behavior this integration replaces"
    )
    fused = root["poe_fused"]
    sequences = np.asarray(fused["sequences"])
    logits = np.asarray(fused["logits"])
    assert sequences.shape[0] == 2  # num_samples
    assert sequences.shape[-1] == logits.shape[-2]  # L matches
    assert logits.shape[-1] == 21
    assert bool(np.all(np.isfinite(logits)))
    assert fused.attrs["structure_id"] == "poe_fused"
    assert fused.attrs["fused_structure_ids"] == ["1ubq", "1mbn"] or len(
      fused.attrs["fused_structure_ids"],
    ) == 2

    # Retry short-circuits via the done marker, without re-running the (expensive) sampling.
    with patch(
      "aminx.sampling.multistate_poe.sample_multistate_poe_bead",
    ) as mock_sample_bead:
      result2 = run_manifest_row(
        manifest_path=str(manifest_path),
        row_hash=row_hash,
        lock_backend="local_fs",
      )
    assert result2["status"] == "already_done"
    mock_sample_bead.assert_not_called()

  def test_spike_in_row_uses_sample_not_the_poe_wrapper(self, tmp_path):
    """A single-structure ("spike-in") row -- len(inputs) == 1 -- must keep going through
    sample()/_sample_batch unchanged; sample_multistate_poe_campaign_row must never be
    called for it. Uses a fake sample() (mirroring
    TestRunManifestRowZarrLifecycle.test_happy_path_creates_zarr_store_and_marker's existing
    pattern) since a real single-structure sample() call is already covered elsewhere and
    is not what this test is checking.
    """
    output_path = tmp_path / "spikein_row.h5"
    manifest_path = tmp_path / "manifest.json"
    row_hash = "test_spikein_row"
    manifest = {
      "schema_version": MANIFEST_SCHEMA_VERSION,
      "rows": [
        {
          "manifest_row_hash": row_hash,
          "job_id": "job0",
          "job_index": 0,
          "sampling_spec": {
            "inputs": [f"{REPO_TESTS_DATA}/1ubq.pdb"],
            "output_h5_path": str(output_path),
          },
        },
      ],
    }
    manifest_path.write_text(json.dumps(manifest))

    def _fake_sample(spec):
      root = zarr.open_group(str(spec.output_h5_path), mode="a")
      arr = root.create_array(name="sequences", shape=(1,), dtype="int32")
      arr[...] = np.array([3], dtype=np.int32)
      return {"status": "completed"}

    with (
      patch("aminx.host.campaign.sample", side_effect=_fake_sample) as mock_sample,
      patch(
        "aminx.host.campaign.sample_multistate_poe_campaign_row",
      ) as mock_poe_row,
    ):
      result = run_manifest_row(
        manifest_path=str(manifest_path),
        row_hash=row_hash,
        lock_backend="local_fs",
      )

    assert result["status"] == "completed"
    mock_sample.assert_called_once()
    mock_poe_row.assert_not_called()
