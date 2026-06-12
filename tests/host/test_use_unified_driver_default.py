"""RS-5 regression: kernel_dispatch getattr default matches SamplingSpecification."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp

from aminx.host.plan import InferencePlan
from aminx.run.specs import SamplingSpecification
from aminx.tiling.strategy import Vmap
from aminx.types.stages import StageSet
from aminx.utils.data_structures import Protein


def _make_fake_protein(batch_size: int = 1, seq_len: int = 10) -> Protein:
    return Protein(
        coordinates=jnp.zeros((batch_size, seq_len, 4, 3), dtype=jnp.float32),
        aatype=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        atom_mask=jnp.ones((batch_size, seq_len, 37), dtype=jnp.float32),
        residue_index=jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0),
        chain_index=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        mask=jnp.ones((batch_size, seq_len), dtype=jnp.float32),
        mapping=None,
    )


def _make_dispatch_monkeypatches(monkeypatch, *, batch_plan: MagicMock) -> None:
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.make_sampling_planner",
        lambda spec, **kwargs: batch_plan,
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.extract_batch_sizes",
        lambda plan: (1, 2, 1, 1),
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._resolve_grid_lineage",
        lambda spec: None,
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._base_sampling_key",
        lambda spec, **kwargs: jax.random.key(0),
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.resolve_target_samples",
        lambda spec, chunk_count, grid: 2,
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._prepare_ligand_context",
        lambda *args, **kwargs: {"Y": None, "Y_t": None, "Y_m": None},
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._prepare_fixed_controls",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.compute_sample_keys",
        lambda *args, **kwargs: jnp.stack([jax.random.key(i) for i in range(2)]),
    )


def _make_batch_plan_mock() -> MagicMock:
    batch_plan = MagicMock()

    def _decision_for(axis_name: str) -> MagicMock:
        decision = MagicMock()
        decision.strategy = Vmap()
        return decision

    batch_plan.decision_for = MagicMock(side_effect=_decision_for)
    batch_plan.log_summary = MagicMock()
    return batch_plan


def test_sampling_spec_use_unified_driver_defaults_true():
    """SamplingSpecification.use_unified_driver defaults to True."""
    spec = SamplingSpecification(
        inputs=["/tmp/test.pdb"],
        checkpoint_id="ckpt_001",
    )
    assert spec.use_unified_driver is True


def test_kernel_dispatch_getattr_use_unified_driver_defaults_true():
    """getattr fallback in _sample_batch must default to True (RS-5)."""
    from aminx.host import kernel_dispatch

    src = inspect.getsource(kernel_dispatch._sample_batch)
    assert 'getattr(spec, "use_unified_driver", True)' in src


def test_kernel_dispatch_unified_path_when_spec_lacks_attr(monkeypatch):
    """Spec without use_unified_driver attribute must take unified driver path."""
    from aminx.host.kernel_dispatch import _sample_batch

    batch_plan = _make_batch_plan_mock()
    _make_dispatch_monkeypatches(monkeypatch, batch_plan=batch_plan)

    dispatch_axis_calls: list[str] = []

    def _spy_dispatch_axis(strategy, body, xs, *, batch_size_fallback: int = 0):
        dispatch_axis_calls.append(type(strategy).__name__)
        return (
            jnp.zeros((1, 1, 1, 2, 10), dtype=jnp.int32),
            jnp.zeros((1, 1, 1, 2, 10, 21), dtype=jnp.float32),
        )

    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._dispatch_axis",
        _spy_dispatch_axis,
    )

    plan = MagicMock(spec=InferencePlan)
    stage_set = MagicMock(spec=StageSet)
    stage_set.encoding_fusion = None
    stage_set.encoder_sink = ()
    plan.stage_set = stage_set
    plan.encode = MagicMock(return_value=MagicMock())
    plan.decode = MagicMock(
        return_value=MagicMock(
            sequence=jnp.zeros(10, dtype=jnp.int32),
            logits=jnp.zeros((10, 21), dtype=jnp.float32),
        )
    )

    spec = SimpleNamespace(
        backbone_noise=[0.0],
        temperature=[1.0],
        bias=None,
        tie_group_map=None,
        structure_mapping=None,
        state_weights=None,
        compute_pseudo_perplexity=False,
    )

    _sample_batch(
        spec,
        _make_fake_protein(),
        plan,
        batch_idx=0,
        structure_batch_count=1,
        emit_structure_batch_io=False,
    )
    jax.effects_barrier()

    assert dispatch_axis_calls, "unified driver must route through _dispatch_axis"
    assert batch_plan.decision_for.call_count >= 4
