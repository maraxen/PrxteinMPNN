"""Guards that BatchLogitsFn is gone and LogitTransformFn is importable."""
import pytest


def test_logit_transform_fn_importable():
    from prxteinmpnn.model_inputs import LogitTransformFn
    assert LogitTransformFn is not None


def test_batch_logits_fn_removed():
    import prxteinmpnn.model_inputs as mi
    assert not hasattr(mi, "BatchLogitsFn"), (
        "BatchLogitsFn must be removed; use LogitTransformFn"
    )


def test_encoder_output_importable():
    from prxteinmpnn.payloads import EncoderOutput
    assert EncoderOutput is not None


def test_encoder_pre_fn_protocol():
    from prxteinmpnn.protocols import EncoderPreFn
    assert EncoderPreFn is not None


def test_encoder_post_fn_protocol():
    from prxteinmpnn.protocols import EncoderPostFn
    assert EncoderPostFn is not None


def test_encoder_state_fn_importable():
    from prxteinmpnn.protocols import EncoderStateFn
    assert EncoderStateFn is not None


def test_pipeline_protocol():
    from prxteinmpnn.protocols import Pipeline
    assert Pipeline is not None


def test_model_protocol():
    from prxteinmpnn.protocols import ModelProtocol
    assert ModelProtocol is not None


def test_model_protocol_runtime_checkable_vs_prxtein_mpnn():
    """PrxteinMPNN satisfies ModelProtocol structurally."""
    import jax
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.protocols import ModelProtocol

    key = jax.random.PRNGKey(0)
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=key)
    assert isinstance(m, ModelProtocol)


def test_pipeline_registry_register_resolve_roundtrip():
    import jax.numpy as jnp
    from prxteinmpnn.pipeline_registry import register_hook, resolve_hook

    def my_transform(state_logits, state_index, state_weights):
        return jnp.mean(state_logits, axis=0)

    uid = register_hook(my_transform, name="test_mean")
    resolved = resolve_hook(uid)
    assert resolved is my_transform
    assert len(uid) == 16


def test_pipeline_registry_idempotent():
    import jax.numpy as jnp
    from prxteinmpnn.pipeline_registry import register_hook

    def fn(state_logits, state_index, state_weights):
        return jnp.sum(state_logits, axis=0)

    uid1 = register_hook(fn, name="idem_test")
    uid2 = register_hook(fn, name="idem_test")
    assert uid1 == uid2


def test_default_logit_transform_uid_exists():
    from prxteinmpnn.pipeline_registry import DEFAULT_LOGIT_TRANSFORM_UID, resolve_hook
    assert isinstance(DEFAULT_LOGIT_TRANSFORM_UID, str)
    assert len(DEFAULT_LOGIT_TRANSFORM_UID) == 16
    fn = resolve_hook(DEFAULT_LOGIT_TRANSFORM_UID)
    assert callable(fn)


def test_pipeline_fns_default_constructible():
    from prxteinmpnn.pipeline_fns import PipelineFns
    fns = PipelineFns.default()
    assert isinstance(fns.logit_transform_uid, str)
    assert len(fns.logit_transform_uid) == 16
    assert fns.encoder_pre_process_uid is None
    assert fns.encoder_post_process_uid is None


def test_pipeline_fns_from_callables():
    import jax.numpy as jnp
    from prxteinmpnn.pipeline_fns import PipelineFns

    def my_transform(state_logits, state_index, state_weights):
        return jnp.mean(state_logits, axis=0)

    fns = PipelineFns.from_callables(logit_transform=my_transform)
    assert isinstance(fns.logit_transform_uid, str)
    assert len(fns.logit_transform_uid) == 16
    assert fns.encoder_pre_process_uid is None


def test_pipeline_fns_resolve_logit_transform():
    import jax.numpy as jnp
    from prxteinmpnn.pipeline_fns import PipelineFns

    fns = PipelineFns.default()
    fn = fns.resolve_logit_transform()
    assert callable(fn)
    state_logits = jnp.ones((2, 4, 21))
    result = fn(state_logits, jnp.arange(2), jnp.ones(2))
    assert result.shape == (4, 21)


def test_pipeline_fns_is_frozen():
    import dataclasses
    import pytest
    from prxteinmpnn.pipeline_fns import PipelineFns
    fns = PipelineFns.default()
    assert dataclasses.is_dataclass(fns)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        fns.logit_transform_uid = "something_else"


def test_pipeline_fns_with_encoder_post():
    from prxteinmpnn.pipeline_fns import PipelineFns

    def my_post(encoded, state_index):
        return encoded

    fns = PipelineFns.from_callables(encoder_post_process=my_post)
    assert fns.encoder_post_process_uid is not None
    resolved = fns.resolve_encoder_post_process()
    assert resolved is my_post


def test_encoder_output_is_pytree():
    import jax
    import jax.numpy as jnp
    from prxteinmpnn.payloads import EncoderOutput

    S, L, K, D, E = 2, 6, 16, 32, 32
    enc = EncoderOutput(
        node_features=jnp.zeros((S, L, D)),
        edge_features=jnp.zeros((S, L, K, E)),
        neighbor_indices=jnp.zeros((S, L, K), dtype=jnp.int32),
        mask=jnp.ones((S, L)),
    )
    leaves, treedef = jax.tree_util.tree_flatten(enc)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.node_features.shape == (S, L, D)
    assert restored.mask.shape == (S, L)


def test_pipeline_top_level_imports():
    """All four pipeline types importable from prxteinmpnn.pipeline."""
    from prxteinmpnn.pipeline import (
        AutoregressivePipeline,
        ConditionalPipeline,
        STEPipeline,
        UnconditionalPipeline,
    )
    assert all(x is not None for x in [
        AutoregressivePipeline, ConditionalPipeline, STEPipeline, UnconditionalPipeline
    ])


def test_pipeline_input_carriers_top_level():
    """Input carrier classes importable from prxteinmpnn.pipeline."""
    from prxteinmpnn.pipeline import (
        AutoregressiveInputs,
        ConditionalInputs,
        STEInputs,
    )
    assert all(x is not None for x in [AutoregressiveInputs, ConditionalInputs, STEInputs])


def test_pipeline_fns_has_encoder_state_fn_uid():
    from prxteinmpnn.pipeline_fns import PipelineFns
    fns = PipelineFns.default()
    assert hasattr(fns, "encoder_state_fn_uid")
    assert fns.encoder_state_fn_uid is None


def test_pipeline_fns_resolve_encoder_state_fn_returns_none_by_default():
    from prxteinmpnn.pipeline_fns import PipelineFns
    fns = PipelineFns.default()
    assert fns.resolve_encoder_state_fn() is None
