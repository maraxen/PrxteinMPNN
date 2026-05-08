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
