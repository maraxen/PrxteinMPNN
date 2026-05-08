"""Sprint MODELINPUTS + EncoderHooks regression tests."""
import jax.numpy as jnp
from prxteinmpnn.pipeline_registry import make_geometric_mean_transform, register_hook


def test_make_geometric_mean_transform_factory():
    T = 0.5
    fn = make_geometric_mean_transform(T)
    state_logits = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])  # (S=2, 3)
    result = fn(state_logits, state_index=None, state_weights=None)
    # Closed-form: mean(state_logits, axis=0) / T
    expected = jnp.mean(state_logits, axis=0) / T
    assert jnp.allclose(result, expected, atol=1e-5)


def test_make_geometric_mean_transform_registerable():
    fn = make_geometric_mean_transform(0.1)
    uid = register_hook(fn, name="test_geom_mean")
    assert isinstance(uid, str) and len(uid) == 16


def test_make_geometric_mean_transform_cache_idempotent():
    """Same temperature must return the same closure object (no registry leak)."""
    fn_a = make_geometric_mean_transform(0.5)
    fn_b = make_geometric_mean_transform(0.5)
    assert fn_a is fn_b, "Same temperature must return the same cached closure"


def test_geometric_mean_transform_temperature_effect():
    state_logits = jnp.array([[0.0, 1.0, -1.0], [0.0, -1.0, 1.0]])
    fn_hot = make_geometric_mean_transform(2.0)
    fn_cold = make_geometric_mean_transform(0.5)
    out_hot = fn_hot(state_logits, None, None)
    out_cold = fn_cold(state_logits, None, None)
    # cold (T=0.5) should produce 4x the magnitude of hot (T=2.0)
    assert jnp.allclose(out_cold, 4.0 * out_hot, atol=1e-5)


def test_mpnn_score_unconditional_no_temperature_param():
    import inspect
    import jax
    from prxteinmpnn.model.mpnn import PrxteinMPNN

    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))
    sig = inspect.signature(m.score_unconditional_state_vmap_exact)
    assert "multi_state_temperature" not in sig.parameters, (
        "multi_state_temperature must not appear in score_unconditional_state_vmap_exact"
    )
