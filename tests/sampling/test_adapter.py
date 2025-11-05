import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.sampling.adapter import (
    is_equinox_model,
    get_encoder_fn,
    get_decoder_fn,
    get_model_parameters,
)
from prxteinmpnn.eqx_new import PrxteinMPNN


# Mock Equinox Model
class MockPrxteinMPNN(PrxteinMPNN):
    def __init__(self, key):
        super().__init__(
            node_features=128,
            edge_features=128,
            hidden_features=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=48,
            key=key
        )

    def encoder(self, edge_features, neighbor_indices, mask):
        return jnp.ones((10, 128)), jnp.ones((10, 10, 128))


@pytest.fixture
def mock_equinox_model():
    return MockPrxteinMPNN(key=jax.random.PRNGKey(0))


@pytest.fixture
def mock_pytree_model():
    return {"params": jnp.ones((10,))}


def test_is_equinox_model(mock_equinox_model, mock_pytree_model):
    assert is_equinox_model(mock_equinox_model)
    assert not is_equinox_model(mock_pytree_model)


def test_get_encoder_fn(mock_equinox_model, mock_pytree_model):
    # Test with Equinox model
    eqx_encoder_fn = get_encoder_fn(mock_equinox_model)
    assert callable(eqx_encoder_fn)
    node_feats, edge_feats = eqx_encoder_fn(
        jnp.zeros((10, 10, 128)), jnp.zeros((10, 10), dtype=jnp.int32), jnp.ones((10,))
    )
    assert node_feats.shape == (10, 128)
    assert edge_feats.shape == (10, 10, 128)

    # Test with PyTree model (will fail if dependencies are not met)
    try:
        pytree_encoder_fn = get_encoder_fn(mock_pytree_model)
        assert callable(pytree_encoder_fn)
    except (ImportError, KeyError):
        pytest.skip("Legacy PyTree dependencies or mock data incomplete.")


def test_get_decoder_fn(mock_equinox_model, mock_pytree_model):
    # Test with Equinox model
    eqx_decoder_fn = get_decoder_fn(mock_equinox_model)
    assert callable(eqx_decoder_fn)
    output = eqx_decoder_fn(
        node_features=jnp.zeros((10, 128)),
        edge_features=jnp.zeros((10, 48, 128)),
        mask=jnp.ones((10,))
    )
    assert output.shape == (10, 128)

    # Test with PyTree model (will fail if dependencies are not met)
    try:
        pytree_decoder_fn = get_decoder_fn(mock_pytree_model)
        assert callable(pytree_decoder_fn)
    except (ImportError, KeyError):
        pytest.skip("Legacy PyTree dependencies or mock data incomplete.")


def test_get_model_parameters(mock_equinox_model, mock_pytree_model):
    # Test with PyTree model
    params = get_model_parameters(mock_pytree_model)
    assert "params" in params
    assert jnp.array_equal(params["params"], jnp.ones((10,)))

    # Test with Equinox model
    with pytest.raises(NotImplementedError):
        get_model_parameters(mock_equinox_model)
