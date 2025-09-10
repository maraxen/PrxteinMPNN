"""Tests for ensemble residue states functionality."""

import jax
import jax.numpy as jnp
import pytest
import chex
from unittest.mock import Mock, patch
from prxteinmpnn.ensemble.residue_states import residue_states_from_ensemble
from prxteinmpnn.utils.data_structures import Protein, ProteinEnsemble
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import ModelParameters

L_GLOBAL = 10  # Define a global sequence length for consistency
K_NEIGHBORS = 48  # Hardcoded in residue_states.py


@pytest.fixture
def mock_model_parameters():
  """Create a complete and structurally correct set of mock model parameters."""
  # Model dimensions
  C_V = 128  # Node feature dimension
  C_E = 128  # Edge feature dimension
  INITIAL_EDGE_FEATURES = 528
  ENCODER_MLP_INPUT_DIM = C_V + C_V + C_E  # 384
  # CORRECTED: Define the decoder's specific input dimension
  DECODER_MLP_INPUT_DIM = C_V + ENCODER_MLP_INPUT_DIM # 128 + 384 = 512
  NUM_AMINO_ACIDS = 21
  MAXIMUM_RELATIVE_FEATURES = 32
  pos_enc_dim = 2 * MAXIMUM_RELATIVE_FEATURES + 2

  # Helper to create mock linear layer parameters
  def _make_linear_params(d_in, d_out):
    return {"w": jax.random.normal(jax.random.PRNGKey(0), (d_in, d_out)), "b": jax.random.normal(jax.random.PRNGKey(0), (d_out,))}

  # Helper to create mock norm layer parameters
  def _make_norm_params(dim=C_V):
    return {"scale": jnp.ones((dim,)), "offset": jnp.zeros((dim,))}

  params = {
    # Feature extraction parameters
    "protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear": _make_linear_params(
      pos_enc_dim, C_E
    ),
    "protein_mpnn/~/protein_features/~/edge_embedding": _make_linear_params(
      INITIAL_EDGE_FEATURES, C_E
    ),
    "protein_mpnn/~/protein_features/~/norm_edges": _make_norm_params(dim=C_E),
    # Main model parameters
    "protein_mpnn/~/W_e": _make_linear_params(C_E, C_E),
    "protein_mpnn/~/embed_token": {"W_s": jnp.ones((NUM_AMINO_ACIDS, C_V))},
    "protein_mpnn/~/W_out": _make_linear_params(C_V, NUM_AMINO_ACIDS),
  }

  # Encoder and Decoder Layers
  for i in range(3):
    enc_l_name = f"enc{i}"
    dec_l_name = f"dec{i}"
    enc_prefix = f"protein_mpnn/~/enc_layer_{i}" if i > 0 else "protein_mpnn/~/enc_layer"
    dec_prefix = f"protein_mpnn/~/dec_layer_{i}" if i > 0 else "protein_mpnn/~/dec_layer"

    # Encoder
    params.update(
      {
        f"{enc_prefix}/~/{enc_l_name}_norm1": _make_norm_params(),
        f"{enc_prefix}/~/{enc_l_name}_norm2": _make_norm_params(),
        f"{enc_prefix}/~/{enc_l_name}_norm3": _make_norm_params(),
        f"{enc_prefix}/~/{enc_l_name}_W1": _make_linear_params(ENCODER_MLP_INPUT_DIM, C_V),
        f"{enc_prefix}/~/{enc_l_name}_W2": _make_linear_params(C_V, C_V),
        f"{enc_prefix}/~/{enc_l_name}_W3": _make_linear_params(C_V, C_V),
        f"{enc_prefix}/~/{enc_l_name}_W11": _make_linear_params(ENCODER_MLP_INPUT_DIM, C_E),
        f"{enc_prefix}/~/{enc_l_name}_W12": _make_linear_params(C_E, C_E),
        f"{enc_prefix}/~/{enc_l_name}_W13": _make_linear_params(C_E, C_E),
        f"{enc_prefix}/~/position_wise_feed_forward/~/{enc_l_name}_dense_W_in": _make_linear_params(
          C_V, C_V
        ),
        f"{enc_prefix}/~/position_wise_feed_forward/~/{enc_l_name}_dense_W_out": _make_linear_params(
          C_V, C_V
        ),
      }
    )
    # Decoder
    params.update(
      {
        f"{dec_prefix}/~/{dec_l_name}_norm1": _make_norm_params(),
        f"{dec_prefix}/~/{dec_l_name}_norm2": _make_norm_params(),
        # CORRECTED: The decoder's W1 layer expects a 512-dim input.
        f"{dec_prefix}/~/{dec_l_name}_W1": _make_linear_params(DECODER_MLP_INPUT_DIM, C_V),
        f"{dec_prefix}/~/{dec_l_name}_W2": _make_linear_params(C_V, C_V),
        f"{dec_prefix}/~/{dec_l_name}_W3": _make_linear_params(C_V, C_V),
        f"{dec_prefix}/~/position_wise_feed_forward/~/{dec_l_name}_dense_W_in": _make_linear_params(
          C_V, C_V
        ),
        f"{dec_prefix}/~/position_wise_feed_forward/~/{dec_l_name}_dense_W_out": _make_linear_params(
          C_V, C_V
        ),
      }
    )

  return params


@pytest.fixture
def mock_decoding_order_fn():
    """Create mock decoding order function."""
    def mock_fn(prng_key, seq_len):
        return jnp.arange(seq_len, dtype=jnp.int32), prng_key
    return mock_fn


@pytest.fixture
def mock_protein_ensemble() -> ProteinEnsemble:
    """Create a mock protein ensemble with 3 frames, all with the same sequence."""

    async def _ensemble_generator() -> ProteinEnsemble:
        # Use a fixed sequence to make alignment predictable (no change in length)
        aatype = jax.random.randint(jax.random.PRNGKey(100), (L_GLOBAL,), 0, 21, dtype=jnp.int8)
        for i in range(3):
            protein = Protein(
                coordinates=jax.random.normal(jax.random.PRNGKey(i), (L_GLOBAL, 37, 3)),
                aatype=aatype,
                atom_mask=jnp.ones((L_GLOBAL, 37), dtype=jnp.int32),
                residue_index=jnp.arange(L_GLOBAL, dtype=jnp.int32),
                chain_index=jnp.zeros(L_GLOBAL, dtype=jnp.int32),
            )
            yield protein, f"protein_{i}"

    return _ensemble_generator()


@pytest.mark.anyio
@patch('prxteinmpnn.ensemble.residue_states.make_conditional_logits_fn')
async def test_residue_states_from_ensemble_structure(
    mock_make_logits_fn, mock_model_parameters, mock_decoding_order_fn, mock_protein_ensemble
):
    """Tests the output structure and length from a standard ensemble."""
    n_frames = 3
    key = jax.random.PRNGKey(42)

    # Mock the function that returns the features tuple
    mock_logits_fn = Mock(return_value=(
        jnp.ones((L_GLOBAL, 21)),
        jnp.zeros((L_GLOBAL, 128)),
        jnp.zeros((L_GLOBAL, K_NEIGHBORS, 128)),
    ))
    mock_make_logits_fn.return_value = mock_logits_fn

    states_generator = residue_states_from_ensemble(
        prng_key=key,
        model_parameters=mock_model_parameters,
        decoding_order_fn=mock_decoding_order_fn,
        ensemble=mock_protein_ensemble,
    )
    states_list = [s async for s in states_generator]

    # 1. Check the number of yielded items
    assert len(states_list) == n_frames, "Should yield one tuple per frame in the ensemble."

    # 2. Check the structure of each yielded item
    ref_seq, (logits, node_features, edge_features) = states_list[0]

    chex.assert_shape(ref_seq, (L_GLOBAL,))
    chex.assert_shape(logits, (L_GLOBAL, 21))
    chex.assert_shape(node_features, (L_GLOBAL, 128))
    chex.assert_shape(edge_features, (L_GLOBAL, K_NEIGHBORS, 128))


@pytest.mark.anyio
async def test_residue_states_empty_ensemble(mock_model_parameters, mock_decoding_order_fn):
  """Tests that an empty ensemble correctly yields no results."""

  async def _empty_generator():
      if False:
          yield

  states_generator = residue_states_from_ensemble(
      prng_key=jax.random.PRNGKey(42),
      model_parameters=mock_model_parameters,
      decoding_order_fn=mock_decoding_order_fn,
      ensemble=_empty_generator(),
  )
  states_list = [s async for s in states_generator]
  assert not states_list


@pytest.mark.anyio
@patch('prxteinmpnn.ensemble.residue_states.make_conditional_logits_fn')
async def test_residue_states_single_frame_ensemble(
    mock_make_logits_fn, mock_model_parameters, mock_decoding_order_fn
):
    """Tests that an ensemble with a single frame yields exactly one result."""
    key = jax.random.PRNGKey(456)

    async def _single_generator():
        protein = Protein(
            coordinates=jax.random.normal(key, (L_GLOBAL, 37, 3)),
            aatype=jax.random.randint(key, (L_GLOBAL,), 0, 21, dtype=jnp.int8),
            atom_mask=jnp.ones((L_GLOBAL, 37), dtype=jnp.int32),
            residue_index=jnp.arange(L_GLOBAL, dtype=jnp.int32),
            chain_index=jnp.zeros(L_GLOBAL, dtype=jnp.int32),
        )
        yield protein, "single_protein"

    mock_make_logits_fn.return_value = Mock(return_value=(None, None, None))

    states_generator = residue_states_from_ensemble(
        prng_key=key,
        model_parameters=mock_model_parameters,
        decoding_order_fn=mock_decoding_order_fn,
        ensemble=_single_generator(),
    )
    states_list = [s async for s in states_generator]

    assert len(states_list) == 1, "A single-frame ensemble should yield exactly one result."


@pytest.mark.anyio
@patch('jax.random.fold_in')
@patch('prxteinmpnn.ensemble.residue_states.make_conditional_logits_fn')
async def test_prng_key_generation_with_fold_in(
    mock_make_logits_fn, mock_fold_in, mock_model_parameters, mock_decoding_order_fn, mock_protein_ensemble
):
    """Tests that PRNG keys are correctly generated using fold_in for each frame."""
    key = jax.random.PRNGKey(789)
    n_frames = 3
    # The actual return value doesn't matter, only the number of calls
    mock_make_logits_fn.return_value.return_value = (None, None, None)

    # Consume the generator to trigger the calls
    states_generator = residue_states_from_ensemble(
        prng_key=key,
        model_parameters=mock_model_parameters,
        decoding_order_fn=mock_decoding_order_fn,
        ensemble=mock_protein_ensemble,
    )
    _ = [s async for s in states_generator]

    # The main key is folded in for each frame.
    assert mock_fold_in.call_count == n_frames

    # Check that it was called with indices 0, 1, 2, ...
    for i in range(n_frames):
        call_index = mock_fold_in.call_args_list[i].args[1]
        assert call_index == i