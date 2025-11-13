"""Shared test fixtures."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax import random

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.types import ModelParameters


@pytest.fixture(scope="session")
def protein_structure() -> Protein:
    """Load a sample protein structure from a PDB file."""
    pdb_path = Path(__file__).parent / "data" / "1ubq.pdb"
    protein_tuple = next(parse_input(str(pdb_path)))
    return Protein.from_tuple(protein_tuple)


@pytest.fixture(scope="session")
def pqr_protein_tuple() -> "ProteinTuple":
    """Load a sample protein structure from a PQR file."""
    pqr_path = Path(__file__).parent / "data" / "1a00.pqr"
    return next(parse_input(str(pqr_path)))


@pytest.fixture(scope="session")
def model_inputs(protein_structure: Protein) -> dict:
    """Create model inputs from a protein structure."""
    return {
        "structure_coordinates": protein_structure.coordinates,
        "mask": protein_structure.mask,
        "residue_index": protein_structure.residue_index,
        "chain_index": protein_structure.chain_index,
        "sequence": protein_structure.aatype,
    }


@pytest.fixture
def rng_key() -> random.PRNGKey:
    """Create a new random key for testing."""
    return random.PRNGKey(0)


@pytest.fixture
def mock_model_parameters() -> ModelParameters:
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
    enc_prefix = f"protein_mpnn/~/enc_layer"
    if i > 0:
        enc_prefix += f"_{i}"
    dec_prefix = f"protein_mpnn/~/dec_layer"
    if i > 0:
        dec_prefix += f"_{i}"

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

