"""Debug autoregressive sampling with detailed prints."""

import jax
import jax.numpy as jnp
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import random_decoding_order


def test_debug_first_positions():
  """Debug the first 3 positions of autoregressive sampling."""
  # Load
  protein_tuple = next(parse_input("tests/data/1ubq.pdb"))
  protein = Protein.from_tuple(protein_tuple)
  model = load_model()

  # Setup - use a simple ordered decoding for clarity
  key = jax.random.key(42)
  decoding_order = jnp.arange(len(protein.aatype))  # Simple 0, 1, 2, ...
  ar_mask = generate_ar_mask(decoding_order, None, None, None)

  print("=" * 80)
  print("DEBUG: Autoregressive sampling (first positions only)")
  print("=" * 80)
  print(f"Decoding order: {decoding_order[:10]}")
  print(f"Native sequence: {protein.aatype[:10]}")
  print()

  # Sample with debug prints
  seq_autoregressive, _ = model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    decoding_approach="autoregressive",
    prng_key=key,
    ar_mask=ar_mask,
    temperature=jnp.array(0.1, dtype=jnp.float32),
  )

  if seq_autoregressive.ndim == 2:
    seq_autoregressive = jnp.argmax(seq_autoregressive, axis=-1)

  print("\n" + "=" * 80)
  print(f"Final sampled sequence: {seq_autoregressive[:10]}")
  print("=" * 80)


if __name__ == "__main__":
  test_debug_first_positions()
