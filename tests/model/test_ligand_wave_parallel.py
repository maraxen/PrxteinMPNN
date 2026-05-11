"""Wave-parallel autoregressive path for ``PrxteinLigandMPNN`` (parity with plain MPNN wiring)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.model.ligand_mpnn import PrxteinLigandMPNN
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.utils.wave_parallel import compute_wave_assignments


def _tie_map_compact(seq_len: int, pairs: list[tuple[int, int]]) -> np.ndarray:
  """Map paired positions to compact group IDs 0..G-1."""
  tie = np.arange(seq_len, dtype=np.int32)
  for a, b in pairs:
    tie[b] = tie[a]
  _, inv = np.unique(tie, return_inverse=True)
  return inv.astype(np.int32, copy=False)


@pytest.mark.parametrize("use_wave", [False, True])
def test_ligand_tied_autoregressive_wave_kwarg_smoke(use_wave: bool) -> None:
  """Ligand MPNN tied sampling runs with or without wave tables (no ``TypeError`` on kwargs)."""
  key = jax.random.PRNGKey(42)
  seq_len = 12
  n_canonical = 6
  k_nbr = 8
  model = PrxteinLigandMPNN(
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=k_nbr,
    num_context_layers=2,
    dropout_rate=0.0,
    key=key,
  )
  rng = np.random.default_rng(0)
  coords = jnp.asarray(rng.normal(size=(seq_len, 4, 3)), dtype=jnp.float32)
  mask = jnp.ones((seq_len,), dtype=jnp.float32)
  res = jnp.arange(seq_len, dtype=jnp.int32)
  chain = jnp.zeros((seq_len,), dtype=jnp.int32)
  y = jnp.asarray(rng.normal(size=(seq_len, 4, 3)), dtype=jnp.float32)
  y_t = jnp.ones((seq_len, 4), dtype=jnp.int32)
  y_m = jnp.ones((seq_len, 4), dtype=jnp.float32)

  pairs = [(i * 2, i * 2 + 1) for i in range(n_canonical)]
  tie_flat = _tie_map_compact(seq_len, pairs)
  num_groups = int(tie_flat.max()) + 1
  assert num_groups == n_canonical
  max_gs = 2
  git = np.full((num_groups, max_gs), -1, dtype=np.int32)
  gvt = np.zeros((num_groups, max_gs), dtype=bool)
  for g in range(num_groups):
    mem = np.where(tie_flat == g)[0]
    git[g, : len(mem)] = mem
    gvt[g, : len(mem)] = True

  ca_for_wave = np.asarray(coords[::2], dtype=np.float32)
  assert ca_for_wave.shape == (n_canonical, 4, 3)
  w_id, w_pos, w_gv, w_pv = compute_wave_assignments(
    ca_for_wave,
    tie_flat,
    git,
    gvt,
    k_neighbors=k_nbr,
    n_canonical=n_canonical,
  )

  _, sk = jax.random.split(key)

  wave_kw: dict = {}
  if use_wave:
    wave_kw = {
      "wave_group_ids": jnp.asarray(w_id, dtype=jnp.int32),
      "wave_group_positions": jnp.asarray(w_pos, dtype=jnp.int32),
      "wave_group_valid": jnp.asarray(w_gv, dtype=bool),
      "wave_position_valid": jnp.asarray(w_pv, dtype=bool),
    }
  else:
    wave_kw = {
      "wave_group_ids": None,
      "wave_group_positions": None,
      "wave_group_valid": None,
      "wave_position_valid": None,
    }

  sampler = make_sample_sequences(model, sampling_strategy="temperature")
  seq, logits, _ = sampler(
    sk,
    coords,
    mask,
    res,
    chain,
    temperature=jnp.array(0.1, dtype=jnp.float32),
    tie_group_map=jnp.asarray(tie_flat, dtype=jnp.int32),
    num_groups=num_groups,
    group_indices_table=jnp.asarray(git, dtype=jnp.int32),
    group_valid_table=jnp.asarray(gvt, dtype=bool),
    Y=y,
    Y_t=y_t,
    Y_m=y_m,
    **wave_kw,
  )
  assert seq.shape == (seq_len,)
  assert logits.shape == (seq_len, 21)
  assert jnp.all(jnp.isfinite(logits))
