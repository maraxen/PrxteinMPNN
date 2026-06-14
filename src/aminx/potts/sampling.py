"""Single-site Gibbs updates and parallel tempering for pairwise Potts models on masked k-NN support.

References
----------
.. [Gibbs] Geman, S. and Geman, D. (1984). "Stochastic relaxation, Gibbs
   distributions, and the Bayesian restoration of images." IEEE TPAMI 6(6):721-741.
   https://doi.org/10.1109/TPAMI.1984.4767596
.. [PT] Hukushima, K. and Nemoto, K. (1996). "Exchange Monte Carlo method and
   application to spin glass simulations." J. Phys. Soc. Jpn. 65:1604-1608.
   https://doi.org/10.1143/JPSJ.65.1604
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def _potts_field_energy(
  h: Float[Array, "n q"],
  seq: Int[Array, " n"],
  mask: Float[Array, " n"],
) -> Float[Array, ""]:
  """Masked sum of local fields ``Σ_i m_i h_{i s_i}``."""
  m = mask.astype(h.dtype).reshape(-1)
  n = int(h.shape[0])
  idx = jnp.arange(n, dtype=jnp.int32)
  return jnp.sum(m * h[idx, seq])


def _potts_pair_energy(
  j: Float[Array, "n n q q"],
  w: Float[Array, "n n"],
  seq: Int[Array, " n"],
  mask: Float[Array, " n"],
) -> Float[Array, ""]:
  """Pair contribution ``0.5 * Σ_{i,k} w_{ik} m_i m_k J_{ik}(s_i,s_k)`` (undirected counting)."""
  m = mask.astype(j.dtype).reshape(-1)
  n = int(j.shape[0])
  idx = jnp.arange(n, dtype=jnp.int32)
  mm = m[:, None] * m[None, :]
  jj = j[idx[:, None], idx[None, :], seq[:, None], seq[None, :]]
  return 0.5 * jnp.sum(w * mm * jj)


def _potts_log_unnormalized(
  h: Float[Array, "n q"],
  j: Float[Array, "n n q q"],
  w: Float[Array, "n n"],
  seq: Int[Array, " n"],
  mask: Float[Array, " n"],
) -> Float[Array, ""]:
  """``log ψ(s)`` for ``ψ(s) ∝ exp(field + pairs)`` matching ``conditional_logits_site`` (symmetric ``J``, ``W``).

  Uses ``0.5 * Σ_{i,k} w_{ik} m_i m_k J_{ik}(s_i,s_k)`` so each undirected edge is counted once when ``W`` is symmetric.
  """
  return _potts_field_energy(h, seq, mask) + _potts_pair_energy(j, w, seq, mask)


def _conditional_logits_site(
  h: Float[Array, "n q"],
  j: Float[Array, "n n q q"],
  w: Float[Array, "n n"],
  seq: Int[Array, " n"],
  mask: Float[Array, " n"],
  i: Int[Array, ""],
  *,
  neighbor_scale: Float[Array, ""] | float = 1.0,
  inverse_temp: Float[Array, ""] | float = 1.0,
) -> Float[Array, " q"]:
  """Log-unnormalized scores for all amino acids at site ``i`` given ``seq`` elsewhere.

  ``neighbor_scale`` scales **pair** contributions only (tempering / soft coupling); ``h[i]`` unchanged.
  ``inverse_temp`` multiplies the **full** vector (field + scaled pairs) — use for **parallel tempering**
  replicas where ``π_k(s) ∝ ψ(s)^{β_k}`` with ascending ``β_k`` (hot → cold).
  """
  ns = jnp.asarray(neighbor_scale, dtype=h.dtype)
  inv = jnp.asarray(inverse_temp, dtype=h.dtype)
  n = int(h.shape[0])
  k_idx = jnp.arange(n, dtype=jnp.int32)

  def row(k: Int[Array, ""]) -> Float[Array, " q"]:
    return ns * w[i, k] * mask[k] * j[i, k, :, seq[k]]

  logits = h[i] + jnp.sum(jax.vmap(row)(k_idx), axis=0)
  return inv * logits


def gibbs_sweep(
  key: Array,
  seq: Int[Array, " n"],
  h: Float[Array, "n q"],
  j: Float[Array, "n n q q"],
  w: Float[Array, "n n"],
  mask: Float[Array, " n"],
  site_order: Int[Array, " n"] | None = None,
  neighbor_scale: Float[Array, ""] | float = 1.0,
  inverse_temp: Float[Array, ""] | float = 1.0,
) -> tuple[Int[Array, " n"], Array]:
  """One deterministic sweep over ``site_order`` (length ``n``); updates masked sites only.

  Args:
    key: JAX random key
    seq: Current sequence of shape (n,)
    h: Field energies of shape (n, q)
    j: Pairwise couplings of shape (n, n, q, q)
    w: Contact weights of shape (n, n)
    mask: Site activity mask of shape (n,)
    site_order: Visitation order of shape (n,); if None, uses [0, 1, ..., n-1]
    neighbor_scale: Scaling factor for pair contributions
    inverse_temp: Inverse temperature for full energy scaling

  Returns:
    Updated sequence and new random key
  """
  n = int(seq.shape[0])
  if site_order is None:
    site_order = jnp.arange(n, dtype=jnp.int32)

  def step(
    carry: tuple[Int[Array, " n"], Array], idx: Int[Array, ""],
  ) -> tuple[tuple[Int[Array, " n"], Array], None]:
    seq_in, k_in = carry
    i = site_order[idx]
    k2, kc = jax.random.split(k_in)
    logits = _conditional_logits_site(
      h,
      j,
      w,
      seq_in,
      mask,
      i,
      neighbor_scale=neighbor_scale,
      inverse_temp=inverse_temp,
    )
    new_aa = jax.random.categorical(kc, logits)
    active = mask[i] > 0.5
    new_aa_eff = jnp.where(active, new_aa, seq_in[i])
    seq_out = seq_in.at[i].set(new_aa_eff)
    return (seq_out, k2), None

  (seq_out, k_out), _ = jax.lax.scan(
    step, (seq, key), jnp.arange(int(site_order.shape[0]), dtype=jnp.int32),
  )
  return seq_out, k_out


def _attempt_adjacent_swap(
  key: Array,
  seqs: Int[Array, "k n"],
  i: Int[Array, ""],
  beta_lo: Float[Array, ""],
  beta_hi: Float[Array, ""],
  h: Float[Array, "n q"],
  j: Float[Array, "n n q q"],
  w: Float[Array, "n n"],
  mask: Float[Array, " n"],
  *,
  swap_pair_energy_only: bool,
) -> tuple[Int[Array, "k n"], Array, Float[Array, ""]]:
  """Metropolis swap between replicas ``i`` and ``i+1`` (``beta_hi > beta_lo``).

  If ``swap_pair_energy_only`` is True, target ``π_β(s) ∝ exp(F(s) + β P(s))`` with ``P`` the pair energy only
  (matches **pair-scaled** Gibbs conditionals ``neighbor_scale=β``). Otherwise ``π_β(s) ∝ ψ(s)^β`` with
  ``log ψ = F + P`` at reference coupling (full-``ψ`` **inverse temperature** on conditionals).
  """
  sa = seqs[i]
  sb = seqs[i + 1]
  if swap_pair_energy_only:
    la = _potts_pair_energy(j, w, sa, mask)
    lb = _potts_pair_energy(j, w, sb, mask)
  else:
    la = _potts_log_unnormalized(h, j, w, sa, mask)
    lb = _potts_log_unnormalized(h, j, w, sb, mask)
  log_alpha = (beta_hi - beta_lo) * (la - lb)
  key, ku = jax.random.split(key)
  u = jax.random.uniform(ku, dtype=log_alpha.dtype)
  accept = u < jnp.exp(jnp.minimum(0.0, log_alpha))
  seq_out = seqs.at[i].set(jnp.where(accept, sb, sa)).at[i + 1].set(jnp.where(accept, sa, sb))
  return seq_out, key, accept.astype(jnp.float32)


def _parallel_tempering_exchange(
  key: Array,
  seqs: Int[Array, "k n"],
  betas: Float[Array, " k"],
  h: Float[Array, "n q"],
  j: Float[Array, "n n q q"],
  w: Float[Array, "n n"],
  mask: Float[Array, " n"],
  *,
  swap_pair_energy_only: bool = False,
) -> tuple[Int[Array, "k n"], Array, Float[Array, " k-1"]]:
  """One exchange phase: even-adjacent then odd-adjacent Metropolis swaps.

  Returns ``accept_edge`` with shape ``(k-1,)``: ``1.0`` if the swap for edge ``(i, i+1)`` was accepted,
  ``0.0`` if rejected (each undirected edge is attempted exactly once per call).
  """
  k_rep = int(seqs.shape[0])
  seq_cur = seqs
  key_cur = key
  accept_edge = jnp.zeros((k_rep - 1,), dtype=jnp.float32)

  def process_parity(parity: int) -> tuple[Int[Array, "k n"], Float[Array, ""], Array]:
    """Process one parity group (even or odd) using vmap over non-overlapping edges."""
    nonlocal seq_cur, accept_edge, key_cur

    # Compute edge indices for this parity: 0,2,4,... (even) or 1,3,5,... (odd)
    edges = jnp.arange(parity, k_rep - 1, 2, dtype=jnp.int32)
    n_edges = edges.shape[0]

    if n_edges == 0:
      return seq_cur, accept_edge, key_cur

    # Split keys for each edge in this parity
    keys = jax.random.split(key_cur, n_edges + 1)
    key_cur_new, subkeys = keys[0], keys[1:]

    # Helper function: perform one swap and return updated full seqs and accept
    def swap_fn(sk: Array, edge_i: Int[Array, ""]) -> tuple[Int[Array, "k n"], Float[Array, ""]]:
      """Swap edge_i and return updated seqs and accept scalar."""
      seq_out, _, acc = _attempt_adjacent_swap(
        sk,
        seq_cur,
        edge_i,
        betas[edge_i],
        betas[edge_i + 1],
        h,
        j,
        w,
        mask,
        swap_pair_energy_only=swap_pair_energy_only,
      )
      return seq_out, acc

    # Vmap over edges: in_axes: keys axis 0, edges axis 0; seq_cur is in_axes=None
    vmapped_fn = jax.vmap(swap_fn)
    seq_updates, accepts_per_edge = vmapped_fn(subkeys, edges)

    # Merge all updates using lax.scan (JAX-friendly, JIT-compatible)
    # seq_updates has shape (n_edges, k_rep, n) where seq_updates[idx] has full seqs with swaps at edges[idx], edges[idx]+1
    def merge_update(seq_state: Int[Array, "k n"], idx_val: int) -> tuple[Int[Array, "k n"], None]:
      # seq_updates[idx_val] is the full (k_rep, n) array with swap at edges[idx_val], edges[idx_val]+1
      update_seq = seq_updates[idx_val]
      edge_idx = edges[idx_val]
      seq_state = seq_state.at[edge_idx].set(update_seq[edge_idx])
      seq_state = seq_state.at[edge_idx + 1].set(update_seq[edge_idx + 1])
      return seq_state, None

    # Create indices array for lax.scan
    indices = jnp.arange(n_edges, dtype=jnp.int32)
    seq_updated, _ = jax.lax.scan(merge_update, seq_cur, indices)

    # Build accept_edge for this parity
    accept_edge_updated = accept_edge.at[edges].set(accepts_per_edge)

    return seq_updated, accept_edge_updated, key_cur_new

  # Process even parity (0, 2, 4, ...)
  seq_cur, accept_edge, key_cur = process_parity(0)

  # Process odd parity (1, 3, 5, ...) — uses seq_cur updated by even parity
  seq_cur, accept_edge, key_cur = process_parity(1)

  return seq_cur, key_cur, accept_edge


def parallel_tempering_round(
  key: Array,
  seqs: Int[Array, "k n"],
  betas: Float[Array, " k"],
  h: Float[Array, "n q"],
  j: Float[Array, "n n q q"],
  w: Float[Array, "n n"],
  mask: Float[Array, " n"],
) -> tuple[Int[Array, "k n"], Array, Float[Array, " k-1"]]:
  """One Gibbs sweep per replica, then one exchange phase.

  **Full-``ψ`` tempering** (default): each replica uses ``inverse_temp=β_k``,
  ``neighbor_scale=1`` (same stationary idea as ``--pt-betas`` pilots).

  Args:
    key: JAX random key
    seqs: Replica sequences of shape (n_replicas, N); n_replicas is STATIC (no Python branch on count)
    betas: Inverse temperatures of shape (n_replicas,)
    h: Field energies of shape (N, q)
    j: Pairwise couplings of shape (N, N, q, q)
    w: Contact weights of shape (N, N)
    mask: Site activity mask of shape (N,)

  Returns:
    Updated sequences, new random key, and acceptance rates for each replica edge
  """
  k_rep, n = int(seqs.shape[0]), int(seqs.shape[1])
  keys = jax.random.split(key, k_rep + 1)
  key_out = keys[0]

  def one_sweep(seq: Int[Array, " n"], kr: Array, beta: Float[Array, ""]) -> Int[Array, " n"]:
    kperm, kg = jax.random.split(kr)
    order = jax.random.permutation(kperm, n)
    # Full-psi tempering: inverse_temp=beta, neighbor_scale=1
    seq2, _ = gibbs_sweep(
      kg,
      seq,
      h=h,
      j=j,
      w=w,
      mask=mask,
      site_order=order,
      neighbor_scale=1.0,
      inverse_temp=beta,
    )
    return seq2

  seq_next = jax.vmap(one_sweep)(seqs, keys[1:], betas)
  return _parallel_tempering_exchange(
    key_out,
    seq_next,
    betas,
    h,
    j,
    w,
    mask,
    swap_pair_energy_only=False,
  )


def _gibbs_one_round_chains(
  seqs: Int[Array, "c n"],
  key: Array,
  h: Float[Array, "n q"],
  j: Float[Array, "n n q q"],
  w: Float[Array, "n n"],
  mask: Float[Array, " n"],
) -> tuple[Int[Array, "c n"], Array]:
  """One randomized synchronous sweep per chain."""
  c, n = int(seqs.shape[0]), int(seqs.shape[1])
  key, kperm, kg = jax.random.split(key, 3)
  kperm_c = jax.random.split(kperm, c)
  kg_c = jax.random.split(kg, c)
  orders = jax.vmap(lambda kk: jax.random.permutation(kk, n), in_axes=0)(kperm_c)

  def _one(s: Int[Array, " n"], g: Array, o: Int[Array, " n"]) -> Int[Array, " n"]:
    s2, _ = gibbs_sweep(
      g,
      s,
      h=h,
      j=j,
      w=w,
      mask=mask,
      site_order=o,
      neighbor_scale=1.0,
      inverse_temp=1.0,
    )
    return s2

  seqs2 = jax.vmap(_one)(seqs, kg_c, orders)
  return seqs2, key


def log_energy(
  seq: Int[Array, " n"],
  h: Float[Array, "n q"],
  j: Float[Array, "n n q q"],
  w: Float[Array, "n n"],
  mask: Float[Array, " n"],
) -> Float[Array, ""]:
  """Compute log-unnormalized energy for a Potts sequence.

  E(seq) = sum_i m_i h[i, seq[i]] + 0.5 * sum_{i,k} m_i m_k w[i,k] J[i,k,seq[i],seq[k]]

  Args:
    seq: Sequence indices of shape (n,)
    h: Field energies of shape (n, q)
    j: Pairwise couplings of shape (n, n, q, q)
    w: Contact weights of shape (n, n)
    mask: Site activity mask of shape (n,)

  Returns:
    Scalar energy (Float32)
  """
  return _potts_log_unnormalized(h, j, w, seq, mask)


def parallel_tempering(
  key: Array,
  seq: Int[Array, " n"],
  h: Float[Array, "n q"],
  j: Float[Array, "n n q q"],
  w: Float[Array, "n n"],
  mask: Float[Array, " n"],
  n_replicas: int = 4,
  temperatures: Float[Array, " n_replicas"] | None = None,
  n_sweeps: int = 100,
) -> tuple[Int[Array, "n_replicas n"], Float[Array, " n_replicas"]]:
  """Parallel tempering MCMC for sampling from Potts distribution.

  Runs independent Gibbs sweeps on n_replicas chains at different temperatures,
  with periodic replica-exchange moves (Metropolis-Hastings).

  Args:
    key: JAX random key
    seq: Initial sequence of shape (n,)
    h: Field energies of shape (n, q)
    j: Pairwise couplings of shape (n, n, q, q)
    w: Contact weights of shape (n, n)
    mask: Site activity mask of shape (n,)
    n_replicas: Number of replicas (default 4)
    temperatures: Inverse temperatures (betas) of shape (n_replicas,).
      If None, uses linspace(0.5, 2.0, n_replicas).
    n_sweeps: Number of Gibbs + exchange rounds per replica (default 100)

  Returns:
    Tuple of (final_sequences, final_energies) where:
      - final_sequences: shape (n_replicas, n)
      - final_energies: shape (n_replicas,)
  """
  # Handle default temperatures
  if temperatures is None:
    temperatures = jnp.linspace(0.5, 2.0, n_replicas, dtype=h.dtype)
  else:
    temperatures = jnp.asarray(temperatures, dtype=h.dtype)

  # Validate temperature count
  if int(temperatures.shape[0]) != n_replicas:
    msg = (
      f"temperatures shape mismatch: got {int(temperatures.shape[0])} "
      f"but n_replicas={n_replicas}"
    )
    raise ValueError(msg)

  # Initialize replicas from the same starting sequence
  seqs = jnp.tile(seq[None, :], (n_replicas, 1)).astype(jnp.int32)

  # Run PT rounds
  def body(
    _i: Int[Array, ""], carry: tuple[Int[Array, "k n"], Array],
  ) -> tuple[Int[Array, "k n"], Array]:
    seqs_in, k_in = carry
    seqs_out, k_out, _ = parallel_tempering_round(
      k_in,
      seqs_in,
      temperatures,
      h,
      j,
      w,
      mask,
    )
    return (seqs_out, k_out)

  (seqs_final, _key_out) = jax.lax.fori_loop(
    jnp.int32(0),
    jnp.int32(n_sweeps),
    body,
    (seqs, key),
  )

  # Compute final energies
  energies = jax.vmap(lambda s: log_energy(s, h, j, w, mask))(seqs_final)

  return seqs_final, energies


@functools.partial(jax.jit, static_argnames=("num_chains", "num_sweeps", "num_aa"))
def gibbs_empirical_marginals(
  key: Array,
  h: Float[Array, "n q"],
  j: Float[Array, "n n q q"],
  w: Float[Array, "n n"],
  mask: Float[Array, " n"],
  *,
  num_chains: int,
  num_sweeps: int,
  num_aa: int,
) -> Float[Array, "n num_aa"]:
  """Mean one-hot empirical marginals from ``num_chains`` parallel chains after ``num_sweeps`` sweeps.

  Used for sampling-based inference: compute empirical marginals that match model predictions.

  Args:
    key: JAX random key
    h: Field energies of shape (n, q)
    j: Pairwise couplings of shape (n, n, q, q)
    w: Contact weights of shape (n, n)
    mask: Site activity mask of shape (n,)
    num_chains: Number of parallel MCMC chains (STATIC)
    num_sweeps: Number of Gibbs sweeps per chain (STATIC)
    num_aa: Number of amino acid states q (STATIC)

  Returns:
    Empirical marginals of shape (n, num_aa) representing empirical distribution over states
  """
  n = int(h.shape[0])
  q = int(num_aa)
  key, kinit = jax.random.split(key)
  seqs = jax.random.randint(kinit, (num_chains, n), 0, q, dtype=jnp.int32)

  def body(
    _i: Int[Array, ""], carry: tuple[Int[Array, "c n"], Array],
  ) -> tuple[Int[Array, "c n"], Array]:
    seq_in, k_in = carry
    seq_out, k2 = _gibbs_one_round_chains(seq_in, k_in, h, j, w, mask)
    return seq_out, k2

  seq_f, _ = jax.lax.fori_loop(
    jnp.int32(0),
    jnp.int32(int(num_sweeps)),
    body,
    (seqs, key),
  )
  oh = jax.nn.one_hot(seq_f, q).astype(jnp.float32)
  return jnp.mean(oh, axis=0)
