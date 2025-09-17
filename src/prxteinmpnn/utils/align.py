"""Utilities for aligning proteins."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from collections.abc import Callable

  from prxteinmpnn.utils.types import OneHotProteinSequence, ProteinSequence


import jax
import jax.numpy as jnp


def smith_waterman_no_gap(unroll_factor: int = 2, *, batch: bool = True) -> Callable:
  """Get a JAX-jit function for Smith-Waterman (local alignment) with no gap penalty.

  Args:
    unroll_factor (int): The unroll parameter for `jax.lax.scan` for performance tuning.
    batch (bool): If True, the returned function will be vmapped for batch processing.

  Returns:
    Callable: A function that performs the alignment traceback.

  """

  def rotate_matrix(
    score_matrix: jax.Array,
    mask: jax.Array | None = None,
  ) -> tuple[dict[str, jax.Array], tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Rotate the score matrix for striped dynamic programming."""
    a, b = score_matrix.shape
    ar, br = jnp.arange(a)[::-1, None], jnp.arange(b)[None, :]
    i, j = (br - ar) + (a - 1), (ar + br) // 2
    n, m = (a + b - 1), (a + b) // 2
    zero = jnp.zeros([n, m])
    if mask is None:
      mask = jnp.ones(score_matrix.shape, dtype=jnp.bool_)
    rotated_data = {
      "score": zero.at[i, j].set(score_matrix),
      "mask": zero.at[i, j].set(mask),
      "parity": (jnp.arange(n) + a % 2) % 2,
    }
    previous_scores = (jnp.zeros(m), jnp.zeros(m))
    return rotated_data, previous_scores, (i, j)

  def compute_scoring_matrix(
    score_matrix: jax.Array,
    masks: tuple[jax.Array, jax.Array],
    temperature: float = 1.0,
  ) -> jax.Array:
    """Compute the scoring matrix for Smith-Waterman alignment.

    Args:
      score_matrix (jax.Array): The input score matrix.
      masks (tuple): A tuple of boolean masks for the two sequences.
      temperature (float): The temperature parameter for the soft maximum function.

    Returns:
      jax.Array: The maximum score in the scoring matrix.

    """

    def _soft_maximum(values: jax.Array, axis: int | None = None) -> jax.Array:
      """Compute the soft maximum of values along a specified axis."""
      return temperature * jax.nn.logsumexp(values / temperature, axis)

    def _conditional_select(
      condition: jax.Array,
      true_value: jax.Array,
      false_value: jax.Array,
    ) -> jax.Array:
      """Select values based on a boolean condition."""
      return condition * true_value + (1 - condition) * false_value

    def _scan_step(
      previous_scores: tuple[jax.Array, jax.Array],
      rotated_data: dict[str, jax.Array],
    ) -> tuple:
      """Perform a single step of the scan for computing the scoring matrix."""
      h_previous, h_current = previous_scores  # previous two rows of scoring (hij) mtx
      h_current_shifted = _conditional_select(
        rotated_data["parity"],
        jnp.pad(h_current[:-1], [1, 0]),
        jnp.pad(h_current[1:], [0, 1]),
      )
      h_combined = jnp.stack([h_previous + rotated_data["score"], h_current, h_current_shifted], -1)
      h_masked = rotated_data["mask"] * _soft_maximum(h_combined, -1)
      return (h_current, h_masked), h_masked

    mask_a, mask_b = masks
    combined_mask = mask_a[:, None] & mask_b[None, :]

    rotated_data, previous_scores, indices = rotate_matrix(score_matrix, mask=combined_mask)
    final_scores = jax.lax.scan(_scan_step, previous_scores, rotated_data, unroll=unroll_factor)[
      -1
    ][indices]
    return final_scores.max()

  traceback_function = jax.grad(compute_scoring_matrix, argnums=0)
  return jax.vmap(traceback_function, (0, 0, None)) if batch else traceback_function


def smith_waterman(unroll_factor: int = 2, ninf: float = -1e30, *, batch: bool = True) -> Callable:
  """Get a JAX-jit function for Smith-Waterman (local alignment) with a gap penalty.

  Args:
    unroll_factor (int): The unroll parameter for `jax.lax.scan` for performance tuning.
    ninf (float): A large negative number representing negative infinity, used for padding.
    batch (bool): If True, the returned function will be vmapped for batch processing.

  Returns:
    Callable: A function that performs the alignment traceback.

  """

  def _rotate_matrix(
    score_matrix: jax.Array,
  ) -> tuple[dict[str, jax.Array], tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Rotate the score matrix for striped dynamic programming."""
    a, b = score_matrix.shape
    ar, br = jnp.arange(a)[::-1, None], jnp.arange(b)[None, :]
    i, j = (br - ar) + (a - 1), (ar + br) // 2
    n, m = (a + b - 1), (a + b) // 2
    rotated_data = {
      "score": jnp.full([n, m], ninf).at[i, j].set(score_matrix),
      "parity": (jnp.arange(n) + a % 2) % 2,
    }
    previous_scores = (jnp.full(m, ninf), jnp.full(m, ninf))
    return rotated_data, previous_scores, (i, j)

  def _compute_scoring_matrix(
    score_matrix: jax.Array,
    masks: tuple[jax.Array, jax.Array],
    gap: float = 0.0,
    temperature: float = 1.0,
  ) -> jax.Array:
    """Compute the scoring matrix for Smith-Waterman alignment with gap penalty.

    Args:
      score_matrix (jax.Array): The input score matrix.
      masks (tuple): A tuple of boolean masks for the two sequences.
      gap (float): The gap penalty.
      temperature (float): The temperature parameter for the soft maximum function.

    Returns:
      jax.Array: The maximum score in the scoring matrix.

    """

    def _soft_maximum(
      values: jax.Array,
      axis: int | None = None,
      mask: jax.Array | None = None,
    ) -> jax.Array:
      """Compute the soft maximum of values along a specified axis."""
      values = jnp.maximum(values, ninf)
      if mask is None:
        return temperature * jax.nn.logsumexp(values / temperature, axis)

      max_values = values.max(axis, keepdims=True)
      return temperature * (
        max_values
        + jnp.log(jnp.sum(mask * jnp.exp((values - max_values) / temperature), axis=axis))
      )

    def _conditional_select(
      condition: jax.Array,
      true_value: jax.Array,
      false_value: jax.Array,
    ) -> jax.Array:
      """Select values based on a boolean condition."""
      return condition * true_value + (1 - condition) * false_value

    def _pad(vals: jax.Array, shape: list) -> jax.Array:
      """Pad an array with negative infinity values."""
      return jnp.pad(vals, shape, constant_values=(ninf, ninf))

    def _step(previous_scores: tuple, rotated_data: dict) -> tuple:
      previous_row, current_row = previous_scores
      shifted_row = _conditional_select(
        rotated_data["parity"],
        _pad(current_row[:-1], [1, 0]),
        _pad(current_row[1:], [0, 1]),
      )
      combined_scores = jnp.stack(
        [
          previous_row + rotated_data["score"],
          current_row + gap,
          shifted_row + gap,
          rotated_data["score"],
        ],
        axis=-1,
      )
      updated_row = _soft_maximum(combined_scores, axis=-1)
      return (
        current_row,
        updated_row,
      ), updated_row

    mask_a, mask_b = masks
    combined_mask = mask_a[:, None] & mask_b[None, :]
    score_matrix = jnp.where(combined_mask, score_matrix, ninf)

    rotated_data, previous_scores, indices = _rotate_matrix(score_matrix[:-1, :-1])
    _final_scores, h_all = jax.lax.scan(
      _step,
      previous_scores,
      rotated_data,
      unroll=unroll_factor,
    )
    final_scores = h_all[indices]

    return _soft_maximum(final_scores + score_matrix[1:, 1:], mask=combined_mask[1:, 1:]).max()

  traceback_function = jax.grad(_compute_scoring_matrix, argnums=0)
  return jax.vmap(traceback_function, (0, 0, None, None)) if batch else traceback_function


def smith_waterman_affine(  # noqa: C901
  unroll: int = 2,
  ninf: float = -1e30,
  *,
  restrict_turns: bool = True,
  penalize_turns: bool = True,
  batch: bool = True,
) -> Callable:
  """Get a JAX-jit function for Smith-Waterman with affine gap penalties.

  Args:
    restrict_turns (bool): Whether to restrict turns in the alignment (e.g., no U-turns).
    penalize_turns (bool): Whether to apply penalties for turns (e.g., for non-diagonal moves).
    unroll (int): The unroll parameter for `jax.lax.scan`.
    ninf (float): A large negative number to represent negative infinity.
    batch (bool): If True, the returned function will be vmapped for batch processing.

  Returns:
    Callable: A function that performs the alignment traceback.

  """

  def _rotate_matrix(
    score_matrix: jax.Array,
  ) -> tuple[dict[str, jax.Array], tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Rotate the score matrix for striped dynamic programming."""
    a, b = score_matrix.shape
    ar, br = jnp.arange(a)[::-1, None], jnp.arange(b)[None, :]
    i, j = (br - ar) + (a - 1), (ar + br) // 2
    n, m = (a + b - 1), (a + b) // 2
    rotated_data = {
      "score": jnp.full([n, m], ninf).at[i, j].set(score_matrix),
      "parity": (jnp.arange(n) + a % 2) % 2,
    }
    previous_scores = (jnp.full((m, 3), ninf), jnp.full((m, 3), ninf))
    return rotated_data, previous_scores, (i, j)

  def _compute_scoring_matrix(
    score_matrix: jax.Array,
    masks: tuple[jax.Array, jax.Array],
    gap: float = 0.0,
    open_penalty: float = 0.0,
    temperature: float = 1.0,
  ) -> jax.Array:
    """Compute the scoring matrix for Smith-Waterman alignment with affine gap penalties."""

    def _soft_maximum(
      values: jax.Array,
      axis: int | None = None,
      mask: jax.Array | None = None,
    ) -> jax.Array:
      def _logsumexp(y: jax.Array) -> jax.Array:
        y = jnp.maximum(y, ninf)
        if mask is None:
          return jax.nn.logsumexp(y, axis=axis)
        max_y = y.max(axis, keepdims=True)
        return y.max(axis) + jnp.log(jnp.sum(mask * jnp.exp(y - max_y), axis=axis))

      return temperature * _logsumexp(values / temperature)

    def _conditional_select(
      condition: jax.Array,
      true_value: jax.Array,
      false_value: jax.Array,
    ) -> jax.Array:
      """Select values based on a boolean condition."""
      return condition * true_value + (1 - condition) * false_value

    def _pad(vals: jax.Array, shape: list) -> jax.Array:
      """Pad an array with negative infinity values."""
      return jnp.pad(vals, shape, constant_values=(ninf, ninf))

    def _scan_step(
      previous_scores: tuple[jax.Array, jax.Array],
      rotated_data: dict[str, jax.Array],
    ) -> tuple:
      """Perform a single step of the scan for computing the scoring matrix."""
      h_previous, h_current = previous_scores
      aligned_score = jnp.pad(h_previous, [[0, 0], [0, 1]]) + rotated_data["score"][:, None]
      right_score = _conditional_select(
        rotated_data["parity"],
        _pad(h_current[:-1], [[1, 0], [0, 0]]),
        h_current,
      )
      down_score = _conditional_select(
        rotated_data["parity"],
        h_current,
        _pad(h_current[1:], [[0, 1], [0, 0]]),
      )

      right = jnp.zeros_like(h_current)
      down = jnp.zeros_like(h_current)

      if penalize_turns:
        right += jnp.stack([open_penalty, gap, open_penalty])
        down += jnp.stack([open_penalty, open_penalty, gap])
      else:
        gap_pen = jnp.stack([open_penalty, gap, gap])
        right += gap_pen
        down += gap_pen

      if restrict_turns:
        right_score = right_score[:, :2]

      h0_aligned = _soft_maximum(aligned_score, -1)
      h0_right = _soft_maximum(right_score, -1)
      h0_down = _soft_maximum(down_score, -1)
      h0 = jnp.stack([h0_aligned, h0_right, h0_down], axis=-1)
      return (h_current, h0), h0

    mask_a, mask_b = masks
    combined_mask = mask_a[:, None] & mask_b[None, :]
    score_matrix = jnp.where(combined_mask, score_matrix, ninf)

    rotated_data, previous_scores, indices = _rotate_matrix(score_matrix[:-1, :-1])
    _final_scores, h_all = jax.lax.scan(_scan_step, previous_scores, rotated_data, unroll=unroll)
    final_scores = h_all[indices]
    return _soft_maximum(
      final_scores + score_matrix[1:, 1:, None],
      mask=combined_mask[1:, 1:, None],
    ).max()

  traceback_function = jax.grad(_compute_scoring_matrix, argnums=0)
  return jax.vmap(traceback_function, (0, 0, None, None, None)) if batch else traceback_function


def needleman_wunsch_alignment(unroll_factor: int = 2, *, batch: bool = True) -> Callable:
  """Get a JAX-jit function for Needleman-Wunsch (global alignment).

  Args:
    unroll_factor (int): The unroll parameter for `jax.lax.scan` for performance tuning.
    batch (bool): If True, the returned function will be vmapped for batch processing.

  Returns:
    Callable: A function that performs the alignment traceback.

  """
  ninf = -1e30

  def prepare_rotated_data(
    score_matrix: jax.Array,
    masks: tuple[jax.Array, jax.Array],
    gap_penalty: float,
  ) -> dict:
    """Prepare the rotated data structure for Needleman-Wunsch alignment."""
    mask_a, mask_b = masks
    num_rows, num_cols = score_matrix.shape

    combined_mask = mask_a[:, None] & mask_b[None, :]
    combined_mask = jnp.pad(combined_mask, [[1, 0], [1, 0]])

    score_matrix = jnp.where(combined_mask[1:, 1:], score_matrix, ninf)
    score_matrix = jnp.pad(score_matrix, [[1, 0], [1, 0]])

    num_rows, num_cols = score_matrix.shape
    row_indices, col_indices = jnp.arange(num_rows)[::-1, None], jnp.arange(num_cols)[None, :]
    i_indices, j_indices = (
      (col_indices - row_indices) + (num_rows - 1),
      (row_indices + col_indices) // 2,
    )
    num_diagonals, max_diagonal_length = (num_rows + num_cols - 1), (num_rows + num_cols) // 2
    zero_matrix = jnp.zeros((num_diagonals, max_diagonal_length))

    rotated_data = {
      "rotated_scores": zero_matrix.at[i_indices, j_indices].set(score_matrix),
      "rotated_mask": zero_matrix.at[i_indices, j_indices].set(combined_mask),
      "parity": (jnp.arange(num_diagonals) + num_rows % 2) % 2,
    }

    initial_row = jnp.where(
      jnp.arange(num_rows) < mask_a.sum(),
      gap_penalty * jnp.arange(num_rows),
      ninf,
    )
    initial_col = jnp.where(
      jnp.arange(num_cols) < mask_b.sum(),
      gap_penalty * jnp.arange(num_cols),
      ninf,
    )

    initial_conditions = (
      jnp.full((num_rows, num_cols), ninf).at[:, 0].set(initial_row).at[0, :].set(initial_col)
    )

    rotated_data["initial_conditions"] = zero_matrix.at[i_indices, j_indices].set(
      initial_conditions,
    )

    return {
      "rotated_data": rotated_data,
      "previous_scores": (jnp.full(max_diagonal_length, ninf), jnp.full(max_diagonal_length, ninf)),
      "indices": (i_indices, j_indices),
      "seq_len_a": mask_a.sum(),
      "seq_len_b": mask_b.sum(),
    }

  def compute_scoring_matrix(
    score_matrix: jax.Array,
    masks: tuple[jax.Array, jax.Array],
    gap_penalty: float = 0.0,
    temperature: float = 1.0,
  ) -> jax.Array:
    """Compute the scoring matrix for Needleman-Wunsch alignment."""

    def _logsumexp(
      values: jax.Array,
      axis: int | None = None,
      mask: jax.Array | None = None,
    ) -> jax.Array:
      if mask is None:
        return jax.nn.logsumexp(values, axis=axis)
      max_values = values.max(axis, keepdims=True)
      return values.max(axis) + jnp.log(jnp.sum(mask * jnp.exp(values - max_values), axis=axis))

    def _soft_maximum(
      values: jax.Array,
      axis: int | None = None,
      mask: jax.Array | None = None,
    ) -> jax.Array:
      return temperature * _logsumexp(values / temperature, axis, mask)

    def _conditional_select(
      condition: jax.Array,
      true_value: jax.Array,
      false_value: jax.Array,
    ) -> jax.Array:
      return condition * true_value + (1 - condition) * false_value

    def _scan_step(previous_scores: tuple, rotated_data: dict) -> tuple:
      previous_row, current_row = previous_scores
      alignment_score = previous_row + rotated_data["rotated_scores"]
      turn_score = _conditional_select(
        rotated_data["parity"],
        jnp.pad(current_row[:-1], [1, 0], constant_values=(ninf, ninf)),
        jnp.pad(current_row[1:], [0, 1], constant_values=(ninf, ninf)),
      )
      combined_scores = jnp.stack(
        [alignment_score, current_row + gap_penalty, turn_score + gap_penalty],
      )
      updated_row = rotated_data["rotated_mask"] * _soft_maximum(combined_scores, axis=0)
      updated_row += rotated_data["initial_conditions"]
      return (current_row, updated_row), updated_row

    rotated_data = prepare_rotated_data(
      score_matrix,
      masks=masks,
      gap_penalty=gap_penalty,
    )
    final_scores = jax.lax.scan(
      _scan_step,
      rotated_data["previous_scores"],
      rotated_data["rotated_data"],
      unroll=unroll_factor,
    )[-1][rotated_data["indices"]]
    return final_scores[rotated_data["seq_len_a"], rotated_data["seq_len_b"]]

  traceback_function = jax.grad(compute_scoring_matrix, argnums=0)
  return jax.vmap(traceback_function, (0, 0, None, None)) if batch else traceback_function


# BLOSUM62 scoring matrix with an added column/row for gaps.
# Order: A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V, Gap
_AA_SCORE_MATRIX = jnp.array(
  [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -4],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -4],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, -4],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, -4],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -4],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, -4],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, -4],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -4],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, -4],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -4],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, -4],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -4],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -4],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -4],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, -4],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -4],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -4],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -4],
    [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1],
  ],
  dtype=jnp.float32,
)

_MINIMUM_PROTEINS_COUNT = 2
_ONE_HOT_NDIM = 3


def align_sequences(
  protein_sequences_stacked: ProteinSequence | OneHotProteinSequence,
  gap_open: float = -1.0,
  gap_extend: float = -0.1,
  temp: float = 0.1,
) -> jax.Array:
  """Generate cross-protein position mapping using batched sequence alignment.

  Creates a mapping array for cross-protein position comparisons using
  Smith-Waterman alignment. This version uses `jax.vmap` for efficient computation
  of all pairwise alignments.

  Args:
      protein_sequences_stacked: Stacked array of protein sequences of shape
          (n_proteins, max_length). Assumes -1 for padded positions.
      gap_open: Gap opening penalty for alignment.
      gap_extend: Gap extension penalty for alignment.
      temp: Temperature parameter for soft alignment.

  Returns:
      Upper triangle mapping array of shape (num_pairs, max_length, 2)
      where num_pairs = n*(n-1)/2. Each entry contains [pos_in_protein_i,
      pos_in_protein_j] or [-1, -1] for unaligned positions.

  """
  if (
    protein_sequences_stacked.dtype == jnp.float32
    and protein_sequences_stacked.ndim == _ONE_HOT_NDIM
  ):
    protein_sequences_stacked = jnp.argmax(protein_sequences_stacked, axis=-1)

  n_proteins, max_seq_len = protein_sequences_stacked.shape

  if n_proteins < _MINIMUM_PROTEINS_COUNT or max_seq_len == 0:
    return jnp.empty((0, max_seq_len, 2), dtype=jnp.int32)

  true_lengths = jnp.sum(protein_sequences_stacked != -1, axis=1)
  sw_aligner = smith_waterman_affine(batch=False)

  def _align_and_map_pair(
    seq_a: jax.Array,
    mask_a: jax.Array,
    seq_b: jax.Array,
    mask_b: jax.Array,
  ) -> jax.Array:
    """Aligns a single pair of sequences and extracts a one-to-one mapping."""
    # Clip values on the full padded sequences
    seq_a_clipped = jnp.clip(seq_a, 0, 20)
    seq_b_clipped = jnp.clip(seq_b, 0, 20)

    # Apply the mask to set invalid positions to -1 for score matrix lookup
    seq_a_masked = jnp.where(mask_a, seq_a_clipped, -1)
    seq_b_masked = jnp.where(mask_b, seq_b_clipped, -1)

    score_matrix = _AA_SCORE_MATRIX[seq_a_masked[:, None], seq_b_masked[None, :]]
    masks_tuple = (mask_a, mask_b)

    traceback = sw_aligner(score_matrix, masks_tuple, gap_extend, gap_open, temp)

    # The traceback is now over the padded, max-length arrays
    best_j_for_i = jnp.argmax(traceback, axis=1)
    best_i_for_j = jnp.argmax(traceback, axis=0)

    # Create indices for the full padded arrays
    i_indices = jnp.arange(traceback.shape[0])
    mutual_alignment_mask = best_i_for_j[best_j_for_i] == i_indices

    scores = jnp.max(traceback, axis=1)
    score_threshold = jnp.max(scores) * 0.1
    final_mask = mutual_alignment_mask & (scores > score_threshold) & mask_a

    # Use jnp.where to generate the padded aligned indices.
    # This avoids the non-concrete boolean index error.
    padded_i = jnp.where(final_mask, i_indices, -1)
    padded_j = jnp.where(final_mask, best_j_for_i, -1)

    return jnp.stack([padded_i, padded_j], axis=-1)

  i_indices, j_indices = jnp.triu_indices(n_proteins, k=1)

  seqs_a = protein_sequences_stacked[i_indices]
  masks_a = jnp.arange(max_seq_len) < true_lengths[i_indices][:, None]
  seqs_b = protein_sequences_stacked[j_indices]
  masks_b = jnp.arange(max_seq_len) < true_lengths[j_indices][:, None]

  return jax.vmap(_align_and_map_pair)(seqs_a, masks_a, seqs_b, masks_b)
