"""Utilities for aligning proteins."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from collections.abc import Callable

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
    """Rotate the score matrix for striped dynamic programming.

    Args:
      score_matrix (jax.Array): The input score matrix.
      mask (jax.Array | None): An optional mask to apply to the matrix.

    Returns:
      tuple: A tuple containing the rotated data, previous scores, and indices.

    """
    a, b = score_matrix.shape
    ar, br = jnp.arange(a)[::-1, None], jnp.arange(b)[None, :]
    i, j = (br - ar) + (a - 1), (ar + br) // 2
    n, m = (a + b - 1), (a + b) // 2
    zero = jnp.zeros([n, m])
    if mask is None:
      mask = jnp.array(1.0)
    rotated_data = {
      "score": zero.at[i, j].set(score_matrix),
      "mask": zero.at[i, j].set(mask),
      "parity": (jnp.arange(n) + a % 2) % 2,
    }
    previous_scores = (jnp.zeros(m), jnp.zeros(m))
    return rotated_data, previous_scores, (i, j)

  def compute_scoring_matrix(
    score_matrix: jax.Array,
    sequence_lengths: jax.Array,
    temperature: float = 1.0,
  ) -> jax.Array:
    """Compute the scoring matrix for Smith-Waterman alignment.

    Args:
      score_matrix (jax.Array): The input score matrix.
      sequence_lengths (jax.Array): The lengths of the two sequences.
      temperature (float): The temperature parameter for the soft maximum function.

    Returns:
      jax.Array: The maximum score in the scoring matrix.

    """

    def _soft_maximum(values: jax.Array, axis: int | None = None) -> jax.Array:
      """Compute the soft maximum of values along a specified axis.

      Args:
        values (jax.Array): The input values.
        axis (int | None): The axis along which to compute the soft maximum.

      Returns:
        jax.Array: The soft maximum values.

      """
      return temperature * jax.nn.logsumexp(values / temperature, axis)

    def _conditional_select(
      condition: jax.Array,
      true_value: jax.Array,
      false_value: jax.Array,
    ) -> jax.Array:
      """Select values based on a boolean condition.

      Args:
        condition (jax.Array): The boolean condition.
        true_value (jax.Array): The value to select if the condition is true.
        false_value (jax.Array): The value to select if the condition is false.

      Returns:
        jax.Array: The selected value.

      """
      return condition * true_value + (1 - condition) * false_value

    def _scan_step(
      previous_scores: tuple[jax.Array, jax.Array],
      rotated_data: dict[str, jax.Array],
    ) -> tuple:
      """Perform a single step of the scan for computing the scoring matrix.

      Args:
        previous_scores (tuple): A tuple containing the previous two rows of scores.
        rotated_data (dict): A dictionary containing the rotated matrix data.

      Returns:
        tuple: A tuple containing the updated previous scores and the current masked scores.

      """
      h_previous, h_current = previous_scores  # previous two rows of scoring (hij) mtx
      h_current_shifted = _conditional_select(
        rotated_data["parity"],
        jnp.pad(h_current[:-1], [1, 0]),
        jnp.pad(h_current[1:], [0, 1]),
      )
      h_combined = jnp.stack([h_previous + rotated_data["score"], h_current, h_current_shifted], -1)
      h_masked = rotated_data["mask"] * _soft_maximum(h_combined, -1)
      return (h_current, h_masked), h_masked

    a, b = score_matrix.shape
    real_a, real_b = sequence_lengths
    mask = (jnp.arange(a) < real_a)[:, None] * (jnp.arange(b) < real_b)[None, :]

    rotated_data, previous_scores, indices = rotate_matrix(score_matrix, mask=mask)
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
    """Rotate the score matrix for striped dynamic programming.

    Args:
      score_matrix (jax.Array): The input score matrix.

    Returns:
      tuple: A tuple containing the rotated data, previous scores, and indices.

    """
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
    sequence_lengths: jax.Array,
    gap: float = 0.0,
    temperature: float = 1.0,
  ) -> jax.Array:
    """Compute the scoring matrix for Smith-Waterman alignment with gap penalty.

    Args:
      score_matrix (jax.Array): The input score matrix.
      sequence_lengths (jax.Array): The lengths of the two sequences.
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
      """Compute the soft maximum of values along a specified axis.

      Args:
        values (jax.Array): The input values.
        axis (int | None): The axis along which to compute the soft maximum.
        mask (jax.Array | None): An optional mask to apply to the values before logsumexp.

      Returns:
        jax.Array: The soft maximum values.

      """
      values = jnp.maximum(values, ninf)
      if mask is None:
        return temperature * jax.nn.logsumexp(values / temperature, axis)

      # Apply mask for logsumexp
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
      """Select values based on a boolean condition.

      Args:
        condition (jax.Array): The boolean condition.
        true_value (jax.Array): The value to select if the condition is true.
        false_value (jax.Array): The value to select if the condition is false.

      Returns:
        jax.Array: The selected value.

      """
      return condition * true_value + (1 - condition) * false_value

    def _pad(vals: jax.Array, shape: list) -> jax.Array:
      """Pad an array with negative infinity values.

      Args:
        vals (jax.Array): The input array.
        shape (list): The padding shape.

      Returns:
        jax.Array: The padded array.

      """
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
      ), updated_row  # Return updated previous scores and current masked scores

    a, b = score_matrix.shape
    real_a, real_b = sequence_lengths
    mask = (jnp.arange(a) < real_a)[:, None] * (jnp.arange(b) < real_b)[None, :]

    # Apply negative infinity to masked out regions of the score matrix
    score_matrix = score_matrix + ninf * (1 - mask)

    # Rotate the score matrix (excluding the first row/column for alignment)
    # The first row/column are implicitly handled by the initial `ninf` values in `prev`.
    rotated_data, previous_scores, indices = _rotate_matrix(score_matrix[:-1, :-1])

    # Perform the scan to compute the scoring matrix
    _final_scores, h_all = jax.lax.scan(
      _step,
      previous_scores,
      rotated_data,
      unroll=unroll_factor,
    )
    final_scores = h_all[indices]

    return _soft_maximum(final_scores + score_matrix[1:, 1:], mask=mask[1:, 1:]).max()

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
    """Rotate the score matrix for striped dynamic programming.

    Args:
      score_matrix (jax.Array): The input score matrix.

    Returns:
      tuple: A tuple containing the rotated data, previous scores, and indices.

    """
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
    lengths: jax.Array,
    gap: float = 0.0,
    open_penalty: float = 0.0,
    temperature: float = 1.0,
  ) -> jax.Array:
    """Compute the scoring matrix for Smith-Waterman alignment with affine gap penalties.

    Args:
      score_matrix (jax.Array): The input score matrix.
      lengths (jax.Array): The lengths of the two sequences.
      gap (float): The gap extension penalty.
      open_penalty (float): The gap opening penalty.
      temperature (float): The temperature parameter for the soft maximum function.

    Returns:
      jax.Array: The maximum score in the scoring matrix.

    """

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
      """Select values based on a boolean condition.

      Args:
        condition (jax.Array): The boolean condition.
        true_value (jax.Array): The value to select if the condition is true.
        false_value (jax.Array): The value to select if the condition is false.

      Returns:
        jax.Array: The selected value.

      """
      return condition * true_value + (1 - condition) * false_value

    def _pad(vals: jax.Array, shape: list) -> jax.Array:
      """Pad an array with negative infinity values.

      Args:
        vals (jax.Array): The input array.
        shape (list): The padding shape.

      Returns:
        jax.Array: The padded array.

      """
      return jnp.pad(vals, shape, constant_values=(ninf, ninf))

    def _scan_step(
      previous_scores: tuple[jax.Array, jax.Array],
      rotated_data: dict[str, jax.Array],
    ) -> tuple:
      """Perform a single step of the scan for computing the scoring matrix.

      Args:
        previous_scores (tuple): A tuple containing the previous two rows of scores.
        rotated_data (dict): A dictionary containing the rotated matrix data.

      Returns:
        tuple: A tuple containing the updated previous scores and the current masked scores.

      """
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

      # Initialize right and down variables
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

    a, b = score_matrix.shape
    real_a, real_b = lengths
    mask = (jnp.arange(a) < real_a)[:, None] * (jnp.arange(b) < real_b)[None, :]
    score_matrix = score_matrix + ninf * (1 - mask)

    rotated_data, previous_scores, indices = _rotate_matrix(score_matrix[:-1, :-1])
    _final_scores, h_all = jax.lax.scan(_scan_step, previous_scores, rotated_data, unroll=unroll)
    final_scores = h_all[indices]
    return _soft_maximum(
      final_scores + score_matrix[1:, 1:, None],
      mask=mask[1:, 1:, None],
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

  def prepare_rotated_data(
    score_matrix: jax.Array,
    sequence_lengths: jax.Array,
    gap_penalty: float,
  ) -> dict:
    """Prepare the rotated data structure for Needleman-Wunsch alignment.

    Args:
      score_matrix (jax.Array): The input score matrix.
      sequence_lengths (jax.Array): The lengths of the two sequences.
      gap_penalty (float): The gap penalty.
      temperature (float): The temperature parameter for soft maximum computation.

    Returns:
      dict: A dictionary containing the rotated score matrix, mask, initial conditions, and indices.

    """
    num_rows, num_cols = score_matrix.shape
    seq_len_a, seq_len_b = sequence_lengths

    # Create a mask for valid sequence positions
    mask = (jnp.arange(num_rows) < seq_len_a)[:, None] * (jnp.arange(num_cols) < seq_len_b)[None, :]
    mask = jnp.pad(mask, [[1, 0], [1, 0]])

    # Pad the score matrix
    score_matrix = jnp.pad(score_matrix, [[1, 0], [1, 0]])

    # Rotate the score matrix for striped dynamic programming
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
      "rotated_mask": zero_matrix.at[i_indices, j_indices].set(mask),
      "parity": (jnp.arange(num_diagonals) + num_rows % 2) % 2,
    }

    # Initialize gap penalties for the first row and column
    initial_row = gap_penalty * jnp.arange(num_rows)
    initial_col = gap_penalty * jnp.arange(num_cols)
    initial_conditions = (
      jnp.zeros((num_rows, num_cols)).at[:, 0].set(initial_row).at[0, :].set(initial_col)
    )
    rotated_data["initial_conditions"] = zero_matrix.at[i_indices, j_indices].set(
      initial_conditions,
    )

    return {
      "rotated_data": rotated_data,
      "previous_scores": (jnp.zeros(max_diagonal_length), jnp.zeros(max_diagonal_length)),
      "indices": (i_indices, j_indices),
      "sequence_lengths": sequence_lengths,
    }

  def compute_scoring_matrix(
    score_matrix: jax.Array,
    sequence_lengths: jax.Array,
    gap_penalty: float = 0.0,
    temperature: float = 1.0,
  ) -> jax.Array:
    """Compute the scoring matrix for Needleman-Wunsch alignment.

    Args:
      score_matrix (jax.Array): The input score matrix.
      sequence_lengths (jax.Array): The lengths of the two sequences.
      gap_penalty (float): The gap penalty.
      temperature (float): The temperature parameter for soft maximum computation.

    Returns:
      jax.Array: The final alignment score.

    """

    def _logsumexp(
      values: jax.Array,
      axis: int | None = None,
      mask: jax.Array | None = None,
    ) -> jax.Array:
      """Compute the log-sum-exp of values along a specified axis."""
      if mask is None:
        return jax.nn.logsumexp(values, axis=axis)
      max_values = values.max(axis, keepdims=True)
      return values.max(axis) + jnp.log(jnp.sum(mask * jnp.exp(values - max_values), axis=axis))

    def _soft_maximum(
      values: jax.Array,
      axis: int | None = None,
      mask: jax.Array | None = None,
    ) -> jax.Array:
      """Compute the soft maximum of values along a specified axis."""
      return temperature * _logsumexp(values / temperature, axis, mask)

    def _conditional_select(
      condition: jax.Array,
      true_value: jax.Array,
      false_value: jax.Array,
    ) -> jax.Array:
      """Select values based on a boolean condition."""
      return condition * true_value + (1 - condition) * false_value

    def _scan_step(previous_scores: tuple, rotated_data: dict) -> tuple:
      """Perform a single step of the scan for computing the scoring matrix."""
      previous_row, current_row = previous_scores
      alignment_score = previous_row + rotated_data["rotated_scores"]
      turn_score = _conditional_select(
        rotated_data["parity"],
        jnp.pad(current_row[:-1], [1, 0]),
        jnp.pad(current_row[1:], [0, 1]),
      )
      combined_scores = jnp.stack(
        [alignment_score, current_row + gap_penalty, turn_score + gap_penalty],
      )
      updated_row = rotated_data["rotated_mask"] * _soft_maximum(combined_scores, axis=0)
      updated_row += rotated_data["initial_conditions"]
      return (current_row, updated_row), updated_row

    rotated_data = prepare_rotated_data(
      score_matrix,
      sequence_lengths=sequence_lengths,
      gap_penalty=gap_penalty,
    )
    final_scores = jax.lax.scan(
      _scan_step,
      rotated_data["previous_scores"],
      rotated_data["rotated_data"],
      unroll=unroll_factor,
    )[-1][rotated_data["indices"]]
    return final_scores[rotated_data["sequence_lengths"][0], rotated_data["sequence_lengths"][1]]

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


def align_sequences(
  protein_sequences_stacked: jax.Array,
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
  n_proteins, max_seq_len = protein_sequences_stacked.shape

  if n_proteins < _MINIMUM_PROTEINS_COUNT:
    return jnp.empty((0, max_seq_len, 2), dtype=jnp.int32)

  true_lengths = jnp.sum(protein_sequences_stacked != -1, axis=1)
  sw_aligner = smith_waterman_affine(batch=False)

  def _align_and_map_pair(
    seq_a: jax.Array,
    len_a: jax.Array,
    seq_b: jax.Array,
    len_b: jax.Array,
  ) -> jax.Array:
    """Aligns a single pair of sequences and extracts a one-to-one mapping."""
    seq_a_clipped = jnp.clip(jax.lax.dynamic_slice(seq_a, (0,), (len_a,)), 0, 20)
    seq_b_clipped = jnp.clip(jax.lax.dynamic_slice(seq_b, (0,), (len_b,)), 0, 20)

    score_matrix = _AA_SCORE_MATRIX[seq_a_clipped[:, None], seq_b_clipped[None, :]]
    lengths = jnp.array([len_a, len_b])

    traceback = sw_aligner(score_matrix, lengths, gap_extend, gap_open, temp)

    best_j_for_i = jnp.argmax(traceback, axis=1)  # shape (len_a,)
    best_i_for_j = jnp.argmax(traceback, axis=0)  # shape (len_b,)

    i_indices = jnp.arange(len_a)
    mutual_alignment_mask = best_i_for_j[best_j_for_i] == i_indices

    scores = jnp.max(traceback, axis=1)
    score_threshold = jnp.max(scores) * 0.1
    final_mask = mutual_alignment_mask & (scores > score_threshold)

    aligned_i = i_indices[final_mask]
    aligned_j = best_j_for_i[final_mask]

    n_aligned = aligned_i.shape[0]
    pad_width = (0, max_seq_len - n_aligned)
    padded_i = jnp.pad(aligned_i, pad_width, constant_values=-1)
    padded_j = jnp.pad(aligned_j, pad_width, constant_values=-1)

    return jnp.stack([padded_i, padded_j], axis=-1)

  i_indices, j_indices = jnp.triu_indices(n_proteins, k=1)

  seqs_a = protein_sequences_stacked[i_indices]
  lengths_a = true_lengths[i_indices]
  seqs_b = protein_sequences_stacked[j_indices]
  lengths_b = true_lengths[j_indices]

  return jax.vmap(_align_and_map_pair)(seqs_a, lengths_a, seqs_b, lengths_b)
