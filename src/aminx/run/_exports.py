"""Public ``sample``/``score`` exports with spec-driven routing and tensor deprecation."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import equinox as eqx

if TYPE_CHECKING:
  from collections.abc import Callable

  from aminx.run.specs import SamplingSpecification, ScoringSpecification


_TENSOR_SAMPLE_DEPRECATION = (
  "Calling aminx.run.sample with tensor arguments (prng_key, model, ...) is deprecated; "
  "use aminx.sampling.sample instead."
)
_TENSOR_SCORE_DEPRECATION = (
  "Calling aminx.run.score with tensor arguments (prng_key, model, ...) is deprecated; "
  "use aminx.scoring.score instead."
)


def _is_tensor_call(
  args: tuple[Any, ...],
  kwargs: dict[str, Any],
  spec_type: type[SamplingSpecification | ScoringSpecification],
  *,
  spec_only_kwargs: frozenset[str],
) -> bool:
  """Return True when the call matches the legacy tensor API signature."""
  if args and isinstance(args[0], spec_type):
    return False
  spec = kwargs.get("spec")
  if isinstance(spec, spec_type):
    return False
  if spec_only_kwargs.intersection(kwargs):
    return False
  return len(args) >= 2 and isinstance(args[1], eqx.Module)


def _dispatch(
  *,
  args: tuple[Any, ...],
  kwargs: dict[str, Any],
  spec_type: type[SamplingSpecification | ScoringSpecification],
  spec_only_kwargs: frozenset[str],
  deprecation_message: str,
  runner_fn: Callable[..., object],
  tensor_fn: Callable[..., object],
) -> object:
  if _is_tensor_call(args, kwargs, spec_type, spec_only_kwargs=spec_only_kwargs):
    warnings.warn(deprecation_message, DeprecationWarning, stacklevel=3)
    return tensor_fn(*args, **kwargs)
  return runner_fn(*args, **kwargs)


def sample(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
  """Sample sequences via :func:`~aminx.host.runner.sample` (spec-driven).

  Legacy tensor calls ``(prng_key, model, coordinates, ...)`` emit
  :class:`DeprecationWarning` and delegate to :func:`aminx.sampling.sample`.
  """
  from aminx.host.runner import sample as runner_sample  # noqa: PLC0415
  from aminx.run.specs import SamplingSpecification  # noqa: PLC0415
  from aminx.sampling import sample as tensor_sample  # noqa: PLC0415

  return _dispatch(
    args=args,
    kwargs=kwargs,
    spec_type=SamplingSpecification,
    spec_only_kwargs=frozenset({"inputs"}),
    deprecation_message=_TENSOR_SAMPLE_DEPRECATION,
    runner_fn=runner_sample,
    tensor_fn=tensor_sample,
  )


def score(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
  """Score sequences via :func:`~aminx.host.runner.score` (spec-driven).

  Legacy tensor calls ``(prng_key, model, coordinates, ...)`` emit
  :class:`DeprecationWarning` and delegate to :func:`aminx.scoring.score`.
  """
  from aminx.host.runner import score as runner_score  # noqa: PLC0415
  from aminx.run.specs import ScoringSpecification  # noqa: PLC0415
  from aminx.scoring.score import score as tensor_score  # noqa: PLC0415

  return _dispatch(
    args=args,
    kwargs=kwargs,
    spec_type=ScoringSpecification,
    spec_only_kwargs=frozenset({"inputs", "sequences_to_score"}),
    deprecation_message=_TENSOR_SCORE_DEPRECATION,
    runner_fn=runner_score,
    tensor_fn=tensor_score,
  )
