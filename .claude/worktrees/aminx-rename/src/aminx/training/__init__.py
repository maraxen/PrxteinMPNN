"""Training utilities and functions for Aminx.

.. warning::
    The training module has not been updated to use the Sprint 2 composable
    inference architecture (StageSet / InferencePlan / make_encode_fn). Importing
    from this module will raise ``NotImplementedError`` until that work is complete.

    Tracked as Sprint 3 tech debt. The target design mirrors the inference contract:
    a ``TrainingPlan`` (analogous to ``InferencePlan``) with composable loss transforms,
    gradient accumulation stages, and a unified ``TrainingStageSet`` that plugs into
    the same encode-once / decode-many pattern used by the inference kernels.
"""

_TRAINING_NOT_READY = (
  "aminx.training is not yet updated for the Sprint 2 composable architecture. "
  "See Sprint 3 backlog item for TrainingPlan / TrainingStageSet design."
)


def __getattr__(name: str) -> object:
  raise NotImplementedError(_TRAINING_NOT_READY)


__all__: list[str] = []
