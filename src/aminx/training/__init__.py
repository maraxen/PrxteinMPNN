"""Training utilities and functions for Aminx.

.. warning::
    The training module has not been updated to use the Sprint 2 composable
    inference architecture (StageSet / InferencePlan / make_encode_fn). Accessing
    an attribute of this module will raise ``AttributeError`` until that work is
    complete.

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
  # AttributeError, not NotImplementedError: a module __getattr__ (PEP 562) must
  # raise AttributeError so that `getattr(obj, name, default)`/`hasattr` degrade
  # gracefully. `pickle.whichmodule` scans all of sys.modules via `getattr(mod,
  # name, None)` when resolving where an object is defined (e.g. Inductor's
  # FxGraphCache pickling compiled artifacts) -- raising anything else there
  # crashes torch.compile in-process for any script that merely imports this
  # module, regardless of whether it ever touches aminx.training on purpose.
  msg = f"module 'aminx.training' has no attribute {name!r}: {_TRAINING_NOT_READY}"
  raise AttributeError(msg)


__all__: list[str] = []
