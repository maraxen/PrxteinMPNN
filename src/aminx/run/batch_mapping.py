"""Generic per-structure batch input mapping ("json-mapped-by-X").

See ``.praxia/docs/specs/260826_chain-selection-gap-closure.md`` (G1) for the design
rationale. Only ``by="path"`` is implemented; other ``X`` values are refused loudly rather
than built speculatively (see :class:`MappedBy`'s docstring for why "path" here means the
canonical structure id, not the literal filesystem path).
"""

from __future__ import annotations

from dataclasses import dataclass

_SUPPORTED_MAPPED_BY: frozenset[str] = frozenset({"path"})


@dataclass(frozen=True, slots=True)
class MappedBy[T]:
  """A per-structure value supplied as ``{key: value}``, keyed by a declared input field.

  Only ``by="path"`` is implemented today. ``by`` is a real field (not a ``Literal["path"]``)
  so the shape survives when a second ``X`` is added -- :func:`resolve_mapped_by` is where
  support is actually gated, not the type.

  ``by="path"`` keys are **canonical structure ids** (``Path(input).stem``, or
  ``f"structure_{index}"`` when a stem can't be derived), matching the identity convention
  ``host/_sampling_helper.py``'s ``_canonical_structure_id`` already uses for ligand-context
  batch keying -- reusing that convention rather than inventing a second one, at the cost of
  deviating from vendor's own literal-absolute-path JSON keys. A caller porting a vendor-style
  ``{"/path/to/1abc.pdb": "..."}`` mapping should key by ``"1abc"`` here, not the full path.

  Deliberately NOT jit-safe: ``mapping`` is a plain ``dict``, and this class is a plain
  dataclass (not an ``eqx.Module`` / registered pytree), so JAX treats an unresolved
  ``MappedBy`` instance as an opaque leaf. It must be resolved to a concrete array via
  :func:`resolve_mapped_by` before it reaches any ``jax.jit`` / ``eqx.filter_jit`` boundary --
  every current call site resolves it inside ``_prepare_fixed_controls``, on the host, before
  any kernel dispatch.
  """

  by: str
  mapping: dict[str, T]

  def __post_init__(self) -> None:
    if self.by not in _SUPPORTED_MAPPED_BY:
      msg = (
        f"MappedBy(by={self.by!r}) is not supported yet -- only "
        f"{sorted(_SUPPORTED_MAPPED_BY)} is implemented. Construct your batch mapping keyed "
        f"by structure id, or file the new `by` value as a feature request rather than "
        f"working around this check."
      )
      raise NotImplementedError(msg)


def resolve_mapped_by(
  value: object,
  *,
  structure_ids: list[str],
  field_name: str,
) -> list[object]:
  """Resolve a possibly-``MappedBy`` field into one value per structure, id-order matched.

  A non-``MappedBy`` value broadcasts to every structure (existing single-value behavior,
  unchanged). A ``MappedBy`` value is resolved per structure id; a structure id absent from
  the mapping raises -- silently defaulting an omitted structure to ``None``/unset is exactly
  the kind of quiet wrong-answer this audit sprint exists to catch (see FA2/FA3, and the
  ``fixed_tokens`` Alanine-collapse guard in ``_sampling_helper.py``).

  Args:
    value: The field's raw value -- a plain array-like, a ``MappedBy``, or ``None``.
    structure_ids: Canonical structure ids for the CURRENT batch, in row order (the same ids
      ``_structure_ids_for_batch`` already produces for ligand-context keying).
    field_name: Name of the field being resolved, for the error message only.

  Returns:
    A list the same length as ``structure_ids``: ``value`` repeated for a non-``MappedBy``
    input, or the per-id mapped values in row order.

  Raises:
    ValueError: A structure id in this batch has no entry in ``value.mapping``.

  """
  if not isinstance(value, MappedBy):
    return [value] * len(structure_ids)
  missing = [sid for sid in structure_ids if sid not in value.mapping]
  if missing:
    msg = (
      f"{field_name}: MappedBy(by={value.by!r}) is missing an entry for {len(missing)} of "
      f"{len(structure_ids)} structures in this batch: "
      f"{missing[:5]}{'...' if len(missing) > 5 else ''}. Every structure in this batch must "
      f"have a mapping entry -- there is no implicit fallback."
    )
    raise ValueError(msg)
  return [value.mapping[sid] for sid in structure_ids]
