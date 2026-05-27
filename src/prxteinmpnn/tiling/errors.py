"""Library-side error base for tiling primitives.

PlanTopologyError (in host/plan.py) and DispatchRejected (in tiling/dispatch.py)
both subclass TilingError so a single ``except TilingError`` catches both,
while preserving the library/app dependency direction (host imports tiling, not vice versa).
"""


class TilingError(Exception):
    """Base for errors raised by tiling-side primitives."""
