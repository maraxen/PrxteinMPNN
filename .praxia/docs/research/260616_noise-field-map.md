# R6-2 Gate: Noise Field Mapping Table

**Sprint:** 260620_rs6-noise-fn  
**Gate for:** Track B — R6-2 FeatureNoiseBundle (#1923)  
**Source:** aminx/src/aminx/run/specs.py:154-161, 231-243  

---

## Field Mapping Table

| old field | type | default | normalization | role | FeatureNoiseBundle slot |
|---|---|---|---|---|---|
| `backbone_noise` | `Sequence[float] \| float` | `(0.0,)` | float → `(float,)` in `__post_init__:233` | Per-residue backbone coordinate noise schedule; passed to every kernel dispatch call (`kernel_dispatch.py:110,197,277,310,372,448,481`) | `noise_levels` on `backbone` bundle |
| `backbone_noise_mode` | `Literal["direct", "thermal"]` | `"direct"` | none | Controls whether noise is applied directly to coordinates or as a thermal perturbation | `mode` on `backbone` bundle |
| `estat_noise` | `Sequence[float] \| float \| None` | `None` | float → `(float,)` in `__post_init__:235-236`; non-None auto-sets `use_electrostatics=True` at `__post_init__:237-238` | Electrostatic potential noise schedule; enables the electrostatic physics path when non-None | `noise_levels` on `electrostatic` bundle (bundle present only when non-None) |
| `estat_noise_mode` | `Literal["direct", "thermal"]` | `"direct"` | none | Noise injection mode for electrostatic channel | `mode` on `electrostatic` bundle |
| `vdw_noise` | `Sequence[float] \| float \| None` | `None` | float → `(float,)` in `__post_init__:240-241`; non-None auto-sets `use_vdw=True` at `__post_init__:242-243` | Van der Waals noise schedule; enables the VDW physics path when non-None | `noise_levels` on `vdw` bundle (bundle present only when non-None) |
| `vdw_noise_mode` | `Literal["direct", "thermal"]` | `"direct"` | none | Noise injection mode for VDW channel | `mode` on `vdw` bundle |
| `use_electrostatics` | `bool` | `False` | auto-set to `True` when `estat_noise` is non-None | Gates electrostatic physics feature; controls `physics_feature_dim += 5` in model weight loading; also exposed as CLI flag (`cli.py:150,200`) | `enabled` on `electrostatic` bundle (bundle present ↔ enabled=True) |
| `use_vdw` | `bool` | `False` | auto-set to `True` when `vdw_noise` is non-None | Gates VDW physics feature; same dimension-addition pattern as electrostatics; CLI flag (`cli.py:151,201`) | `enabled` on `vdw` bundle |

---

## FeatureNoiseBundle Dataclass

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class FeatureNoiseBundle:
    feature_type: Literal["backbone", "electrostatic", "vdw"]
    noise_levels: tuple[float, ...]
    mode: Literal["direct", "thermal"] = "direct"
    enabled: bool = True
```

**Notes:**
- Frozen `@dataclass` (NOT `eqx.Module`) — lives on `RunSpecification` (serializable plain dataclass)
- `noise_levels` is always a `tuple[float, ...]` (same normalized form as current field values after `__post_init__`)
- `enabled=True` is the default — the bundle's presence in `noise: list[FeatureNoiseBundle]` implies intent to use it
- Backbone bundle is always implicit (noise is `(0.0,)` by default, representing no-noise); electrostatic and VDW bundles are opt-in

---

## Design Decisions

### bundle presence = feature enabled

The old design used two independent signals for electrostatics/VDW:
1. A noise schedule (`estat_noise`, `vdw_noise`) — `None` means disabled
2. An explicit boolean flag (`use_electrostatics`, `use_vdw`) — also redundantly tracks enablement

In `FeatureNoiseBundle`, these collapse: **a bundle in the `noise` list with `enabled=True` means the feature is active**. No separate boolean needed.

### `use_electrostatics`/`use_vdw` as deprecated aliases

These booleans are still used by:
- CLI (`cli.py:150,200`) — users can pass `--use-electrostatics` without providing noise levels
- Downstream consumers that check `spec.use_electrostatics` directly

They must remain as computed attributes (not `__init__` parameters) set from the bundle list in `__post_init__`.

---

## Migration Notes

### Old construction → new construction

Old:
```python
RunSpecification(
    inputs=["protein.pdb"],
    estat_noise=(0.05,),
    estat_noise_mode="thermal",
    use_electrostatics=True,
    vdw_noise=(0.02,),
    vdw_noise_mode="direct",
    use_vdw=True,
)
```

New:
```python
RunSpecification(
    inputs=["protein.pdb"],
    noise=[
        FeatureNoiseBundle(feature_type="electrostatic", noise_levels=(0.05,), mode="thermal"),
        FeatureNoiseBundle(feature_type="vdw", noise_levels=(0.02,), mode="direct"),
    ],
)
```

Backbone bundle is optional in the list (default behavior is no backbone noise = `(0.0,)` implied). To set backbone noise explicitly:
```python
noise=[FeatureNoiseBundle(feature_type="backbone", noise_levels=(0.1, 0.2))]
```

### Backward-compat aliases in `__post_init__`

`RunSpecification.__post_init__` must continue to set these computed attributes from the `noise` list:

```python
# Backward-compat computed attributes (not __init__ params)
object.__setattr__(self, "backbone_noise",
    _extract_noise_levels(self.noise, "backbone"))
object.__setattr__(self, "use_electrostatics",
    _has_enabled(self.noise, "electrostatic"))
object.__setattr__(self, "use_vdw",
    _has_enabled(self.noise, "vdw"))
```

### `build_run_spec()` bridge

`build_run_spec()` (aminx/run/spec.py) must derive `use_electrostatics`/`use_vdw` from the bundle list. The existing `spec.use_electrostatics` computed attr handles this transparently.

### CLI backward compatibility

`cli.py` currently accepts `--estat-noise`, `--use-electrostatics`, etc. as separate arguments that are passed directly to `RunSpecification`. Post-R6-2, the CLI bridge function (`cli.py:281-282`) must convert these flat kwargs into a `noise` list before constructing `RunSpecification`. This is **CLI migration work** — out of scope for R6-2 code, but noted here so Track B fixer knows not to break the CLI path.

### `kernel_dispatch.py` compatibility

`kernel_dispatch.py` accesses `spec.backbone_noise` at 6 call sites (lines 110, 197, 277, 310, 372, 448, 481). The backward-compat computed attribute handles this — no changes needed in `kernel_dispatch.py` for R6-2.

---

## RS-6 RunSpec Noise Field Names

Based on the analysis above, Track B (R6-2) MUST use:

- `noise: list[FeatureNoiseBundle]` on `RunSpecification` (replaces all 8 old fields as init params)
- `backbone_noise`, `use_electrostatics`, `use_vdw` remain as computed attributes in `__post_init__`
- `FeatureNoiseBundle.feature_type` discriminates the three noise domains: `"backbone"`, `"electrostatic"`, `"vdw"`
