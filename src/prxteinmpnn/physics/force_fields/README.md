# Force Fields

This directory contains force field parameters used by PrxteinMPNN.

## Structure

- `eqx/`: Converted Equinox (`.eqx`) force field files. These are binary PyTrees loaded directly by JAX.
- `xml/`: Original OpenMM XML force field definitions. These are the source of truth for conversion.

## Usage

To load a force field:

```python
from prxteinmpnn.physics import force_fields

# Load local file
ff = force_fields.load_force_field("src/prxteinmpnn/physics/force_fields/eqx/ff19SB.eqx")

# Or from Hub (if uploaded)
# ff = force_fields.load_force_field_from_hub("ff19SB")
```

## Conversion

To convert XML files to `.eqx` format, run:

```bash
python scripts/convert_all_xmls.py
```

This script reads from `openmmforcefields` (or `xml/` if configured) and outputs to `eqx/`.
