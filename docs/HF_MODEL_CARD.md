---
license: mit
language:
- en
library_name: jax
tags:
- biology
- protein-design
- mpnn
- ligandmpnn
- structural-biology
---

# PrxteinMPNN: A Functional JAX/Equinox Interface for ProteinMPNN

PrxteinMPNN is a modular, high-performance implementation of the ProteinMPNN architecture and its variants (LigandMPNN, MembraneMPNN) using the **JAX** and **Equinox** frameworks. It provides a transparent and functional interface for protein sequence design, optimized for accelerated computation and integration into modern machine learning workflows.

## 🎯 Key Features

- **Functional Design**: Clean separation of state and logic following the JAX paradigm.
- **Unified Architecture**: Single implementation supporting backbone-only, ligand-aware, and membrane-specific design.
- **Equinox Native**: Models are structured as Equinox modules for easy inspection and transformation.
- **Validated Parity**: Rigorously tested for numerical parity against the original PyTorch reference implementations (>0.95 Pearson correlation).

## 🚀 Model Variants

This repository hosts pre-trained weights for several ProteinMPNN families, converted to `.eqx` format:

### 1. ProteinMPNN (Backbone-only)
Optimized for design based on protein backbone coordinates.
- **Original**: Trained on the standard PDB dataset.
- **Soluble**: Trained on filtered sets for improved protein expression and solubility.
- *Checkpoints*: `original_v_48_020`, `soluble_v_48_020` (recommended).

### 2. LigandMPNN (Context-aware)
Conditioned on atomic context from ligands, small molecules, and DNA/RNA.
- *Checkpoints*: `ligandmpnn_v_32_005_25`, `ligandmpnn_v_32_010_25`, `ligandmpnn_v_32_020_25`, `ligandmpnn_v_32_030_25`.

### 3. MembraneMPNN
Specialized for membrane protein design with per-residue or global label conditioning.
- *Checkpoints*: `global_label_membrane_mpnn_v_48_020`, `per_residue_label_membrane_mpnn_v_48_020`.

### 4. Sidechain Packer
Predicts sidechain conformations (torsion distributions) for a given sequence and backbone.
- *Checkpoint*: `ligandmpnn_sc_v_32_002_16`.

## 📚 Installation & Usage

For the full package and high-level API, visit the [PrxteinMPNN GitHub repository](https://github.com/maraxen/PrxteinMPNN).

```bash
pip install prxteinmpnn
```

### Quick Loading Example
```python
from prxteinmpnn.io.weights import load_model, load_ligand_model

# Load standard ProteinMPNN
model = load_model(model_version="v_48_020", model_weights="original")

# Load LigandMPNN
ligand_model = load_ligand_model(checkpoint_id="ligandmpnn_v_32_020_25")
```

## 📜 Citations

If you use these models or the PrxteinMPNN framework, please cite the original ProteinMPNN/LigandMPNN publications:

**ProteinMPNN**
```bibtex
@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```

**LigandMPNN**
```bibtex
@article{dauparas2023atomic,
  title={Atomic context-conditioned protein sequence design using LigandMPNN},
  author={Dauparas, Justas and Lee, Gyu Rie and Pecoraro, Robert and An, Linna and Anishchenko, Ivan and Glasscock, Cameron and Baker, David},
  journal={bioRxiv},
  pages={2023--12},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```
