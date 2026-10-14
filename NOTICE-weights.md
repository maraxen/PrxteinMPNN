# Third-party model weights

aminx code is MIT-licensed (see `LICENSE`). The model checkpoints below are **converted copies of
third-party weights**, redistributed under the upstream MIT licences. Each upstream copyright and
permission notice is reproduced here and must travel with any copy of these weights.

Converted files are the upstream parameters re-serialised as Equinox `.eqx` / `.eqx.zst`; they are not
retrained. Parity of each checkpoint against its upstream is measured (see the parity ledger and
`docs/MODEL_FAMILIES.md`) and is not complete.

| aminx checkpoint id | Upstream | Upstream file |
|---|---|---|
| `pottsmpnn_vanilla_20`, `pottsmpnn_vanilla_30`, `pottsmpnn_sol_20`, `pottsmpnn_sol_30`, `pottsmpnn_ft` | PottsMPNN, <https://github.com/KeatingLab/PottsMPNN> @ `0cb0a58874e373d664114f1735deab77614e4ab5` | `vanilla_model_weights/pottsmpnn_{20,30}.pt`, `soluble_model_weights/sol_pottsmpnn_{20,30}.pt`, `ft_model_weights/potts_ft.pt` |
| `lasermpnn_nothing_heldout`, `lasermpnn_noise_ligandmpnn_split`, `lasermpnn_soluble_65000` | LASErMPNN, <https://github.com/polizzilab/LASErMPNN> @ `e70f2c6d765416f7e29d51bfd6d4e08496438878` | `model_weights/laser_weights_0p1A_nothing_heldout.pt`, `..._noise_ligandmpnn_split.pt`, `soluble_weights_no_heldout_drop_clusters_optstep_65000.pt` |
| `protonpottsmpnn_v6_30` | Proton-PottsMPNN (Jacobsen et al., 2026) | `checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt` |

The Proton-PottsMPNN licence states that its MIT grant covers the `checkpoints/` directory. Its bundled
`foundry/` directory (IPD rc-foundry, BSD 3-Clause, Copyright (c) 2025 Institute for Protein Design,
University of Washington) is not redistributed here.

If you use these weights in academic work, cite the upstream publications.

---

## PottsMPNN

MIT License

Copyright (c) 2022 Justas Dauparas
Copyright (c) 2026 Foster Birnbaum, Amy E. Keating.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## LASErMPNN

Copyright (c) 2025 Benjamin Fry, Nicholas Polizzi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Proton-PottsMPNN

MIT License

Copyright (c) 2026 Christian P. Jacobsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
