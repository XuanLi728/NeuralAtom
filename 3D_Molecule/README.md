# 3D scenario Neural Atoms

Models for which Neural Atoms is currently implemented:

- SchNet [[`arXiv`](https://arxiv.org/abs/1706.08566)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/schnet.py)]
- DimeNet++ [[`arXiv`](https://arxiv.org/abs/2011.14115)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/dimenet_plus_plus.py)]
- PaiNN [[`arXiv`](https://arxiv.org/abs/2102.03150)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/tree/main/ocpmodels/models/painn)]

Currently supported datasets:
 - OE62 [[`arXiv`](https://arxiv.org/abs/2001.08954)] [[`download`](https://mediatum.ub.tum.de/1507656)]


## Installation

To setup a `conda` environment with the required dependencies, please follow the [OCP installation instructions](https://github.com/Open-Catalyst-Project/ocp/blob/main/INSTALL.md). They should work identically in this repository. We further recommend installing the `jupyter` package to access our example training and evaluation notebooks, as well as the `seml` package [[`github`](https://github.com/TUM-DAML/seml)] to run and manage (especially longer) experiments from the CLI. To reproduce the long-range binning analyses from the Ewald message passing paper, please install the `simple-dftd3` package [[`installation instructions`](https://dftd3.readthedocs.io/en/latest/installation.html)] including the Python API.


## Commands

For SchNet model.
```bash
CUDA_VISIBLE_DEVICES=0 python train_and_evaluate.py --cfg=schnet_oe62_na.yml
```

For PaiNN model.
```bash
CUDA_VISIBLE_DEVICES=0 python train_and_evaluate.py --cfg=painn_oe62_na.yml
```

For DimeNet++ model.
```bash
CUDA_VISIBLE_DEVICES=0 python train_and_evaluate.py --cfg=dpp_oe62_na.yml
```

One can modify the configs in `3D_Molecule/configs_oe62`.  

The NeuralAtom block is in `3D_Molecule/ocpmodels/models/neural_atom_block.py`, while the attention components is in `3D_Molecule/ocpmodels/models/na_pooling.py`.

To change the number of NeuralAtoms, please refers to `lines 65 and 66` in `neural_atom_block.py`.

## Acknowledge
This repo is based on the [Ewald Message Passing](https://github.com/arthurkosmala/EwaldMP), we thanks the author for publishing their codes and data.