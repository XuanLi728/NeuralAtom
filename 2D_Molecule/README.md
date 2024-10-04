# NeuralAtom 2D

The repo is based on [GraphGPS](https://github.com/rampasek/GraphGPS) which is built using [PyG](https://www.pyg.org/) and [GraphGym from PyG2](https://pytorch-geometric.readthedocs.io/en/2.0.0/notes/graphgym.html).


### Overview of Datasets
Data set downloads automatically, you can setup the downloading root in each config file. For example, in `./configs/GCN/peptides-func-GCN.yaml` the default root is `dataset:
  dir: ./data`.

| Dataset         | Domain            | Task                 | Node Feat. (dim) | Edge Feat. (dim) | Perf. Metric |
| --------------- | ----------------- | -------------------- | ---------------- | ---------------- | ------------ |
| PCQM-Contact    | Quantum Chemistry | Link Prediction      | Atom Encoder (9) | Bond Encoder (3) | Hits@K, MRR  |
| Peptides-func   | Chemistry         | Graph Classification | Atom Encoder (9) | Bond Encoder (3) | AP           |
| Peptides-struct | Chemistry         | Graph Regression     | Atom Encoder (9) | Bond Encoder (3) | MAE          |


### Statistics of Datasets

| Dataset         | # Graphs |    # Nodes | μ Nodes | μ Deg. |     # Edges |  μ Edges | μ Short. Path |  μ Diameter |
| --------------- | -------: | ---------: | ------: | :----: | ----------: | -------: | ------------: | ----------: |
| PascalVOC-SP    |   11,355 |  5,443,545 |  479.40 |  5.65  |  30,777,444 | 2,710.48 |    10.74±0.51 |  27.62±2.13 |
| COCO-SP         |  123,286 | 58,793,216 |  476.88 |  5.65  | 332,091,902 | 2,693.67 |    10.66±0.55 |  27.39±2.14 |
| PCQM-Contact    |  529,434 | 15,955,687 |   30.14 |  2.03  |  32,341,644 |    61.09 |     4.63±0.63 |   9.86±1.79 |
| Peptides-func   |   15,535 |  2,344,859 |  150.94 |  2.04  |   4,773,974 |   307.30 |    20.89±9.79 | 56.99±28.72 |
| Peptides-struct |   15,535 |  2,344,859 |  150.94 |  2.04  |   4,773,974 |   307.30 |    20.89±9.79 | 56.99±28.72 |


### Python environment setup with Conda

```bash
conda create -n lrgb python=3.9
conda activate lrgb

conda install pytorch=1.9 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.2 -c pyg -c conda-forge
conda install pandas scikit-learn

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

# Check https://www.dgl.ai/pages/start.html to install DGL based on your CUDA requirements
pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html

pip install performer-pytorch
pip install torchmetrics==0.7.2
pip install ogb
pip install wandb

conda clean --all
```

### Running NeuralAtom

Given `GCN` as example, using `Decremental (desc)` setting shown in Fig.4 and pool ratio is setted as `0.9` with random `seed` as `0`, we have:
```bash
conda activate lrgb
cd 2D_Molecule
# Running GCN w/ Neural Atoms for Peptides-func.
python main.py --cfg ./configs/GCN/peptides-func-GCN.yaml seed 0
```