# FEX_Ep

![License](https://github.com/Daviddjddu/FEX_Ep/blob/main/LICENSE)

By [Jianda Du], [Senwei Liang], and [Chunmei Wang]

This repository is the implementation of article "Learning Epidemiological Dynamics via the Finite Expression Method" published on Journal of Machine Learning for Modeling and Computing. (https://doi.org/10.48550/arXiv.2412.21049)

## Introduction

 Finite Expression Method is a symbolic learning framework that leverages reinforcement learning to derive explicit mathematical expressions for epidemiological dynamics. It is used to discover the physics law behind covid.

## Environment

* Python ≥ 3.11
* [PyTorch](https://pytorch.org/)
* NumPy, Pandas, Matplotlib
* (Optional) Jupyter / ipykernel for notebooks

Quick start with Miniconda/Mambaforge:

```bash
conda create -n ml python=3.11 -y
conda activate ml
conda install numpy pandas matplotlib -y
pip install torch torchvision torchaudio ipykernel
python -m ipykernel install --user --name ml --display-name "Python (ml)"
```

## Code structure

```
FEX_Ep
│   README.md              <-- You are here
│
├── Data/                  
│   ├── sir_training_data.npz
│   ├── seir_training_data.npz
│   ├── seird_training_data.npz
│   └── Covid data
│   
│
│
├── Plot/                  <-- exported figures (PNG)
│
│
├── scripts/               <-- Python scripts (generation & training)
│   ├── generate_sir_data.py
│   ├── generate_seir_data.py
│   ├── generate_seird_data.py
│   ├── FEX for all three models (SIR, SEIR, SEIRD) and Covid data
│   ├── NN for all three models (SIR, SEIR, SEIRD)
│   ├── RNN for all three models (SIR, SEIR, SEIRD)
│   └── SEIQRDP for Covid data prediction (compared with FEX)
│
└── notebooks/
    └── Running_Order.ipynb  <-- run order to reproduce results
```

## Data

To reproduce the figure in the paper, use datasets included in the "Data" folder.
To test other possibilities, generate the new datasets via the three .ipynb files

## Running order

Open **`notebooks/Running_Order.ipynb`** and execute cells **top‑to‑bottom**. It will:

1. (Optional) **Generate** datasets into `Data/` and save initial conditions to CSV.
2. **Train** FEX and NN/RNN models; save `.pt` weights and `.npy` metrics.
3. **Plot** figures into `Plot/`.

## Citing

If this code is useful to your work, please kindly cite :

```
@article{du2025learning,
  title={Learning epidemiological dynamics via the finite expression method},
  author={Du, Jianda and Liang, Senwei and Wang, Chunmei},
  journal={Journal of machine learning for modeling and computing},
  volume={6},
  number={1},
  year={2025},
  publisher={Begel House Inc.}
}
```

If one is interested in the Finite expression method, one can read :

```
@article{liang2022finite,
  title={Finite Expression Method for Solving High-Dimensional Partial Differential Equations},
  author={Liang, Senwei and Yang, Haizhao},
  journal={arXiv preprint arXiv:2206.10121},
  year={2022}
}
```

## Acknowledgments

* Built with [PyTorch](https://pytorch.org/).
* Thanks to related open‑source projects and prior baselines that inspired organization and tooling.

---
