# Fourier-Neural-Operator-Surrogate-for-Automated-Conversion-Model Dummy Demo (4 scripts)

![Pipeline overview](assets/pipeline.png)


*Figure: End-to-end demo pipeline using synthetic complex transfer functions (X:45 → Y:24).*

This repository reproduces the full pipeline (data format, phase augmentation, training, and UQ reporting) using synthetic dummy transfer functions that match the tensor shapes used in the paper. It is intended for pipeline/implementation verification rather than reproducing proprietary FE results.

You only need the following 5 files in the GitHub repository:
- README.md
- generate_dummy_tf.py
- make_dataset.py
- train_fno.py
- uq_report.py

## Installation
```bash
pip install numpy torch
```

## Run
```bash
python generate_dummy_tf.py --out_dir data/raw --n_train_cases 2 --n_test_cases 1 --seed 42
python make_dataset.py --in_npz data/raw/dummy_tf.npz --out_dir data/processed --n_phi 40 --val_ratio 0.2 --seed 42
python train_fno.py --data_dir data/processed --out_dir outputs/default --quick
python uq_report.py --data_dir data/processed --out_dir outputs/default --T 20 --Q 0.95
```

> Note: This repository uses **synthetic dummy data**, so it is **not intended to reproduce the exact numerical results** reported in the paper.  
> The purpose is to enable **pipeline/implementation reproducibility** (i.e., to verify code structure, tensor formats, preprocessing/augmentation, training, and UQ reporting).
