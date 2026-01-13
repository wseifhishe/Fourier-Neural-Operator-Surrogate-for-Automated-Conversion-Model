# AC-DBM FNO Dummy Demo (4 scripts)

![Pipeline overview](assets/pipeline.png)


업로드할 파일은 아래 5개만 있으면 됩니다.
- README.md
- generate_dummy_tf.py
- make_dataset.py
- train_fno.py
- uq_report.py

## 설치
```bash
pip install numpy torch
```

## 실행
```bash
python generate_dummy_tf.py --out_dir data/raw --n_train_cases 2 --n_test_cases 1 --seed 42
python make_dataset.py --in_npz data/raw/dummy_tf.npz --out_dir data/processed --n_phi 40 --val_ratio 0.2 --seed 42
python train_fno.py --data_dir data/processed --out_dir outputs/default --quick
python uq_report.py --data_dir data/processed --out_dir outputs/default --T 20 --Q 0.95
```

> 더미 데이터이므로 논문 수치 재현이 목적이 아닙니다. 목적은 pipeline/구현 검증입니다.
