# rsna-ihd

Author: Owen Hughes
Contact: obh@uoregon.edu

````markdown
# RSNA Intracranial Hemorrhage Detection Pipeline

End-to-end code for detecting intracranial hemorrhage in head CT slices using three models: logistic regression, a small CNN from scratch, and ResNet-18 fine-tuned on RSNA’s 2019 competition data.

## Setup

1. **Clone & enter the repo**  
   ```bash
   git clone https://github.com/owbhu/rsna-ihd.git
   cd rsna-ihd
````

2. **Create & activate** the Conda environment

   ```bash
   conda env create -f environment.yml
   conda activate ihd
   ```

## Data Download

1. **Install & configure** Kaggle CLI

   ```bash
   pip install kaggle
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Download RSNA data**

   ```bash
   kaggle competitions download -c rsna-intracranial-hemorrhage-detection -p data/raw
   unzip data/raw/rsna-intracranial-hemorrhage-detection.zip -d data/raw/stage_2_train
   unzip data/raw/stage_2_test.zip -d data/raw/stage_2_test
   ```

3. **Verify** files

   ```
   ls data/raw/stage_2_train/*.dcm
   head -n6 data/raw/stage_2_train.csv
   ```

## Preprocessing

Convert DICOM → 3-channel PNG (brain, subdural, bone windows) at 160×160:

```bash
python -m src.preprocess \
       --csv       data/raw/stage_2_train.csv \
       --dicom_root data/raw/stage_2_train \
       --out_dir   data/processed/160
```

## Train/Dev/Test Split

Patient-stratified 60/20/20 split of the 20 000-slice subset:

```bash
python -m src.sample_data      # builds data/small_prs.csv if used
python -m src.split_data \
       --csv        data/processed/160/sample_30k.csv \
       --dicom_root data/processed/160
```

Output: `data/splits/train.csv`, `dev.csv`, `test.csv`.

## Training & Tuning

1. **Logistic Regression Baseline**

   ```bash
   python -m src.train --model logreg --lr 1.0
   python -m src.evaluate --model logreg --ckpt runs/logreg.pkl
   ```

2. **Small CNN (from scratch)**

   ```bash
   python -m src.tune  --model small_cnn --epochs 6
   python -m src.train --model small_cnn --lr 1e-3 --batch 32 --epochs 15
   python -m src.evaluate --model small_cnn --ckpt runs/small_cnn_best.ckpt
   ```

3. **ResNet-18 Fine-Tuning**

   ```bash
   python -m src.tune  --model resnet18 --epochs 6
   python -m src.train --model resnet18 --lr 1e-4 --batch 16 --epochs 15
   python -m src.evaluate --model resnet18 --ckpt runs/resnet18_best.ckpt
   ```

## Grad-CAM Visualization

Generate heatmaps for a given slice ID:

```bash
python -m src.cam --model small_cnn --ckpt runs/small_cnn_best.ckpt --slice_id ID_xxxxxxxx
python -m src.cam --model resnet18  --ckpt runs/resnet18_best.ckpt  --slice_id ID_xxxxxxxx
```

Outputs saved to `reports/figures/ID_xxxxxxxx_model_cam.png`.

## Repo Structure

```
rsna-ihd/
├── data/
│   ├── raw/                 # DICOMs & CSVs (ignored by Git)
│   ├── processed/160/       # PNGs
│   └── splits/              # train.csv, dev.csv, test.csv
├── reports/figures/         # Grad-CAM outputs
├── runs/                    # model checkpoints & logs (ignored)
├── src/
│   ├── config.py            # paths & constants
│   ├── preprocess.py        # DICOM→PNG
│   ├── sample_data.py       # (optional) build sample CSV
│   ├── split_data.py        # patient-stratified split
│   ├── train_logreg.py      # logistic regression
│   ├── train.py             # CNN training loop
│   ├── tune.py              # hyperparameter sweep
│   ├── evaluate.py          # AUC/F1 evaluation
│   ├── cam.py               # Grad-CAM generation
│   └── models.py            # SmallCNN & ResNet-18
├── environment.yml          # Conda spec
├── .gitignore
└── README.md
```
