"""
python -m src.split_data --csv data/raw/sample_20k.csv \
                         --dicom_root data/raw/train
Creates 60 / 20 / 20 train-dev-test splits.
If --dicom_root is omitted, the script skips patient-level grouping
and just does a random stratified split.
"""

import argparse, pandas as pd, numpy as np, pathlib, tqdm, pydicom
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.model_selection import GroupShuffleSplit
from src.config import SEED


ap = argparse.ArgumentParser()
ap.add_argument("--csv",        required=True, help="slice-list CSV")
ap.add_argument("--dicom_root", required=False, help="folder with DICOMs "
                                                     "(optional; if omitted skip patient grouping)")
ap.add_argument("--out_dir",    default="data/splits", help="where to write train/dev/test CSVs")
args = ap.parse_args()
pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)


sample = pd.read_csv(args.csv).sample(frac=1, random_state=SEED).reset_index(drop=True)


def patient_id_of(slice_id: str, dicom_root: str) -> str:
    dcm_path = f"{dicom_root}/{slice_id}.dcm"
    return pydicom.dcmread(dcm_path, stop_before_pixels=True).PatientID

if args.dicom_root:
    print("• building patient-level split …")
    tqdm.tqdm.pandas()
    sample["patient"] = sample.slice_id.progress_apply(lambda sid: patient_id_of(sid, args.dicom_root))




    gss = GroupShuffleSplit(n_splits=1, train_size=0.60, random_state=SEED)
    train_idx, hold_idx = next(
        gss.split(sample, y=sample["any"], groups=sample["patient"])
    )



    hold_df = sample.iloc[hold_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=SEED)
    dev_rel, test_rel = next(
        gss2.split(hold_df, y=hold_df["any"], groups=hold_df["patient"])
    )

    dev_idx  = hold_df.iloc[dev_rel].index
    test_idx = hold_df.iloc[test_rel].index
else:
    print("• no dicom_root supplied: random stratified 60/20/20")
    train_df, hold_df = train_test_split(
        sample, test_size=0.4, stratify=sample["any"], random_state=SEED
    )
    dev_df, test_df = train_test_split(
        hold_df, test_size=0.5, stratify=hold_df["any"], random_state=SEED
    )
    train_idx = train_df.index
    dev_idx   = dev_df.index
    test_idx  = test_df.index


sample.iloc[train_idx].to_csv(f"{args.out_dir}/train.csv", index=False)
sample.iloc[dev_idx  ].to_csv(f"{args.out_dir}/dev.csv",   index=False)
sample.iloc[test_idx ].to_csv(f"{args.out_dir}/test.csv",  index=False)
print("train/dev/test CSVs written:",
      len(train_idx), len(dev_idx), len(test_idx))
