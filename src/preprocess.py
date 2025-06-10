"""
python -m src.preprocess --csv data/raw/sample_20k.csv \
                         --dicom_root data/raw/train \
                         --out_dir    data/processed/160

Reads each DICOM slice, applies three window/level settings
(brain, subdural, bone), stacks them into an RGB image,
resizes to 160×160, and writes 3-channel PNGs.
"""

import os
import argparse
import pandas as pd
import pydicom
import numpy as np
import cv2
from tqdm import tqdm

def window(arr: np.ndarray, wl: float, ww: float) -> np.ndarray:
    lo, hi = wl - ww/2, wl + ww/2
    arr_clipped = np.clip(arr, lo, hi)
    scaled = (arr_clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",        required=True,
                    help="Path to slice-list CSV")
    ap.add_argument("--dicom_root", required=True,
                    help="Directory of DICOM slices")
    ap.add_argument("--out_dir",    required=True,
                    help="Where to write 3-channel PNGs")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)

    for sid in tqdm(df.slice_id, desc="Preprocessing"):
        dcm = pydicom.dcmread(os.path.join(args.dicom_root, f"{sid}.dcm"))
        arr = dcm.pixel_array.astype(np.float32)

        c1 = window(arr,  40,    80)    # brain
        c2 = window(arr,  75,   215)    # subdural
        c3 = window(arr, 600,  2000)    # bone/high-contrast



        rgb = np.stack([c1, c2, c3], axis=2)
        rgb = cv2.resize(rgb, (160, 160),
                         interpolation=cv2.INTER_AREA)

        out_path = os.path.join(args.out_dir, f"{sid}.png")
        cv2.imwrite(out_path, rgb)
