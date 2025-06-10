"""
CLI: python -m src.sample_data --n_pos 10000 --n_neg 10000
Outputs sample_20k.csv with slice_id, any, patient_id
"""

import argparse, pandas as pd, numpy as np, random
from collections import defaultdict
from src.config import RAW_CSV, SAMPLE_CSV, SEED

def build_patient_lookup(csv_path):
    df = pd.read_csv(csv_path)
    df[['slice_id', 'type']] = df.Id.str.rsplit('_', n=1, expand=True)
    any_df = df[df.type == 'any'][['slice_id', 'Label']].rename(columns={'Label':'any'})
    return any_df

def sample_balanced(any_df, n_pos, n_neg):
    pos = any_df[any_df.any == 1].sample(n_pos, random_state=SEED)
    neg = any_df[any_df.any == 0].sample(n_neg, random_state=SEED)
    return pd.concat([pos, neg]).sample(frac=1, random_state=SEED)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pos", type=int, default=5000)
    ap.add_argument("--n_neg", type=int, default=5000)
    args = ap.parse_args()

    any_df = build_patient_lookup(RAW_CSV)
    sample = sample_balanced(any_df, args.n_pos, args.n_neg)
    sample.to_csv(SAMPLE_CSV, index=False)
    print(f"✓ wrote {SAMPLE_CSV} ({len(sample)} rows)")
