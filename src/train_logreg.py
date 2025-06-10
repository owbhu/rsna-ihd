import os, pandas as pd, cv2, numpy as np, joblib, argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from src.config import IMG_SIZE, PROC_DIR, TRAIN_CSV, DEV_CSV

def load_flat(csv):
    df = pd.read_csv(csv)
    X = np.stack([cv2.imread(f"{PROC_DIR}/{sid}.png", cv2.IMREAD_GRAYSCALE).flatten()
                  for sid in df.slice_id])
    y = df["any"].values
    return X, y

def train_logreg(args):
    Xtr, ytr = load_flat(TRAIN_CSV)
    Xdv, ydv = load_flat(DEV_CSV)

    clf = LogisticRegression(max_iter=3000, C=args.lr, class_weight='balanced')
    clf.fit(Xtr, ytr)
    dv_pred = clf.predict_proba(Xdv)[:, 1]
    print("LogReg dev AUC", roc_auc_score(ydv, dv_pred))

    os.makedirs("runs", exist_ok=True)
    joblib.dump(clf, "runs/logreg.pkl")
