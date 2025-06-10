"""
Evaluate a trained model on the held-out test set.

Usage:
  python -m src.evaluate --model small_cnn --ckpt runs/small_cnn_best.ckpt
"""
import argparse, torch, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader
from src.config import PROC_DIR, TEST_CSV
from src.datamodules import SliceDataset
from src.models      import SmallResNet, resnet18_ft

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["small_cnn","resnet18"])
    ap.add_argument("--ckpt",  required=True, help="path to .ckpt file")
    args = ap.parse_args()

    device = torch.device("cpu")
    model  = {"small_cnn":SmallResNet,"resnet18":resnet18_ft}[args.model]()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    test_ds = SliceDataset(TEST_CSV, PROC_DIR, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )



    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            all_logits.append(logits.sigmoid())
            all_labels.append(y)

    logits = torch.cat(all_logits).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()


    auc = roc_auc_score(labels, logits)
    f1  = f1_score(labels, (logits > 0.5).astype(int))
    print(f"Test AUC: {auc:.4f}")
    print(f"Test F1:  {f1:.4f}")
