"""
Train or fine-tune a model.

Usage examples
--------------
# logistic-reg baseline (dev only)
python -m src.train --model logreg --lr 1.0

# small CNN quick run
python -m src.train --model small_cnn --lr 3e-4 --batch 32 --epochs 12
"""

import argparse, os, torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score

from src.config      import PROC_DIR, TRAIN_CSV, DEV_CSV
from src.datamodules import make_loaders
from src.models      import SmallResNet, resnet18_ft


def get_model(name: str):
    """Return an un-initialised model (no .to(device) yet)."""
    return {"small_cnn": SmallResNet,
            "resnet18": resnet18_ft}[name]()

def run_epoch(model, loader, criterion, optimizer=None):
    model.train(bool(optimizer))
    logits, labels, losses = [], [], []

    for x, y in loader:

        x, y = x.to(model.device), y.to(model.device)
        out  = model(x)
        out  = out.squeeze(1)

        loss = criterion(out, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logits.append(out.detach().cpu())
        labels.append(y.cpu())
        losses.append(loss.item())

    logits = torch.cat(logits)
    labels = torch.cat(labels)
    auc    = roc_auc_score(labels, logits.sigmoid())
    f1     = f1_score(labels, (logits > 0).int())
    return sum(losses) / len(losses), auc, f1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=True,
                    choices=["small_cnn", "resnet18", "logreg"])
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--batch",  type=int,   default=32)
    ap.add_argument("--epochs", type=int,   default=10)
    args = ap.parse_args()


    if args.model == "logreg":
        from src.train_logreg import train_logreg
        train_logreg(args)
        exit()


    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Training on:", device)


    loaders = make_loaders(args.batch, PROC_DIR, train=TRAIN_CSV, dev=DEV_CSV)
    model   = get_model(args.model).to(device)
    model.device = device

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("runs", exist_ok=True)
    best_auc = 0.0
    patience = 2
    stale    = 0

    for ep in range(args.epochs):
        tr_loss, tr_auc, _  = run_epoch(model, loaders["train"], criterion, optimizer)
        dv_loss, dv_auc, _  = run_epoch(model, loaders["dev"],   criterion)

        print(f"[{ep+1}/{args.epochs}] train AUC {tr_auc:.3f}  dev AUC {dv_auc:.3f}")

        if dv_auc > best_auc:
            best_auc = dv_auc
            stale    = 0
            torch.save(model.state_dict(), f"runs/{args.model}_best.ckpt")
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping (patience reached).")
                break

    print(f"best_dev_auc {best_auc:.4f}")
