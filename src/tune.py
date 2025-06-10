"""
python -m src.tune --model small_cnn
"""
import itertools, subprocess, json, os, argparse
grids = {
    "small_cnn": {"lr":[1e-3,3e-4], "batch":[32]},
    "resnet18":  {"lr":[1e-4,3e-5], "batch":[16]}
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Grid-search small CNN / ResNet-18")
    ap.add_argument("--model",  required=True, choices=["small_cnn", "resnet18"])
    ap.add_argument("--epochs", type=int, default=6,
                    help="epochs per trial (default 6)")
    args = ap.parse_args()

    rows = []
    for vals in itertools.product(*grids[args.model].values()):
        cfg = dict(zip(grids[args.model].keys(), vals))


        cmd = [
            "python", "-m", "src.train",
            "--model", args.model,
            "--lr",    str(cfg["lr"]),
            "--batch", str(cfg["batch"]),
            "--epochs", str(args.epochs)
        ]


        res = subprocess.run(cmd, capture_output=True, text=True)
        out = res.stdout + res.stderr


        if "best_dev_auc" in out:
            auc = float(out.split("best_dev_auc")[-1].split()[0])
        elif "dev AUC" in out:
            auc = float(out.split("dev AUC")[-1].split()[0])
        else:
            print("⚠️  could not parse AUC for", cfg, "\n", out[:400], "…")
            auc = 0.0

        rows.append({**cfg, "auc": auc})



    os.makedirs("runs", exist_ok=True)
    json.dump(rows, open(f"runs/grid_{args.model}.json", "w"), indent=2)
