"""Evaluate a full-quant QAT DeepSets checkpoint (produced by qat_deepsets.py)
on the top-tagging test split.

The model is rebuilt from the checkpoint's arch_config (shape and quantizer
settings) with the same wrapping QAT used, then the weights are restored.

Must run with the omnilearned-fpga/env python (has Brevitas).

Usage:
    /global/homes/t/twamorka/omnilearned-fpga/env/bin/python qat_deepsets_eval.py \
        --tag qat_top_deepsets_distillnet_fpga_a05_T4_8bit_fullQuant
"""

import argparse

import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

from omnilearned.dataloader import load_data
from omnilearned.utils import ddp_setup
from qat_deepsets import load_qat_model

CHECKPOINT_DIR = "/pscratch/sd/t/twamorka/omnilearned/checkpoints/"
DATA_PATH = "/global/cfs/cdirs/m4567/www/"
SIGNAL_EFFS = [0.50, 0.30]


def compute_metrics(preds, labels):
    scores = preds[:, 1]
    pred_classes = preds.argmax(axis=1)
    acc = (pred_classes == labels).mean()
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    rej = {}
    for eff in SIGNAL_EFFS:
        idx = tpr.searchsorted(eff)
        rej[eff] = float("inf") if idx >= len(fpr) or fpr[idx] == 0 else 1.0 / fpr[idx]
    return acc, auc, rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="QAT checkpoint save-tag")
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR, help="dir holding the QAT --tag checkpoint")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-path", default=DATA_PATH, help="dataset root holding top/")
    args = ap.parse_args()

    local_rank, rank, size = ddp_setup()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model, cfg = load_qat_model(args.checkpoint_dir, args.tag)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.tag} ({cfg['size']}, fixed_n={cfg['fixed_n']}, quant={cfg['quant']}): {n_params:,} params")

    test_loader = load_data(
        "top", dataset_type="test", use_cond=True, path=args.data_path, batch=args.batch,
        num_workers=args.num_workers, rank=0, size=1, mode="classifier", shuffle=False,
    )

    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="QAT eval", total=len(test_loader)):
            X = batch["X"].to(device, dtype=torch.float)
            y = batch["y"].to(device)
            cond = batch["cond"].to(device) if batch.get("cond") is not None else None
            out = model(X, y, cond=cond)
            preds.append(out["y_pred"].float().softmax(-1).cpu())
            labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    acc, auc, rej = compute_metrics(preds, labels)
    print(f"\n=== {args.tag} (full-quant QAT) ===")
    print(f"  N events  : {len(labels):,}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC       : {auc:.4f}")
    for eff in SIGNAL_EFFS:
        print(f"  1/FPR @ {int(eff*100)}% sig eff : {rej[eff]:.1f}")


if __name__ == "__main__":
    main()
