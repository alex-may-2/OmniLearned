"""Truth-labeled ROC/AUC/max-SIC for one fine-tuned lhco_ad checkpoint.

Combines the real test-set evaluate output's pid==0 rows (pure background,
already held out) with the leftover-signal evaluate output (pure, never
injected -- see build_lhco_eval_signal.py + evaluate_lhco_eval_signal.sh).
No retraining involved; both point at the same already-fine-tuned checkpoint.

EDIT THESE PER RUN -- must match the fine_tune_lhco_ad.sh/evaluate_lhco_ad.sh/
evaluate_lhco_eval_signal.sh run being scored.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

BLUE = "#2a78d6"
GRAY = "#8a8a86"

# Every run's (nsig, auc, max_sic) lands here, keyed by SAVE_TAG -- so
# plot_lhco_sic_curve.py can build the nsig-vs-max-SIC curve from files
# instead of pasted-in numbers (the original OmniLearn plot_lhco_sic.py's
# approach, which goes stale the moment a model is re-evaluated).
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lhco_sic_results.json")

# ============================================================
SAVE_TAG_BASE = "fine_tune_pretrain_s"
DATASET = "lhco_ad"
NSIG = 10000
DESCRIPT_TAG = "test7"
QUANTIZATION = "none"

# Drop FPR below this before max-SIC -- matches evaluate_classifiers_lhco.py
# (uncommented in that repo; not stated in the paper text either). Their own
# background eval set (val_background_SR_extended.h5, 119,004 events) is
# only ~2x ours, so 1e-4 there is already just ~12 surviving background
# events -- a noisy regime in both cases, not something ours needs a
# different floor to compensate for.
FPR_FLOOR = 1e-4
# ============================================================

SAVE_TAG = f"{SAVE_TAG_BASE}_{DATASET}_nsig{NSIG}_{DESCRIPT_TAG}"
INDIR_TEST = f"/pscratch/sd/m/mbenyas/{SAVE_TAG}_{QUANTIZATION}"
INDIR_SIGNAL = f"/pscratch/sd/m/mbenyas/{SAVE_TAG}_eval_signal_{QUANTIZATION}"


def load_npz_shards(indir, tag, dataset=DATASET, dataset_type="test"):
    """Concatenate prediction/pid across every per-rank npz for one run."""
    pattern = os.path.join(indir, f"outputs_{tag}_{dataset}_{dataset_type}_rank*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching: {pattern}")
    pred, pid = [], []
    for p in paths:
        z = np.load(p)
        pred.append(z["prediction"].astype(np.float32))
        pid.append(z["pid"])
    return np.concatenate(pred), np.concatenate(pid)


def max_sic(fpr, tpr, fpr_floor):
    """Peak of TPR/sqrt(FPR) above fpr_floor -- below it the ratio is noise-
    dominated by the last surviving background events. Random classifier's
    baseline is 1.0, not 0."""
    keep = fpr > fpr_floor
    tpr, fpr = tpr[keep], fpr[keep]
    if len(fpr) == 0:
        return float("nan"), 0
    return np.max(tpr / np.sqrt(fpr)), len(fpr)


def save_result(save_tag, nsig, auc, sic, n_bkg, n_sig):
    """Upsert this run's numbers into RESULTS_FILE, keyed by save_tag -- so
    rerunning the same (nsig, descript_tag) overwrites cleanly instead of
    accumulating duplicates, while different tags (different nsig, or
    different seeds at the same nsig) all persist side by side."""
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    results[save_tag] = {
        "nsig": nsig,
        "auc": auc,
        "max_sic": sic,
        "n_background": n_bkg,
        "n_signal": n_sig,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved result to {RESULTS_FILE}")


def plot_roc_sic(fpr, tpr, auc, sic, outfile):
    """ROC (with the random-classifier diagonal) beside SIC=TPR/sqrt(FPR)
    vs FPR (with the random baseline at SIC=1) -- both x-axes log-scaled,
    standard for HEP background rejection."""
    sic_curve = np.divide(tpr, np.sqrt(fpr), out=np.zeros_like(tpr), where=fpr > 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.plot(fpr, tpr, color=BLUE, linewidth=2, label="classifier")
    ax.plot([1e-4, 1], [1e-4, 1], color=GRAY, linestyle="--", linewidth=1.5, label="random")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("false positive rate (background efficiency)")
    ax.set_ylabel("true positive rate (signal efficiency)")
    ax.set_title(f"ROC  (AUC = {auc:.4f})")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(fpr, sic_curve, color=BLUE, linewidth=2, label="classifier")
    ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1.5, label="random (SIC=1)")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("false positive rate (background efficiency)")
    ax.set_ylabel("SIC = TPR / sqrt(FPR)")
    ax.set_title(f"Max SIC = {sic:.4f}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")


def main():
    pred_bkg_pool, pid_bkg_pool = load_npz_shards(INDIR_TEST, SAVE_TAG)
    bkg_mask = pid_bkg_pool == 0
    n_bkg = int(bkg_mask.sum())
    if n_bkg == 0:
        raise SystemExit(f"No pid==0 rows in {INDIR_TEST} -- check SAVE_TAG/DATASET.")
    score_bkg = pred_bkg_pool[bkg_mask, 1]

    pred_sig, pid_sig = load_npz_shards(INDIR_SIGNAL, SAVE_TAG)
    if not (pid_sig == 1).all():
        raise SystemExit(f"Expected all-pid==1 in {INDIR_SIGNAL} -- got a mix; wrong --path at evaluate time?")
    score_sig = pred_sig[:, 1]

    scores = np.concatenate([score_bkg, score_sig])
    truth = np.concatenate([np.zeros(n_bkg, dtype=np.int64), np.ones(len(score_sig), dtype=np.int64)])

    auc = roc_auc_score(truth, scores)
    fpr, tpr, _ = roc_curve(truth, scores)
    sic, n_surviving = max_sic(fpr, tpr, FPR_FLOOR)

    print(f"save_tag: {SAVE_TAG}")
    print(f"n_background (pure, held-out)  : {n_bkg:,}")
    print(f"n_signal (pure, never injected): {len(score_sig):,}")
    print(f"AUC     : {auc:.4f}")
    print(f"Max SIC : {sic:.4f}  (FPR floor={FPR_FLOOR:g}, {n_surviving:,} ROC points survive)")
    print("Reference: a random classifier's max-SIC is 1.0, not 0.")

    save_result(SAVE_TAG, NSIG, auc, sic, n_bkg, len(score_sig))
    plot_roc_sic(fpr, tpr, auc, sic, f"roc_sic_{SAVE_TAG}.png")


if __name__ == "__main__":
    main()
