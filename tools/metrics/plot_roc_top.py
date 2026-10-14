"""Background rejection 1/eps_B vs signal efficiency eps_S for groups of top-tagging runs.

Each group is several runs of one recipe (e.g. seeds); it is drawn as its median curve with a min-max band. Scores
are the per-jet npz files `outputs_<tag>_top_test_rank*.npz` (omnilearned evaluate, or qat_deepsets_eval.py
--save-scores). The --ref run is drawn bold and is the baseline of the ratio panel.

    python tools/metrics/plot_roc_top.py --indir $PSCRATCH/omnilearned/eval/ps \\
        --group "KD float:ps_a_s1,ps_a_s2" --group "KD QAT:qat_a_s1,qat_a_s2" --ref qat_a_s2 --out roc_kd_vs_scratch
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from _common import load_rank_files, rej_at_eff

GRID = np.linspace(0.1, 0.9, 801)
EFFS = (0.3, 0.5)
COLORS = ["#2a9d6f", "#7b4fc0", "#3a6fd8", "#e07040", "#777777", "#c03060"]
STYLES = ["-", "-", "--", ":", "-.", "-"]


def run_curve(indir, tag):
    preds, labels = load_rank_files(indir, tag, "top")
    fpr, tpr, _ = roc_curve(labels, preds[:, 1])
    with np.errstate(divide="ignore"):
        rej = 1.0 / np.interp(GRID, tpr, fpr)
    stats = [roc_auc_score(labels, preds[:, 1])] + [rej_at_eff(fpr, tpr, e) for e in EFFS]
    return rej, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="dir with outputs_<tag>_top_test_rank*.npz")
    ap.add_argument("--group", action="append", required=True, help='"label:tag1,tag2,..."; repeatable')
    ap.add_argument("--ref", help="tag drawn bold and used as ratio baseline")
    ap.add_argument("--out", default="roc_top", help="output stem; writes .png and .pdf")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    groups = []
    for g in args.group:
        label, tags = g.split(":", 1)
        runs = {t: run_curve(args.indir, t) for t in tags.split(",")}
        groups.append((label, runs))
    ref = next((r[t][0] for _, r in groups for t in r if t == args.ref), None)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    print(f"{'group':40s} {'n':>2s}  {'AUC':>15s}  {'1/eB@30%':>13s}  {'1/eB@50%':>13s}")
    for i, (label, runs) in enumerate(groups):
        c, ls = COLORS[i % len(COLORS)], STYLES[i % len(STYLES)]
        curves = np.array([r[0] for r in runs.values()])
        med = np.median(curves, axis=0)
        ax.plot(GRID, med, color=c, ls=ls, lw=2, label=f"{label} ({len(runs)} runs)")
        ax.fill_between(GRID, curves.min(0), curves.max(0), color=c, alpha=0.2, lw=0)
        if ref is not None:
            axr.plot(GRID, med / ref, color=c, ls=ls, lw=2)
            axr.fill_between(GRID, curves.min(0) / ref, curves.max(0) / ref, color=c, alpha=0.2, lw=0)
        if args.ref in runs:
            ax.plot(GRID, runs[args.ref][0], color=c, lw=3.5, label=f"{args.ref} (ref)")
        s = np.array([r[1] for r in runs.values()])
        m, sd = s.mean(0), s.std(0)
        print(f"{label:40s} {len(runs):2d}  {m[0]:.4f}±{sd[0]:.4f}  {m[1]:6.1f}±{sd[1]:5.1f}  {m[2]:6.1f}±{sd[2]:5.1f}")
        for t, r in runs.items():
            print(f"    {t:36s}     {r[1][0]:.4f}         {r[1][1]:6.1f}         {r[1][2]:6.1f}")

    for a in (ax, axr):
        for e in EFFS:
            a.axvline(e, color="grey", lw=0.8, ls=":")
        a.grid(alpha=0.3)
    for e in EFFS:
        ax.text(e, 0.97, f"{int(e * 100)}%", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel(r"Background rejection $1/\varepsilon_B$")
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(args.title)
    axr.set_ylabel(f"ratio to\n{args.ref}" if ref is not None else "")
    axr.set_xlabel(r"Signal efficiency $\varepsilon_S$")
    axr.set_xlim(GRID[0], GRID[-1])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print(f"wrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
