"""Max-SIC vs injected-signal-events curve, read from lhco_sic_results.json
(written by compute_roc_sic_lhco.py) rather than pasted-in numbers -- so this
plot is always regenerable from the actual eval outputs on disk, unlike the
original OmniLearn plot_lhco_sic.py.

Groups entries by nsig (multiple tags can share an nsig -- e.g. repeated
seeds). With >=2 seeds at an nsig, plots median + 16/84% quantile band,
matching the original methodology. With exactly 1 seed, plots a bare point/
line and says so explicitly in the legend -- never fakes a band from a single
run.

Usage:
    python plot_lhco_sic_curve.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"
GRAY = "#8a8a86"

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lhco_sic_results.json")

# ============================================================
#  EDIT THIS PER PLOT -- exact SAVE_TAG keys (from lhco_sic_results.json) to
#  include. Superseded/debugging reruns at the same nsig (e.g. r1, r2 before
#  a working r3) won't silently sneak into the curve as extra "seeds" unless
#  listed here. Empty set = include everything in the file (not recommended
#  once you have more than one attempt per nsig).
# ============================================================
INCLUDE_TAGS = {
    "fine_tune_pretrain_s_lhco_ad_nsig500_test8",
    "fine_tune_pretrain_s_lhco_ad_nsig1000_r3",
    "fine_tune_pretrain_s_lhco_ad_nsig2000_r1",
    "fine_tune_pretrain_s_lhco_ad_nsig10000_test7",
}
# ============================================================


def load_by_nsig():
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    by_nsig = {}
    for tag, entry in results.items():
        if INCLUDE_TAGS and tag not in INCLUDE_TAGS:
            continue
        by_nsig.setdefault(entry["nsig"], []).append(entry["max_sic"])
    return by_nsig


def main():
    by_nsig = load_by_nsig()
    if not by_nsig:
        raise SystemExit(f"No entries in {RESULTS_FILE} -- run compute_roc_sic_lhco.py first.")

    nsigs = sorted(by_nsig)
    median = [np.median(by_nsig[n]) for n in nsigs]
    lo = [np.quantile(by_nsig[n], 0.16) if len(by_nsig[n]) >= 2 else None for n in nsigs]
    hi = [np.quantile(by_nsig[n], 0.84) if len(by_nsig[n]) >= 2 else None for n in nsigs]

    has_band = any(l is not None for l in lo)
    label = "median (with 16/84% band where >=2 seeds)" if has_band else "classifier"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(nsigs, median, "o-", color=BLUE, linewidth=2, label=label)
    if has_band:
        band_lo = [l if l is not None else m for l, m in zip(lo, median)]
        band_hi = [h if h is not None else m for h, m in zip(hi, median)]
        ax.fill_between(nsigs, band_lo, band_hi, color=BLUE, alpha=0.3)
    ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1.5, label="random (SIC=1)")

    ax.set_xlabel("injected signal events")
    ax.set_ylabel("Max SIC")
    # ax.set_title("Max SIC vs injected signal events")
    ax.legend(fontsize=8)
    fig.tight_layout()

    outfile = "lhco_sic_curve.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")
    for n, vals in sorted(by_nsig.items()):
        print(f"nsig={n:<6} seeds={len(vals)}  values={vals}")


if __name__ == "__main__":
    main()
