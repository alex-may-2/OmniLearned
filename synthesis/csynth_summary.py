"""One-line csynth summary per hls4ml project: II, latency, clock estimate, % of one SLR, peak csynth RAM.

Usage (from synthesis/): python csynth_summary.py hls_prj/<project> [logs/<tag>.txt ...]
The optional log (from `/usr/bin/time -v`) gives the peak RAM. Logs and projects pair up in order.
"""

import glob
import re
import sys


def row(text, name):
    m = re.search(rf"^\|{re.escape(name)}\s*\|(.+)$", text, re.M)
    return [c.strip() for c in m.group(1).split("|") if c.strip()] if m else None


def summary(prj, log=None):
    rpts = [r for r in glob.glob(f"{prj}/*_prj/solution1/syn/report/*_csynth.rpt") if "_s_csynth" not in r]
    top = min(rpts, key=len)  # the top function's report has the shortest name
    text = open(top).read()
    clk = re.search(r"\|ap_clk\s*\|\s*([\d.]+) ns\|\s*([\d.]+) ns\|", text)
    lat = re.search(r"Latency \(cycles\).*?\n.*?\n.*?\n\s*\|\s*(\d+)\|\s*(\d+)\|[^|]*\|[^|]*\|\s*(\d+)\|\s*(\d+)\|\s*(\S+)\|", text, re.S)
    slr = row(text, "Utilization SLR (%)")  # BRAM_18K, DSP, FF, LUT, URAM
    tot = row(text, "Total")
    target, est = float(clk.group(1)), float(clk.group(2))
    ii = int(lat.group(4))
    out = (
        f"{prj.rstrip('/').split('/')[-1]}: II {ii} cyc = {ii * target:g} ns, latency {lat.group(2)} cyc, "
        f"clock {target:g} -> est {est:g} ns{' MISS' if est > target else ''}, "
        f"%SLR LUT {slr[3]} FF {slr[2]} DSP {slr[1]} BRAM {slr[0]} (LUT {tot[3]}, FF {tot[2]}, DSP {tot[1]})"
    )
    if log:
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", open(log, errors="ignore").read())
        t = re.search(r"Elapsed \(wall clock\) time.*?: (\S+)", open(log, errors="ignore").read())
        out += f", peak RAM {int(m.group(1)) / 2**20:.1f} GB" if m else ""
        out += f", wall {t.group(1)}" if t else ""
    return out


if __name__ == "__main__":
    prjs = [a for a in sys.argv[1:] if not a.endswith(".txt")]
    logs = [a for a in sys.argv[1:] if a.endswith(".txt")]
    for i, prj in enumerate(prjs):
        print(summary(prj, logs[i] if i < len(logs) else None))
