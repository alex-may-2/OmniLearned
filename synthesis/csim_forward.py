"""Forward pass of a compiled hls4ml project's C-sim library over a jet npz (default: full 404k top test set).

    python csim_forward.py hls_prj/<project> [--npz top_test_full_n64.npz]

Rebuilds the C-sim library first (build_lib.sh bakes in the absolute weights path, which a project rename breaks).
Writes <project>/csim_forward_<npz stem>.npz with the logits.
"""
import argparse
import ctypes
import glob
import os
import re
import subprocess

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

p = argparse.ArgumentParser()
p.add_argument("project")
p.add_argument("--npz", default="top_test_full_n64.npz")
args = p.parse_args()

prj = os.path.abspath(args.project)
subprocess.run(["bash", "build_lib.sh"], cwd=prj, check=True, capture_output=True)
top = re.search(r"PROJECT=(\w+)", open(f"{prj}/build_lib.sh").read()).group(1)
lib = ctypes.CDLL(glob.glob(f"{prj}/firmware/{top}-*.so")[0])
f = getattr(lib, f"{top}_float")
ptr = np.ctypeslib.ndpointer(np.float32, flags="C")
f.argtypes, f.restype = [ptr, ptr], None

n_part = int(re.search(r"convert_data<float, input_t, (\d+)\*", open(f"{prj}/{top}_bridge.cpp").read()).group(1))
d = np.load(args.npz)
X, y = np.ascontiguousarray(d["x"][:, :n_part], dtype=np.float32), d["y"]  # leading-pT slots, like convert.py
logits = np.zeros((len(X), 2), np.float32)
for i in range(len(X)):
    f(X[i], logits[i])
np.savez(f"{prj}/csim_forward_{os.path.basename(args.npz)[:-4]}.npz", logits=logits, y=y)

e = np.exp(logits - logits.max(1, keepdims=True))
score = e[:, 1] / e.sum(1)
fpr, tpr, _ = roc_curve(y, score)
rej = "  ".join(f"1/eB@eS={s}={1 / np.interp(s, tpr, fpr):.1f}" for s in (0.3, 0.5, 0.7))
print(f"{os.path.basename(prj)} n_jets={len(y)} n_part={n_part} acc={np.mean(logits.argmax(1) == y):.4f} "
      f"AUC={roc_auc_score(y, score):.4f}  {rej}")
