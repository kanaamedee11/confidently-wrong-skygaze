"""Figures for the real-data section: (1) example pair (drone frame + satellite tile), (2) outcome bars, (3) risk-coverage."""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
import analyze as A
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "pdf.fonttype": 42})
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "paper", "figures"); TAU = 25.0
rows = [json.loads(l) for f in glob.glob(os.path.join(ROOT, "results_real", "results_real_*.jsonl")) for l in open(f)]
def oc(r): return "none" if not r["H"] else ("acc" if r["err_m"] <= TAU else "silent")
M = ["SIFT", "ORB", "SuperGlue", "SuperGlueTTA"]; LAB = {"SIFT": "SIFT", "ORB": "ORB", "SuperGlue": "SuperGlue", "SuperGlueTTA": "SuperGlue + rot. hyp."}
COL = {"SIFT": "#0072B2", "ORB": "#E69F00", "SuperGlue": "#D55E00", "SuperGlueTTA": "#009E73"}

# --- data figure: satellite map with the flight-03 frame positions + 3 example drone frames
rws = load_rows(); S = sat(True)
fig = plt.figure(figsize=(7.0, 2.6))
ax = fig.add_axes([0.0, 0.02, 0.50, 0.93]); ax.imshow(cv2.cvtColor(S, cv2.COLOR_BGR2RGB)); ax.axis("off")
pts = np.array([gt_px(float(r["lat"]), float(r["lon"]), True) for r in rws])
ax.scatter(pts[:, 0], pts[:, 1], s=6, c="yellow", edgecolors="k", linewidths=0.3)
ax.set_title("Satellite map (flight 03) with the 40 frame positions used", fontsize=7.5)
for i, idx in enumerate([0, 13, 27]):
    a = fig.add_axes([0.52 + i * 0.16, 0.30, 0.155, 0.45])
    im = cv2.imread(os.path.join(D, "drone", rws[idx]["filename"])); a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); a.axis("off")
    a.set_title(rws[idx]["filename"].replace(".JPG", ""), fontsize=6.5)
fig.text(0.76, 0.17, "Example drone frames (real, resized for display)", ha="center", fontsize=7)
plt.savefig(os.path.join(OUT, "fig_realdata_v2.pdf"), bbox_inches="tight"); plt.savefig(os.path.join(OUT, "fig_realdata_v2.png"), dpi=300, bbox_inches="tight"); plt.close()

# --- outcome bars + risk-coverage
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.25, 1]})
x = np.arange(len(M)); w = 0.27
for j, (k, lab, c) in enumerate([("acc", "accurate", "#2a9d8f"), ("silent", "silent failure", "#d62828"), ("none", "detected failure", "#9a9a9a")]):
    v = []
    for m in M:
        rs = [r for r in rows if r["method"] == m]; v.append(100 * sum(oc(r) == k for r in rs) / max(1, len(rs)))
    a1.bar(x + (j - 1) * w, v, w, label=lab, color=c)
a1.set_xticks(x); a1.set_xticklabels([LAB[m].replace(" + ", "\n+ ") for m in M], fontsize=7); a1.set_ylabel("Trials (%)"); a1.set_ylim(0, 100)
a1.set_title("(a) Outcomes, real drone vs satellite (error $\\leq$ 25 m)"); a1.legend(loc="upper left", frameon=True)
for m in M:
    rs = [r for r in rows if r["method"] == m]
    if not rs: continue
    ts = sorted({0} | {r["n_in"] for r in rs if r["H"]})
    rc = []
    for t in ts:
        acc_ = [r for r in rs if r["H"] and r["n_in"] >= t]
        if acc_: rc.append((100 * len(acc_) / len(rs), 100 * sum(oc(r) != "acc" for r in acc_) / len(acc_)))
    a2.plot([p[0] for p in rc], [p[1] for p in rc], color=COL[m], label=LAB[m], lw=1.1)
a2.set_xlabel("Coverage: trials accepted (%)"); a2.set_ylabel("Wrong among accepted (%)"); a2.set_title("(b) Inlier-count gate"); a2.grid(alpha=0.3, lw=0.4); a2.set_ylim(0, 102)
a2.legend(loc="lower right", fontsize=6.5)
plt.tight_layout(w_pad=1.2)
plt.savefig(os.path.join(OUT, "fig_realres_v2.pdf"), bbox_inches="tight"); plt.savefig(os.path.join(OUT, "fig_realres_v2.png"), dpi=300, bbox_inches="tight"); plt.close()
print("figs ok")
