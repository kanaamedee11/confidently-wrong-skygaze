"""
make_qual.py : qualitative failure-case figure.
Picks a trial where plain SuperGlue returns a confident-looking but wrong homography at a
large rotation while the rotation-hypothesis variant recovers it, then draws all footprints.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SUPERGLUE_PATH", os.path.abspath(os.path.join(HERE, "..", "..", "SuperGluePretrainedNetwork")))
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analyze as A
import run_exp as R
from data_utils import generate_synthetic_reference
from gen2 import make_query, gt_center_and_quad

OUT = os.path.join(A.RES, "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "pdf.fonttype": 42})

sg = A.load("rot_sg.jsonl")
tta = A.load("rot_tta.jsonl")
key = lambda r: (r["level"], r["ref_seed"], r["seed"])
tta_ok = {key(r): r for r in tta if A.outcome(r) == "accurate"}
cands = [r for r in sg if A.outcome(r) == "silent" and key(r) in tta_ok and r["level"] in (45, 60, 75)]
cands.sort(key=lambda r: (-r["err_c"]))
pick = cands[min(1, len(cands) - 1)] if cands else None
if pick is None:
    print("no candidate found"); sys.exit(1)
print("picked", key(pick), "SG err", pick["err_c"])

ref = generate_synthetic_reference(size=800, seed=pick["ref_seed"])
cfg = dict(R.BASE, rotation_deg=float(pick["level"]))
q, gtH = make_query(ref, seed=1000 * pick["ref_seed"] + pick["seed"], **cfg)
gc, gq = gt_center_and_quad(q.shape, gtH)
ms = R.build_methods(["SIFT", "SuperGlue", "SuperGlueTTA"])

fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.0), gridspec_kw={"width_ratios": [1, 1.6, 1.6, 1.6]})
axs[0].imshow(cv2.cvtColor(q, cv2.COLOR_BGR2RGB)); axs[0].set_title("Query (%d$^\\circ$ rot.)" % pick["level"], fontsize=7)
axs[0].axis("off")
for ax, (name, ttl, col) in zip(axs[1:], [("SIFT", "SIFT", "#0072B2"), ("SuperGlue", "SuperGlue", "#D55E00"),
                                           ("SuperGlueTTA", "SuperGlue + rot. hyp.", "#009E73")]):
    r = ms[name](ref, q)
    ax.imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    gqc = np.vstack([gq, gq[:1]])
    ax.plot(gqc[:, 0], gqc[:, 1], color="white", lw=1.6, ls="--")
    if r["success"]:
        qq = np.vstack([r["quad_ref"], r["quad_ref"][:1]])
        ax.plot(qq[:, 0], qq[:, 1], color=col, lw=1.4)
        c_in = np.float32([[q.shape[1] / 2, q.shape[0] / 2]]).reshape(-1, 1, 2)
        ec = cv2.perspectiveTransform(c_in, np.asarray(r["homography"], np.float64)).reshape(2)
        err = float(np.linalg.norm(ec - gc))
        ax.set_title("%s\n%d inliers, error %.1f px" % (ttl, r["num_inliers"], err), fontsize=6.5)
    else:
        ax.set_title("%s\nno homography" % ttl, fontsize=6.5)
    ax.set_xlim(0, 800); ax.set_ylim(800, 0); ax.axis("off")
plt.tight_layout(w_pad=0.4)
plt.savefig(os.path.join(OUT, "fig_qualitative_v2.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(OUT, "fig_qualitative_v2.png"), dpi=300, bbox_inches="tight")
print("saved")
