"""
make_figs.py : figures + LaTeX table fragments for the SkyGaze v2 paper.
Run after the experiments:  python3 src/make_figs.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import analyze as A

OUT = os.path.join(A.RES, "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
STYLE = {  # colour-blind-safe palette, distinct markers for greyscale print
    "SIFT": dict(color="#0072B2", marker="o", ls="-"),
    "ORB": dict(color="#E69F00", marker="s", ls="--"),
    "SuperGlue": dict(color="#D55E00", marker="^", ls="-."),
    "SuperGlueTTA": dict(color="#009E73", marker="D", ls="-"),
}
LABEL = {"SIFT": "SIFT", "ORB": "ORB", "SuperGlue": "SuperGlue", "SuperGlueTTA": "SuperGlue + rot. hyp."}


# ------------------------------------------------------------------ pipeline figure
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 1.75))
    ax.set_xlim(0, 14); ax.set_ylim(0, 3.5); ax.axis("off")

    def box(x, y, w, h, txt, fc="white", fs=7, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec="black", lw=0.7))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.15)

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                     lw=0.7, color="black"))

    box(0.1, 2.05, 1.9, 0.95, "Reference\ntile (georef.)", "#eef2f7")
    box(0.1, 0.5, 1.9, 0.95, "Query frame\n(UAV camera)", "#eef2f7")
    box(2.6, 0.5, 2.1, 0.95, "Rotation\nhypotheses\n0/90/180/270" + "°", "#e6f4ee", bold=False)
    box(5.3, 1.3, 2.4, 1.0, "Matcher\nSIFT | ORB |\nSuperPoint+SuperGlue", "#fdf3e3")
    box(8.3, 1.3, 1.9, 1.0, "RANSAC\nhomography H", "#fdf3e3")
    box(10.8, 1.3, 1.5, 1.0, "Confidence\ngate  n_in ≥ τ", "#e6f4ee")
    box(12.7, 2.15, 1.2, 0.85, "Location\n(accept)", "#eef2f7")
    box(12.7, 0.45, 1.2, 0.85, "Reject\n(fall back)", "#f7e9e9")
    arr(2.0, 2.5, 5.3, 2.0)
    arr(2.0, 0.97, 2.6, 0.97)
    arr(4.7, 0.97, 5.3, 1.6)
    arr(7.7, 1.8, 8.3, 1.8)
    arr(10.2, 1.8, 10.8, 1.8)
    arr(12.3, 1.95, 12.7, 2.55)
    arr(12.3, 1.65, 12.7, 0.9)
    ax.text(3.65, 0.22, "optional", ha="center", fontsize=6, style="italic")
    
    plt.savefig(os.path.join(OUT, "fig_pipeline_v2.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.savefig(os.path.join(OUT, "fig_pipeline_v2.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()


# ------------------------------------------------------------------ rotation figure
def fig_rotation(rot):
    methods = [m for m in ["SIFT", "ORB", "SuperGlue", "SuperGlueTTA"] if any(r["method"] == m for r in rot)]
    g = A.group(rot, ["method", "level"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for m in methods:
        lv = sorted({k[1] for k in g if k[0] == m})
        cs = [A.cell(g[(m, l)]) for l in lv]
        acc = np.array([c["acc"] for c in cs]) * 100
        lo = np.array([c["acc_lo"] for c in cs]) * 100
        hi = np.array([c["acc_hi"] for c in cs]) * 100
        st = STYLE[m]
        a1.errorbar(lv, acc, yerr=[acc - lo, hi - acc], capsize=2, lw=0.9, ms=3.5, elinewidth=0.5,
                    label=LABEL[m], **st)
        sil = np.array([c["silent"] for c in cs]) * 100
        a2.plot(lv, sil, lw=0.9, ms=3.5, label=LABEL[m], **st)
    a1.set_xlabel("In-plane rotation of query (deg)"); a1.set_ylabel("Accurate localizations (%)")
    a1.set_ylim(-3, 104); a1.grid(alpha=0.3, lw=0.4); a1.set_title("(a) Accuracy (error $\\leq$ 10 px), 95% Wilson CI")
    a2.set_xlabel("In-plane rotation of query (deg)"); a2.set_ylabel("Silent failures (%)")
    a2.set_ylim(-3, 104); a2.grid(alpha=0.3, lw=0.4)
    a2.set_title("(b) Wrong homography returned as a success")
    a1.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.7")
    plt.tight_layout(w_pad=1.2)
    plt.savefig(os.path.join(OUT, "fig_rotation_v2.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT, "fig_rotation_v2.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ stress-axes figure
AX_TITLE = {"scale": "Scale (query zoom)", "persp": "Perspective coefficient", "noise": "Sensor noise $\\sigma$ (grey levels)",
            "blur": "Blur $\\sigma$ (px)", "occlusion": "Occluded fraction", "appearance": "Appearance gap (gain/gamma)"}


def fig_axes(ax_rows):
    methods = [m for m in ["SIFT", "ORB", "SuperGlue"] if any(r["method"] == m for r in ax_rows)]
    g = A.group(ax_rows, ["method", "axis", "level"])
    axes_names = ["scale", "persp", "noise", "blur", "occlusion", "appearance"]
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.9))
    for ax, name in zip(axs.ravel(), axes_names):
        for m in methods:
            lv = sorted({k[2] for k in g if k[0] == m and k[1] == name})
            if not lv:
                continue
            cs = [A.cell(g[(m, name, l)]) for l in lv]
            acc = np.array([c["acc"] for c in cs]) * 100
            lo = np.array([c["acc_lo"] for c in cs]) * 100
            hi = np.array([c["acc_hi"] for c in cs]) * 100
            ax.errorbar(range(len(lv)), acc, yerr=[acc - lo, hi - acc], capsize=1.5, lw=0.9, ms=3.2,
                        elinewidth=0.45, label=LABEL[m], **STYLE[m])
        lv0 = sorted({k[2] for k in g if k[1] == name})
        ax.set_xticks(range(len(lv0))); ax.set_xticklabels([("%g" % l) for l in lv0])
        ax.set_title(AX_TITLE[name]); ax.set_ylim(-4, 104); ax.grid(alpha=0.3, lw=0.4)
    for ax in axs[:, 0]:
        ax.set_ylabel("Accurate (%)")
    axs[0, 0].legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.7")
    plt.tight_layout(h_pad=1.0, w_pad=0.8)
    plt.savefig(os.path.join(OUT, "fig_axes_v2.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT, "fig_axes_v2.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ confidence-gate figure
def gate_stats(rows, methods):
    summary = {}
    curves = {}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            continue
        ts = sorted({0} | {r["n_in"] for r in rs if r["H"]})
        rc = A.risk_coverage(rs, ts)
        curves[m] = [(c * 100, w * 100) for t, c, w in rc if not np.isnan(w)]
        pos = [r["n_in"] for r in rs if A.outcome(r) == "accurate"]
        neg = [r["n_in"] if r["H"] else 0 for r in rs if A.outcome(r) != "accurate"]
        d = {"n": len(rs), "n_wrong": len(neg), "base_wrong": len(neg) / len(rs),
             "auroc": A.auroc(pos, neg)}
        for tgt in (5.0, 1.0):
            okc = [c for (t, c, w) in rc if not np.isnan(w) and w * 100 <= tgt]
            d["cov@%d" % int(tgt)] = max(okc) if okc else 0.0
        summary[m] = d
    return summary, curves


def fig_gate(rot, axr):
    rot_common = [r for r in rot if r["level"] in (0, 10, 20, 30, 45, 60, 75, 90, 135, 180)]
    methods = ["SIFT", "ORB", "SuperGlue", "SuperGlueTTA"]
    s_rot, c_rot = gate_stats(rot_common, methods)
    s_ax, c_ax = gate_stats(axr, methods[:3])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, curves, ttl in ((a1, c_rot, "(a) Rotation sweep (0 to 180 deg)"), (a2, c_ax, "(b) Six stress axes")):
        for m, pts in curves.items():
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.1, label=LABEL[m],
                    color=STYLE[m]["color"], ls=STYLE[m]["ls"])
        ax.set_xlabel("Coverage: trials accepted (%)"); ax.set_title(ttl)
        ax.set_xlim(0, 101); ax.set_ylim(0, None); ax.grid(alpha=0.3, lw=0.4)
    a1.set_ylabel("Wrong among accepted (%)")
    a1.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="0.7")
    plt.tight_layout(w_pad=1.2)
    plt.savefig(os.path.join(OUT, "fig_gate_v2.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT, "fig_gate_v2.png"), dpi=300, bbox_inches="tight")
    plt.close()
    return {"rot": s_rot, "axes": s_ax}


if __name__ == "__main__":
    rot = A.load("rot_*.jsonl")
    axr = A.load("axes_*.jsonl")
    fig_pipeline()
    if rot:
        fig_rotation(rot)
    if axr:
        fig_axes(axr)
    s = fig_gate(rot, axr)
    json.dump(s, open(os.path.join(A.RES, 'gate_summary.json'), 'w'), indent=1, default=float)
    print(json.dumps(s, indent=1, default=float))
