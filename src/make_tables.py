"""
make_tables.py : emit LaTeX table bodies straight from the trial records (no manual transcription).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analyze as A

OUT = os.path.join(A.RES, "..", "paper")
os.makedirs(OUT, exist_ok=True)
NAME = {"SIFT": "SIFT", "ORB": "ORB", "SuperGlue": "SuperGlue", "SuperGlueTTA": "SuperGlue + rot.\\ hyp."}


def pct(x):
    return "%d" % round(100 * x)


def table_rot(rot, levels, fname, key="acc"):
    g = A.group(rot, ["method", "level"])
    lines = []
    for m in ["SIFT", "ORB", "SuperGlue", "SuperGlueTTA"]:
        cells = []
        for lv in levels:
            rs = g.get((m, lv))
            if not rs:
                cells.append("n/a")
            else:
                c = A.cell(rs)
                cells.append(pct(c[key]))
        lines.append(NAME[m] + " & " + " & ".join(cells) + " \\\\")
    open(os.path.join(OUT, fname), "w").write("\n".join(lines) + "\n")


def table_axes(axr, fname):
    g = A.group(axr, ["axis", "method", "level"])
    from run_exp import AXES
    label = {"scale": "Scale (zoom)", "persp": "Perspective", "noise": "Sensor noise", "blur": "Blur",
             "occlusion": "Occlusion", "appearance": "Appearance gap"}
    lines = []
    for ax in ["scale", "persp", "noise", "blur", "occlusion", "appearance"]:
        lv = AXES[ax][-1]
        cells = []
        for m in ["SIFT", "ORB", "SuperGlue"]:
            rs = g.get((ax, m, lv))
            if not rs:
                cells.append("n/a")
            else:
                c = A.cell(rs)
                cells.append("%s / %s" % (pct(c["acc"]), pct(c["silent"])))
        lines.append("%s & %g & %s \\\\" % (label[ax], lv, " & ".join(cells)))
    open(os.path.join(OUT, fname), "w").write("\n".join(lines) + "\n")


def table_gate(summary, fname):
    lines = []
    for m in ["SIFT", "ORB", "SuperGlue", "SuperGlueTTA"]:
        if m not in summary:
            continue
        s = summary[m]
        lines.append("%s & %d & %s & %.2f & %s & %s \\\\" % (
            NAME[m], s["n"], pct(s["base_wrong"]), s["auroc"], pct(s["cov@5"]), pct(s["cov@1"])))
    open(os.path.join(OUT, fname), "w").write("\n".join(lines) + "\n")


def table_gate2(fname):
    gs = json.load(open(os.path.join(A.RES, "gate_summary.json")))
    def fmt(d):
        if d is None:
            return "n/a & n/a & n/a"
        au = "n/a" if d["auroc"] is None or d["auroc"] != d["auroc"] else "%.3f" % d["auroc"]
        return "%d/%d & %s & %s" % (d["n_wrong"], d["n"], au, pct(d["cov@1"]))
    lines = []
    for m in ["SIFT", "ORB", "SuperGlue", "SuperGlueTTA"]:
        lines.append("%s & %s & %s \\\\" % (NAME[m], fmt(gs["rot"].get(m)), fmt(gs["axes"].get(m))))
    open(os.path.join(OUT, fname), "w").write("\n".join(lines) + "\n")


def table_runtime(fname):
    p = os.path.join(A.RES, "runtime.json")
    if not os.path.exists(p):
        return
    rt = json.load(open(p))
    lines = []
    for m in ["SIFT", "ORB", "SuperGlue", "SuperGlueTTA"]:
        if m in rt:
            r = rt[m]
            lines.append("%s & %.0f & %.0f to %.0f \\\\" % (NAME[m], r["median_ms"], r["q1"], r["q3"]))
    open(os.path.join(OUT, fname), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    rot = A.load("rot_*.jsonl")
    axr = A.load("axes_*.jsonl")
    table_rot(rot, [0, 30, 45, 50, 55, 60, 75, 90, 135, 180], "tab_rot_acc.tex", "acc")
    table_rot(rot, [0, 30, 45, 50, 55, 60, 75, 90, 135, 180], "tab_rot_silent.tex", "silent")
    table_axes(axr, "tab_axes.tex")
    table_gate2("tab_gate.tex")
    table_runtime("tab_runtime.tex")
    print("tables written")
