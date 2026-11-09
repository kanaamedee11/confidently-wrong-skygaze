"""
analyze.py
==========
Aggregate the JSONL trial records into the tables and statistics used in the paper.

Outcome taxonomy (per trial), with accuracy threshold TAU on centre error (px):
  accurate : homography returned AND centre error <= TAU
  silent   : homography returned BUT centre error  >  TAU   (wrong answer, looks like a success)
  none     : no homography returned (the pipeline knows it failed)
"""
import json, glob, os, sys
import numpy as np

TAU = 10.0          # px on an 800 px reference (1.25 % of width)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def load(pattern):
    rows = []
    for fn in sorted(glob.glob(os.path.join(RES, pattern))):
        for line in open(fn):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def outcome(r, tau=TAU):
    if not r["H"]:
        return "none"
    return "accurate" if r["err_c"] <= tau else "silent"


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cell(rows):
    n = len(rows)
    oc = [outcome(r) for r in rows]
    acc = oc.count("accurate")
    sil = oc.count("silent")
    non = oc.count("none")
    errs = [r["err_c"] for r, o in zip(rows, oc) if o == "accurate"]
    lo, hi = wilson(acc, n)
    return {"n": n, "acc": acc / n, "acc_lo": lo, "acc_hi": hi, "silent": sil / n, "none": non / n,
            "med_err": float(np.median(errs)) if errs else np.nan,
            "n_silent": sil, "n_none": non, "n_acc": acc}


def group(rows, keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def auroc(scores_pos, scores_neg):
    """P(score_pos > score_neg) with ties counted 0.5 (Mann-Whitney U)."""
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return np.nan
    a = np.asarray(scores_pos, float)[:, None]
    b = np.asarray(scores_neg, float)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))


def risk_coverage(rows, thresholds):
    """Accept a trial if its RANSAC inlier count >= t. Returns coverage and the fraction of
    ACCEPTED trials that are wrong (silent failures)."""
    n = len(rows)
    out = []
    for t in thresholds:
        acc = [r for r in rows if r["H"] and r["n_in"] >= t]
        if not acc:
            out.append((t, 0.0, np.nan))
            continue
        wrong = sum(1 for r in acc if outcome(r) != "accurate")
        out.append((t, len(acc) / n, wrong / len(acc)))
    return out


if __name__ == "__main__":
    rot = load("rot_*.jsonl")
    print("rot trials:", len(rot))
    for (m, lv), rs in sorted(group([r for r in rot if True], ["method", "level"]).items()):
        c = cell(rs)
        print(f"{m:13s} {lv:>5} n={c['n']:3d} acc={c['acc']:.2f} silent={c['silent']:.2f} none={c['none']:.2f} med={c['med_err']:.2f}")
