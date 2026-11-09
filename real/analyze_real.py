import json, glob, sys, numpy as np
sys.path.insert(0, "src")
from analyze import wilson, auroc
rows = [json.loads(l) for f in glob.glob("results_real_*.jsonl") for l in open(f)]
TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
def oc(r): return "none" if not r["H"] else ("acc" if r["err_m"] <= TAU else "silent")
for m in ["SIFT", "ORB", "SuperGlue", "SuperGlueTTA"]:
    rs = [r for r in rows if r["method"] == m]
    if not rs: continue
    n = len(rs); a = sum(oc(r) == "acc" for r in rs); s = sum(oc(r) == "silent" for r in rs); d = n - a - s
    lo, hi = wilson(a, n)
    pos = [r["n_in"] for r in rs if oc(r) == "acc"]; neg = [r["n_in"] if r["H"] else 0 for r in rs if oc(r) != "acc"]
    au = auroc(pos, neg) if pos and neg else float("nan")
    print(f"{m:13s} n={n:3d} acc={a:3d} ({100*a/n:.0f}%, CI {100*lo:.0f}-{100*hi:.0f})  silent={s:3d} ({100*s/n:.0f}%)  detected={d:3d} ({100*d/n:.0f}%)  AUROC={au:.2f}  med_inl_acc={np.median(pos) if pos else None} med_inl_silent={np.median([r['n_in'] for r in rs if oc(r)=='silent']) if s else None}")
