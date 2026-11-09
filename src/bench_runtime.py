"""
bench_runtime.py : quiet, single-process wall-clock benchmark (run when nothing else is using the CPU).
800x800 reference, 280x280 query, 15 deg rotation. Reports median and IQR over N trials per method.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SUPERGLUE_PATH", os.path.abspath(os.path.join(HERE, "..", "..", "SuperGluePretrainedNetwork")))
os.environ.setdefault("SG_THREADS", "2")
import numpy as np
import run_exp as R
from data_utils import generate_synthetic_reference
from gen2 import make_query

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
ref = generate_synthetic_reference(size=800, seed=0)
ms = R.build_methods(["SIFT", "ORB", "SuperGlue", "SuperGlueTTA"])
out = {}
for name, fn in ms.items():
    n = N if name != "SuperGlueTTA" else max(5, N // 3)
    fn(ref, make_query(ref, seed=99, **R.BASE)[0])            # warm-up
    ts = []
    for s in range(n):
        q, _ = make_query(ref, seed=s, **R.BASE)
        t0 = time.perf_counter(); fn(ref, q); ts.append((time.perf_counter() - t0) * 1000)
    out[name] = {"median_ms": float(np.median(ts)), "q1": float(np.percentile(ts, 25)),
                 "q3": float(np.percentile(ts, 75)), "n": n}
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(HERE, "..", "results", "runtime.json"), "w"), indent=1)
