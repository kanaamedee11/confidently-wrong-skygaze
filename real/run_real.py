"""Real-data experiment: UAV-VisLoc flight 03 drone frames vs the real satellite map.
Per trial: 900x900 reference tile cut from the satellite map around the true position with a random offset
(the 'last known position' prior), drone frame resized to a fixed width, one pass per matcher.
Calibration frames (every 5th) were used only to choose the drone resize width and are excluded here."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SG_THREADS", "2")
import numpy as np, cv2
from common import *
import run_exp as R

WQ = 280; TILE = 900; NOFF = 3; OFFR = 250
out = sys.argv[1]; methods = sys.argv[2].split(",")
rows = load_rows(); test = [r for i, r in enumerate(rows) if i % 5 != 0]
S = sat(True); mpp = m_per_px(True)
fns = R.build_methods(methods)
done = set()
if os.path.exists(out):
    for l in open(out):
        d = json.loads(l); done.add((d["method"], d["frame"], d["off"]))
f = open(out, "a")
for r in test:
    q0 = cv2.imread(os.path.join(D, "drone", r["filename"]))
    q = cv2.resize(q0, (WQ, int(round(q0.shape[0] * WQ / q0.shape[1]))), interpolation=cv2.INTER_AREA)
    g = gt_px(float(r["lat"]), float(r["lon"]), True)
    for k in range(NOFF):
        rng = np.random.default_rng(1000 * int(r["num"]) + k)
        ox, oy = rng.integers(-OFFR, OFFR + 1, 2)
        x0 = int(np.clip(g[0] + ox - TILE // 2, 0, S.shape[1] - TILE)); y0 = int(np.clip(g[1] + oy - TILE // 2, 0, S.shape[0] - TILE))
        tile = S[y0:y0 + TILE, x0:x0 + TILE]
        for m, fn in fns.items():
            if (m, r["filename"], k) in done: continue
            t0 = time.time(); o = fn(tile, q); rt = (time.time() - t0) * 1000
            rec = {"method": m, "frame": r["filename"], "off": k, "H": bool(o["success"]), "n_in": int(o.get("num_inliers", 0)),
                   "rt": rt, "err_m": None, "rot": None, "phi": float(r["Phi1"])}
            if o["success"]:
                H = np.asarray(o["homography"], np.float64)
                c = cv2.perspectiveTransform(np.float32([[[q.shape[1] / 2, q.shape[0] / 2]]]), H).reshape(2) + [x0, y0]
                rec["err_m"] = float(np.linalg.norm(c - g) * mpp)
                rec["rot"] = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
            f.write(json.dumps(rec) + "\n"); f.flush()
print("done")
