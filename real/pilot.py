import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import *
from matcher import VisualLocalizer
rows = load_rows()
calib = rows[::5]
sift = VisualLocalizer(detector="sift")
rng = np.random.default_rng(0)
print("m/px iso:", m_per_px(True))
for iso in (True, False):
    S = sat(iso)
    for Wq in (200, 280, 380):
        res = []
        for r in calib:
            q = cv2.imread(os.path.join(D, "drone", r["filename"]))
            h = int(round(q.shape[0] * Wq / q.shape[1])); q = cv2.resize(q, (Wq, h), interpolation=cv2.INTER_AREA)
            g = gt_px(float(r["lat"]), float(r["lon"]), iso)
            ox, oy = rng.integers(-250, 250, 2)
            cx, cy = int(g[0] + ox), int(g[1] + oy)
            x0 = int(np.clip(cx - 450, 0, S.shape[1] - 900)); y0 = int(np.clip(cy - 450, 0, S.shape[0] - 900))
            tile = S[y0:y0 + 900, x0:x0 + 900]
            out = sift.localize(tile, q)
            if out["success"]:
                H = np.asarray(out["homography"])
                c = cv2.perspectiveTransform(np.float32([[[q.shape[1] / 2, q.shape[0] / 2]]]), H).reshape(2) + [x0, y0]
                err = np.linalg.norm(c - g) * m_per_px(True) if iso else float("nan")
                sv = np.linalg.svd(H[:2, :2], compute_uv=False)
                res.append((out["num_inliers"], err, sv[0], sv[1]))
            else:
                res.append((0, None, None, None))
        ok = [x for x in res if x[0] >= 15]
        print(f"iso={iso} Wq={Wq}: n_in={[x[0] for x in res]}  ok15={len(ok)}  err_m={[round(x[1]) if x[1] is not None and x[1]==x[1] else None for x in res]}")
        if ok: print("   H singular values (med):", np.median([x[2] for x in ok]), np.median([x[3] for x in ok]))
