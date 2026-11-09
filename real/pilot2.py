import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import *
from matcher import VisualLocalizer
rows = load_rows(); calib = rows[::5]
S = sat(True); mpp = m_per_px(True)
rng = np.random.default_rng(0)
tiles = []
for r in calib:
    g = gt_px(float(r["lat"]), float(r["lon"]), True)
    ox, oy = rng.integers(-250, 250, 2)
    x0 = int(np.clip(g[0] + ox - 450, 0, S.shape[1] - 900)); y0 = int(np.clip(g[1] + oy - 450, 0, S.shape[0] - 900))
    tiles.append((g, x0, y0))
for name in ("sift",):
    for clahe in (False, True):
        loc = VisualLocalizer(detector=name)
        for Wq in (100, 140, 180, 240, 320, 450, 600):
            line = []
            for r, (g, x0, y0) in zip(calib, tiles):
                q = cv2.imread(os.path.join(D, "drone", r["filename"]))
                q = cv2.resize(q, (Wq, int(round(q.shape[0] * Wq / q.shape[1]))), interpolation=cv2.INTER_AREA)
                tile = S[y0:y0 + 900, x0:x0 + 900]
                if clahe:
                    cl = cv2.createCLAHE(3.0, (8, 8))
                    f = lambda im: cv2.cvtColor(cl.apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)
                    tile, q = f(tile), f(q)
                o = loc.localize(tile, q)
                if o["success"]:
                    c = cv2.perspectiveTransform(np.float32([[[q.shape[1]/2, q.shape[0]/2]]]), np.asarray(o["homography"])).reshape(2) + [x0, y0]
                    line.append((o["num_inliers"], round(float(np.linalg.norm(c - g) * mpp))))
                else: line.append((0, None))
            good = sum(1 for n, e in line if e is not None and e <= 25)
            print(f"{name} clahe={clahe} Wq={Wq}: correct<=25m: {good}/8  {line}")
