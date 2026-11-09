"""
run_exp.py
==========
SkyGaze v2 experiment runner. Writes one JSON record per trial (JSONL) so partial
results survive interruption.

  python3 run_exp.py --exp rot   --methods SIFT,ORB          --nref 3 --nseed 10 --out r.jsonl
  python3 run_exp.py --exp axes  --methods SuperGlue         --nref 2 --nseed 4  --out a.jsonl

Methods: SIFT, ORB, SuperGlue, SuperGlueTTA (4 test-time query rotations 0/90/180/270,
keep the hypothesis with the most RANSAC inliers).
"""

import os, sys, time, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import cv2
from data_utils import generate_synthetic_reference
from matcher import VisualLocalizer
from gen2 import make_query, gt_center_and_quad

ROT_LEVELS = [0, 10, 20, 30, 45, 60, 75, 90, 135, 180]
AXES = {
    "scale":      [0.5, 0.7, 0.9, 1.3, 1.8],
    "persp":      [0.0, 0.0002, 0.0004, 0.0008, 0.0012],
    "noise":      [0, 10, 20, 35],
    "blur":       [0.8, 2.0, 3.5, 5.0],
    "occlusion":  [0.0, 0.2, 0.4, 0.6],
    "appearance": [0.0, 0.2, 0.4, 0.6],
}
BASE = dict(rotation_deg=15.0, scale=0.9, persp=0.0, brightness=15,
            noise_sigma=0.0, blur_sigma=0.8, occl_frac=0.0, appearance=0.0)
AXIS_KEY = {"scale": "scale", "persp": "persp", "noise": "noise_sigma",
            "blur": "blur_sigma", "occlusion": "occl_frac", "appearance": "appearance"}


class SuperGlueTTA:
    """SuperGlue with test-time rotation hypotheses. Cost = 4x SuperGlue."""
    def __init__(self, deep):
        self.deep = deep

    def localize(self, reference, query):
        t0 = time.time()
        h, w = query.shape[:2]
        # (rotated image, 3x3 matrix mapping ORIGINAL query px -> ROTATED query px)
        variants = [
            (query, np.eye(3)),
            (cv2.rotate(query, cv2.ROTATE_90_CLOCKWISE),
             np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], float)),
            (cv2.rotate(query, cv2.ROTATE_180),
             np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], float)),
            (cv2.rotate(query, cv2.ROTATE_90_COUNTERCLOCKWISE),
             np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], float)),
        ]
        best, best_M = None, None
        for img, M in variants:
            r = self.deep.localize(reference, img)
            if not r["success"]:
                continue
            if best is None or r["num_inliers"] > best["num_inliers"]:
                best, best_M = r, M
        if best is None:
            return {"success": False, "num_inliers": 0, "inlier_ratio": 0.0,
                    "num_good_matches": 0, "runtime_ms": (time.time() - t0) * 1000}
        Htot = best["homography"] @ best_M
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(corners, Htot).reshape(-1, 2)
        best = dict(best)
        best.update({"homography": Htot, "quad_ref": quad, "center_ref": tuple(quad.mean(0)),
                     "runtime_ms": (time.time() - t0) * 1000})
        return best


def texture_mosaic():
    """800x800 mosaic of four photographic ground textures from the scikit-image sample data
    (gravel, grass, brick, and a rock/soil-like retina crop would be off-domain, so the 4th is
    gravel rotated 90 deg). Real photographic texture, still a single-source synthetic query."""
    import skimage.data as skd
    tiles = [skd.gravel(), skd.grass(), skd.brick(), np.rot90(skd.gravel())]
    tiles = [cv2.resize(np.ascontiguousarray(t), (400, 400), interpolation=cv2.INTER_AREA) for t in tiles]
    top = np.hstack(tiles[:2]); bot = np.hstack(tiles[2:])
    m = np.vstack([top, bot])
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) if m.ndim == 2 else m


def build_methods(names):
    out = {}
    deep = None
    for n in names:
        if n == "SIFT":
            out[n] = VisualLocalizer(detector="sift").localize
        elif n == "ORB":
            out[n] = VisualLocalizer(detector="orb").localize
        elif n in ("SuperGlue", "SuperGlueTTA"):
            if deep is None:
                os.environ.setdefault("SUPERGLUE_PATH", os.path.abspath(os.path.join(HERE, "..", "..", "SuperGluePretrainedNetwork")))
                from deep_matcher import DeepLocalizer
                import torch
                torch.set_num_threads(int(os.environ.get("SG_THREADS", "1")))
                deep = DeepLocalizer(weights="outdoor")
            out[n] = deep.localize if n == "SuperGlue" else SuperGlueTTA(deep).localize
        else:
            raise ValueError(n)
    return out


def run_one(fn, reference, cfg, seed):
    q, gtH = make_query(reference, seed=seed, **cfg)
    gc, gq = gt_center_and_quad(q.shape, gtH)
    t0 = time.time()
    r = fn(reference, q)
    rt = (time.time() - t0) * 1000
    rec = {"H": bool(r["success"]), "err_c": None, "err_q": None,
           "n_in": int(r.get("num_inliers", 0)), "ratio": float(r.get("inlier_ratio", 0.0)),
           "n_m": int(r.get("num_good_matches", 0)), "rt": rt}
    if r["success"]:
        # Estimated location = the query CENTRE pushed through the estimated homography.
        # (The pipeline's own `center_ref` is the mean of the four projected corners, which
        #  equals the projected centre only for similarity transforms, not under perspective.)
        h_, w_ = q.shape[:2]
        c_in = np.float32([[w_ / 2, h_ / 2]]).reshape(-1, 1, 2)
        est_c = cv2.perspectiveTransform(c_in, np.asarray(r["homography"], np.float64)).reshape(2)
        rec["err_c"] = float(np.linalg.norm(est_c - gc))
        rec["err_q"] = float(np.mean(np.linalg.norm(np.array(r["quad_ref"]) - gq, axis=1)))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["rot", "axes"])
    ap.add_argument("--methods", required=True)
    ap.add_argument("--nref", type=int, default=3)
    ap.add_argument("--nseed", type=int, default=8)
    ap.add_argument("--rot_levels", default="")
    ap.add_argument("--axes", default="")
    ap.add_argument("--ref", default="synthetic", choices=["synthetic", "retina", "texture"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    methods = build_methods(a.methods.split(","))
    if a.ref == "retina":
        import skimage.data as skd
        rimg = cv2.resize(skd.retina(), (800, 800), interpolation=cv2.INTER_AREA)
        refs = {10: cv2.cvtColor(rimg, cv2.COLOR_RGB2BGR)}
    elif a.ref == "texture":
        refs = {20: texture_mosaic()}
    else:
        refs = {s: generate_synthetic_reference(size=800, seed=s) for s in range(a.nref)}

    jobs = []
    if a.exp == "rot":
        levels = [int(x) for x in a.rot_levels.split(",")] if a.rot_levels else ROT_LEVELS
        for lv in levels:
            jobs.append(("rotation", lv, dict(BASE, rotation_deg=float(lv), scale=0.9)))
    else:
        names = a.axes.split(",") if a.axes else list(AXES)
        for ax in names:
            for lv in AXES[ax]:
                jobs.append((ax, lv, dict(BASE, **{AXIS_KEY[ax]: lv})))

    done = 0
    total = len(jobs) * len(methods) * a.nref * a.nseed
    with open(a.out, "a") as f:
        for axis, lv, cfg in jobs:
            for mname, fn in methods.items():
                for rs, ref in refs.items():
                    for sd in range(a.nseed):
                        rec = run_one(fn, ref, cfg, seed=1000 * rs + sd)
                        rec.update({"method": mname, "axis": axis, "level": lv,
                                    "ref_seed": rs, "seed": sd})
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        done += 1
                if done % 20 == 0:
                    print(f"[{a.methods}] {done}/{total}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
