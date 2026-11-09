"""
deep_matcher.py
================
Deep-learning localizer using SuperPoint + SuperGlue, matching the exact same
interface as VisualLocalizer in matcher.py so results are directly comparable
in the paper (classical baseline vs. learned matcher, same eval code).

IMPORTANT — SETUP REQUIRED (not bundled in this repo):
SuperGlue's code/weights are released under an "academic/non-commercial
research use only" license by Magic Leap, with the source files marked
confidential/no-redistribution. That means we do NOT vendor their code here —
you clone it yourself (this is exactly what the license permits: your own
personal non-commercial research use):

    git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git

Then either:
  - place it as a sibling folder next to this project, e.g.
      parent/
        skygaze/
        SuperGluePretrainedNetwork/
  - or set the environment variable SUPERGLUE_PATH to wherever you cloned it.

Keep this to academic/research use per their license — do not ship or sell
anything that bundles their code/weights.
"""

import os
import sys
import time
import numpy as np
import cv2
import torch


def _locate_superglue_repo():
    env_path = os.environ.get("SUPERGLUE_PATH")
    candidates = []
    if env_path:
        candidates.append(env_path)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "..", "..", "SuperGluePretrainedNetwork"))
    candidates.append(os.path.join(here, "..", "SuperGluePretrainedNetwork"))

    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(os.path.join(c, "models")):
            return c
    raise FileNotFoundError(
        "Could not find SuperGluePretrainedNetwork. Clone it "
        "(git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git) "
        "and either place it next to this project or set SUPERGLUE_PATH."
    )


class DeepLocalizer:
    def __init__(self, weights="outdoor", max_keypoints=1024,
                 keypoint_threshold=0.005, match_threshold=0.2, device=None):
        """
        weights: 'outdoor' (MegaDepth-trained — the right choice for aerial/
                  satellite/drone imagery) or 'indoor' (ScanNet-trained).
        """
        repo_path = _locate_superglue_repo()
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from models.matching import Matching  # noqa: import from cloned repo

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        config = {
            "superpoint": {
                "nms_radius": 4,
                "keypoint_threshold": keypoint_threshold,
                "max_keypoints": max_keypoints,
            },
            "superglue": {
                "weights": weights,
                "sinkhorn_iterations": 20,
                "match_threshold": match_threshold,
            },
        }
        self.matching = Matching(config).eval().to(self.device)

    def _to_tensor(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        tensor = torch.from_numpy(gray / 255.0).float()[None, None].to(self.device)
        return tensor

    def localize(self, reference, query, ransac_thresh=5.0):
        t0 = time.time()
        ref_t = self._to_tensor(reference)
        qry_t = self._to_tensor(query)

        with torch.no_grad():
            pred = self.matching({"image0": qry_t, "image1": ref_t})

        kpts_qry = pred["keypoints0"][0].cpu().numpy()
        kpts_ref = pred["keypoints1"][0].cpu().numpy()
        matches = pred["matches0"][0].cpu().numpy()
        confidence = pred["matching_scores0"][0].cpu().numpy()

        valid = matches > -1
        mkpts_qry = kpts_qry[valid]
        mkpts_ref = kpts_ref[matches[valid]]
        mconf = confidence[valid]

        result = {
            "success": False,
            "homography": None,
            "center_ref": None,
            "quad_ref": None,
            "num_keypoints_ref": len(kpts_ref),
            "num_keypoints_qry": len(kpts_qry),
            "num_good_matches": int(valid.sum()),
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "mean_match_confidence": float(mconf.mean()) if len(mconf) else 0.0,
            "runtime_ms": None,
        }

        if valid.sum() < 4:
            result["runtime_ms"] = (time.time() - t0) * 1000
            return result

        H, mask = cv2.findHomography(
            mkpts_qry.reshape(-1, 1, 2), mkpts_ref.reshape(-1, 1, 2),
            cv2.RANSAC, ransac_thresh
        )
        if H is None:
            result["runtime_ms"] = (time.time() - t0) * 1000
            return result

        inliers = int(mask.sum())
        result["num_inliers"] = inliers
        result["inlier_ratio"] = inliers / int(valid.sum())

        h, w = (query.shape[0], query.shape[1])
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        quad_ref = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        center_ref = quad_ref.mean(axis=0)

        result.update({
            "success": True,
            "homography": H,
            "center_ref": tuple(center_ref),
            "quad_ref": quad_ref,
        })
        result["runtime_ms"] = (time.time() - t0) * 1000
        return result
