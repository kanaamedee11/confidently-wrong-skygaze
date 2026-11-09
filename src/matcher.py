"""
matcher.py
==========
Baseline visual localization: given a reference image (map/satellite tile)
and a query image (drone frame / live feed crop), find WHERE the query sits
inside the reference, and return an estimated center location + confidence.

This is the classical-CV baseline (SIFT/ORB + RANSAC homography). It is the
right FIRST thing to build and report in the paper: it is fast, interpretable,
and gives you a concrete number to beat once you swap in a deep learned
matcher (SuperGlue / LoFTR) later. Always keep this as the reported baseline.
"""

import time
import numpy as np
import cv2


class VisualLocalizer:
    def __init__(self, detector="sift", ratio_thresh=0.75, ransac_thresh=5.0):
        """
        detector: "sift" or "orb". SIFT is more accurate but slower and
                  patent-free since 2020 (available in opencv-contrib-python).
                  ORB is faster and license-free, better for embedded/onboard
                  use once you move toward real hardware.
        """
        self.detector_name = detector
        self.ratio_thresh = ratio_thresh
        self.ransac_thresh = ransac_thresh

        if detector == "sift":
            self.detector = cv2.SIFT_create()
            self.norm = cv2.NORM_L2
        elif detector == "orb":
            self.detector = cv2.ORB_create(nfeatures=4000)
            self.norm = cv2.NORM_HAMMING
        else:
            raise ValueError("detector must be 'sift' or 'orb'")

        self.matcher = cv2.BFMatcher(self.norm)

    def _to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    def localize(self, reference, query):
        """
        Returns a dict with:
          success            bool
          homography         3x3 array mapping query px -> reference px (or None)
          center_ref         (x, y) estimated query-center location in reference coords
          quad_ref           4x2 array: query image corners projected into reference
          num_keypoints_ref  int
          num_keypoints_qry  int
          num_good_matches   int
          num_inliers        int
          inlier_ratio       float  (quality/confidence signal)
          runtime_ms         float
        """
        t0 = time.time()
        ref_gray = self._to_gray(reference)
        qry_gray = self._to_gray(query)

        kp_ref, des_ref = self.detector.detectAndCompute(ref_gray, None)
        kp_qry, des_qry = self.detector.detectAndCompute(qry_gray, None)

        result = {
            "success": False,
            "homography": None,
            "center_ref": None,
            "quad_ref": None,
            "num_keypoints_ref": len(kp_ref) if kp_ref else 0,
            "num_keypoints_qry": len(kp_qry) if kp_qry else 0,
            "num_good_matches": 0,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "runtime_ms": None,
        }

        if des_ref is None or des_qry is None or len(kp_qry) < 4 or len(kp_ref) < 4:
            result["runtime_ms"] = (time.time() - t0) * 1000
            return result

        # Lowe's ratio test
        knn = self.matcher.knnMatch(des_qry, des_ref, k=2)
        good = [m for m, n in knn if m.distance < self.ratio_thresh * n.distance]
        result["num_good_matches"] = len(good)

        if len(good) < 4:
            result["runtime_ms"] = (time.time() - t0) * 1000
            return result

        src_pts = np.float32([kp_qry[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)

        if H is None:
            result["runtime_ms"] = (time.time() - t0) * 1000
            return result

        inliers = int(mask.sum())
        result["num_inliers"] = inliers
        result["inlier_ratio"] = inliers / len(good)

        h, w = qry_gray.shape[:2]
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

    def draw_result(self, reference, query, result):
        """Visualize: reference image with the query's projected footprint drawn on it."""
        vis = reference.copy()
        if result["success"]:
            quad = result["quad_ref"].astype(int)
            cv2.polylines(vis, [quad], isClosed=True, color=(0, 255, 0), thickness=3)
            cx, cy = map(int, result["center_ref"])
            cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 25, 3)
        return vis
