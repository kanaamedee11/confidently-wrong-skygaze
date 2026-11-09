"""
gen2.py
=======
Extended synthetic query generator for the SkyGaze v2 stress tests.

Builds on data_utils.generate_synthetic_query but exposes additional, independently
controllable degradation axes so each one can be swept on its own:

  rotation_deg   in-plane rotation (viewpoint heading difference)
  scale          zoom factor of the query relative to the reference (altitude difference)
  persp          perspective tilt term (oblique camera vs. nadir map), 0 = none
  brightness     additive brightness shift
  noise_sigma    Gaussian sensor noise (std, in grey levels)
  blur_sigma     Gaussian blur (std, px)
  occl_frac      fraction of the query covered by flat opaque rectangles (cloud / shadow / mask)
  appearance     per-channel gain and gamma change of strength a (season / illumination gap)

Ground truth is exact by construction: the homography query -> reference is returned.
With persp = 0, appearance = 0 and the original defaults this reproduces the geometry of
data_utils.generate_synthetic_query (same crop offsets for the same seed).
"""

import numpy as np
import cv2


def make_query(reference, crop_frac=0.35, rotation_deg=15.0, scale=0.9, persp=0.0,
               brightness=15, noise_sigma=0.0, blur_sigma=0.6, occl_frac=0.0,
               appearance=0.0, seed=1):
    rng = np.random.default_rng(seed)
    h, w = reference.shape[:2]
    cs = int(min(h, w) * crop_frac)
    x0 = int(rng.integers(0, w - cs))
    y0 = int(rng.integers(0, h - cs))
    patch = reference[y0:y0 + cs, x0:x0 + cs].copy()

    c = (cs // 2, cs // 2)
    rot = cv2.getRotationMatrix2D(c, rotation_deg, scale)
    R3 = np.vstack([rot, [0.0, 0.0, 1.0]])
    Tc = np.array([[1, 0, c[0]], [0, 1, c[1]], [0, 0, 1]], dtype=np.float64)
    Tci = np.linalg.inv(Tc)
    P = np.array([[1, 0, 0], [0, 1, 0], [persp, persp * 0.5, 1]], dtype=np.float64)
    aug = Tc @ P @ Tci @ R3                      # patch px -> query px

    q = cv2.warpPerspective(patch, aug, (cs, cs), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    q = q.astype(np.float32)

    if appearance > 0:
        gains = 1.0 + rng.uniform(-appearance, appearance, 3)
        gamma = float(np.exp(rng.uniform(-appearance, appearance)))
        q = 255.0 * np.power(np.clip(q / 255.0, 0, 1), gamma) * gains[None, None, :]

    q = q + brightness
    if blur_sigma > 0:
        k = int(2 * round(3 * blur_sigma) + 1)
        q = cv2.GaussianBlur(q, (k, k), blur_sigma)
    if noise_sigma > 0:
        q = q + rng.normal(0, noise_sigma, q.shape)
    q = np.clip(q, 0, 255).astype(np.uint8)

    if occl_frac > 0:
        target = occl_frac * cs * cs
        covered = np.zeros((cs, cs), bool)
        while covered.sum() < target:
            rw, rh = rng.integers(cs // 8, cs // 3, 2)
            rx = int(rng.integers(0, cs - rw))
            ry = int(rng.integers(0, cs - rh))
            val = int(rng.integers(150, 255))
            q[ry:ry + rh, rx:rx + rw] = val
            covered[ry:ry + rh, rx:rx + rw] = True

    gt_H = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64) @ np.linalg.inv(aug)
    return q, gt_H


def gt_center_and_quad(query_shape, gt_H):
    h, w = query_shape[:2]
    center = np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2)
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    c = cv2.perspectiveTransform(center, gt_H).reshape(2)
    q = cv2.perspectiveTransform(corners, gt_H).reshape(4, 2)
    return c, q
