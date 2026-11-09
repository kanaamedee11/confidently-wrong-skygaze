"""
data_utils.py
=============
Utilities for loading reference ("map") images and query ("live feed") images,
plus a synthetic data generator so the matching pipeline can be developed and
unit-tested BEFORE real Sentinel-2 / drone footage is plugged in.

Once you have real data (e.g. a Sentinel-2 GeoTIFF exported as PNG, or frames
extracted from drone video), load_reference() / load_query() are the two
functions to point at real files. Everything downstream (matcher.py) doesn't
care whether the image came from synthetic generation or a real sensor.
"""

import numpy as np
import cv2


def generate_synthetic_reference(size=800, seed=0):
    """
    Build a synthetic 'map' image with enough texture/corners for feature
    matching to be meaningful (flat, textureless regions are the #1 reason
    SIFT/ORB fail on real satellite imagery over water/desert/uniform crops).

    Draws fields (rectangles), a road network (lines), and buildings (small
    squares) on a noisy background, roughly mimicking the *structure* of an
    aerial/satellite scene without using any copyrighted imagery.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 60, dtype=np.uint8)

    # base terrain noise (so flat regions still have some texture)
    noise = rng.integers(-15, 15, (size, size, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # fields
    for _ in range(25):
        x, y = rng.integers(0, size, 2)
        w, h = rng.integers(40, 160, 2)
        color = tuple(int(c) for c in rng.integers(40, 200, 3))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)

    # road network
    for _ in range(12):
        x1, y1 = rng.integers(0, size, 2)
        x2, y2 = rng.integers(0, size, 2)
        cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), thickness=6)

    # buildings
    for _ in range(60):
        x, y = rng.integers(0, size - 20, 2)
        s = rng.integers(8, 22)
        color = tuple(int(c) for c in rng.integers(100, 255, 3))
        cv2.rectangle(img, (x, y), (x + s, y + s), color, -1)
        cv2.rectangle(img, (x, y), (x + s, y + s), (0, 0, 0), 1)

    return img


def generate_synthetic_query(reference, crop_frac=0.35, rotation_deg=15,
                              scale=0.9, brightness_shift=15, seed=1):
    """
    Simulate a drone/live-feed view of a SUB-REGION of the reference map:
    crop a patch, then apply rotation + scale + brightness change + blur to
    mimic a different viewpoint, altitude, and lighting condition.

    Returns (query_image, ground_truth_homography) where the homography maps
    query pixel coords -> reference pixel coords, so you can score the
    matcher's estimate against a known-correct answer.
    """
    rng = np.random.default_rng(seed)
    h, w = reference.shape[:2]
    crop_size = int(min(h, w) * crop_frac)

    x0 = rng.integers(0, w - crop_size)
    y0 = rng.integers(0, h - crop_size)
    patch = reference[y0:y0 + crop_size, x0:x0 + crop_size].copy()

    # rotate + scale the patch
    center = (crop_size // 2, crop_size // 2)
    rot_mat = cv2.getRotationMatrix2D(center, rotation_deg, scale)
    query = cv2.warpAffine(patch, rot_mat, (crop_size, crop_size),
                            borderMode=cv2.BORDER_REPLICATE)

    # lighting / sensor differences
    query = np.clip(query.astype(np.int16) + brightness_shift, 0, 255).astype(np.uint8)
    query = cv2.GaussianBlur(query, (3, 3), 0)

    # ground-truth homography: query -> reference
    # affine (crop offset + inverse rotation/scale) expressed as a 3x3 homography
    inv_rot = cv2.invertAffineTransform(rot_mat)
    affine_3x3 = np.vstack([inv_rot, [0, 0, 1]])
    translate = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64)
    gt_homography = translate @ affine_3x3

    return query, gt_homography


def load_reference(path):
    """Load a real reference (map/satellite) image from disk."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read reference image: {path}")
    return img


def load_query(path):
    """Load a real query (drone/live feed frame) image from disk."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read query image: {path}")
    return img
