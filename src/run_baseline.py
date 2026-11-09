"""
run_baseline.py
================
End-to-end smoke test: generate a synthetic reference/query pair with a KNOWN
ground-truth transform, run the localizer, and report the localization error.
This is the sanity check to run first, and the script to build the paper's
"baseline results" table from once real imagery is swapped in.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
from data_utils import generate_synthetic_reference, generate_synthetic_query
from matcher import VisualLocalizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

try:
    from deep_matcher import DeepLocalizer
    DEEP_AVAILABLE = True
except Exception as e:
    DEEP_AVAILABLE = False
    _DEEP_IMPORT_ERROR = e


def ground_truth_center(query_shape, gt_homography):
    h, w = query_shape[:2]
    center = np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2)
    gt_center = cv2.perspectiveTransform(center, gt_homography).reshape(2)
    return gt_center


def main():
    print("Generating synthetic reference + query...")
    reference = generate_synthetic_reference(size=800, seed=0)
    query, gt_H = generate_synthetic_query(
        reference, crop_frac=0.35, rotation_deg=15, scale=0.9, seed=1
    )

    rows = []

    for detector in ["sift", "orb"]:
        print(f"\n--- Detector: {detector.upper()} ---")
        localizer = VisualLocalizer(detector=detector)
        result = localizer.localize(reference, query)
        row = {"method": detector.upper()}

        print(f"  keypoints (ref/qry): {result['num_keypoints_ref']} / {result['num_keypoints_qry']}")
        print(f"  good matches:        {result['num_good_matches']}")
        print(f"  RANSAC inliers:      {result['num_inliers']} (ratio {result['inlier_ratio']:.2f})")
        print(f"  runtime:             {result['runtime_ms']:.1f} ms")

        if result["success"]:
            gt_center = ground_truth_center(query.shape, gt_H)
            est_center = np.array(result["center_ref"])
            error_px = np.linalg.norm(gt_center - est_center)
            print(f"  estimated center:    ({est_center[0]:.1f}, {est_center[1]:.1f})")
            print(f"  ground-truth center: ({gt_center[0]:.1f}, {gt_center[1]:.1f})")
            print(f"  LOCALIZATION ERROR:  {error_px:.2f} px")

            vis = localizer.draw_result(reference, query, result)
            out_path = os.path.join(RESULTS_DIR, f"baseline_{detector}.png")
            cv2.imwrite(out_path, vis)
            print(f"  saved visualization: {out_path}")
            row.update({"success": True, "error_px": error_px, **result})
        else:
            print("  LOCALIZATION FAILED")
            row.update({"success": False, "error_px": None, **result})
        rows.append(row)

    if DEEP_AVAILABLE:
        print(f"\n--- Deep matcher: SuperPoint+SuperGlue (outdoor) ---")
        try:
            deep_localizer = DeepLocalizer(weights="outdoor")
            result = deep_localizer.localize(reference, query)
            row = {"method": "SuperGlue"}
            print(f"  keypoints (ref/qry): {result['num_keypoints_ref']} / {result['num_keypoints_qry']}")
            print(f"  good matches:        {result['num_good_matches']}")
            print(f"  RANSAC inliers:      {result['num_inliers']} (ratio {result['inlier_ratio']:.2f})")
            print(f"  runtime:             {result['runtime_ms']:.1f} ms")
            if result["success"]:
                gt_center = ground_truth_center(query.shape, gt_H)
                est_center = np.array(result["center_ref"])
                error_px = np.linalg.norm(gt_center - est_center)
                print(f"  LOCALIZATION ERROR:  {error_px:.2f} px")
                vis = deep_localizer_vis = reference.copy()
                quad = result["quad_ref"].astype(int)
                cv2.polylines(vis, [quad], True, (0, 255, 0), 3)
                cx, cy = map(int, result["center_ref"])
                cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 25, 3)
                cv2.imwrite(os.path.join(RESULTS_DIR, "baseline_superglue.png"), vis)
                row.update({"success": True, "error_px": error_px, **result})
            else:
                row.update({"success": False, "error_px": None, **result})
            rows.append(row)
        except Exception as e:
            print(f"  Skipped: {e}")
    else:
        print(f"\n--- Deep matcher: SKIPPED (SuperGlue not set up: {_DEEP_IMPORT_ERROR}) ---")
        print("  See README for setup instructions.")

    cv2.imwrite(os.path.join(RESULTS_DIR, "reference.png"), reference)
    cv2.imwrite(os.path.join(RESULTS_DIR, "query.png"), query)

    # Write a markdown results table for the paper
    table_path = os.path.join(RESULTS_DIR, "baseline_results.md")
    with open(table_path, "w") as f:
        f.write("# Baseline results (synthetic validation set)\n\n")
        f.write("| Method | Good matches | Inliers | Inlier ratio | Error (px) | Runtime (ms) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            err = f"{r['error_px']:.2f}" if r.get("error_px") is not None else "FAILED"
            f.write(f"| {r['method']} | {r['num_good_matches']} | {r['num_inliers']} | "
                     f"{r['inlier_ratio']:.2f} | {err} | {r['runtime_ms']:.1f} |\n")
    print(f"\nResults table saved: {table_path}")


if __name__ == "__main__":
    main()
