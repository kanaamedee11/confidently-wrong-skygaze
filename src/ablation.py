"""
ablation.py
============
Systematic controlled ablation: sweep viewpoint rotation angle (the single
largest driver of real cross-view matching difficulty) across multiple
random seeds per condition, for all three matchers, and report AGGREGATE
statistics (mean/median/std) per condition instead of a single lucky pair.

This is what turns "one demo pair worked" into an actual result: N=5 seeds
per angle x 6 angles x 3 methods = 90 independent trials.

Usage:
    SUPERGLUE_PATH=/path/to/SuperGluePretrainedNetwork python3 src/ablation.py
"""

import sys
import os
import time
import csv
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
    print(f"[ablation] SuperGlue not available, skipping deep matcher: {e}")

ROTATION_ANGLES = [0, 10, 20, 30, 45, 60]
SEEDS_PER_CONDITION = 5


def ground_truth_center(query_shape, gt_H):
    h, w = query_shape[:2]
    center = np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(center, gt_H).reshape(2)


def run_trial(localizer_fn, reference, angle, seed):
    query, gt_H = generate_synthetic_query(
        reference, crop_frac=0.35, rotation_deg=angle, scale=0.9, seed=seed
    )
    t0 = time.time()
    result = localizer_fn(reference, query)
    runtime_ms = (time.time() - t0) * 1000

    if not result["success"]:
        return {"success": False, "error_px": None, "runtime_ms": runtime_ms,
                "inlier_ratio": 0.0}

    gt_center = ground_truth_center(query.shape, gt_H)
    est_center = np.array(result["center_ref"])
    error_px = float(np.linalg.norm(gt_center - est_center))
    return {"success": True, "error_px": error_px, "runtime_ms": runtime_ms,
            "inlier_ratio": result["inlier_ratio"]}


def main():
    reference = generate_synthetic_reference(size=800, seed=0)

    methods = {
        "SIFT": VisualLocalizer(detector="sift").localize,
        "ORB": VisualLocalizer(detector="orb").localize,
    }
    if DEEP_AVAILABLE:
        deep = DeepLocalizer(weights="outdoor")
        methods["SuperGlue"] = deep.localize

    rows = []
    for method_name, localize_fn in methods.items():
        for angle in ROTATION_ANGLES:
            trial_results = [
                run_trial(localize_fn, reference, angle, seed)
                for seed in range(SEEDS_PER_CONDITION)
            ]
            successes = [r for r in trial_results if r["success"]]
            errors = [r["error_px"] for r in successes]
            success_rate = len(successes) / len(trial_results)

            row = {
                "method": method_name,
                "rotation_deg": angle,
                "n_trials": len(trial_results),
                "success_rate": success_rate,
                "mean_error_px": float(np.mean(errors)) if errors else None,
                "median_error_px": float(np.median(errors)) if errors else None,
                "std_error_px": float(np.std(errors)) if errors else None,
                "mean_runtime_ms": float(np.mean([r["runtime_ms"] for r in trial_results])),
                "mean_inlier_ratio": float(np.mean([r["inlier_ratio"] for r in trial_results])),
            }
            rows.append(row)
            print(f"{method_name:10s} angle={angle:3d} deg  "
                  f"success={success_rate:.0%}  "
                  f"error(px) mean={row['mean_error_px']}  "
                  f"median={row['median_error_px']}")

    csv_path = os.path.join(RESULTS_DIR, "ablation_rotation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # Plot
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    for method_name in methods:
        m_rows = [r for r in rows if r["method"] == method_name]
        angles = [r["rotation_deg"] for r in m_rows]
        means = [r["mean_error_px"] if r["mean_error_px"] is not None else np.nan for r in m_rows]
        stds = [r["std_error_px"] if r["std_error_px"] is not None else 0 for r in m_rows]
        ax1.errorbar(angles, means, yerr=stds, marker='o', label=method_name, capsize=3)

        success_rates = [r["success_rate"] * 100 for r in m_rows]
        ax2.plot(angles, success_rates, marker='o', label=method_name)

    ax1.set_xlabel("Rotation angle (deg)")
    ax1.set_ylabel("Localization error (px, mean +/- std)")
    ax1.set_title("Accuracy vs. viewpoint rotation")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Rotation angle (deg)")
    ax2.set_ylabel("Success rate (%)")
    ax2.set_title("Robustness vs. viewpoint rotation")
    ax2.set_ylim(-5, 105)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "ablation_rotation.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_rotation.pdf"), bbox_inches='tight')
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
