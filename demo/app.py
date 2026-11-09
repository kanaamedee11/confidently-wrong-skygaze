"""
SkyGaze demo — interactive visual localization.

Run with:
    streamlit run demo/app.py

Upload a reference (map/satellite) image and a query (drone/live-feed) image,
or click "Use synthetic example" to see it work instantly with no data of
your own. This is the portfolio-facing piece: what an admissions reviewer
(or a paper reader) actually clicks around in.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import cv2
import streamlit as st
from PIL import Image

from data_utils import generate_synthetic_reference, generate_synthetic_query
from matcher import VisualLocalizer

st.set_page_config(page_title="SkyGaze — Visual Localization", layout="wide")

st.title("SkyGaze — image-based visual localization")
st.caption(
    "Match a live camera view against a stored reference image (satellite tile, "
    "map, or pre-recorded scan) to estimate where the live view sits inside it."
)

with st.sidebar:
    st.header("Settings")
    detector_name = st.selectbox("Feature detector", ["sift", "orb"], index=0)
    ratio_thresh = st.slider("Lowe's ratio test threshold", 0.5, 0.95, 0.75, 0.05)
    ransac_thresh = st.slider("RANSAC reprojection threshold (px)", 1.0, 10.0, 5.0, 0.5)
    use_synthetic = st.button("Use synthetic example")

col1, col2 = st.columns(2)
with col1:
    ref_file = st.file_uploader("Reference image (map / satellite tile)", type=["png", "jpg", "jpeg"])
with col2:
    qry_file = st.file_uploader("Query image (drone frame / live feed)", type=["png", "jpg", "jpeg"])

reference, query = None, None

if use_synthetic:
    reference = generate_synthetic_reference(size=800, seed=0)
    query, _ = generate_synthetic_query(reference, crop_frac=0.35, rotation_deg=15, scale=0.9, seed=1)
elif ref_file is not None and qry_file is not None:
    reference = cv2.cvtColor(np.array(Image.open(ref_file).convert("RGB")), cv2.COLOR_RGB2BGR)
    query = cv2.cvtColor(np.array(Image.open(qry_file).convert("RGB")), cv2.COLOR_RGB2BGR)

if reference is not None and query is not None:
    localizer = VisualLocalizer(detector=detector_name, ratio_thresh=ratio_thresh, ransac_thresh=ransac_thresh)
    result = localizer.localize(reference, query)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Good matches", result["num_good_matches"])
    c2.metric("RANSAC inliers", result["num_inliers"])
    c3.metric("Inlier ratio", f"{result['inlier_ratio']:.2f}")
    c4.metric("Runtime", f"{result['runtime_ms']:.1f} ms")

    if result["success"]:
        st.success(f"Localized. Estimated center in reference frame: "
                   f"({result['center_ref'][0]:.1f}, {result['center_ref'][1]:.1f})")
        vis = localizer.draw_result(reference, query, result)
        vc1, vc2 = st.columns([2, 1])
        with vc1:
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                      caption="Reference image with query footprint (green) and estimated center (red)",
                      use_container_width=True)
        with vc2:
            st.image(cv2.cvtColor(query, cv2.COLOR_BGR2RGB), caption="Query image", use_container_width=True)
    else:
        st.error("Localization failed — not enough reliable matches. Try SIFT, "
                 "lower the ratio threshold, or check the images actually overlap.")
else:
    st.info("Upload a reference + query image, or click **Use synthetic example** in the sidebar to try it instantly.")
