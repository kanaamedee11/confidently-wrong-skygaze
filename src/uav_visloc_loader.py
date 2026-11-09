"""
uav_visloc_loader.py
=====================
Loader for the UAV-VisLoc dataset (Xu et al. 2024, arXiv:2405.11936):
6,742 real drone images + 11 georeferenced satellite maps across 11 sites in
China — built specifically for this exact task (UAV visual localization
against a satellite map). This is the dataset the paper should ultimately
report numbers on.

NOT bundled here: the dataset itself (16.4GB / 2.04GB sample) is hosted on
Google Drive / Baidu Netdisk, which this build environment cannot reach.
Download it yourself from:
    https://github.com/IntelliSensing/UAV-VisLoc
(get the "dataset example" 2.04GB sample first — it's the fast way to get
real, working end-to-end results before committing to the full 16.4GB set)

Expected directory layout (matches the dataset's own structure):
    UAV-VisLoc/
      satellite_coordinates_range.csv   # filename, lat range, lon range for each satellite map
      01/
        drone/
          01_0001.JPG
          ...
        satellite01.tif
        01.csv                         # per-image: filename, lat, lon, height, heading...

Usage:
    from uav_visloc_loader import UAVVisLocFlight
    flight = UAVVisLocFlight("/path/to/UAV-VisLoc", flight_id="01")
    satellite_img = flight.satellite_image()
    for sample in flight.iter_samples():
        drone_img = sample["image"]
        gt_pixel = sample["gt_pixel"]        # (x, y) ground-truth location in satellite image pixels
        gt_latlon = sample["gt_latlon"]      # (lat, lon)
"""

import os
import math
import pandas as pd
import cv2


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters — use this for the paper's reported error metric, not raw pixels."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class UAVVisLocFlight:
    def __init__(self, dataset_root, flight_id):
        self.root = dataset_root
        self.flight_id = flight_id
        self.flight_dir = os.path.join(dataset_root, flight_id)
        if not os.path.isdir(self.flight_dir):
            raise FileNotFoundError(
                f"Flight dir not found: {self.flight_dir}. "
                f"Check dataset_root points at the UAV-VisLoc top-level folder."
            )

        self._satellite_path = os.path.join(self.flight_dir, f"satellite{flight_id}.tif")
        self._drone_csv_path = os.path.join(self.flight_dir, f"{flight_id}.csv")
        self._range_csv_path = os.path.join(dataset_root, "satellite_coordinates_range.csv")

        self.drone_meta = pd.read_csv(self._drone_csv_path)
        self.range_meta = pd.read_csv(self._range_csv_path)

        # Match this flight's satellite map row. Column names are dataset-defined;
        # inspect self.range_meta.columns and adjust this filter if the real CSV
        # uses different header names than assumed here.
        match = self.range_meta[
            self.range_meta.iloc[:, 0].astype(str).str.contains(flight_id)
        ]
        if len(match) == 0:
            raise ValueError(
                f"Could not find satellite map bounds for flight {flight_id} in "
                f"{self._range_csv_path}. Inspect the CSV columns and adjust the matching logic."
            )
        self._bounds_row = match.iloc[0]

    def satellite_image(self):
        img = cv2.imread(self._satellite_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read {self._satellite_path}")
        return img

    def latlon_to_pixel(self, lat, lon, satellite_shape, lat_min, lat_max, lon_min, lon_max):
        """
        Linear interpolation of lat/lon into pixel coordinates. Valid for the
        small geographic extent of a single orthorectified satellite tile
        (a few km) — do NOT reuse this for large-area maps where a proper
        map projection (not linear interpolation) is required.
        """
        h, w = satellite_shape[:2]
        x = (lon - lon_min) / (lon_max - lon_min) * w
        y = (lat_max - lat) / (lat_max - lat_min) * h  # image y grows downward, lat grows upward
        return x, y

    def iter_samples(self, limit=None):
        """
        Yields dicts: {image, image_path, gt_latlon, filename}. Ground-truth
        pixel location on the satellite map is left for the caller to compute
        via latlon_to_pixel once you've confirmed the actual column names in
        satellite_coordinates_range.csv against your downloaded copy.
        """
        n = 0
        for _, row in self.drone_meta.iterrows():
            filename = row.iloc[0]
            path = os.path.join(self.flight_dir, "drone", filename)
            if not os.path.exists(path):
                continue
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            yield {
                "filename": filename,
                "image_path": path,
                "image": img,
                "row": row,  # full metadata row (lat, lon, height, heading, ...) — inspect columns directly
            }
            n += 1
            if limit and n >= limit:
                break
