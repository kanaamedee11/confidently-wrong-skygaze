"""Shared loaders for the UAV-VisLoc flight-03 subset (real drone frames vs a real satellite map)."""
import os, json, csv, sys
import numpy as np, cv2
D = os.environ.get("VISLOC_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "visloc_03_subset"))
META = json.load(open(os.path.join(D, "meta.json")))
LON0, LAT0, LON1, LAT1 = META["sat_bounds"]          # left, bottom, right, top (EPSG:4326)
SW, SH = META["sat_small_wh"]
M_LAT = 111_320.0
COS = np.cos(np.radians((LAT0 + LAT1) / 2))

def load_rows():
    rows = list(csv.DictReader(open(os.path.join(D, "03.csv"), encoding="utf-8-sig")))
    return rows

def sat(iso=True):
    """Satellite map (BGR). If iso, resample horizontally so one pixel is the same ground distance in x and y
    (the GeoTIFF is in degrees, so a degree of longitude is shorter than a degree of latitude)."""
    im = cv2.imread(os.path.join(D, "satellite03_small.jpg"))
    if iso:
        im = cv2.resize(im, (int(round(SW * COS)), SH), interpolation=cv2.INTER_AREA)
    return im

def gt_px(lat, lon, iso=True):
    x = (lon - LON0) / (LON1 - LON0) * SW
    y = (LAT1 - lat) / (LAT1 - LAT0) * SH
    if iso:
        x *= (int(round(SW * COS)) / SW)
    return np.array([x, y])

def m_per_px(iso=True):
    # metres per pixel of the (iso) small satellite map
    y = (LAT1 - LAT0) * M_LAT / SH
    return y if iso else None
