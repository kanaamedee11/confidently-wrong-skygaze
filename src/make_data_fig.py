import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_exp as R
from data_utils import generate_synthetic_reference
from gen2 import make_query
OUT = os.path.join(HERE, "..", "paper", "figures")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 7, "pdf.fonttype": 42})
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
refs = [generate_synthetic_reference(size=800, seed=s) for s in range(3)] + [R.texture_mosaic()]
titles = ["Reference A (seed 0)", "Reference B (seed 1)", "Reference C (seed 2)", "Texture mosaic (photographic)"]
ref0 = refs[0]
B = R.BASE
qs = [("Base (15 deg)", dict(B)), ("Rotation 60 deg", dict(B, rotation_deg=60.0)), ("Rotation 180 deg", dict(B, rotation_deg=180.0)),
      ("Scale 1.8", dict(B, scale=1.8)), ("Perspective 0.0012", dict(B, persp=0.0012)), ("Noise 35 grey levels", dict(B, noise_sigma=35.0)),
      ("Blur 5 px", dict(B, blur_sigma=5.0)), ("Occlusion 40%", dict(B, occl_frac=0.4)), ("Appearance 0.6", dict(B, appearance=0.6))]
fig = plt.figure(figsize=(7.0, 4.4))
top = fig.add_gridspec(1, 4, left=0.0, right=1.0, top=1.0, bottom=0.60, wspace=0.05)
bot = fig.add_gridspec(2, 5, left=0.0, right=1.0, top=0.54, bottom=0.0, hspace=0.28, wspace=0.05)
for i, (r, t) in enumerate(zip(refs, titles)):
    ax = fig.add_subplot(top[0, i]); ax.imshow(rgb(r)); ax.set_title(t, fontsize=7); ax.axis("off")
for i, (t, cfg) in enumerate(qs):
    q, _ = make_query(ref0, seed=3, **cfg)
    ax = fig.add_subplot(bot[i // 5, i % 5]); ax.imshow(rgb(q)); ax.set_title(t, fontsize=6.5); ax.axis("off")
plt.savefig(os.path.join(OUT, "fig_data_v2.pdf"), bbox_inches="tight", pad_inches=0.03)
plt.savefig(os.path.join(OUT, "fig_data_v2.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
print("ok")
