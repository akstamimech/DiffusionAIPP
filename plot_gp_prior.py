"""
The GP prior: current live simulation scripts (CMAES_classic_singlemap.py,
ImitateTrans_singlemap.py, lawnmower_singlemap.py) all initialize the belief
mean as a single constant value (mean = np.full(N, utility_threshold + 0.1),
the optimistic UCB/GRF prior) before any measurements are taken - so the
"prior" is literally a flat field, not a spatially varying one, hence no
colorbar (a single value has no meaningful range to show) and just a
corner label giving that constant instead.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PATH = SCRIPT_DIR / "gp_prior.png"

UTILITY_THRESHOLD = 0.5  # current live default across CMAES_classic_singlemap.py etc.
PRIOR_MAGNITUDE = UTILITY_THRESHOLD + 0.1  # optimistic UCB/GRF prior

plt.rcParams.update({
    "mathtext.fontset": "cm",
})

field = np.full((51, 51), PRIOR_MAGNITUDE)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(field, origin="lower", cmap="viridis", vmin=0, vmax=1, extent=[0, 100, 0, 100])

ax.text(
    0.05, 0.95, rf"$\mu_0 = {PRIOR_MAGNITUDE:g}$",
    transform=ax.transAxes, ha="left", va="top", fontsize=26, color="white",
)

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("2D GP prior mean")

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUT_PATH}")
