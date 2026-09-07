"""
Companion to plot_gp_prior.py: the GP prior VARIANCE field. initialize_gp()'s
kernel is ConstantKernel(sigma2=0.05) * Matern(...) - Matern's self-covariance
k(x,x) is always 1, so the prior covariance diagonal is sigma2 everywhere, a
flat field just like the prior mean. Same convention as the mean panel: no
colorbar (a single value has no range to show), just the magnitude labeled
in the corner. vmax is set to sigma2 itself (not 1) so the prior renders at
peak saturation - the natural "maximum uncertainty, before any measurements"
reading, matching the vmin=0/vmax=diag_max convention used for variance
fields elsewhere in this project.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PATH = SCRIPT_DIR / "gp_prior_variance.png"

SIGMA2 = 0.05  # current live default: initialize_gp(sigma2=0.05, ...)

plt.rcParams.update({
    "mathtext.fontset": "cm",
})

field = np.full((51, 51), SIGMA2)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(field, origin="lower", cmap="viridis", vmin=0, vmax=SIGMA2, extent=[0, 100, 0, 100])

ax.text(
    0.05, 0.95, rf"$\sigma_0^2 = {SIGMA2:g}$",
    transform=ax.transAxes, ha="left", va="top", fontsize=26, color="black",
)

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("2D GP prior variance")

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUT_PATH}")
