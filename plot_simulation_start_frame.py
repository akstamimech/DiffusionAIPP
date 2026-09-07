"""
The shared starting frame for all three future-plan animations (lawnmower,
CMA-ES, greedy) - identical across all three since none have planned or
moved yet at t=0: same map 51 GRF prior mean field, same start position
(4,4), same initial FOV box. No title, since it's shared rather than
belonging to one specific planner.
"""
import os

os.environ.setdefault("MAPTYPE", "grf")
os.environ.setdefault("SELECTED_MAP", "51")

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import CMAES_classic_singlemap as sim
from gaussianprocesstraining import fov_lateral_radius, initialize_gp

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PNG = SCRIPT_DIR / "simulation_start_frame_map51_grf.png"


def main():
    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()
    mu = np.full(X_test.shape[0], sim.utility_threshold + 0.1)

    cx, cy, cz = 4.0, 4.0, sim.INIT_ALTITUDE
    ny, nx = len(ys), len(xs)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(
        mu.reshape(ny, nx), origin="lower", cmap="viridis",
        extent=[xmin, xmax, ymin, ymax], vmin=0, vmax=1, interpolation="bicubic", zorder=1,
    )
    r = fov_lateral_radius(cz, 60)
    ax.add_patch(Rectangle((cx - r, cy - r), 2 * r, 2 * r, facecolor="none", edgecolor="white", linewidth=1.4, alpha=0.6, zorder=3))
    ax.scatter([cx], [cy], color="white", edgecolors="#ff5a4e", linewidths=2, s=170, zorder=4)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    fig.savefig(OUT_PNG, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
