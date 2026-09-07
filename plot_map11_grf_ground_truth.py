"""
Ground truth for GRF map 11, with a contour around the important region
(>=0.5, matching UTILITY_THRESHOLD's live default for GRF/UCB used
throughout the sim scripts). Raw values are already in [0,1] for this
maptype, unlike halffield's 0-30 raw counts, so no normalization needed.
Companion to plot_map11_naip_ground_truth.py - same clean style (no title,
axes, or colorbar), for a matched pair.
"""
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])

from gaussianprocesstraining import initialize_gp

SCRIPT_DIR = Path(__file__).resolve().parent
MAP_ID = int(os.environ.get("GRF_MAP_ID", "11"))
MAPTYPE = "grf"
IMPORTANT_THRESHOLD = 0.5
OUT_PNG = SCRIPT_DIR / f"map_{MAP_ID}_{MAPTYPE}_ground_truth.png"


def main():
    csv_path = SCRIPT_DIR / "csv" / f"map_{MAP_ID}_{MAPTYPE}_grid_counts.csv"
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()

    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]

    true_map_flat = np.zeros(X_test.shape[0], dtype=float)
    for x_true, y_true, value in pts:
        idx = np.where(np.isclose(X_test[:, 0], x_true) & np.isclose(X_test[:, 1], y_true))[0]
        if idx.size:
            true_map_flat[idx[0]] = value

    ny, nx = len(ys), len(xs)
    true_grid = true_map_flat.reshape(ny, nx)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        true_grid, origin="lower", cmap="viridis",
        extent=[xmin, xmax, ymin, ymax], interpolation="bicubic",
    )
    ax.contour(
        X, Y, true_grid, levels=[IMPORTANT_THRESHOLD],
        colors="red", linewidths=1.8, linestyles="--",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
