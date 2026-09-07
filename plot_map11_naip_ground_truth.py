"""
Ground truth NDVI for NAIP map 11 (the map used elsewhere for the hero-shot /
thesis cover figures), with a contour around the low-NDVI ("stressed crop")
region. Threshold is 0.2, matching the project's own convention baked into
Datasets/NAIP_dataset (the "ndvi_below_0.2" selected-tile naming) rather than
the generic UTILITY_THRESHOLD=0.5 used for the synthetic halffield/grf maps
- map 11's raw NDVI only ranges ~[-0.03, 0.47], so 0.5 would never trigger.
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
MAP_ID = int(os.environ.get("NAIP_MAP_ID", "11"))
MAPTYPE = "NAIP"
LOW_NDVI_THRESHOLD = 0.2
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
        true_grid, origin="lower", cmap="RdYlGn",
        extent=[xmin, xmax, ymin, ymax], interpolation="bicubic",
    )
    ax.contour(
        X, Y, true_grid, levels=[LOW_NDVI_THRESHOLD],
        colors="blue", linewidths=1.8, linestyles="--",
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
