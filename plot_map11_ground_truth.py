import os

os.environ.setdefault("MAPTYPE", "halffield")

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])

from gaussianprocesstraining import initialize_gp

SCRIPT_DIR = Path(__file__).resolve().parent
MAP_ID = 11
MAPTYPE = "halffield"
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

    grid_min, grid_max = true_grid.min(), true_grid.max()
    norm_grid = (true_grid - grid_min) / (grid_max - grid_min)
    norm_threshold = (IMPORTANT_THRESHOLD - grid_min) / (grid_max - grid_min)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        norm_grid, origin="lower", cmap="viridis",
        extent=[xmin, xmax, ymin, ymax], interpolation="bicubic",
        vmin=0, vmax=1,
    )
    ax.contour(
        X, Y, norm_grid, levels=[norm_threshold],
        colors="red", linewidths=1.8, linestyles="--",
    )
    ax.set_title(f"Map {MAP_ID} ground truth ({MAPTYPE})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("crop stress")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
