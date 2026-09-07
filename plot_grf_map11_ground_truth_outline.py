"""
Standalone GRF map 11 ground truth panel (companion to
plot_naip_map8_ground_truth_outline.py, meant as a matched pair rather than
one combined figure). No colorbar, no title; a red dashed contour marks the
important region (>=0.5) instead. Axis ticks/labels kept.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
GRF_MAP_ID = 11
IMPORTANT_THRESHOLD = 0.5
OUT_PATH = SCRIPT_DIR / f"grf_map{GRF_MAP_ID}_ground_truth_outline.png"


def load_grid(map_id, maptype, step=2.0):
    path = SCRIPT_DIR / "csv" / f"map_{map_id}_{maptype}_grid_counts.csv"
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    mask = (
        np.isclose(np.mod(data[:, 0], step), 0.0, atol=1e-9)
        & np.isclose(np.mod(data[:, 1], step), 0.0, atol=1e-9)
    )
    data = data[mask]
    xs = np.unique(data[:, 0])
    ys = np.unique(data[:, 1])
    grid = np.full((len(ys), len(xs)), np.nan)
    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}
    for x, y, val in data:
        grid[y_idx[y], x_idx[x]] = val
    return xs, ys, grid


def main():
    xs, ys, grid = load_grid(GRF_MAP_ID, "grf")
    X, Y = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.imshow(
        grid, origin="lower", cmap="viridis",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()], aspect="equal",
    )
    ax.contour(X, Y, grid, levels=[IMPORTANT_THRESHOLD], colors="red", linewidths=1.8, linestyles="--")
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
