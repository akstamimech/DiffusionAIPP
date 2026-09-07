"""
Standalone NAIP map 8 ground truth panel (companion to
plot_grf_map11_ground_truth_outline.py). Uses the CSV's native 1m resolution
(101x101) rather than sub-sampling to the simulation's 2m/51x51 grid, since
the raw satellite source tile for any given map_id couldn't be confidently
matched (map-id renumbering on 2026-08-15 plus no verified NDVI-scale raw
raster - see naip_shuffle_mapping_2026-08-15.csv and the investigation notes
in this session; the 1m CSV is the finest verified-authentic version of this
data). No colorbar, no title; a red dashed contour marks the low-NDVI region
(<0.3) instead. Axis ticks/labels kept.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
NAIP_MAP_ID = 14
LOW_NDVI_THRESHOLD = 0.3
OUT_PATH = SCRIPT_DIR / f"naip_map{NAIP_MAP_ID}_ground_truth_outline.png"


def load_native_grid(map_id, maptype):
    """No step-based subsampling - uses every point in the CSV as-is."""
    path = SCRIPT_DIR / "csv" / f"map_{map_id}_{maptype}_grid_counts.csv"
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    xs = np.unique(data[:, 0])
    ys = np.unique(data[:, 1])
    grid = np.full((len(ys), len(xs)), np.nan)
    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}
    for x, y, val in data:
        grid[y_idx[y], x_idx[x]] = val
    return xs, ys, grid


def main():
    xs, ys, grid = load_native_grid(NAIP_MAP_ID, "NAIP")
    X, Y = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.imshow(
        grid, origin="lower", cmap="viridis",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()], aspect="equal",
    )
    ax.contour(X, Y, grid, levels=[LOW_NDVI_THRESHOLD], colors="red", linewidths=1.8, linestyles="--")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
