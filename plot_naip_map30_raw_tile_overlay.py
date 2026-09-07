"""
NAIP map 30 ground truth, shown over its actual raw NDVI tile rather than the
2m/51x51-subsampled grid_counts heatmap. Unlike the earlier investigation
(see plot_naip_map8_ground_truth_outline.py's docstring), the raw source tile
IS verified here: Datasets/NAIP_dataset/selected_tiles/ + selected_tiles_csv/
turned out to hold the actual 65 candidate NDVI tiles (and their grid_counts
CSVs, numbered by the OLD pre-shuffle map id) used to build the sim's 65 NAIP
maps - a directory the earlier session never checked. Old map id 13 (= new
map 30, per naip_shuffle_mapping_2026-08-15.csv) gives
selected_tiles_csv/map_13_NAIP_grid_counts.csv, byte-identical to the live
csv/map_30_NAIP_grid_counts.csv. Correlating every tile in selected_tiles/
(resampled to the CSV's 101x101 grid) against that CSV found exactly one
decisive match: ndvi_below_0.2_r0005_c0000.tif at r=0.97 (next-best r=0.67).
That tif is a real NDVI raster (unlike the earlier session's candidate,
Map_1_NAIP/map_1_NAIP.tif, which turned out to be raw reflectance/DN values,
not NDVI - explaining why it never correlated).

No colorbar, no title, no axes; a red dashed contour marks the low-NDVI
region (<0.3), computed from the grid_counts CSV, overlaid on the raw tile.
"""
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
NAIP_MAP_ID = 30
RAW_TIF_PATH = (
    SCRIPT_DIR.parent
    / "Datasets" / "NAIP_dataset" / "selected_tiles" / "ndvi_below_0.2_r0005_c0000.tif"
)
LOW_NDVI_THRESHOLD = 0.3
OUT_PATH = SCRIPT_DIR / f"naip_map{NAIP_MAP_ID}_raw_tile_outline.png"


def load_native_grid(map_id, maptype):
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

    with rasterio.open(RAW_TIF_PATH) as src:
        raw = src.read(1).astype(float)
        px_w, px_h = src.res
    extent_w = raw.shape[1] * px_w
    extent_h = raw.shape[0] * px_h

    fig, ax = plt.subplots(figsize=(6, 5.5))
    # origin="lower" matches the grid_counts CSV's row convention (verified
    # via correlation - the winning orientation was the identity/no-flip one).
    ax.imshow(
        raw, origin="lower", cmap="viridis",
        extent=[0, extent_w, 0, extent_h], aspect="equal",
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
