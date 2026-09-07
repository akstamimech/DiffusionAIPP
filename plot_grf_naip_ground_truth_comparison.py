"""
Side-by-side ground-truth comparison: GRF map vs. NAIP map, viridis for
both panels, each with its own colorbar (separate vmin/vmax per panel -
NAIP's real NDVI range is narrower than and offset from GRF's clean [0,1]
range).

Map choices: GRF map 11 and NAIP map 8. NAIP map 8 was picked by scanning
maps 1-40 for value distribution (see the printed table this script
produces from that scan) - map 8 has only 3.3% of cells below 0.3 (mean
0.476), unlike the original map 11 pairing (96.4% below 0.3, heavily
dominated by low values).
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
GRF_MAP_ID = 11
NAIP_MAP_ID = 8
OUT_PATH = SCRIPT_DIR / "grf_naip_ground_truth_comparison.png"


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


def draw_panel(ax, fig, xs, ys, grid, title):
    vmin, vmax = float(np.nanmin(grid)), float(np.nanmax(grid))
    im = ax.imshow(
        grid, origin="lower", cmap="viridis",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        aspect="equal", vmin=vmin, vmax=vmax,
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Ground-truth value", fontsize=10)
    return vmin, vmax


def main():
    grf_x, grf_y, grf_grid = load_grid(GRF_MAP_ID, "grf")
    naip_x, naip_y, naip_grid = load_grid(NAIP_MAP_ID, "NAIP")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    grf_range = draw_panel(ax1, fig, grf_x, grf_y, grf_grid, f"GRF map {GRF_MAP_ID}")
    naip_range = draw_panel(ax2, fig, naip_x, naip_y, naip_grid, f"NAIP map {NAIP_MAP_ID}")

    print(f"GRF value range: [{grf_range[0]:.3f}, {grf_range[1]:.3f}]")
    print(f"NAIP value range: [{naip_range[0]:.3f}, {naip_range[1]:.3f}]")

    fig.suptitle("Ground-truth field comparison: GRF vs. NAIP", fontsize=14.5, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
