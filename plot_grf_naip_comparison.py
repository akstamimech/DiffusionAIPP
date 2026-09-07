"""
Side-by-side ground-truth comparison: a synthetic GRF map vs. a real NAIP map, each with its own
axes and colorbar (separate vmin/vmax per panel deliberately - NAIP's real value range is narrower
than and offset from GRF's clean [0,1] range, not something to paper over with a shared scale).

Uses the same example maps as this session's other figures (map 51 for GRF, map 11 for NAIP) for
consistency across the thesis's figure set, rather than picking new arbitrary map IDs.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = Path(__file__).resolve().parent
GRF_MAP_ID = 51
NAIP_MAP_ID = 11
OUT_PATH = SCRIPT_DIR / "grf_naip_ground_truth_comparison.png"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK_PRIMARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


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


def draw_panel(ax, fig, xs, ys, grid, title, cmap):
    vmin, vmax = float(np.nanmin(grid)), float(np.nanmax(grid))
    im = ax.imshow(
        grid, origin="lower", cmap=cmap,
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        aspect="equal", vmin=vmin, vmax=vmax,
    )
    ax.set_title(title, fontsize=13, color=INK_PRIMARY, pad=10)
    ax.set_xlabel("x (m)", color=INK_SECONDARY, fontsize=11)
    ax.set_ylabel("y (m)", color=INK_SECONDARY, fontsize=11)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(INK_MUTED)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("field value", color=INK_SECONDARY, fontsize=10)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=9)
    return vmin, vmax


def main():
    grf_x, grf_y, grf_grid = load_grid(GRF_MAP_ID, "grf")
    naip_x, naip_y, naip_grid = load_grid(NAIP_MAP_ID, "NAIP")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    grf_range = draw_panel(ax1, fig, grf_x, grf_y, grf_grid, f"Synthetic GRF (map {GRF_MAP_ID})", SEQUENTIAL_CMAP)
    naip_range = draw_panel(ax2, fig, naip_x, naip_y, naip_grid, f"NAIP-derived (map {NAIP_MAP_ID})", "Greys")

    print(f"GRF value range: [{grf_range[0]:.3f}, {grf_range[1]:.3f}]")
    print(f"NAIP value range: [{naip_range[0]:.3f}, {naip_range[1]:.3f}]")

    fig.suptitle("Ground-truth field comparison: synthetic vs. real-world", fontsize=14.5, color=INK_PRIMARY, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
