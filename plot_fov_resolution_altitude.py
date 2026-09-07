"""
How field-of-view and sensor resolution change with altitude - built directly
from the real simulation mechanics in gaussianprocesstraining.py and the
ground-truth loading convention in CMAES_classic_singlemap.py, not a schematic
redrawing of the numbers.

Top panel: side-view cross-section. Three drones at ZMIN/mid/ZMAX altitude,
each casting its true angle_of_view=60deg FOV cone (fov_lateral_radius) down
to the ground - shows the footprint widening with altitude.

Bottom panels: for each altitude, the *actual* sensor reading each grid cell
in the FOV would get, computed by calling fov_grid_points + build_sensor_matrix
on map 11 (NAIP)'s real ground truth and reading off `sensor @ true_map_flat`
- i.e. this is the literal block-averaged observation CMAES_classic_singlemap.py's
own Kalman update would receive at that altitude, not a synthetic checkerboard.
resolution_block_size(altitude) (1/2/4 grid cells per block, step=2m each) is
what produces the coarser "pixelation" visible at higher altitude.

Reads: csv/map_11_NAIP_grid_counts.csv (same load+mask convention as
CMAES_classic_singlemap.py). ZMIN/ZMAX mirror that script's current
module-level constants (10.0/40.0).

Palette: same categorical slots 2/3/7 (orange/aqua/violet) as
plot_lattice_over_map.py's low/middle/top tiers, so the two figures read as a
matched pair encoding the same three altitude bands.

FIG_WIDTH matches plot_sensor_noise_model.py's figure width, and text here is
kept to the minimum needed to stay legible (large font, one shared legend
instead of per-point sentences) since combine_noise_fov_resolution.py stacks
the two into one image sized for an IEEE-style paper column.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from gaussianprocesstraining import (
    initialize_gp,
    fov_lateral_radius,
    resolution_block_size,
    fov_grid_points,
    build_sensor_matrix,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MAP_ID = 11
MAPTYPE = "NAIP"
ZMIN, ZMAX = 10.0, 40.0
ANGLE_OF_VIEW = 60.0
CENTER = (50.0, 50.0)  # map center, so even the widest (z=ZMAX) footprint stays inside [0,100]
CSV_PATH = SCRIPT_DIR / "csv" / f"map_{MAP_ID}_{MAPTYPE}_grid_counts.csv"
OUT_PATH = SCRIPT_DIR / f"fov_resolution_vs_altitude_map{MAP_ID}_{MAPTYPE}.png"
FIG_WIDTH = 7.5  # inches; matches plot_sensor_noise_model.py's FIG_WIDTH for clean vertical stacking

ALTITUDES = [ZMIN, (ZMIN + ZMAX) / 2, ZMAX]  # 10, 25, 40 - matches the lattice's tier heights

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
TIER_COLORS = ["#eb6834", "#1baf7a", "#4a3aa7"]  # slot 2/3/7: orange/aqua/violet

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


def load_true_map_flat(csv_path, X_test, step=2.0):
    """Mirrors CMAES_classic_singlemap.py's own ground-truth load block exactly."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    pts = data[:, 0:3]
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=1e-9)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=1e-9)
    )
    pts = pts[mask]

    true_map_flat = np.zeros(X_test.shape[0], dtype=float)
    for x_true, y_true, value in pts:
        idx = np.where(
            np.isclose(X_test[:, 0], x_true) & np.isclose(X_test[:, 1], y_true)
        )[0]
        if idx.size > 0:
            true_map_flat[idx[0]] = value
    return true_map_flat


def observed_image_for_altitude(cx, cy, z, xs, ys, true_map_flat, step=2.0):
    fov = fov_grid_points(cx, cy, z, xs, ys, angle_of_view=ANGLE_OF_VIEW)
    sensor, _ = build_sensor_matrix(fov, z, xs, ys, return_block_ids=True)
    observed = np.asarray(sensor @ true_map_flat).ravel()
    assert len(fov) == len(observed), "fov points and sensor rows must align 1:1"

    fov_arr = np.asarray(fov)
    xs_vis = np.unique(fov_arr[:, 0])
    ys_vis = np.unique(fov_arr[:, 1])
    x_idx = {v: i for i, v in enumerate(xs_vis)}
    y_idx = {v: i for i, v in enumerate(ys_vis)}
    img = np.full((len(ys_vis), len(xs_vis)), np.nan)
    for (x, y), val in zip(fov_arr, observed):
        img[y_idx[y], x_idx[x]] = val

    extent = [
        xs_vis.min() - step / 2, xs_vis.max() + step / 2,
        ys_vis.min() - step / 2, ys_vis.max() + step / 2,
    ]
    return img, extent


def draw_side_view(ax):
    ax.axhspan(-3, 0, color=BLUE_RAMP[1], alpha=0.6, zorder=1)
    ax.axhline(0, color=INK_MUTED, linewidth=1.2, zorder=2)

    max_radius = fov_lateral_radius(ZMAX, ANGLE_OF_VIEW)
    for i, (z, color) in enumerate(zip(ALTITUDES, TIER_COLORS)):
        radius = fov_lateral_radius(z, ANGLE_OF_VIEW)
        ax.plot([0, -radius], [z, 0], color=color, linewidth=2.0, zorder=3)
        ax.plot([0, radius], [z, 0], color=color, linewidth=2.0, zorder=3)
        ax.add_patch(Circle((0, z), radius=1.1, color=color, zorder=5, ec=SURFACE, linewidth=1.2))

        bracket_y = -3 - 2.6 * i
        ax.plot([-radius, radius], [bracket_y, bracket_y], color=color, linewidth=1.8, zorder=3)
        ax.plot([-radius, -radius], [bracket_y - 0.5, bracket_y + 0.5], color=color, linewidth=1.8, zorder=3)
        ax.plot([radius, radius], [bracket_y - 0.5, bracket_y + 0.5], color=color, linewidth=1.8, zorder=3)
        ax.annotate(
            f"{2*radius:.0f} m", (radius + 1.2, bracket_y),
            color=color, fontsize=13, fontweight="bold", va="center", zorder=6,
        )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=c, markeredgecolor=SURFACE, label=f"{z:.0f} m")
        for z, c in zip(ALTITUDES, TIER_COLORS)
    ]
    ax.legend(
        handles=legend_handles, loc="upper left", frameon=False,
        fontsize=13, labelcolor=INK_SECONDARY, handletextpad=0.4,
        borderaxespad=0.2,
    )

    ax.set_xlim(-max_radius - 12, max_radius + 20)
    ax.set_ylim(-3 - 2.6 * len(ALTITUDES) - 1, ZMAX + 6)
    ax.set_xlabel("Ground distance (m)", color=INK_SECONDARY, fontsize=14)
    ax.set_ylabel("Altitude (m)", color=INK_SECONDARY, fontsize=14)
    ax.set_title("FOV vs. altitude", color=INK_PRIMARY, fontsize=16)
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelsize=12)


def main():
    _, X_test, _, _, xs, ys, _, _, _, _, _, _, _ = initialize_gp()
    true_map_flat = load_true_map_flat(CSV_PATH, X_test)
    cx, cy = CENTER

    panels = []
    for z in ALTITUDES:
        img, extent = observed_image_for_altitude(cx, cy, z, xs, ys, true_map_flat)
        panels.append((z, img, extent))

    vmin = min(np.nanmin(img) for _, img, _ in panels)
    vmax = max(np.nanmax(img) for _, img, _ in panels)
    max_radius = fov_lateral_radius(ZMAX, ANGLE_OF_VIEW)
    half_window = max_radius + 4

    fig = plt.figure(figsize=(FIG_WIDTH, 6.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.4, wspace=0.12)
    ax_side = fig.add_subplot(gs[0, :])
    draw_side_view(ax_side)

    for col, ((z, img, extent), color) in enumerate(zip(panels, TIER_COLORS)):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(
            img, extent=extent, origin="lower", cmap=SEQUENTIAL_CMAP,
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        block = resolution_block_size(z)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(cx - half_window, cx + half_window)
        ax.set_ylim(cy - half_window, cy + half_window)
        ax.set_title(
            f"{z:.0f} m\nblock {block}x{block}",
            color=color, fontsize=14, fontweight="bold", linespacing=1.4,
        )

    mappable = plt.cm.ScalarMappable(cmap=SEQUENTIAL_CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=fig.get_axes()[1:], shrink=0.75, pad=0.02, location="right", aspect=20)
    cbar.set_label("Sensed value", color=INK_SECONDARY, fontsize=13)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=11)

    fig.suptitle("Resolution vs. altitude", color=INK_PRIMARY, fontsize=17, y=0.99)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
