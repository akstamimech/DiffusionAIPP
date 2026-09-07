"""
3D visualization of the pyramid candidate-waypoint lattice
(build_pyramid_lattice_3d, gaussianprocesstraining.py) over a NAIP ground-truth
landscape. This is the actual candidate set real_receding_horizon_planner's
grid_search_3d step searches over every replan (CMAES_classic_singlemap.py),
reused directly here rather than reconstructed by hand.

Map 11 is one of the "moderate" NAIP maps (~20-50% of cells below the 0.3
importance threshold - a genuine search/discrimination problem, not a trivial
near-100%-important map). See CARRYOVER_NAIP_SWITCH_AND_BASELINES.md sec 1.4.

Reads: csv/map_11_NAIP_grid_counts.csv, masked to the same 2m-step grid
CMAES_classic_singlemap.py itself filters down to before building true_map_flat.

ZMIN/ZMAX/step below mirror CMAES_classic_singlemap.py's current module-level
constants (10.0/40.0/2.0) - not re-imported directly, since importing that
script's module also imports torch/evalmetrics for its guarded __main__ block.

Palette: dataviz skill's sequential blue ramp for the ground-truth floor
(matching plot_cmaes_3d_map53.py's convention), categorical slots 2/3/7
(orange/aqua/violet - chosen to skip yellow, which the palette's own notes
flag as a poor neighbor for orange) for the lattice's low/middle/top tiers.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

from gaussianprocesstraining import initialize_gp, build_pyramid_lattice_3d

SCRIPT_DIR = Path(__file__).resolve().parent
MAP_ID = 11
MAPTYPE = "NAIP"
ZMIN, ZMAX = 10.0, 40.0
GRID_CSV = SCRIPT_DIR / "csv" / f"map_{MAP_ID}_{MAPTYPE}_grid_counts.csv"
OUT_PATH = SCRIPT_DIR / f"lattice_over_map_{MAP_ID}_{MAPTYPE}.png"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
COLOR_LOW = "#eb6834"   # categorical slot 2 (orange) - low tier, densest (9x9)
COLOR_MID = "#1baf7a"   # categorical slot 3 (aqua) - middle tier (3x3)
COLOR_TOP = "#4a3aa7"   # categorical slot 7 (violet) - top tier (2x2)

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


def load_ground_truth_grid(path, step=2.0):
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
    X, Y = np.meshgrid(xs, ys)
    return X, Y, grid


def main():
    X, Y, Z_field = load_ground_truth_grid(GRID_CSV)
    vmin, vmax = float(np.nanmin(Z_field)), float(np.nanmax(Z_field))

    _, _, _, _, xs, ys, _, _, _, _, _, _, _ = initialize_gp()
    lattice = np.asarray(build_pyramid_lattice_3d(xs, ys, ZMIN, ZMAX))
    print(f"Lattice has {len(lattice)} candidate waypoints")

    z_layers = sorted(set(np.round(lattice[:, 2], 6).tolist()))
    tier_style = [
        (COLOR_LOW, "Low tier"),
        (COLOR_MID, "Middle tier"),
        (COLOR_TOP, "Top tier"),
    ]

    fig = plt.figure(figsize=(10, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    # Disable mplot3d's per-artist auto depth-sort (unreliable for a flat
    # surface + scattered points/lines at similar depth) so explicit zorder
    # values below are respected instead.
    ax.computed_zorder = False

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    ax.plot_surface(
        X, Y, np.zeros_like(Z_field),
        facecolors=SEQUENTIAL_CMAP(norm(Z_field)),
        rstride=1, cstride=1, shade=False, antialiased=False, zorder=1,
    )

    for z_val, (color, label) in zip(z_layers, tier_style):
        tier_pts = lattice[np.isclose(lattice[:, 2], z_val)]
        for x, y, z in tier_pts:
            ax.plot([x, x], [y, y], [0, z], color=INK_MUTED, linewidth=0.5, alpha=0.3, zorder=2)
        ax.scatter(
            tier_pts[:, 0], tier_pts[:, 1], tier_pts[:, 2],
            color=color, s=42, zorder=5, edgecolors=SURFACE, linewidths=0.6,
            label=f"{label} (z={z_val:.0f}m, n={len(tier_pts)})",
        )

    ax.set_xlabel("x (m)", color=INK_SECONDARY, labelpad=10)
    ax.set_ylabel("y (m)", color=INK_SECONDARY, labelpad=10)
    ax.set_zlabel("altitude z (m)", color=INK_SECONDARY, labelpad=6)
    ax.set_title(
        f"Pyramid candidate-waypoint lattice over map",
        color=INK_PRIMARY, fontsize=13, pad=14,
    )
    ax.set_zlim(0, ZMAX + 5)
    ax.view_init(elev=26, azim=-58)
    ax.xaxis.pane.set_facecolor(SURFACE)
    ax.yaxis.pane.set_facecolor(SURFACE)
    ax.zaxis.pane.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED)

    mappable = plt.cm.ScalarMappable(cmap=SEQUENTIAL_CMAP, norm=norm)
    mappable.set_array([])
    # cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.02, location="left")
    # cbar.set_label("Ground-truth map value", color=INK_SECONDARY)
    # cbar.ax.tick_params(colors=INK_MUTED)

    ax.legend(
        loc="upper left", frameon=True, facecolor=SURFACE, edgecolor=INK_MUTED,
        framealpha=0.92, fontsize=15, labelcolor=INK_SECONDARY,
        markerscale=1.8, handletextpad=0.6, borderpad=0.9, labelspacing=0.7,
    )

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=170)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
