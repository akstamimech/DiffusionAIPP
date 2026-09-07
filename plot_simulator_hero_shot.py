"""
A single "hero shot" of the simulator environment: the real ground truth
terrain (map 11, NAIP) with a freshly generated, longer CMA-ES flight
(generate_hero_trajectory.py's output, hero_shot_data/ - a real run of
CMAES_classic_singlemap.py's own planner/dynamics, not synthetic; replaces the
first attempt's 105-timestep run, which was cut short by the default
WALLCLOCK_SECONDS=50 budget and read as two disconnected loops rather than a
real coverage mission) swooping over it, altitude varying freely, with the
sensor's actual FOV footprint (fov_lateral_radius, gaussianprocesstraining.py)
dropped onto the terrain at a few sampled points, and the actual 3D control
waypoints CMA-ES chose at each replan highlighted along the path.

Styling matches plot_cmaes_3d_map53.py (sequential blue terrain, categorical
path/start/end colors, Gaussian-smoothed path for visual clarity - see that
script's own docstring for why the smoothing is cosmetic only, not a
reconstruction of the unsnapped CMA-ES spline).
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter1d

from gaussianprocesstraining import fov_lateral_radius

SCRIPT_DIR = Path(__file__).resolve().parent
MAP_ID = 11
MAPTYPE = "NAIP"
GRID_CSV = SCRIPT_DIR / "csv" / f"map_{MAP_ID}_{MAPTYPE}_grid_counts.csv"
TRAJ_CSV = SCRIPT_DIR / "hero_shot_data" / f"map_{MAP_ID}_hero_trajectory.csv"
WAYPOINTS_CSV = SCRIPT_DIR / "hero_shot_data" / f"map_{MAP_ID}_hero_control_waypoints.csv"
OUT_PATH = SCRIPT_DIR / f"simulator_hero_shot_map{MAP_ID}_{MAPTYPE}.png"

SMOOTH_SIGMA = 1.4
ANGLE_OF_VIEW = 60.0
N_FOOTPRINTS = 5  # sampled evenly along the flight, incl. start and end

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
COLOR_PATH = "#eb6834"       # categorical slot 2 (orange) - flight path
COLOR_START = "#1baf7a"      # categorical slot 3 (aqua) - start marker
COLOR_END = "#e34948"        # categorical slot 8 (red) - end marker
COLOR_RAW = "#c3c2b7"
COLOR_WAYPOINT = "#4a3aa7"   # categorical slot 7 (violet) - control waypoints
COLOR_FOOTPRINT_EDGE = "#c98500"   # warm gold - sensor "viewfinder" highlight
COLOR_FOOTPRINT_FACE = "#fff0c2"

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


def load_trajectory(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]  # ts, x, y, z


def load_waypoints(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 2], data[:, 3], data[:, 4]  # x, y, z


def add_footprint(ax, cx, cy, cz, angle_of_view=ANGLE_OF_VIEW, zorder=4):
    radius = fov_lateral_radius(cz, angle_of_view)
    corners = np.array([
        [cx - radius, cy - radius, 0.05],
        [cx + radius, cy - radius, 0.05],
        [cx + radius, cy + radius, 0.05],
        [cx - radius, cy + radius, 0.05],
    ])
    quad = Poly3DCollection(
        [corners], facecolor=COLOR_FOOTPRINT_FACE, edgecolor=COLOR_FOOTPRINT_EDGE,
        linewidth=1.6, alpha=0.55, zorder=zorder,
    )
    ax.add_collection3d(quad)
    ax.plot([cx, cx], [cy, cy], [0.05, cz], color=COLOR_FOOTPRINT_EDGE,
            linewidth=1.1, linestyle=(0, (3, 2)), alpha=0.8, zorder=zorder)


def main():
    X, Y, Z_field = load_ground_truth_grid(GRID_CSV)
    ts, tx, ty, tz = load_trajectory(TRAJ_CSV)
    wx, wy, wz = load_waypoints(WAYPOINTS_CSV)
    print(
        f"Loaded {len(ts)} trajectory points (timestep 0-{int(ts[-1])}), "
        f"altitude range {tz.min():.1f}-{tz.max():.1f} m, {len(wx)} control waypoints"
    )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    vmin, vmax = float(np.nanmin(Z_field)), float(np.nanmax(Z_field))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    ax.plot_surface(
        X, Y, np.zeros_like(Z_field),
        facecolors=SEQUENTIAL_CMAP(norm(Z_field)),
        rstride=1, cstride=1, shade=False, antialiased=False, zorder=1,
    )

    plot_x = gaussian_filter1d(tx, sigma=SMOOTH_SIGMA)
    plot_y = gaussian_filter1d(ty, sigma=SMOOTH_SIGMA)
    plot_z = gaussian_filter1d(tz, sigma=SMOOTH_SIGMA)
    plot_x[0], plot_y[0], plot_z[0] = tx[0], ty[0], tz[0]
    plot_x[-1], plot_y[-1], plot_z[-1] = tx[-1], ty[-1], tz[-1]
    ax.scatter(tx, ty, tz, color=COLOR_RAW, s=4, alpha=0.3, linewidths=0, zorder=3)

    for i in range(0, len(plot_x), 8):
        ax.plot([plot_x[i], plot_x[i]], [plot_y[i], plot_y[i]], [0, plot_z[i]],
                 color=INK_MUTED, linewidth=0.5, alpha=0.3, zorder=2)

    ax.plot(plot_x, plot_y, plot_z, color=COLOR_PATH, linewidth=2.6, zorder=5, label="Flight path")

    ax.scatter(
        wx, wy, wz, color=COLOR_WAYPOINT, s=55, marker="D", zorder=6,
        edgecolors=SURFACE, linewidths=0.8, label="CMA-ES control waypoints",
    )

    footprint_idx = np.linspace(0, len(tx) - 1, N_FOOTPRINTS, dtype=int)
    for i in footprint_idx:
        add_footprint(ax, tx[i], ty[i], tz[i])

    ax.scatter([tx[0]], [ty[0]], [tz[0]], color=COLOR_START, s=100, zorder=6,
               edgecolors=SURFACE, linewidths=1.2, label="Start")
    ax.scatter([tx[-1]], [ty[-1]], [tz[-1]], color=COLOR_END, s=100, zorder=6,
               edgecolors=SURFACE, linewidths=1.2, label="End")

    ax.set_zlim(0, max(45, tz.max() + 5))
    ax.view_init(elev=25, azim=-55)
    ax.set_axis_off()

    ax.legend(
        loc="upper left", frameon=True, facecolor=SURFACE, edgecolor=INK_MUTED,
        framealpha=0.9, fontsize=13, labelcolor=INK_SECONDARY, markerscale=1.3,
    )

    ax.set_box_aspect(None, zoom=1.3)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
