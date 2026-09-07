"""
3D visualization of the greedy one-step-horizon expert planner's path on map
53, up to simulation timestep 747 (matched to plot_lawnmower_3d_map53.py and
plot_cmaes_3d_map53.py's range), with the ground-truth map rendered as a
colored surface on the z=0 floor beneath the flight path. Third of a matched
three-figure baseline comparison set (fixed coverage vs. myopic one-step
expert vs. full 8-waypoint expert).

Reads:
  - csv/map_53_grf_grid_counts.csv (ground truth field, 51x51 grid, 2m spacing)
  - Vizualization/greedy_map_grf_53_viz/map_53_executed_trajectory.csv
    (from realgreedy_singlemap.py, SELECTED_MAP=53, default PLANNING_HORIZON=1,
    EXECUTION_CHUNK=5, same beta/step-size CMA-ES-refinement settings as the
    full expert planner - it still calls cma_es_refine_waypoints_3d, just on a
    single waypoint instead of an eight-point spline)

Like CMA-ES and unlike the lawnmower (fixed altitude, z=10 throughout), the
greedy planner's altitude is a real free decision variable.

NOTE on grid-snapping: dynamics_3d (shared with CMAES_classic_singlemap.py
via realgreedy_singlemap.py's own import) rounds (x,y,z) to the nearest
`step`=2m grid point every single timestep, even while chasing a continuous
target - this is real simulated behavior, baked into executed_trajectory.csv
itself, not a plotting artifact. SMOOTH_SIGMA below applies a Gaussian
smoothing filter directly to the recorded (snapped) x/y/z sequences purely
for visual clarity - an approximation of the executed path, not a
reconstruction of the original per-replan plan. Set to 0 to disable and plot
the raw snapped positions as-is.

Palette: dataviz skill's sequential blue ramp for the ground-truth field,
categorical slot 2 (orange) for the flight path so it reads clearly against
the blue floor - identical styling to the lawnmower/CMA-ES companion figures
so the three read as a matched set.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from scipy.ndimage import gaussian_filter1d

SCRIPT_DIR = Path(__file__).resolve().parent
MAP_ID = 53
MAX_TIMESTEP = 747
SMOOTH_SIGMA = 1.5  # in samples (timesteps); 0 disables smoothing entirely
SHOW_RAW_POINTS = True  # overlay the true (snapped) points faintly behind the smoothed line
GRID_CSV = SCRIPT_DIR / "csv" / f"map_{MAP_ID}_grf_grid_counts.csv"
TRAJ_CSV = SCRIPT_DIR / "Vizualization" / f"greedy_map_grf_{MAP_ID}_viz" / f"map_{MAP_ID}_executed_trajectory.csv"
OUT_PATH = SCRIPT_DIR / f"greedy_3d_map{MAP_ID}_timestep{MAX_TIMESTEP}.png"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
COLOR_PATH = "#eb6834"       # categorical slot 2 (orange) - flight path
COLOR_START = "#1baf7a"      # categorical slot 3 (aqua) - start marker
COLOR_END = "#e34948"        # categorical slot 8 (red) - end marker
COLOR_RAW = "#c3c2b7"        # muted axis gray - faint raw (grid-snapped) points

# Sequential blue ramp (steps 100->700) for the ground-truth field surface.
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


def load_ground_truth_grid(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    xs = np.unique(data[:, 0])
    ys = np.unique(data[:, 1])
    n = len(xs)
    grid = np.full((len(ys), n), np.nan)
    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}
    for x, y, val in data:
        grid[y_idx[y], x_idx[x]] = val
    X, Y = np.meshgrid(xs, ys)
    return X, Y, grid


def load_trajectory(path, max_timestep):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    mask = data[:, 0] <= max_timestep
    data = data[mask]
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]  # timestep, wall_time_seconds, x, y, z


def main():
    X, Y, Z_field = load_ground_truth_grid(GRID_CSV)
    ts, wall_t, tx, ty, tz = load_trajectory(TRAJ_CSV, MAX_TIMESTEP)
    print(
        f"Loaded {len(ts)} trajectory points (timestep 0-{int(ts[-1])}, "
        f"wall-clock 0-{wall_t[-1]:.1f}s), altitude range {tz.min():.1f}-{tz.max():.1f} m"
    )

    fig = plt.figure(figsize=(10, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    # mplot3d's default automatic depth-sorting is per-artist and unreliable
    # for a flat surface + thin line/markers sharing similar depth - disabling
    # it makes rendering respect the explicit zorder values set below instead.
    ax.computed_zorder = False

    ax.plot_surface(
        X, Y, np.zeros_like(Z_field),
        facecolors=SEQUENTIAL_CMAP(Z_field),
        rstride=1, cstride=1, shade=False, antialiased=False, zorder=1,
    )

    if SMOOTH_SIGMA > 0:
        plot_x = gaussian_filter1d(tx, sigma=SMOOTH_SIGMA)
        plot_y = gaussian_filter1d(ty, sigma=SMOOTH_SIGMA)
        plot_z = gaussian_filter1d(tz, sigma=SMOOTH_SIGMA)
        # Anchor the smoothed curve's endpoints back to the true recorded
        # start/end positions - Gaussian smoothing pulls interior points
        # toward their neighbors' mean, which also drags the very first/last
        # samples slightly off their real values.
        plot_x[0], plot_y[0], plot_z[0] = tx[0], ty[0], tz[0]
        plot_x[-1], plot_y[-1], plot_z[-1] = tx[-1], ty[-1], tz[-1]
        if SHOW_RAW_POINTS:
            ax.scatter(tx, ty, tz, color=COLOR_RAW, s=4, alpha=0.35, linewidths=0, zorder=3,
                       label="Raw grid-snapped positions")
    else:
        plot_x, plot_y, plot_z = tx, ty, tz

    ax.plot(plot_x, plot_y, plot_z, color=COLOR_PATH, linewidth=2.2, zorder=5, label="Greedy expert path")
    ax.scatter([tx[0]], [ty[0]], [tz[0]], color=COLOR_START, s=70, zorder=6,
               edgecolors=SURFACE, linewidths=1, label="Start (t=0)")
    ax.scatter([tx[-1]], [ty[-1]], [tz[-1]], color=COLOR_END, s=70, zorder=6,
               edgecolors=SURFACE, linewidths=1,
               label=f"End (timestep={int(ts[-1])}, wall-clock={wall_t[-1]:.0f}s)")

    # Drop faint vertical guide lines from the (smoothed) path down to the
    # floor so altitude is legible against the flat ground-truth surface.
    for i in range(0, len(plot_x), 8):
        ax.plot([plot_x[i], plot_x[i]], [plot_y[i], plot_y[i]], [0, plot_z[i]],
                 color=INK_MUTED, linewidth=0.5, alpha=0.35, zorder=2)

    ax.set_xlabel("x (m)", color=INK_SECONDARY, labelpad=10)
    ax.set_ylabel("y (m)", color=INK_SECONDARY, labelpad=10)
    ax.set_zlabel("altitude z (m)", color=INK_SECONDARY, labelpad=6)
    smoothing_note = " (Gaussian-smoothed for display)" if SMOOTH_SIGMA > 0 else ""
    ax.set_title(
        f"Greedy one-step expert path - map {MAP_ID}, timesteps 0-{MAX_TIMESTEP}{smoothing_note}",
        color=INK_PRIMARY, fontsize=13, pad=14,
    )
    ax.set_zlim(0, max(45, tz.max() + 5))
    ax.view_init(elev=28, azim=-58)
    ax.xaxis.pane.set_facecolor(SURFACE)
    ax.yaxis.pane.set_facecolor(SURFACE)
    ax.zaxis.pane.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED)

    mappable = plt.cm.ScalarMappable(cmap=SEQUENTIAL_CMAP, norm=plt.Normalize(vmin=0, vmax=1))
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.02, location="left")
    cbar.set_label("Ground-truth map value", color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_MUTED)

    ax.legend(loc="upper left", frameon=False, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=170)
    print(f"Wrote {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
