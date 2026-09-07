"""
3D visualization of the fixed lawnmower (boustrophedon) coverage path on map
53, up to real wall-clock time 150s (not simulation timestep), with the
ground-truth map rendered as a colored surface on the z=0 floor beneath the
flight path.

Reads:
  - csv/map_53_grf_grid_counts.csv (ground truth field, 51x51 grid, 2m spacing)
  - Vizualization/lawnmower_map_grf_53_viz/map_53_executed_trajectory.csv
    (from lawnmower_singlemap.py, SELECTED_MAP=53 - this is
    lawnmower_singlemap.py's own default RESULTS_ROOT/output-dir naming when
    no RESULTS_ROOT/RUN_OUTPUT_TAG env override is given, so re-running the
    regeneration command below with no extra flags overwrites this same file
    in place)

IMPORTANT: the trajectory CSV's wall_time_seconds column only reflects real
flight time if the run that produced it had ENFORCE_MIN_STEP_TIME=1 (so each
step is floored to real flight speed, STEP_DISTANCE_METERS/FLIGHT_SPEED_MPS
per step - see lawnmower_singlemap.py). Regenerate with, e.g.:
  SELECTED_MAP=53 MAPTYPE=grf TIMEALLOTED=3000 WALLCLOCK_SECONDS=0 \
  ENFORCE_MIN_STEP_TIME=1 SKIP_VIZ=1 python lawnmower_singlemap.py
(TIMEALLOTED set generously high since the stopping condition here is wall
time, not timestep count; WALLCLOCK_SECONDS left unconstrained so the flight
itself isn't cut off before it reaches the coverage-path's own end. Currently
748 timesteps / ~250s wall-clock covers the full map at this pacing.)

Palette: dataviz skill's sequential blue ramp for the ground-truth field,
categorical slot 2 (orange) for the flight path so it reads clearly against
the blue floor.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

SCRIPT_DIR = Path(__file__).resolve().parent
MAP_ID = 53
MAX_WALLCLOCK_SECONDS = 250
GRID_CSV = SCRIPT_DIR / "csv" / f"map_{MAP_ID}_grf_grid_counts.csv"
TRAJ_CSV = SCRIPT_DIR / "Vizualization" / f"lawnmower_map_grf_{MAP_ID}_viz" / f"map_{MAP_ID}_executed_trajectory.csv"
OUT_PATH = SCRIPT_DIR / f"lawnmower_3d_map{MAP_ID}_wallclock{MAX_WALLCLOCK_SECONDS}.png"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
COLOR_PATH = "#eb6834"       # categorical slot 2 (orange) - flight path
COLOR_START = "#1baf7a"      # categorical slot 3 (aqua) - start marker
COLOR_END = "#e34948"        # categorical slot 8 (red) - end marker

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


def load_trajectory(path, max_wallclock_seconds):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    mask = data[:, 1] <= max_wallclock_seconds
    if mask.all():
        print(
            f"WARNING: every row has wall_time_seconds <= {max_wallclock_seconds} - "
            "the source trajectory likely wasn't generated with ENFORCE_MIN_STEP_TIME=1, "
            "so this wall-clock filter isn't doing anything meaningful. See the module "
            "docstring for the regeneration command."
        )
    data = data[mask]
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]  # timestep, wall_time_seconds, x, y, z


def main():
    X, Y, Z_field = load_ground_truth_grid(GRID_CSV)
    ts, wall_t, tx, ty, tz = load_trajectory(TRAJ_CSV, MAX_WALLCLOCK_SECONDS)
    print(
        f"Loaded {len(ts)} trajectory points (timestep 0-{int(ts[-1])}, "
        f"wall-clock 0-{wall_t[-1]:.1f}s), altitude range {tz.min():.1f}-{tz.max():.1f} m"
    )

    fig = plt.figure(figsize=(10, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    # mplot3d's default automatic depth-sorting is per-artist and unreliable
    # for a flat surface + thin line/markers sharing similar depth - it was
    # hiding the end-of-path marker behind the surface. Disabling it makes
    # rendering respect the explicit zorder values set below instead.
    ax.computed_zorder = False

    ax.plot_surface(
        X, Y, np.zeros_like(Z_field),
        facecolors=SEQUENTIAL_CMAP(Z_field),
        rstride=1, cstride=1, shade=False, antialiased=False, zorder=1,
    )

    ax.plot(tx, ty, tz, color=COLOR_PATH, linewidth=2.2, zorder=5, label="Lawnmower path")
    ax.scatter([tx[0]], [ty[0]], [tz[0]], color=COLOR_START, s=70, zorder=6,
               edgecolors=SURFACE, linewidths=1, label="Start (t=0)")
    ax.scatter([tx[-1]], [ty[-1]], [tz[-1]], color=COLOR_END, s=70, zorder=6,
               edgecolors=SURFACE, linewidths=1,
               label=f"End (wall-clock={wall_t[-1]:.0f}s, timestep={int(ts[-1])})")

    # Drop faint vertical guide lines from the path down to the floor so
    # altitude is legible against the flat ground-truth surface.
    for i in range(0, len(tx), 8):
        ax.plot([tx[i], tx[i]], [ty[i], ty[i]], [0, tz[i]],
                 color=INK_MUTED, linewidth=0.5, alpha=0.35, zorder=2)

    ax.set_xlabel("x (m)", color=INK_SECONDARY, labelpad=10)
    ax.set_ylabel("y (m)", color=INK_SECONDARY, labelpad=10)
    ax.set_zlabel("altitude z (m)", color=INK_SECONDARY, labelpad=6)
    ax.set_title(
        f"Fixed lawnmower coverage path - map {MAP_ID}, wall-clock 0-{MAX_WALLCLOCK_SECONDS}s",
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
