"""
4-panel top-down progression figure for the paper: GP belief mean + flown-so-far trajectory,
snapshotted at ~4 evenly-spaced points across a WALLCLOCK_SECONDS=40 diffusion-planner mission,
with the >UTILITY_THRESHOLD (0.5, UCB/grf) region contoured.

Reuses Diffusionplanner_singlemap.py's own functions/constants directly (imported, not
reimplemented) and replicates its __main__ simulation loop verbatim, since that loop is only
guarded by `if __name__ == "__main__":` and isn't itself a reusable function. Env vars are set
before the import so the module's own constants (SELECTED_MAP, MAPTYPE, WALLCLOCK_SECONDS,
ENFORCE_MIN_STEP_TIME) pick them up exactly as a normal run would.

Compressed-for-print styling: no axes/ticks/colorbar, just the belief-mean field, the trajectory,
a >0.5 contour, and a small per-panel time label.

Set REPLOT_ONLY=1 to skip re-running the ~40s simulation and just re-render the figure from the
last run's cached snapshots (CACHE_PATH) - use this for any purely cosmetic plotting tweak.
"""
import os

os.environ.setdefault("SELECTED_MAP", "51")
os.environ.setdefault("MAPTYPE", "grf")
os.environ.setdefault("WALLCLOCK_SECONDS", "40")
os.environ.setdefault("ENFORCE_MIN_STEP_TIME", "0")
os.environ.setdefault("SKIP_VIZ", "1")

import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

# Computer-Modern mathtext: matplotlib's built-in stand-in for real LaTeX math fonts, no LaTeX
# install required (there isn't one on this machine - no latex/pdflatex/xelatex on PATH).
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

import Diffusionplanner_singlemap as dp

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PATH = SCRIPT_DIR / f"diffusion_belief_progression_map{dp.selected_map}_{dp.MAPTYPE}.png"
CACHE_PATH = SCRIPT_DIR / f"diffusion_belief_progression_map{dp.selected_map}_{dp.MAPTYPE}_cache.npz"
SNAPSHOT_TIMES = [5.0, 10.0, 20.0]  # wall-clock seconds into the mission
REPLOT_ONLY = os.environ.get("REPLOT_ONLY", "0") == "1"

SURFACE = "#fcfcfb"
COLOR_PATH = "#eb6834"       # categorical slot 2 (orange)
COLOR_START = "#1baf7a"      # categorical slot 3 (aqua)
COLOR_NOW = "#e34948"        # categorical slot 8 (red)

BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)


def run_simulation():
    print(f"Loading diffusion checkpoint: {dp.diffusion_path}", flush=True)
    diffusion_model = dp.load_diffusion_model(checkpoint_path=dp.diffusion_path)

    csv_path = Path(os.environ.get("CSV_DIR", str(SCRIPT_DIR / "csv")))
    data = np.loadtxt(
        csv_path / f"map_{dp.selected_map}_{dp.MAPTYPE}_grid_counts.csv",
        delimiter=",", skiprows=1,
    )
    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = dp.initialize_gp()

    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]
    true_map_flat = dp.build_true_map_flat(pts, X_test)

    rng = np.random.default_rng(dp.SENSORNOISE_SEED + dp.selected_map)

    mean = np.full(X_test.shape[0], dp.utility_threshold + 0.1)
    mu = mean.copy()
    P = cov.copy()

    cx, cy, cz = dp.START_X, dp.START_Y, dp.INIT_ALTITUDE
    grad_x, grad_y, grad_z = 0.0, 0.0, 0.0
    current_heading_velocity = np.zeros(3, dtype=np.float32)
    pos_history = [(cx, cy)]

    spline_path, control_waypoints = [], []
    spline_idx = 0

    snapshots = []
    next_snapshot_idx = 0

    flight_start_time = time.time()
    for ts in range(0, dp.timealloted):
        elapsed = time.time() - flight_start_time
        if elapsed >= dp.WALLCLOCK_SECONDS:
            print(f"Wall-clock budget reached at timestep {ts}", flush=True)
            break
        if next_snapshot_idx >= len(SNAPSHOT_TIMES):
            print(f"All {len(SNAPSHOT_TIMES)} snapshots captured at timestep {ts}; stopping early", flush=True)
            break

        if ts <= 1:
            grad_x, grad_y, grad_z, _ = dp.waypoint_3d(
                cx, cy, cz, goal_x=80.0, goal_y=80.0, goal_z=dp.INIT_ALTITUDE, step=step,
            )
            previous_pose = np.array([cx, cy, cz], dtype=np.float32)
            cx, cy, cz = dp.dynamics_3d(
                cx, cy, cz, grad_x, grad_y, grad_z, step, xmin, xmax, ymin, ymax,
                dp.ZMIN, dp.ZMAX, buffer=step * 2,
            )
            current_heading_velocity = np.array([cx, cy, cz], dtype=np.float32) - previous_pose
            pos_history.append((cx, cy))
        else:
            if ts == 2 or spline_idx >= dp.execution_chunk or spline_idx >= len(spline_path):
                current_mean = mu.reshape(X.shape)
                current_var = np.diag(P).reshape(X.shape)
                dense_traj, sparse_controls = dp.sample_diffusion_trajectory(
                    diffusion_model,
                    current_position=(cx, cy, cz),
                    current_mean=current_mean,
                    current_var=current_var,
                    current_heading_velocity=current_heading_velocity,
                    grid_step=step,
                    bounds=(xmin, xmax, ymin, ymax, dp.ZMIN, dp.ZMAX),
                )
                spline_path = dense_traj.T.tolist()
                spline_idx = 0
                control_waypoints = sparse_controls.T.tolist()
                print(f"ts={ts}: replanned ({elapsed:.1f}s elapsed)", flush=True)

            if spline_idx < len(spline_path):
                goal_x, goal_y, goal_z = spline_path[spline_idx]
                grad_x, grad_y, grad_z, waypoint_reached = dp.waypoint_3d(
                    cx, cy, cz, goal_x=goal_x, goal_y=goal_y, goal_z=goal_z, step=step,
                )
                if waypoint_reached:
                    spline_idx += 1
                    if spline_idx < len(spline_path):
                        goal_x, goal_y, goal_z = spline_path[spline_idx]
                        grad_x, grad_y, grad_z, _ = dp.waypoint_3d(
                            cx, cy, cz, goal_x=goal_x, goal_y=goal_y, goal_z=goal_z, step=step,
                        )
                    else:
                        grad_x, grad_y, grad_z = 0.0, 0.0, 0.0

                previous_pose = np.array([cx, cy, cz], dtype=np.float32)
                cx, cy, cz = dp.dynamics_3d(
                    cx, cy, cz, grad_x, grad_y, grad_z, step, xmin, xmax, ymin, ymax,
                    dp.ZMIN, dp.ZMAX, buffer=step / 2,
                )
                current_heading_velocity = np.array([cx, cy, cz], dtype=np.float32) - previous_pose
                pos_history.append((cx, cy))
            else:
                pos_history.append((cx, cy))

        mu, P = dp.apply_measurement_update_3d(cx, cy, cz, mu, P, true_map_flat, xs, ys, rng)

        elapsed_after = time.time() - flight_start_time
        while next_snapshot_idx < len(SNAPSHOT_TIMES) and elapsed_after >= SNAPSHOT_TIMES[next_snapshot_idx]:
            snapshots.append({
                "time": elapsed_after,
                "mu": mu.copy(),
                "traj": np.asarray(pos_history, dtype=float),
            })
            print(f"  snapshot {next_snapshot_idx} captured at {elapsed_after:.1f}s (timestep {ts})", flush=True)
            next_snapshot_idx += 1

    while len(snapshots) < len(SNAPSHOT_TIMES):
        snapshots.append({
            "time": time.time() - flight_start_time,
            "mu": mu.copy(),
            "traj": np.asarray(pos_history, dtype=float),
        })

    grid = {"X": X, "Y": Y, "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "mean": mean}
    save_cache(snapshots, grid)
    return snapshots, grid


def save_cache(snapshots, grid):
    payload = {"mean": grid["mean"], "X": grid["X"], "Y": grid["Y"],
               "xmin": grid["xmin"], "xmax": grid["xmax"], "ymin": grid["ymin"], "ymax": grid["ymax"]}
    for i, snap in enumerate(snapshots):
        payload[f"time_{i}"] = snap["time"]
        payload[f"mu_{i}"] = snap["mu"]
        payload[f"traj_{i}"] = snap["traj"]
    np.savez(CACHE_PATH, **payload)
    print(f"Cached snapshots to {CACHE_PATH}")


def load_cache():
    payload = np.load(CACHE_PATH)
    grid = {"X": payload["X"], "Y": payload["Y"], "xmin": float(payload["xmin"]),
            "xmax": float(payload["xmax"]), "ymin": float(payload["ymin"]), "ymax": float(payload["ymax"]),
            "mean": payload["mean"]}
    snapshots = []
    i = 0
    while f"time_{i}" in payload:
        snapshots.append({
            "time": float(payload[f"time_{i}"]),
            "mu": payload[f"mu_{i}"],
            "traj": payload[f"traj_{i}"],
        })
        i += 1
    return snapshots, grid


def render_figure(snapshots, grid):
    X, Y = grid["X"], grid["Y"]
    xmin, xmax, ymin, ymax = grid["xmin"], grid["xmax"], grid["ymin"], grid["ymax"]
    mean = grid["mean"]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    fig.patch.set_facecolor(SURFACE)
    vmin = min(float(np.min(mean)), float(min(np.min(s["mu"]) for s in snapshots)))
    vmax = float(max(np.max(s["mu"]) for s in snapshots))

    # Leave a real gap between panels for the z_t transition arrows drawn afterward.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.83, bottom=0.02, wspace=0.45)

    for ax, snap in zip(axes, snapshots):
        mu_grid = snap["mu"].reshape(X.shape)
        ax.imshow(
            mu_grid, extent=[xmin, xmax, ymin, ymax], origin="lower",
            cmap=SEQUENTIAL_CMAP, vmin=vmin, vmax=vmax, aspect="equal",
        )
        contour_set = ax.contour(
            X, Y, mu_grid, levels=[dp.utility_threshold],
            colors="white", linewidths=1.6,
        )
        collections = contour_set.collections if hasattr(contour_set, "collections") else [contour_set]
        for collection in collections:
            collection.set_path_effects([pe.withStroke(linewidth=3.0, foreground="black")])

        traj = np.asarray(snap["traj"], dtype=float)
        ax.plot(traj[:, 0], traj[:, 1], color=COLOR_PATH, linewidth=1.8, zorder=5)
        ax.scatter([traj[0, 0]], [traj[0, 1]], color=COLOR_START, s=26, zorder=6, edgecolors=SURFACE, linewidths=0.6)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], color=COLOR_NOW, s=26, zorder=6, edgecolors=SURFACE, linewidths=0.6)

        # Label the trajectory itself, anchored near its midpoint so it reads as pointing at the
        # orange line rather than floating in empty space. Offset direction adapts to which
        # quadrant the anchor falls in, so the label always grows back toward the panel interior
        # instead of running off the edge.
        mid_idx = len(traj) // 2
        anchor_x, anchor_y = traj[mid_idx, 0], traj[mid_idx, 1]
        x_frac = (anchor_x - xmin) / (xmax - xmin)
        y_frac = (anchor_y - ymin) / (ymax - ymin)
        dx = 10 if x_frac < 0.5 else -10
        dy = 10 if y_frac < 0.5 else -10
        ax.annotate(
            r"$\psi^{*}_t$",
            xy=(anchor_x, anchor_y), xycoords="data",
            xytext=(dx, dy), textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            fontsize=12, color="white", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
            annotation_clip=False,
            zorder=7,
        )

        ax.set_title(
            rf"$b_{{{int(round(snap['time']))}}}$",
            fontsize=15, pad=10, color="#0b0b0b",
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis("off")

    # z_t transition arrows drawn in figure coordinates, in the gaps left by wspace above - the
    # arrow itself is the actual LaTeX \longrightarrow glyph (via mathtext), not a drawn patch, so
    # it matches real LaTeX arrow weight/shape instead of looking like a matplotlib annotation.
    positions = [ax.get_position() for ax in axes]
    arrow_y = np.mean([positions[0].y0, positions[0].y1])
    for left_pos, right_pos in zip(positions[:-1], positions[1:]):
        x_mid = (left_pos.x1 + right_pos.x0) / 2
        fig.text(
            x_mid, arrow_y, r"$\longrightarrow$",
            ha="center", va="center", fontsize=22, color="#0b0b0b",
        )
        fig.text(
            x_mid, arrow_y + 0.07, r"$z_t$",
            ha="center", va="bottom", fontsize=13, color="#0b0b0b",
        )

    fig.savefig(OUT_PATH, dpi=220, facecolor=SURFACE)
    print(f"Wrote {OUT_PATH}")


def main():
    if REPLOT_ONLY and CACHE_PATH.exists():
        print(f"REPLOT_ONLY=1: loading cached snapshots from {CACHE_PATH}")
        snapshots, grid = load_cache()
    else:
        snapshots, grid = run_simulation()
    render_figure(snapshots, grid)


if __name__ == "__main__":
    main()
