"""
Top-down animation of Diffusionplanner_singlemap_headingremoved.py over a GRF
map, showing the FUTURE plan the planner has committed to (the remaining,
not-yet-executed portion of the current spline_path) rather than a trail of
past positions - same convention as record_cmaes_future_plan_animation.py and
record_realgreedy_future_plan_animation.py.

Uses the _headingremoved module rather than plain Diffusionplanner_singlemap:
that module's own live default checkpoint is
sparse_trans_waypoints_epoch_1500_headingremoved.pth, but its NoisePredictor
still has the heading_mlp input branch, so loading that checkpoint into it
fails with missing heading_mlp keys. The _headingremoved sibling module pairs
the same checkpoint with the matching (no heading-conditioning) architecture
and loads cleanly. Mirrors that module's own __main__ loop (3D waypoint/
dynamics from ts=0, no heading-velocity tracking - this architecture doesn't
condition on it). Background is the GP posterior mean field.
"""
import os

os.environ.setdefault("MAPTYPE", "grf")
os.environ.setdefault("SELECTED_MAP", "51")
os.environ.setdefault("TIMEALLOTED", "400")
os.environ.setdefault("WALLCLOCK_SECONDS", "0")
os.environ.setdefault("ENFORCE_MIN_STEP_TIME", "0")
os.environ.setdefault("SKIP_VIZ", "1")

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

import imageio_ffmpeg
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import Diffusionplanner_singlemap_headingremoved as sim
from gaussianprocesstraining import fov_lateral_radius, initialize_gp

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_MP4 = SCRIPT_DIR / f"diffusion_future_plan_map{sim.selected_map}_{sim.MAPTYPE}.mp4"
TITLE = "Diffusion Planner"

FRAME_STRIDE = 2
FPS = 20
PLAN_COLOR = "white"
STROKE = [pe.withStroke(linewidth=3.5, foreground="black")]


def build_true_map_flat(pts, X_test):
    true_map_flat = np.zeros(X_test.shape[0], dtype=float)
    for x_true, y_true, value in pts:
        idx = np.where(np.isclose(X_test[:, 0], x_true) & np.isclose(X_test[:, 1], y_true))[0]
        if idx.size:
            true_map_flat[idx[0]] = value
    return true_map_flat


def main():
    csv_path = Path(os.environ.get("CSV_DIR", str(SCRIPT_DIR / "csv")))
    data = np.loadtxt(csv_path / f"map_{sim.selected_map}_{sim.MAPTYPE}_grid_counts.csv", delimiter=",", skiprows=1)

    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()

    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]
    true_map_flat = build_true_map_flat(pts, X_test)

    rng = np.random.default_rng(sim.SENSORNOISE_SEED + sim.selected_map)
    mean = np.full(X_test.shape[0], sim.utility_threshold + 0.1)
    mu = mean.copy()
    P = cov.copy()

    print(f"Loading diffusion checkpoint: {sim.diffusion_path}")
    model = sim.load_diffusion_model(checkpoint_path=sim.diffusion_path)

    cx, cy, cz = sim.START_X, sim.START_Y, sim.INIT_ALTITUDE

    pose_frames = [(cx, cy, cz)]
    mu_frames = [mu.copy()]
    plan_frames = [[]]

    spline_path = []
    spline_idx = 0

    print("Running Diffusion Planner flight over GRF map (real sampling each replan)...")
    for ts in range(0, sim.timealloted):
        if ts <= 1:
            grad_x, grad_y, grad_z, _ = sim.waypoint_3d(
                cx, cy, cz, goal_x=80.0, goal_y=80.0, goal_z=sim.INIT_ALTITUDE, step=step
            )
            cx, cy, cz = sim.dynamics_3d(
                cx, cy, cz, grad_x, grad_y, grad_z, step, xmin, xmax, ymin, ymax, sim.ZMIN, sim.ZMAX, buffer=step * 2
            )
            future_plan = []
        else:
            if ts == 2 or spline_idx >= sim.execution_chunk or spline_idx >= len(spline_path):
                current_mean = mu.reshape(X.shape)
                current_var = np.diag(P).reshape(X.shape)
                dense_traj, sparse_controls = sim.sample_diffusion_trajectory(
                    model,
                    current_position=(cx, cy, cz),
                    current_mean=current_mean,
                    current_var=current_var,
                    grid_step=step,
                    bounds=(xmin, xmax, ymin, ymax, sim.ZMIN, sim.ZMAX),
                )
                spline_path = dense_traj.T.tolist()
                spline_idx = 0
                print(f"  Replanning at ts={ts}")

            if spline_idx < len(spline_path):
                goal_x, goal_y, goal_z = spline_path[spline_idx]
                grad_x, grad_y, grad_z, waypoint_reached = sim.waypoint_3d(
                    cx, cy, cz, goal_x=goal_x, goal_y=goal_y, goal_z=goal_z, step=step
                )
                if waypoint_reached:
                    spline_idx += 1
                    if spline_idx < len(spline_path):
                        goal_x, goal_y, goal_z = spline_path[spline_idx]
                        grad_x, grad_y, grad_z, _ = sim.waypoint_3d(
                            cx, cy, cz, goal_x=goal_x, goal_y=goal_y, goal_z=goal_z, step=step
                        )
                    else:
                        grad_x, grad_y, grad_z = 0.0, 0.0, 0.0

                if spline_idx < len(spline_path):
                    x_next, y_next, z_next = spline_path[spline_idx]
                    padding = step * 2
                    clamped_x = min(max(x_next, xmin + padding), xmax - padding)
                    clamped_y = min(max(y_next, ymin + padding), ymax - padding)
                    clamped_z = min(max(z_next, sim.ZMIN), sim.ZMAX)
                    if clamped_x != x_next or clamped_y != y_next or clamped_z != z_next:
                        spline_path[spline_idx] = [clamped_x, clamped_y, clamped_z]
            else:
                grad_x, grad_y, grad_z = 0.0, 0.0, 0.0

            cx, cy, cz = sim.dynamics_3d(
                cx, cy, cz, grad_x, grad_y, grad_z, step, xmin, xmax, ymin, ymax, sim.ZMIN, sim.ZMAX, buffer=step / 2
            )
            future_plan = [(x, y) for x, y, _ in spline_path[spline_idx:]]

        mu, P = sim.apply_measurement_update_3d(cx, cy, cz, mu, P, true_map_flat, xs, ys, rng)

        pose_frames.append((cx, cy, cz))
        mu_frames.append(mu.copy())
        plan_frames.append(future_plan)

    print(f"Captured {len(pose_frames)} frames.")

    pose_arr = np.asarray(pose_frames)
    mu_arr = np.asarray(mu_frames)
    ny, nx = len(ys), len(xs)
    true_grid = true_map_flat.reshape(ny, nx)

    frame_idx = list(range(0, len(pose_frames), FRAME_STRIDE))

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    vmin, vmax = true_grid.min(), true_grid.max()
    im = ax.imshow(
        mu_arr[0].reshape(ny, nx), origin="lower", cmap="viridis",
        extent=[xmin, xmax, ymin, ymax], vmin=vmin, vmax=vmax, interpolation="bicubic", zorder=1,
    )
    plan_line, = ax.plot([], [], color=PLAN_COLOR, linewidth=2.6, alpha=0.95, path_effects=STROKE, zorder=3)
    drone_dot = ax.scatter([], [], color="white", edgecolors="#ff5a4e", linewidths=2, s=170, zorder=4)
    footprint = Rectangle((0, 0), 0, 0, facecolor="none", edgecolor="white", linewidth=1.4, alpha=0.6, zorder=3)
    ax.add_patch(footprint)
    ax.set_title(TITLE, color="black", fontsize=17, pad=12)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)

    def update(frame_num):
        i = frame_idx[frame_num]
        im.set_data(mu_arr[i].reshape(ny, nx))
        cx_i, cy_i, cz_i = pose_arr[i]
        drone_dot.set_offsets([[cx_i, cy_i]])
        r = fov_lateral_radius(cz_i, 60)
        footprint.set_bounds(cx_i - r, cy_i - r, 2 * r, 2 * r)

        plan = plan_frames[i]
        if plan:
            plan_x = [cx_i] + [p[0] for p in plan]
            plan_y = [cy_i] + [p[1] for p in plan]
            plan_line.set_data(plan_x, plan_y)
        else:
            plan_line.set_data([], [])
        return im, plan_line, drone_dot, footprint

    anim = animation.FuncAnimation(fig, update, frames=len(frame_idx), interval=1000 / FPS, blit=False)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=4000)
    anim.save(str(OUT_MP4), writer=writer, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_MP4} ({len(frame_idx)} frames at {FPS}fps, ~{len(frame_idx)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
