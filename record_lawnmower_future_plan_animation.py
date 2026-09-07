"""
Top-down animation of lawnmower_singlemap.py over a GRF map, showing the
FUTURE plan (the remaining, not-yet-visited sweep waypoints in flight_plan)
rather than a trail of past positions. Lawnmower has no replanning - the
flight_plan is computed once at ts=2 and only ever shrinks as waypoints are
reached, so "the future plan" here is just its remaining tail each frame.

Background is the GP posterior mean field, matching the established
convention for these presentation animations. Uses lawnmower_singlemap.py's
own simpler (uncorrelated) sensor noise model, matching that script exactly -
unlike CMAES_classic_singlemap.py/realgreedy_singlemap.py's correlated block
noise pipeline.
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

import lawnmower_singlemap as sim
from gaussianprocesstraining import (
    build_sensor_matrix, fov_grid_points, fov_lateral_radius, noise_model, kalman_update,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_MP4 = SCRIPT_DIR / f"lawnmower_future_plan_map{sim.selected_map}_{sim.MAPTYPE}.mp4"
TITLE = "Lawnmower Pattern"

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

    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = sim.initialize_gp()

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

    cx, cy, cz = 4.0, 4.0, sim.INIT_ALTITUDE
    sensor_footprint_radius = fov_lateral_radius(cz)

    pose_frames = [(cx, cy, cz)]
    mu_frames = [mu.copy()]
    plan_frames = [[]]

    flight_plan = []
    for ts in range(0, sim.timealloted):
        if ts <= 1:
            grad_x, grad_y, grad_z, _ = sim.waypoint_3d(
                cx, cy, cz, goal_x=80.0, goal_y=80.0, goal_z=sim.INIT_ALTITUDE, step=step
            )
            cx, cy, cz = sim.dynamics_3d(
                cx, cy, cz, grad_x, grad_y, grad_z, step, xmin, xmax, ymin, ymax, sim.ZMIN, sim.ZMAX, buffer=step * 2
            )
        else:
            if ts == 2:
                flight_plan = sim.lawnmower_planner(
                    cx, cy, xmin, xmax, ymin, ymax, step=step, fov_radius=sensor_footprint_radius,
                )
            if len(flight_plan) > 0:
                grad_x, grad_y, grad_z, waypoint_reached = sim.waypoint_3d(
                    cx, cy, cz, goal_x=flight_plan[0][0], goal_y=flight_plan[0][1], goal_z=sim.INIT_ALTITUDE, step=step
                )
                if waypoint_reached:
                    flight_plan.pop(0)
                    if len(flight_plan) > 0:
                        grad_x, grad_y, grad_z, _ = sim.waypoint_3d(
                            cx, cy, cz, goal_x=flight_plan[0][0], goal_y=flight_plan[0][1], goal_z=sim.INIT_ALTITUDE, step=step
                        )
                    else:
                        grad_x, grad_y, grad_z = 0.0, 0.0, 0.0
            else:
                grad_x, grad_y, grad_z = 0.0, 0.0, 0.0

            cx, cy, cz = sim.dynamics_3d(
                cx, cy, cz, grad_x, grad_y, grad_z, step, xmin, xmax, ymin, ymax, sim.ZMIN, sim.ZMAX, buffer=step * 2
            )

        fov = fov_grid_points(cx, cy, cz, xs, ys)
        sensor = build_sensor_matrix(fov, cz, xs, ys)
        R = noise_model(cz)
        z_meas = sensor @ true_map_flat
        z_meas += rng.normal(0, np.sqrt(R), size=sensor.shape[0])
        mu, P = kalman_update(mu, P, sensor, z_meas, R)

        pose_frames.append((cx, cy, cz))
        mu_frames.append(mu.copy())
        plan_frames.append(list(flight_plan))

        if len(flight_plan) == 0 and ts > 2:
            print(f"Lawnmower sweep complete at ts={ts}")
            break

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
    plan_line, = ax.plot([], [], color=PLAN_COLOR, linewidth=2.2, alpha=0.95, path_effects=STROKE, zorder=3)
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
        r = fov_lateral_radius(cz_i)
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
