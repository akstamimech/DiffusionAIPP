"""
Top-down animation of a lawnmower_singlemap.py survey, current settings
(MAPTYPE stays whatever's live in that script - currently "halffield" - and
UTILITY_THRESHOLD/etc. are left at their defaults too). SELECTED_MAP is
overridden to 11: the script's own default (55) has no halffield ground
truth file on disk, so running its literal defaults together would crash -
11 is the map already used for the hero-shot figures elsewhere in this repo.

Reuses lawnmower_singlemap.py's actual functions (lawnmower_planner,
dynamics_3d, waypoint_3d, kalman_update, ...) via import, exactly like the
generate_hero_trajectory*.py scripts do - this only adds animation-frame
capture. WALLCLOCK_SECONDS/ENFORCE_MIN_STEP_TIME are disabled here (pure
real-time pacing for a live flight, not a "setting" that changes survey
behavior) so this runs at full speed instead of the ~150s the live-paced
version would take.

Captures only (x, y, z) and mu (the GP posterior mean) per timestep, not the
full P (2601x2601 per step would be tens of GB across a few hundred steps) -
mu is all a top-down "reconstructed map filling in" animation needs.
"""
import os

os.environ.setdefault("SELECTED_MAP", "11")
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
from matplotlib.patches import Rectangle

import imageio_ffmpeg
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import lawnmower_singlemap as sim
from gaussianprocesstraining import (
    build_sensor_matrix, fov_grid_points, fov_lateral_radius, noise_model, kalman_update,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "hero_shot_data"
OUT_DIR.mkdir(exist_ok=True)
OUT_MP4 = SCRIPT_DIR / f"lawnmower_topdown_survey_map{sim.selected_map}_{sim.MAPTYPE}.mp4"


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

    from gaussianprocesstraining import initialize_gp
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

    cx, cy, cz = 4.0, 4.0, sim.INIT_ALTITUDE
    sensor_footprint_radius = fov_lateral_radius(cz)

    pose_frames = [(cx, cy, cz)]
    mu_frames = [mu.copy()]

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

        if len(flight_plan) == 0 and ts > 2:
            print(f"Lawnmower sweep complete at ts={ts}")
            break

    print(f"Captured {len(pose_frames)} frames.")

    pose_arr = np.asarray(pose_frames)
    mu_arr = np.asarray(mu_frames)
    ny, nx = len(ys), len(xs)
    true_grid = true_map_flat.reshape(ny, nx)

    # --- Animation ---
    FRAME_STRIDE = 2  # every 2nd sim step -> smoother-feeling motion at a sane render time/file size
    frame_idx = list(range(0, len(pose_frames), FRAME_STRIDE))
    TRAIL_LEN = 40

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    vmin, vmax = true_grid.min(), true_grid.max()
    im = ax.imshow(
        mu_arr[0].reshape(ny, nx), origin="lower", cmap="viridis",
        extent=[xmin, xmax, ymin, ymax], vmin=vmin, vmax=vmax, interpolation="bicubic", zorder=1,
    )
    trail_line, = ax.plot([], [], color="#ff5a4e", linewidth=2.2, alpha=0.85, zorder=3)
    drone_dot = ax.scatter([], [], color="white", edgecolors="#ff5a4e", linewidths=2, s=170, zorder=4)
    footprint = Rectangle((0, 0), 0, 0, facecolor="none", edgecolor="white", linewidth=1.4, alpha=0.8, zorder=3)
    ax.add_patch(footprint)
    title = ax.set_title(
        f"Lawnmower survey — map {sim.selected_map} ({sim.MAPTYPE})",
        color="black", fontsize=15, pad=12,
    )

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
        lo = max(0, i - TRAIL_LEN)
        trail_line.set_data(pose_arr[lo:i + 1, 0], pose_arr[lo:i + 1, 1])
        return im, trail_line, drone_dot, footprint

    fps = 20
    anim = animation.FuncAnimation(fig, update, frames=len(frame_idx), interval=1000 / fps, blit=False)

    writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
    anim.save(str(OUT_MP4), writer=writer, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_MP4} ({len(frame_idx)} frames at {fps}fps, ~{len(frame_idx)/fps:.1f}s)")


if __name__ == "__main__":
    main()
