"""
Diffusion counterpart of generate_hero_trajectory.py: a fresh, longer
Diffusion-planner flight over map 11 (NAIP), for
plot_simulator_hero_shot_diffusion.py. Same map, same ground truth, same
sensor-noise seed, same unconstrained-wallclock/longer-TIMEALLOTED treatment
as the CMA-ES version - only the planner differs, so the two hero shots are
a fair side-by-side.

Reuses Diffusionplanner_singlemap.py's actual planner/dynamics functions
directly (sample_diffusion_trajectory, dynamics_3d, waypoint_3d,
compute_fov) rather than reimplementing them - this script only adds control
-waypoint capture (which the original script prints but never saves to a
file) and the longer, unthrottled time budget.

Env vars must be set before importing Diffusionplanner_singlemap, since that
module reads them into module-level constants at import time.
"""
import os

os.environ.setdefault("SELECTED_MAP", "11")
os.environ.setdefault("MAPTYPE", "NAIP")
os.environ.setdefault("UTILITY_THRESHOLD", "0.3")
os.environ.setdefault("TIMEALLOTED", "400")
os.environ.setdefault("WALLCLOCK_SECONDS", "0")  # unconstrained; loop bounded by TIMEALLOTED
os.environ.setdefault("ENFORCE_MIN_STEP_TIME", "0")
os.environ.setdefault("SKIP_VIZ", "1")

from pathlib import Path
import numpy as np

import Diffusionplanner_singlemap as sim
from gaussianprocesstraining import (
    build_correlated_noise_covariance,
    build_sensor_matrix,
    noise_model,
    sample_correlated_sensor_noise,
    kalman_update,
    initialize_gp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "hero_shot_data"
OUT_DIR.mkdir(exist_ok=True)


def main():
    selected_map = sim.selected_map
    MAPTYPE = sim.MAPTYPE
    utility_threshold = sim.utility_threshold
    execution_chunk = sim.execution_chunk
    timealloted = sim.timealloted
    step = sim.step

    diffusion_model = sim.load_diffusion_model(checkpoint_path=sim.diffusion_path)
    print(f"Loaded diffusion checkpoint: {sim.diffusion_path}")

    csv_path = SCRIPT_DIR / "csv"
    data = np.loadtxt(csv_path / f"map_{selected_map}_{MAPTYPE}_grid_counts.csv", delimiter=",", skiprows=1)
    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()

    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]

    N = X_test.shape[0]
    true_map_flat = np.zeros(N, dtype=float)
    for x_true, y_true, value in pts:
        idx = np.where(
            np.isclose(X_test[:, 0], x_true) & np.isclose(X_test[:, 1], y_true)
        )[0]
        if idx.size > 0:
            true_map_flat[idx[0]] = value

    rng = np.random.default_rng(sim.SENSORNOISE_SEED + selected_map)

    mean = np.full(X_test.shape[0], utility_threshold - 0.1)
    mu = mean.copy()
    P = cov.copy()

    pose_history = []
    control_waypoint_records = []  # (replan_ts, waypoint_index, x, y, z)

    cx, cy, cz = sim.START_X, sim.START_Y, sim.INIT_ALTITUDE
    current_heading_velocity = np.zeros(3, dtype=np.float32)
    pose_history.append((0, cx, cy, cz))

    control_waypoints = []
    spline_path = []
    spline_idx = 0

    for ts in range(0, timealloted):
        if ts <= 1:
            grad_x, grad_y, grad_z, _ = sim.waypoint_3d(
                cx, cy, cz, goal_x=80.0, goal_y=80.0, goal_z=sim.INIT_ALTITUDE, step=step,
            )
            previous_pose = np.array([cx, cy, cz], dtype=np.float32)
            cx, cy, cz = sim.dynamics_3d(
                cx, cy, cz, grad_x, grad_y, grad_z, step,
                xmin, xmax, ymin, ymax, sim.ZMIN, sim.ZMAX, buffer=step * 2,
            )
            current_heading_velocity = np.array([cx, cy, cz], dtype=np.float32) - previous_pose
        else:
            if ts == 2 or spline_idx >= execution_chunk or spline_idx >= len(spline_path):
                current_mean = mu.reshape(X.shape)
                current_var = np.diag(P).reshape(X.shape)
                dense_traj, sparse_controls = sim.sample_diffusion_trajectory(
                    diffusion_model,
                    current_position=(cx, cy, cz),
                    current_mean=current_mean,
                    current_var=current_var,
                    current_heading_velocity=current_heading_velocity,
                    grid_step=step,
                    bounds=(xmin, xmax, ymin, ymax, sim.ZMIN, sim.ZMAX),
                )
                spline_path = dense_traj.T.tolist()
                spline_idx = 0
                control_waypoints = sparse_controls.T.tolist()
                for wi, (wx, wy, wz) in enumerate(control_waypoints):
                    control_waypoint_records.append((ts, wi, wx, wy, wz))
                print(f"ts={ts}: replanned, {len(control_waypoints)} control waypoints")

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

                previous_pose = np.array([cx, cy, cz], dtype=np.float32)
                cx, cy, cz = sim.dynamics_3d(
                    cx, cy, cz, grad_x, grad_y, grad_z, step,
                    xmin, xmax, ymin, ymax, sim.ZMIN, sim.ZMAX, buffer=step / 2,
                )
                current_heading_velocity = np.array([cx, cy, cz], dtype=np.float32) - previous_pose

        pose_history.append((ts + 1, cx, cy, cz))

        fov = sim.compute_fov(cz=cz, xs=xs, ys=ys, angle_of_view=60, step=step, cx=cx, cy=cy)
        sensor, sensor_block_ids = build_sensor_matrix(fov, cz, xs, ys, return_block_ids=True)
        R = noise_model(cz)
        z_meas = sensor @ true_map_flat
        z_meas += sample_correlated_sensor_noise(sensor_block_ids, R, rng)
        R_cov = build_correlated_noise_covariance(sensor_block_ids, R)
        mu, P = kalman_update(mu, P, sensor, z_meas, R_cov, block_ids=sensor_block_ids)

    pose_arr = np.asarray(pose_history, dtype=float)
    np.savetxt(
        OUT_DIR / f"map_{selected_map}_hero_trajectory_diffusion.csv",
        pose_arr, delimiter=",", header="timestep,x,y,z", comments="",
    )

    wp_arr = np.asarray(control_waypoint_records, dtype=float)
    np.savetxt(
        OUT_DIR / f"map_{selected_map}_hero_control_waypoints_diffusion.csv",
        wp_arr, delimiter=",", header="replan_timestep,waypoint_index,x,y,z", comments="",
    )

    print(f"Wrote {len(pose_arr)} trajectory points and {len(wp_arr)} control waypoints")


if __name__ == "__main__":
    main()
