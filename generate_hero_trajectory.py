"""
Generates a fresh, longer CMA-ES flight over map 11 (NAIP) for
plot_simulator_hero_shot.py, since the previous run
(Vizualization/classic_map_NAIP_11_viz/map_11_executed_trajectory.csv) was cut
short by the default WALLCLOCK_SECONDS=50 budget after only 105 timesteps -
too short to read as a real coverage mission, just two isolated climb/dive
loops.

Reuses CMAES_classic_singlemap.py's actual planner/dynamics functions
directly (real_receding_horizon_planner, dynamics_3d, waypoint_3d,
compute_fov) rather than reimplementing them - this script only adds control
-waypoint capture (the 3D waypoints CMA-ES actually chose at each replan,
which the original script prints but never saves to a file) and a longer,
unthrottled time budget (ENFORCE_MIN_STEP_TIME=0 so it doesn't sleep to
match real flight speed, TIMEALLOTED raised so the mission runs to a natural
stopping point instead of a wall-clock cutoff).

Env vars must be set before importing CMAES_classic_singlemap, since that
module reads them into module-level constants at import time.
"""
import os

os.environ.setdefault("SELECTED_MAP", "11")
os.environ.setdefault("MAPTYPE", "NAIP")
os.environ.setdefault("TIMEALLOTED", "400")
os.environ.setdefault("WALLCLOCK_SECONDS", "0")  # unconstrained; loop bounded by TIMEALLOTED
os.environ.setdefault("ENFORCE_MIN_STEP_TIME", "0")
os.environ.setdefault("SKIP_VIZ", "1")

from pathlib import Path
import numpy as np

import CMAES_classic_singlemap as sim
from gaussianprocesstraining import (
    importance_filter,
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
    beta = sim.beta
    planning_horizon = sim.planning_horizon
    execution_chunk = sim.execution_chunk
    cma_seed = sim.cma_seed
    timealloted = sim.timealloted
    step = sim.step

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
    pose_history.append((0, cx, cy, cz))

    control_waypoints = []
    spline_path = []
    spline_idx = 0

    for ts in range(0, timealloted):
        if ts <= 1:
            grad_x, grad_y, _ = sim.waypoint(cx, cy, goal_x=80.0, goal_y=80.0, step=step)
            cx, cy = sim.dynamics(cx, cy, grad_x, grad_y, step, step, xmin, xmax, ymin, ymax)
        else:
            if ts == 2 or spline_idx >= execution_chunk or spline_idx >= len(spline_path):
                control_waypoints = sim.real_receding_horizon_planner(
                    cx, cy, cz, mu, P, xs, ys, utility_threshold, beta,
                    planning_horizon, alpha=sim.alpha, seed=cma_seed,
                )
                for wi, (wx, wy, wz) in enumerate(control_waypoints):
                    control_waypoint_records.append((ts, wi, wx, wy, wz))
                spline_path = sim.build_spline_trajectory_3d(
                    cx, cy, cz, control_waypoints, samples_per_segment=sim.samples_per_segment
                )
                spline_idx = 0
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

                cx, cy, cz = sim.dynamics_3d(
                    cx, cy, cz, grad_x, grad_y, grad_z, step,
                    xmin, xmax, ymin, ymax, sim.ZMIN, sim.ZMAX, buffer=step / 2,
                )

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
        OUT_DIR / f"map_{selected_map}_hero_trajectory.csv",
        pose_arr, delimiter=",", header="timestep,x,y,z", comments="",
    )

    wp_arr = np.asarray(control_waypoint_records, dtype=float)
    np.savetxt(
        OUT_DIR / f"map_{selected_map}_hero_control_waypoints.csv",
        wp_arr, delimiter=",", header="replan_timestep,waypoint_index,x,y,z", comments="",
    )

    print(f"Wrote {len(pose_arr)} trajectory points and {len(wp_arr)} control waypoints")


if __name__ == "__main__":
    main()
