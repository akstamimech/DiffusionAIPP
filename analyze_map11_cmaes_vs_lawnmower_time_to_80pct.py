"""
How many simulation steps did CMA-ES vs. lawnmower take to cut occupied
("important region") variance by 80% on map 11 (halffield)? Fresh,
unconstrained (WALLCLOCK_SECONDS=0, full 400-step) runs of
CMAES_classic_singlemap.py and lawnmower_singlemap.py under CURRENT live
settings, replayed through important_region_variance_from_trajectories.py's
coarse-grid Kalman-replay methodology (exact reconstruction from recorded
poses alone - no need to have stored full P during the run).
"""
import os

os.environ.setdefault("MAPTYPE", "halffield")

from pathlib import Path

import numpy as np

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
MAP_ID = 11
VIZ_DIR = SCRIPT_DIR / "Vizualization"

RUNS = [
    ("CMA-ES", VIZ_DIR / "classic_map_halffield_11_viz" / f"map_{MAP_ID}_executed_trajectory.csv"),
    ("Lawnmower", VIZ_DIR / "lawnmower_map_halffield_11_viz" / f"map_{MAP_ID}_executed_trajectory.csv"),
]


def replay_variance_curve(traj_path, mask, xs_c, ys_c, cov_c):
    traj = np.loadtxt(traj_path, delimiter=",", skiprows=1)
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    P = cov_c.copy()
    timesteps = [0.0]
    variance_curve = [float(np.diag(P)[mask].sum())]
    for row in traj[1:]:
        ts, wall_t, x, y, z = row[0], row[1], row[2], row[3], row[4]
        fov = m.fov_grid_points(x, y, z, xs_c, ys_c, angle_of_view=m.ANGLE_OF_VIEW)
        if fov:
            sensor, block_ids = m.build_sensor_matrix(fov, z, xs_c, ys_c, return_block_ids=True)
            if sensor.shape[0] > 0:
                R = m.noise_model(z)
                sensor_c, _, R_c = m.compress_shared_sensor_rows(sensor, None, R, block_ids)
                P = m._covariance_step(P, sensor_c, R_c)
        timesteps.append(ts)
        variance_curve.append(float(np.diag(P)[mask].sum()))
    return np.array(timesteps), np.array(variance_curve)


def main():
    xs_c, ys_c, cov_c, coords_c = m.build_coarse_grid()
    mask, _ = m._important_mask_for_map(MAP_ID, coords_c)
    print(f"Map {MAP_ID}: {int(mask.sum())} of {len(mask)} coarse-grid cells are 'important' (LCB={m.LCB})\n")

    print(f"{'planner':>10} {'v0':>10} {'target(20%)':>12} {'steps_to_80pct':>15} {'final_v':>10} {'final_%red':>11}")
    for label, traj_path in RUNS:
        ts, var_curve = replay_variance_curve(traj_path, mask, xs_c, ys_c, cov_c)
        v0 = var_curve[0]
        target = 0.2 * v0
        hit = np.where(var_curve <= target)[0]
        if hit.size:
            steps_to_80pct = int(ts[hit[0]])
            steps_str = str(steps_to_80pct)
        else:
            steps_str = "never reached"
        final_pct_reduced = 100.0 * (v0 - var_curve[-1]) / v0
        print(
            f"{label:>10} {v0:>10.3f} {target:>12.3f} {steps_str:>15} "
            f"{var_curve[-1]:>10.3f} {final_pct_reduced:>10.1f}%"
        )


if __name__ == "__main__":
    main()
