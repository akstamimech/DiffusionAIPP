"""
Analysis for the single-map (55), single-run-per-config sigma sweep
(sweep_cmaes_stepsize_map55.py): ground-truth occupied-region final variance
and wall-clock-time-integrated AUC, across 5 step-size sets spanning the
current sub-pixel default through 4x the Popovic-scaling-derived target.
"""
import re
from pathlib import Path

import numpy as np

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
SWEEP_ROOT = SCRIPT_DIR / "results_cmaes_stepsize_map55" / "planner_outputs"

DIR_PATTERN = re.compile(
    r"^classic_map_grf_(?P<mapid>\d+)_viz_hpc_cmaes_stepsize_(?P<name>[\d._]+)_map\d+_repeat0_seed\d+$"
)
MAP_ID = 55
ORDER = ["1.5_1.2", "6.0_4.8", "10_4", "20_8", "40_16"]


def discover_runs():
    runs = {}
    for d in sorted(SWEEP_ROOT.iterdir()):
        if not d.is_dir():
            continue
        match = DIR_PATTERN.match(d.name)
        if not match:
            continue
        traj_path = d / f"map_{MAP_ID}_executed_trajectory.csv"
        if not traj_path.exists():
            continue
        runs[match.group("name")] = str(traj_path)
    return runs


def final_and_auc(traj_path, mask, xs_c, ys_c, cov_c):
    """AUC here is integrated over TIMESTEP index (TIMEALLOTED=200 shared by
    every run), not wall-clock time - appropriate for this sweep since step
    size doesn't change per-replan compute cost the way maxiter did, so there
    is no confound between timestep progress and real time spent."""
    traj = np.loadtxt(traj_path, delimiter=",", skiprows=1)
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    P = cov_c.copy()
    timesteps = [0.0]
    variance_curve = [float(np.diag(P)[mask].sum())]
    for row in traj[1:]:
        ts, _, x, y, z = row[0], row[1], row[2], row[3], row[4]
        fov = m.fov_grid_points(x, y, z, xs_c, ys_c, angle_of_view=m.ANGLE_OF_VIEW)
        if fov:
            sensor, block_ids = m.build_sensor_matrix(fov, z, xs_c, ys_c, return_block_ids=True)
            if sensor.shape[0] > 0:
                R = m.noise_model(z)
                sensor_c, _, R_c = m.compress_shared_sensor_rows(sensor, None, R, block_ids)
                P = m._covariance_step(P, sensor_c, R_c)
        timesteps.append(ts)
        variance_curve.append(float(np.diag(P)[mask].sum()))

    timesteps = np.array(timesteps)
    variance_curve = np.array(variance_curve)
    final_val = float(variance_curve[-1])
    if len(variance_curve) <= 1 or timesteps[-1] <= timesteps[0]:
        auc_val = final_val
    else:
        auc_val = float(np.trapezoid(variance_curve, timesteps) / (timesteps[-1] - timesteps[0]))
    return final_val, auc_val


def main():
    runs = discover_runs()
    print(f"Discovered {len(runs)} runs: {list(runs.keys())}")
    if not runs:
        raise SystemExit("No runs found - has the sweep finished?")

    xs_c, ys_c, cov_c, coords_c = m.build_coarse_grid()
    mask, _ = m._important_mask_for_map(MAP_ID, coords_c)
    initial_var = float(np.diag(cov_c)[mask].sum())

    results = {}
    for name in ORDER:
        if name not in runs:
            continue
        final_val, auc_val = final_and_auc(runs[name], mask, xs_c, ys_c, cov_c)
        pct = 100.0 * (initial_var - final_val) / initial_var
        results[name] = (final_val, auc_val, pct)
        print(f"stepsize={name}: final={final_val:.4f} auc={auc_val:.4f} pct_reduced={pct:.2f}%")

    print()
    print(f"{'stepsize (xy_z)':>16} {'final':>10} {'pct_reduced':>12} {'auc(timestep)':>15}")
    for name in ORDER:
        if name not in results:
            continue
        final_val, auc_val, pct = results[name]
        print(f"{name:>16} {final_val:>10.4f} {pct:>11.2f}% {auc_val:>15.4f}")


if __name__ == "__main__":
    main()
