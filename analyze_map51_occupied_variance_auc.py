"""
Computes the occupied-region variance AUC (wall-clock-time-averaged, ground
truth mask value > 0.5) for the three planners on the standard GRF map 51
(run-tag mission300_steps3000, 300s wall clock each - the pre-existing
data behind the original map_51_planner_fov_time_spent_heatmaps.png /
map_51_planner_time_spent_heatmaps.png reference images, not a fresh run).
Same coarse-grid Kalman replay methodology as
analyze_map200_occupied_variance_auc.py, for direct comparison against the
synthetic four-corner map result.
"""
import csv
from pathlib import Path

import numpy as np

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
MAP_ID = 51
VIZ_DIR = SCRIPT_DIR / "Vizualization"

PLANNERS = [
    ("cmaes", "CMA-ES expert", VIZ_DIR / f"classic_map_grf_{MAP_ID}_viz_mission300_steps3000"),
    ("diffusion", "Diffusion planner", VIZ_DIR / f"diffusion_map_grf_{MAP_ID}_viz_mission300_steps3000"),
    ("imitation", "ImitateTrans", VIZ_DIR / f"imitate_trans_map_grf_{MAP_ID}_viz_mission300_steps3000"),
]


def replay_variance_curve(traj_path, mask, xs_c, ys_c, cov_c):
    traj = np.loadtxt(traj_path, delimiter=",", skiprows=1)
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    P = cov_c.copy()
    wall_times = [0.0]
    variance_curve = [float(np.diag(P)[mask].sum())]
    for row in traj[1:]:
        _, wall_t, x, y, z = row[0], row[1], row[2], row[3], row[4]
        fov = m.fov_grid_points(x, y, z, xs_c, ys_c, angle_of_view=m.ANGLE_OF_VIEW)
        if fov:
            sensor, block_ids = m.build_sensor_matrix(fov, z, xs_c, ys_c, return_block_ids=True)
            if sensor.shape[0] > 0:
                R = m.noise_model(z)
                sensor_c, _, R_c = m.compress_shared_sensor_rows(sensor, None, R, block_ids)
                P = m._covariance_step(P, sensor_c, R_c)
        wall_times.append(wall_t)
        variance_curve.append(float(np.diag(P)[mask].sum()))
    return np.array(wall_times), np.array(variance_curve)


def main():
    xs_c, ys_c, cov_c, coords_c = m.build_coarse_grid()
    mask, _ = m._important_mask_for_map(MAP_ID, coords_c)
    print(f"Map {MAP_ID}: {int(mask.sum())} of {len(mask)} coarse-grid cells are 'important' (ground truth > 0.5)")

    results = []
    for key, label, output_dir in PLANNERS:
        traj_path = output_dir / f"map_{MAP_ID}_executed_trajectory.csv"
        wall_times, variance_curve = replay_variance_curve(traj_path, mask, xs_c, ys_c, cov_c)
        v0 = float(variance_curve[0])
        v_final = float(variance_curve[-1])
        if len(variance_curve) > 1 and wall_times[-1] > wall_times[0]:
            variance_auc = float(np.trapezoid(variance_curve, wall_times) / (wall_times[-1] - wall_times[0]))
        else:
            variance_auc = v_final
        results.append({
            "planner": key,
            "label": label,
            "v0": v0,
            "v_final": v_final,
            "final_wall_time_s": float(wall_times[-1]),
            "occupied_variance_auc": variance_auc,
        })

    header = f"{'planner':>18} {'v0':>10} {'v_final':>10} {'wall_time_s':>13} {'occ_var_AUC':>13}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['label']:>18} {r['v0']:>10.3f} {r['v_final']:>10.3f} "
            f"{r['final_wall_time_s']:>13.1f} {r['occupied_variance_auc']:>13.3f}"
        )

    out_csv = SCRIPT_DIR / "results_map51_occupied_variance_auc.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
