"""
Analyzes sweep_diffusion_horizon_multimap.py output (maps 51-59, execution
horizons 10/20/30/40, wallclock=100s, ENFORCE_MIN_STEP_TIME=1, single run per
map/horizon) under RESULTS_ROOT (default: results_diffusion_horizon_maps51-59).

For each (map, horizon) run, computes:
  - Occupied Tr(P): final occupied-region variance (ground-truth mask,
    value > 0.5), via the coarse-grid Kalman replay used throughout this
    session (important_region_variance_from_trajectories.py helpers).
  - Final Occupied RMSE: taken directly from the run's own
    map_X_rmse_over_time.csv (last row, occupied_rmse column) - this is the
    planner's own full-resolution reconstruction metric, not the coarse
    replay grid.
  - Occupied Tr(P)-AUC: wall-clock-time-averaged occupied variance
    (trapezoidal over real seconds), i.e. how low the occupied variance
    stayed across the whole flight, not just at the end.
  - Total replanning time: sum of "Replanning time: X seconds" lines in the
    run's log.
Averages all four across the 9 maps per execution horizon and emits a LaTeX
table matching the user's requested format.
"""
import csv
import os
import re
from pathlib import Path

import numpy as np

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", str(SCRIPT_DIR / "results_diffusion_horizon_maps51-59")))
SUMMARY_CSV = RESULTS_ROOT / "summaries" / "diffusion_horizon_multimap_summary.csv"
HORIZONS = [10, 20, 30, 40]

REPLAN_TIME_RE = re.compile(r"Replanning time:\s*([\d.]+)\s*seconds")


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


def final_occupied_rmse(metrics_path):
    data = np.loadtxt(metrics_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return float(data[-1, 3])  # header: timestep,wall_time_seconds,global_rmse,occupied_rmse,global_variance


def replan_stats(log_path):
    times = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = REPLAN_TIME_RE.search(line)
            if match:
                times.append(float(match.group(1)))
    return {
        "num_replans": len(times),
        "total_replan_time": float(np.sum(times)) if times else 0.0,
    }


def main():
    if not SUMMARY_CSV.exists():
        raise SystemExit(f"No summary found at {SUMMARY_CSV} - has the sweep finished?")

    with SUMMARY_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))

    xs_c, ys_c, cov_c, coords_c = m.build_coarse_grid()
    mask_cache = {}

    per_run = []
    for row in rows:
        if row["status"] != "ok":
            print(f"SKIPPING non-ok run: map={row['map_id']} horizon={row['execution_horizon']} status={row['status']}")
            continue
        map_id = int(row["map_id"])
        horizon = int(row["execution_horizon"])
        output_dir = Path(row["output_dir"])
        traj_path = output_dir / f"map_{map_id}_executed_trajectory.csv"
        metrics_path = output_dir / f"map_{map_id}_rmse_over_time.csv"
        log_path = Path(row["log_file"])

        if map_id not in mask_cache:
            mask_cache[map_id], _ = m._important_mask_for_map(map_id, coords_c)
        mask = mask_cache[map_id]

        wall_times, variance_curve = replay_variance_curve(traj_path, mask, xs_c, ys_c, cov_c)
        v0 = float(variance_curve[0])
        v_final = float(variance_curve[-1])
        variance_drop = v0 - v_final
        pct_drop = 100.0 * variance_drop / v0 if v0 else float("nan")
        if len(variance_curve) > 1 and wall_times[-1] > wall_times[0]:
            variance_auc = float(np.trapezoid(variance_curve, wall_times) / (wall_times[-1] - wall_times[0]))
        else:
            variance_auc = v_final

        occ_rmse_final = final_occupied_rmse(metrics_path)
        rstats = replan_stats(log_path)

        per_run.append({
            "map_id": map_id,
            "horizon": horizon,
            "occupied_v0": v0,
            "occupied_trP_drop": variance_drop,
            "occupied_trP_pct_drop": pct_drop,
            "final_occupied_rmse": occ_rmse_final,
            "occupied_trP_auc": variance_auc,
            "num_replans": rstats["num_replans"],
            "total_replan_time": rstats["total_replan_time"],
        })
        print(
            f"map={map_id} horizon={horizon}: occ_trP_drop={variance_drop:.3f} ({pct_drop:.2f}%) "
            f"occ_rmse_final={occ_rmse_final:.4f} occ_trP_auc={variance_auc:.3f} "
            f"replans={rstats['num_replans']} total_replan_s={rstats['total_replan_time']:.3f}"
        )

    cols = ["occupied_v0", "occupied_trP_drop", "occupied_trP_pct_drop", "final_occupied_rmse",
            "occupied_trP_auc", "num_replans", "total_replan_time"]

    averaged = []
    for horizon in HORIZONS:
        rows_h = [r for r in per_run if r["horizon"] == horizon]
        if not rows_h:
            continue
        entry = {"horizon": horizon, "n_maps": len(rows_h)}
        for c in cols:
            vals = [r[c] for r in rows_h]
            entry[c] = float(np.mean(vals))
            entry[c + "_std"] = float(np.std(vals))
        averaged.append(entry)

    header = (
        f"{'horizon':>8} {'occ_TrP_%drop':>14} {'final_occ_RMSE':>16} "
        f"{'occ_TrP_AUC':>13} {'#replans':>9} {'total_replan_s':>15}"
    )
    print(f"\nAveraged across maps {sorted({r['map_id'] for r in per_run})}:")
    print(header)
    print("-" * len(header))
    for e in averaged:
        print(
            f"{e['horizon']:>8} {e['occupied_trP_pct_drop']:>13.2f}% {e['final_occupied_rmse']:>16.4f} "
            f"{e['occupied_trP_auc']:>13.3f} {e['num_replans']:>9.2f} {e['total_replan_time']:>15.3f}"
        )

    out_csv = SCRIPT_DIR / "results_diffusion_horizon_maps51-59_analysis.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(averaged[0].keys()))
        writer.writeheader()
        writer.writerows(averaged)
    print(f"\nWrote {out_csv}")

    per_run_csv = SCRIPT_DIR / "results_diffusion_horizon_maps51-59_per_run.csv"
    with per_run_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run[0].keys()))
        writer.writeheader()
        writer.writerows(per_run)
    print(f"Wrote {per_run_csv}")


if __name__ == "__main__":
    main()
