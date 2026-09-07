"""
Analyzes a sweep_cmaes_maxiter_multimap.py run (step size fixed at 10/4 via
CMA_STEP_SIZE_XY/CMA_STEP_SIZE_Z env vars, beta=1, single run per
map/maxiter) written under RESULTS_ROOT (default:
results_cmaes_maxiter_stepsize10-4_map55).

For each (map, maxiter) run, computes:
  - occupied-region variance drop (ground-truth mask, value > 0.5): V0 - V_final,
    via the same coarse-grid Kalman replay used elsewhere this session
    (important_region_variance_from_trajectories.py helpers).
  - occupied-region variance-drop AUC: V0 - time_avg(V(t)) over real wall-clock
    time (trapezoidal), i.e. the wall-clock-time-averaged drop maintained
    over the run, not just the final value.
  - average real replan time and number of replans, parsed directly from the
    run's own log ("Replanning time: X seconds" lines, printed by
    CMAES_classic_singlemap.py every time it replans).
  - variance_drop / total_replan_time: variance removed per second of
    compute actually spent replanning.
"""
import csv
import os
import re
from pathlib import Path

import numpy as np

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", str(SCRIPT_DIR / "results_cmaes_maxiter_stepsize10-4_map55")))
SUMMARY_CSV = RESULTS_ROOT / "summaries" / "cmaes_maxiter_multimap_summary.csv"

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


def replan_stats(log_path):
    times = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = REPLAN_TIME_RE.search(line)
            if match:
                times.append(float(match.group(1)))
    if not times:
        return {"num_replans": 0, "avg_replan_time": float("nan"), "total_replan_time": 0.0}
    return {
        "num_replans": len(times),
        "avg_replan_time": float(np.mean(times)),
        "total_replan_time": float(np.sum(times)),
    }


def main():
    if not SUMMARY_CSV.exists():
        raise SystemExit(f"No summary found at {SUMMARY_CSV} - has the sweep finished?")

    with SUMMARY_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))

    xs_c, ys_c, cov_c, coords_c = m.build_coarse_grid()
    mask_cache = {}

    results = []
    for row in rows:
        map_id = int(row["map_id"])
        maxiter = int(row["maxiter"])
        traj_path = Path(row["output_dir"]) / f"map_{map_id}_executed_trajectory.csv"
        log_path = Path(row["log_file"])

        if map_id not in mask_cache:
            mask_cache[map_id], _ = m._important_mask_for_map(map_id, coords_c)
        mask = mask_cache[map_id]

        wall_times, variance_curve = replay_variance_curve(traj_path, mask, xs_c, ys_c, cov_c)
        v0 = float(variance_curve[0])
        v_final = float(variance_curve[-1])
        variance_drop = v0 - v_final
        pct_reduced = 100.0 * variance_drop / v0 if v0 else float("nan")

        if len(variance_curve) > 1 and wall_times[-1] > wall_times[0]:
            time_avg_variance = float(np.trapezoid(variance_curve, wall_times) / (wall_times[-1] - wall_times[0]))
        else:
            time_avg_variance = v_final
        drop_auc = v0 - time_avg_variance

        rstats = replan_stats(log_path)
        efficiency = variance_drop / rstats["total_replan_time"] if rstats["total_replan_time"] else float("nan")

        results.append({
            "map_id": map_id,
            "maxiter": maxiter,
            "v0": v0,
            "v_final": v_final,
            "variance_drop": variance_drop,
            "pct_reduced": pct_reduced,
            "drop_auc": drop_auc,
            "avg_replan_time": rstats["avg_replan_time"],
            "num_replans": rstats["num_replans"],
            "total_replan_time": rstats["total_replan_time"],
            "drop_per_replan_second": efficiency,
            "process_wall_seconds": float(row["process_wall_seconds"]),
        })

    results.sort(key=lambda r: (r["map_id"], r["maxiter"]))

    header = (
        f"{'map':>4} {'maxiter':>7} {'var_drop':>10} {'%reduced':>9} {'drop_AUC':>10} "
        f"{'avg_replan_s':>13} {'#replans':>9} {'total_replan_s':>15} {'drop/replan_s':>14}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['map_id']:>4} {r['maxiter']:>7} {r['variance_drop']:>10.3f} {r['pct_reduced']:>8.2f}% "
            f"{r['drop_auc']:>10.3f} {r['avg_replan_time']:>13.3f} {r['num_replans']:>9} "
            f"{r['total_replan_time']:>15.3f} {r['drop_per_replan_second']:>14.4f}"
        )

    out_csv = RESULTS_ROOT / "analysis_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
