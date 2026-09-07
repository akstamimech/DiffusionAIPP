"""
Combines the map-55-only maxiter sweep (results_cmaes_maxiter_stepsize10-4_map55)
with the maps-56-59 sweep (results_cmaes_maxiter_stepsize10-4_maps56-59) - both
run with CMA_STEP_SIZE_XY=10/CMA_STEP_SIZE_Z=4, beta=1, single run per
map/maxiter, TIMEALLOTED=200 - and reports the per-maxiter table averaged
across all 5 maps (55-59), same columns as the map-55-only pilot table.
"""
import csv
import re
from pathlib import Path

import numpy as np

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOTS = [
    SCRIPT_DIR / "results_cmaes_maxiter_stepsize10-4_map55",
    SCRIPT_DIR / "results_cmaes_maxiter_stepsize10-4_maps56-59",
    SCRIPT_DIR / "results_cmaes_maxiter_stepsize10-4_maxiter5",
]
MAXITERS = [0, 5, 10, 20, 30, 40]

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


def load_rows(results_root):
    summary_csv = results_root / "summaries" / "cmaes_maxiter_multimap_summary.csv"
    if not summary_csv.exists():
        raise SystemExit(f"No summary found at {summary_csv} - has the sweep finished?")
    with summary_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    xs_c, ys_c, cov_c, coords_c = m.build_coarse_grid()
    mask_cache = {}

    per_run = []
    for results_root in RESULT_ROOTS:
        for row in load_rows(results_root):
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
                variance_auc = float(np.trapezoid(variance_curve, wall_times) / (wall_times[-1] - wall_times[0]))
            else:
                variance_auc = v_final

            rstats = replan_stats(log_path)
            efficiency = variance_drop / rstats["total_replan_time"] if rstats["total_replan_time"] else float("nan")

            per_run.append({
                "map_id": map_id,
                "maxiter": maxiter,
                "variance_drop": variance_drop,
                "pct_reduced": pct_reduced,
                "variance_auc": variance_auc,
                "avg_replan_time": rstats["avg_replan_time"],
                "num_replans": rstats["num_replans"],
                "total_replan_time": rstats["total_replan_time"],
                "drop_per_replan_second": efficiency,
            })
            print(
                f"map={map_id} maxiter={maxiter}: var_drop={variance_drop:.3f} "
                f"({pct_reduced:.2f}%) variance_auc={variance_auc:.3f} replans={rstats['num_replans']} "
                f"total_replan_s={rstats['total_replan_time']:.2f}"
            )

    map_ids_seen = sorted({r["map_id"] for r in per_run})
    print(f"\nMaps included: {map_ids_seen} ({len(map_ids_seen)} maps)")

    cols = ["variance_drop", "pct_reduced", "variance_auc", "avg_replan_time",
            "num_replans", "total_replan_time", "drop_per_replan_second"]

    averaged = []
    for maxiter in MAXITERS:
        rows_mi = [r for r in per_run if r["maxiter"] == maxiter]
        if not rows_mi:
            continue
        entry = {"maxiter": maxiter, "n_maps": len(rows_mi)}
        for c in cols:
            vals = [r[c] for r in rows_mi]
            entry[c] = float(np.mean(vals))
            entry[c + "_std"] = float(np.std(vals))
        averaged.append(entry)

    header = (
        f"{'maxiter':>7} {'var_drop':>10} {'%reduced':>9} {'var_AUC':>10} "
        f"{'avg_replan_s':>13} {'#replans':>9} {'total_replan_s':>15} {'drop/replan_s':>14}"
    )
    print("\nAveraged across maps 55-59 (mean of 1 run per map):")
    print(header)
    print("-" * len(header))
    for e in averaged:
        print(
            f"{e['maxiter']:>7} {e['variance_drop']:>10.3f} {e['pct_reduced']:>8.2f}% "
            f"{e['variance_auc']:>10.3f} {e['avg_replan_time']:>13.3f} {e['num_replans']:>9.2f} "
            f"{e['total_replan_time']:>15.3f} {e['drop_per_replan_second']:>14.4f}"
        )

    print("\nStd dev across maps 55-59 (map-to-map spread):")
    print(header)
    print("-" * len(header))
    for e in averaged:
        print(
            f"{e['maxiter']:>7} {e['variance_drop_std']:>10.3f} {e['pct_reduced_std']:>8.2f}% "
            f"{e['variance_auc_std']:>10.3f} {e['avg_replan_time_std']:>13.3f} {e['num_replans_std']:>9.2f} "
            f"{e['total_replan_time_std']:>15.3f} {e['drop_per_replan_second_std']:>14.4f}"
        )

    out_csv = SCRIPT_DIR / "results_cmaes_maxiter_stepsize10-4_maps55-59_analysis.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(averaged[0].keys()))
        writer.writeheader()
        writer.writerows(averaged)
    print(f"\nWrote {out_csv}")

    per_run_csv = SCRIPT_DIR / "results_cmaes_maxiter_stepsize10-4_maps55-59_per_run.csv"
    with per_run_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run[0].keys()))
        writer.writeheader()
        writer.writerows(per_run)
    print(f"Wrote {per_run_csv}")


if __name__ == "__main__":
    main()
