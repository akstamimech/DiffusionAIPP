"""
Same data/replay as analyze_maxiter_stepsize10-4_maps55-59.py, but reports the
RAW (un-normalized) variance AUC - trapz(variance_curve, wall_times), in units
of variance*seconds - instead of dividing by each run's own elapsed wall time.
Runs at different maxiter have different total wall-clock duration (more
replans, longer replans), so the raw integral is not directly comparable
across maxiter on its own; total_wall_time is reported alongside it for that
reason.
"""
import csv
import importlib.util
from pathlib import Path

import numpy as np

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "_maxiter_base", SCRIPT_DIR / "analyze_maxiter_stepsize10-4_maps55-59.py"
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
RESULT_ROOTS = _base.RESULT_ROOTS
MAXITERS = _base.MAXITERS
replay_variance_curve = _base.replay_variance_curve
replan_stats = _base.replan_stats
load_rows = _base.load_rows


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
            total_wall_time = float(wall_times[-1] - wall_times[0])

            if len(variance_curve) > 1 and total_wall_time > 0:
                variance_auc_raw = float(np.trapezoid(variance_curve, wall_times))
            else:
                variance_auc_raw = float(variance_curve[-1])

            rstats = replan_stats(log_path)

            per_run.append({
                "map_id": map_id,
                "maxiter": maxiter,
                "variance_auc_raw": variance_auc_raw,
                "total_wall_time": total_wall_time,
                "num_replans": rstats["num_replans"],
                "total_replan_time": rstats["total_replan_time"],
            })

    cols = ["variance_auc_raw", "total_wall_time", "num_replans", "total_replan_time"]

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

    header = f"{'maxiter':>7} {'raw_var_AUC':>13} {'total_wall_s':>13} {'#replans':>9} {'total_replan_s':>15}"
    print("Averaged across maps 55-59 (mean of 1 run per map), RAW (un-normalized) AUC:")
    print(header)
    print("-" * len(header))
    for e in averaged:
        print(
            f"{e['maxiter']:>7} {e['variance_auc_raw']:>13.3f} {e['total_wall_time']:>13.3f} "
            f"{e['num_replans']:>9.2f} {e['total_replan_time']:>15.3f}"
        )

    print("\nStd dev across maps 55-59 (map-to-map spread):")
    print(header)
    print("-" * len(header))
    for e in averaged:
        print(
            f"{e['maxiter']:>7} {e['variance_auc_raw_std']:>13.3f} {e['total_wall_time_std']:>13.3f} "
            f"{e['num_replans_std']:>9.2f} {e['total_replan_time_std']:>15.3f}"
        )

    out_csv = SCRIPT_DIR / "results_cmaes_maxiter_stepsize10-4_maps55-59_raw_auc.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(averaged[0].keys()))
        writer.writeheader()
        writer.writerows(averaged)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
