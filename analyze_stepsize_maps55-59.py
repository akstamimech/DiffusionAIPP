"""
Combines the map-55 sigma sweep (sweep_cmaes_stepsize_map55.py,
results_cmaes_stepsize_map55/) with the maps-56-59 sweep
(sweep_cmaes_stepsize_maps56-59.py, results_cmaes_stepsize_maps56-59/) - same
5 step-size sets, single run per (map, stepsize), TIMEALLOTED=200, maxiter=20,
beta=1 - and averages final occupied variance / pct reduced / timestep-AUC
across all 5 maps (55-59) per step-size set, to check whether the map-55
trend (small step size wins) generalizes.
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
ROOTS = [
    SCRIPT_DIR / "results_cmaes_stepsize_map55" / "planner_outputs",
    SCRIPT_DIR / "results_cmaes_stepsize_maps56-59" / "planner_outputs",
]

DIR_PATTERN = re.compile(
    r"^classic_map_grf_(?P<mapid>\d+)_viz_hpc_cmaes_stepsize_(?P<name>[\d._]+)_map\d+_repeat0_seed\d+$"
)
MAP_IDS = [55, 56, 57, 58, 59]
ORDER = ["1.5_1.2", "6.0_4.8", "10_4", "20_8", "40_16"]


def discover_runs():
    runs = []
    for root in ROOTS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            match = DIR_PATTERN.match(d.name)
            if not match:
                continue
            map_id = int(match.group("mapid"))
            traj_path = d / f"map_{map_id}_executed_trajectory.csv"
            if not traj_path.exists():
                continue
            runs.append({"map_id": map_id, "name": match.group("name"), "traj_path": str(traj_path)})
    return runs


def final_and_auc_timestep(traj_path, mask, xs_c, ys_c, cov_c):
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
    found_maps = sorted({r["map_id"] for r in runs})
    print(f"Discovered {len(runs)} runs across maps {found_maps}")
    missing_maps = set(MAP_IDS) - set(found_maps)
    if missing_maps:
        print(f"WARNING: missing maps entirely: {sorted(missing_maps)}")

    xs_c, ys_c, cov_c, coords_c = m.build_coarse_grid()
    mask_cache = {}

    # results[name][map_id] = (final, auc, pct)
    results = defaultdict(dict)

    for run in runs:
        map_id = run["map_id"]
        name = run["name"]
        if map_id not in mask_cache:
            mask_cache[map_id], _ = m._important_mask_for_map(map_id, coords_c)
        mask = mask_cache[map_id]
        initial_var = float(np.diag(cov_c)[mask].sum())

        final_val, auc_val = final_and_auc_timestep(run["traj_path"], mask, xs_c, ys_c, cov_c)
        pct = 100.0 * (initial_var - final_val) / initial_var
        results[name][map_id] = (final_val, auc_val, pct)
        print(f"map={map_id} stepsize={name}: final={final_val:.4f} auc={auc_val:.4f} pct_reduced={pct:.2f}%")

    print()
    print(f"{'stepsize (xy_z)':>16} {'n_maps':>7} {'mean_final':>11} {'std_final':>10} "
          f"{'mean_pct':>9} {'mean_auc':>9} {'std_auc':>8}")
    summary_rows = []
    for name in ORDER:
        per_map = results.get(name, {})
        if not per_map:
            continue
        finals = [v[0] for v in per_map.values()]
        aucs = [v[1] for v in per_map.values()]
        pcts = [v[2] for v in per_map.values()]
        summary_rows.append({
            "name": name,
            "n_maps": len(per_map),
            "mean_final": float(np.mean(finals)),
            "std_final": float(np.std(finals)),
            "mean_pct": float(np.mean(pcts)),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
        })
        print(f"{name:>16} {len(per_map):>7} {np.mean(finals):>11.4f} {np.std(finals):>10.4f} "
              f"{np.mean(pcts):>8.2f}% {np.mean(aucs):>9.4f} {np.std(aucs):>8.4f}")

    # --- LaTeX table ---
    best_final_name = min(summary_rows, key=lambda r: r["mean_final"])["name"]
    best_auc_name = min(summary_rows, key=lambda r: r["mean_auc"])["name"]
    best_pct_name = max(summary_rows, key=lambda r: r["mean_pct"])["name"]

    label_map = {"1.5_1.2": "1.5 / 1.2", "6.0_4.8": "6.0 / 4.8", "10_4": "10 / 4",
                 "20_8": "20 / 8", "40_16": "40 / 16"}

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{CMA-ES step-size sweep, maps 55-59, single run per configuration per map, "
                 r"maxiter=20, $\beta=1$, 200 timesteps. Occupied-region variance final value, percent "
                 r"reduced, and AUC integrated over simulation timestep, averaged across all 5 maps. "
                 r"Bold = best mean value per column.}")
    lines.append(r"\label{tab:stepsize-sweep-maps55-59}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Step size ($\sigma_{xy}$/$\sigma_{z}$, m) & Mean final variance & Mean \% reduced & Mean AUC (timestep) \\")
    lines.append(r"\midrule")
    for row in summary_rows:
        final_cell = f"{row['mean_final']:.2f}"
        pct_cell = f"{row['mean_pct']:.2f}\\%"
        auc_cell = f"{row['mean_auc']:.2f}"
        if row["name"] == best_final_name:
            final_cell = rf"\textbf{{{final_cell}}}"
        if row["name"] == best_pct_name:
            pct_cell = rf"\textbf{{{pct_cell}}}"
        if row["name"] == best_auc_name:
            auc_cell = rf"\textbf{{{auc_cell}}}"
        lines.append(f"{label_map[row['name']]} & {final_cell} & {pct_cell} & {auc_cell} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    latex_table = "\n".join(lines)

    out_dir = SCRIPT_DIR / "results_cmaes_stepsize_maps56-59" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "stepsize_sweep_maps55-59_table.tex"
    table_path.write_text(latex_table, encoding="utf-8")
    print(f"\nWrote {table_path}\n")
    print(latex_table)


if __name__ == "__main__":
    main()
