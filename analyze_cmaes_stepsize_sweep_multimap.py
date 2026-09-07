"""
Computes variance-within-ground-truth-important-cells (>0.5) over timestep
for each (map, condition) run of sweep_cmaes_stepsize_multimap.py, then
averages final/pct_reduced/AUC across maps per condition (stepsize_default
1.5/1.2 vs stepsize_popovic_scaled 10/4).
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
SWEEP_DIR_NAME = sys.argv[1] if len(sys.argv) > 1 else "results_cmaes_stepsize_multimap"
SWEEP_ROOT = SCRIPT_DIR / SWEEP_DIR_NAME / "planner_outputs"
OUT_DIR = SCRIPT_DIR / SWEEP_DIR_NAME / "analysis"

DIR_PATTERN = re.compile(r"^classic_map_grf_(?P<mapid>\d+)_viz_hpc_cmaes_stepsize_(?P<condition>[a-z0-9_]+?)_map\d+_repeat0_seed\d+$")
CONDITION_ORDER = ["stepsize_default", "stepsize_popovic_scaled"]


def discover_runs():
    runs = []
    for d in sorted(SWEEP_ROOT.iterdir()):
        if not d.is_dir():
            continue
        match = DIR_PATTERN.match(d.name)
        if not match:
            continue
        map_id = int(match.group("mapid"))
        traj_path = d / f"map_{map_id}_executed_trajectory.csv"
        if not traj_path.exists():
            continue
        runs.append({"condition": match.group("condition"), "map_id": map_id, "traj_path": str(traj_path)})
    order = {c: i for i, c in enumerate(CONDITION_ORDER)}
    return sorted(runs, key=lambda r: (order.get(r["condition"], 99), r["map_id"]))


def main():
    runs = discover_runs()
    map_ids = sorted({r["map_id"] for r in runs})
    conditions = sorted({r["condition"] for r in runs}, key=lambda c: CONDITION_ORDER.index(c) if c in CONDITION_ORDER else 99)
    print(f"Discovered {len(runs)} runs: {len(conditions)} conditions x {len(map_ids)} maps {map_ids}")
    if not runs:
        raise SystemExit("No stepsize multimap runs found - has sweep_cmaes_stepsize_multimap.py finished?")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xs_c, ys_c, cov_c, coords_c = m.build_coarse_grid()
    mask_cache = {}

    per_run_rows = []
    curves_by_condition = defaultdict(list)

    for run in runs:
        map_id = run["map_id"]
        if map_id not in mask_cache:
            mask_cache[map_id], _ = m._important_mask_for_map(map_id, coords_c)
        mask = mask_cache[map_id]

        traj = np.loadtxt(run["traj_path"], delimiter=",", skiprows=1)
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)

        P = cov_c.copy()
        timesteps = [0.0]
        important_variance = [float(np.diag(P)[mask].sum())]

        for row in traj[1:]:
            ts, wall_t, x, y, z = row[0], row[1], row[2], row[3], row[4]
            fov = m.fov_grid_points(x, y, z, xs_c, ys_c, angle_of_view=m.ANGLE_OF_VIEW)
            if fov:
                sensor, block_ids = m.build_sensor_matrix(fov, z, xs_c, ys_c, return_block_ids=True)
                if sensor.shape[0] > 0:
                    R = m.noise_model(z)
                    sensor_c, _, R_c = m.compress_shared_sensor_rows(sensor, None, R, block_ids)
                    P = m._covariance_step(P, sensor_c, R_c)
            diag = np.diag(P)
            timesteps.append(ts)
            important_variance.append(float(diag[mask].sum()))

        timesteps = np.array(timesteps)
        important_variance = np.array(important_variance)
        curves_by_condition[run["condition"]].append((timesteps, important_variance))

        def trapz_mean(y, x):
            if len(y) <= 1 or x[-1] <= x[0]:
                return float(y[-1])
            return float(np.trapezoid(y, x) / (x[-1] - x[0]))

        per_run_rows.append({
            "condition": run["condition"],
            "map_id": map_id,
            "n_important_cells": int(mask.sum()),
            "important_variance_initial": important_variance[0],
            "important_variance_final": important_variance[-1],
            "important_variance_pct_reduced": 100.0 * (important_variance[0] - important_variance[-1]) / important_variance[0],
            "important_variance_auc_timestep": trapz_mean(important_variance, timesteps),
        })
        print(f"condition={run['condition']} map={map_id}: final={important_variance[-1]:.4f} auc={per_run_rows[-1]['important_variance_auc_timestep']:.4f}")

    per_run_csv = OUT_DIR / "cmaes_stepsize_multimap_per_run.csv"
    with per_run_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)
    print(f"Wrote {per_run_csv}")

    by_condition = defaultdict(list)
    for row in per_run_rows:
        by_condition[row["condition"]].append(row)

    summary_rows = []
    for condition in conditions:
        group = by_condition[condition]
        summary_rows.append({
            "condition": condition,
            "n_maps": len(group),
            "final_mean": float(np.mean([r["important_variance_final"] for r in group])),
            "final_std": float(np.std([r["important_variance_final"] for r in group])),
            "pct_reduced_mean": float(np.mean([r["important_variance_pct_reduced"] for r in group])),
            "auc_mean": float(np.mean([r["important_variance_auc_timestep"] for r in group])),
            "auc_std": float(np.std([r["important_variance_auc_timestep"] for r in group])),
        })

    summary_csv = OUT_DIR / "cmaes_stepsize_multimap_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_csv}")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = {"stepsize_default": "tab:gray", "stepsize_popovic_scaled": "tab:red"}
    for condition in conditions:
        curve_list = curves_by_condition[condition]
        min_len = min(len(c[1]) for c in curve_list)
        ts_ref = curve_list[0][0][:min_len]
        stacked = np.stack([c[1][:min_len] for c in curve_list], axis=0)
        mean_curve = stacked.mean(axis=0)
        ax.plot(ts_ref, mean_curve, label=condition, color=colors.get(condition), linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean variance in ground-truth-important region (log scale)")
    ax.set_title(f"CMA-ES step-size comparison averaged over {len(map_ids)} maps ({SWEEP_DIR_NAME})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    plot_path = OUT_DIR / "cmaes_stepsize_multimap_mean_curves.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")

    print()
    print(f"{'condition':>24} {'n':>3} {'final_mean':>11} {'final_std':>10} {'pct_reduced':>12} {'auc_mean':>10} {'auc_std':>9}")
    for row in sorted(summary_rows, key=lambda r: r["auc_mean"]):
        print(f"{row['condition']:>24} {row['n_maps']:>3} {row['final_mean']:>11.4f} {row['final_std']:>10.4f} "
              f"{row['pct_reduced_mean']:>11.2f}% {row['auc_mean']:>10.4f} {row['auc_std']:>9.4f}")


if __name__ == "__main__":
    main()
