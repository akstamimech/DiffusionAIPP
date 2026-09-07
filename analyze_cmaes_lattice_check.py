"""
Computes variance-within-ground-truth-important-cells (>0.5) over timestep
for each condition in the CMA-ES lattice-densification check
(sweep_cmaes_lattice_check.py: lattice_only / cma_maxiter2 / cma_maxiter20),
reusing the same Kalman-covariance replay approach as
important_region_variance_from_trajectories.py.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
SWEEP_DIR_NAME = sys.argv[1] if len(sys.argv) > 1 else "results_cmaes_lattice_check"
SWEEP_ROOT = SCRIPT_DIR / SWEEP_DIR_NAME / "planner_outputs"
OUT_DIR = SCRIPT_DIR / SWEEP_DIR_NAME / "analysis"

DIR_PATTERN = re.compile(r"^classic_map_grf_(?P<mapid>\d+)_viz_hpc_cmaes_lattice_(?P<condition>[a-z0-9_]+?)_map\d+_repeat0_seed\d+$")

CONDITION_ORDER = ["lattice_only", "cma_maxiter2", "cma_maxiter20"]


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
    return sorted(runs, key=lambda r: order.get(r["condition"], 99))


def main():
    runs = discover_runs()
    print(f"Discovered {len(runs)} runs: {[r['condition'] for r in runs]}")
    if not runs:
        raise SystemExit("No lattice-check runs found - has sweep_cmaes_lattice_check.py finished?")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xs_c, ys_c, cov_c, coords_c = m.build_coarse_grid()

    curves = {}
    summary_rows = []
    for run in runs:
        mask, _ = m._important_mask_for_map(run["map_id"], coords_c)
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
        curves[run["condition"]] = (timesteps, important_variance)

        def trapz_mean(y, x):
            if len(y) <= 1 or x[-1] <= x[0]:
                return float(y[-1])
            return float(np.trapezoid(y, x) / (x[-1] - x[0]))

        summary_rows.append({
            "condition": run["condition"],
            "map_id": run["map_id"],
            "n_important_cells": int(mask.sum()),
            "important_variance_initial": important_variance[0],
            "important_variance_final": important_variance[-1],
            "important_variance_pct_reduced": 100.0 * (important_variance[0] - important_variance[-1]) / important_variance[0],
            "important_variance_auc_timestep": trapz_mean(important_variance, timesteps),
        })
        print(f"condition={run['condition']}: final={important_variance[-1]:.4f} auc={summary_rows[-1]['important_variance_auc_timestep']:.4f}")

    csv_path = OUT_DIR / "cmaes_lattice_check_important_variance.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = {"lattice_only": "tab:gray", "cma_maxiter2": "tab:blue", "cma_maxiter20": "tab:red"}
    for condition in CONDITION_ORDER:
        if condition not in curves:
            continue
        ts, iv = curves[condition]
        ax.plot(ts, iv, label=condition, color=colors.get(condition), linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Variance in ground-truth-important region (log scale)")
    ax.set_title(f"CMA-ES lattice-densification check ({SWEEP_DIR_NAME}, map {runs[0]['map_id']}, fixed 250 timesteps)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    plot_path = OUT_DIR / "cmaes_lattice_check_important_variance.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")

    print()
    print(f"{'condition':>14} {'final':>10} {'pct_reduced':>12} {'AUC':>10}")
    for row in sorted(summary_rows, key=lambda r: r["important_variance_auc_timestep"]):
        print(f"{row['condition']:>14} {row['important_variance_final']:>10.4f} {row['important_variance_pct_reduced']:>11.2f}% {row['important_variance_auc_timestep']:>10.4f}")


if __name__ == "__main__":
    main()
