"""
Computes variance-within-ground-truth-important-cells (>0.5) over timestep
for each (map, beta) run of sweep_cmaes_beta_multimap.py, then averages
final/pct_reduced/AUC across maps per beta to check whether a beta trend
found on one map generalizes.
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
SWEEP_DIR_NAME = sys.argv[1] if len(sys.argv) > 1 else "results_cmaes_beta_sweep_multimap"
SWEEP_ROOT = SCRIPT_DIR / SWEEP_DIR_NAME / "planner_outputs"
OUT_DIR = SCRIPT_DIR / SWEEP_DIR_NAME / "analysis"

DIR_PATTERN = re.compile(r"^classic_map_grf_(?P<mapid>\d+)_viz_hpc_cmaes_beta(?P<beta>[\d.]+)_map\d+_repeat0_seed\d+$")


def discover_beta_runs():
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
        runs.append({"beta": float(match.group("beta")), "map_id": map_id, "traj_path": str(traj_path)})
    return sorted(runs, key=lambda r: (r["beta"], r["map_id"]))


def main():
    runs = discover_beta_runs()
    map_ids = sorted({r["map_id"] for r in runs})
    betas = sorted({r["beta"] for r in runs})
    print(f"Discovered {len(runs)} runs: {len(betas)} betas x {len(map_ids)} maps {map_ids}")
    if not runs:
        raise SystemExit("No beta multimap runs found - has sweep_cmaes_beta_multimap.py finished?")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xs_c, ys_c, cov_c, coords_c = m.build_coarse_grid()
    mask_cache = {}

    per_run_rows = []
    curves_by_beta = defaultdict(list)  # beta -> list of (timesteps, important_variance) per map

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
        curves_by_beta[run["beta"]].append((timesteps, important_variance))

        def trapz_mean(y, x):
            if len(y) <= 1 or x[-1] <= x[0]:
                return float(y[-1])
            return float(np.trapezoid(y, x) / (x[-1] - x[0]))

        per_run_rows.append({
            "beta": run["beta"],
            "map_id": map_id,
            "n_important_cells": int(mask.sum()),
            "important_variance_initial": important_variance[0],
            "important_variance_final": important_variance[-1],
            "important_variance_pct_reduced": 100.0 * (important_variance[0] - important_variance[-1]) / important_variance[0],
            "important_variance_auc_timestep": trapz_mean(important_variance, timesteps),
        })
        print(f"beta={run['beta']} map={map_id}: final={important_variance[-1]:.4f} auc={per_run_rows[-1]['important_variance_auc_timestep']:.4f}")

    per_run_csv = OUT_DIR / "cmaes_beta_multimap_per_run.csv"
    with per_run_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)
    print(f"Wrote {per_run_csv}")

    # Average across maps per beta.
    by_beta = defaultdict(list)
    for row in per_run_rows:
        by_beta[row["beta"]].append(row)

    summary_rows = []
    for beta in betas:
        group = by_beta[beta]
        n = len(group)
        summary_rows.append({
            "beta": beta,
            "n_maps": n,
            "final_mean": float(np.mean([r["important_variance_final"] for r in group])),
            "final_std": float(np.std([r["important_variance_final"] for r in group])),
            "pct_reduced_mean": float(np.mean([r["important_variance_pct_reduced"] for r in group])),
            "auc_mean": float(np.mean([r["important_variance_auc_timestep"] for r in group])),
            "auc_std": float(np.std([r["important_variance_auc_timestep"] for r in group])),
        })

    summary_csv = OUT_DIR / "cmaes_beta_multimap_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_csv}")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, beta in enumerate(betas):
        color = cmap(i / max(1, len(betas) - 1))
        # Average the per-map curves onto a common timestep grid (all runs
        # share the same fixed timestep count, so indices line up).
        curve_list = curves_by_beta[beta]
        min_len = min(len(c[1]) for c in curve_list)
        ts_ref = curve_list[0][0][:min_len]
        stacked = np.stack([c[1][:min_len] for c in curve_list], axis=0)
        mean_curve = stacked.mean(axis=0)
        ax.plot(ts_ref, mean_curve, label=f"beta={beta}", color=color, linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean variance in ground-truth-important region (log scale)")
    ax.set_title(f"CMA-ES beta sweep averaged over {len(map_ids)} maps ({SWEEP_DIR_NAME})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    plot_path = OUT_DIR / "cmaes_beta_multimap_mean_curves.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")

    print()
    print(f"{'beta':>6} {'n':>3} {'final_mean':>11} {'final_std':>10} {'pct_reduced':>12} {'auc_mean':>10} {'auc_std':>9}")
    for row in sorted(summary_rows, key=lambda r: r["auc_mean"]):
        print(f"{row['beta']:>6} {row['n_maps']:>3} {row['final_mean']:>11.4f} {row['final_std']:>10.4f} "
              f"{row['pct_reduced_mean']:>11.2f}% {row['auc_mean']:>10.4f} {row['auc_std']:>9.4f}")


if __name__ == "__main__":
    main()
