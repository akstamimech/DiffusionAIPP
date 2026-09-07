"""
Combined figure for the beta sweep on maps 51-55: final occupied-region
variance (left) and occupied-region variance AUC (right) as side-by-side
subplots, sharing one legend. Computes both metrics in a single replay pass
per run (same coarse-grid Kalman replay methodology as
important_region_variance_from_trajectories.py).
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
SWEEP_ROOT = SCRIPT_DIR / "results_cmaes_beta_sweep_maps51-55" / "planner_outputs"
OUT_DIR = SCRIPT_DIR / "results_cmaes_beta_sweep_maps51-55" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIR_PATTERN = re.compile(
    r"^classic_map_grf_(?P<mapid>\d+)_viz_hpc_cmaes_beta(?P<beta>[\d.]+)_map\d+_repeat(?P<repeat>\d+)_seed\d+$"
)
MAP_IDS = [51, 52, 53, 54, 55]
BETAS = [0.1, 0.5, 1.0, 2.0, 3.0]


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
        runs.append({
            "map_id": map_id,
            "beta": float(match.group("beta")),
            "repeat": int(match.group("repeat")),
            "traj_path": str(traj_path),
        })
    return runs


def occupied_variance_final_and_auc(traj_path, mask, xs_c, ys_c, cov_c):
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


def draw_panel(ax, mean_table, ylabel, title, winner_label, ylim_bottom=0.0):
    n_betas = len(BETAS)
    bar_width = 0.8 / n_betas
    x = np.arange(len(MAP_IDS))
    cmap = plt.get_cmap("viridis")
    bars = []
    for i, beta in enumerate(BETAS):
        values = [mean_table[map_id].get(beta, np.nan) for map_id in MAP_IDS]
        bar = ax.bar(x + i * bar_width, values, width=bar_width, align="edge",
                     label=f"beta={beta}", color=cmap(i / max(1, n_betas - 1)))
        bars.append(bar)

    ax.set_ylim(bottom=ylim_bottom)
    winner_xs, winner_ys = [], []
    for j, map_id in enumerate(MAP_IDS):
        best_beta = min(mean_table[map_id], key=mean_table[map_id].get)
        best_idx = BETAS.index(best_beta)
        best_val = mean_table[map_id][best_beta]
        winner_xs.append(x[j] + best_idx * bar_width + bar_width / 2)
        # centered within the VISIBLE part of the bar, i.e. between the
        # y-axis floor and the bar's top - not the same as best_val/2 once
        # the floor is no longer 0.
        winner_ys.append((ylim_bottom + best_val) / 2.0)
    winner_scatter = ax.scatter(winner_xs, winner_ys, marker="o", s=90, color="red", zorder=5,
                                 edgecolors="white", linewidths=1.2, label=winner_label)

    ax.set_xticks(x + bar_width * n_betas / 2)
    ax.set_xticklabels([f"map {m_id}" for m_id in MAP_IDS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return bars, winner_scatter


def main():
    runs = discover_runs()
    print(f"Discovered {len(runs)} runs")
    if not runs:
        raise SystemExit("No runs found - has the sweep finished?")

    xs_c, ys_c, cov_c, coords_c = m.build_coarse_grid()
    mask_cache = {}

    final_results = defaultdict(lambda: defaultdict(list))
    auc_results = defaultdict(lambda: defaultdict(list))

    for run in runs:
        map_id = run["map_id"]
        if map_id not in mask_cache:
            mask_cache[map_id], _ = m._important_mask_for_map(map_id, coords_c)
        mask = mask_cache[map_id]

        final_val, auc_val = occupied_variance_final_and_auc(run["traj_path"], mask, xs_c, ys_c, cov_c)
        final_results[map_id][run["beta"]].append(final_val)
        auc_results[map_id][run["beta"]].append(auc_val)
        print(f"map={map_id} beta={run['beta']} repeat={run['repeat']}: final={final_val:.4f} auc={auc_val:.4f}")

    final_mean_table = {
        map_id: {beta: float(np.mean(final_results[map_id][beta])) for beta in BETAS if beta in final_results[map_id]}
        for map_id in MAP_IDS
    }
    auc_mean_table = {
        map_id: {beta: float(np.mean(auc_results[map_id][beta])) for beta in BETAS if beta in auc_results[map_id]}
        for map_id in MAP_IDS
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))
    draw_panel(
        ax1, final_mean_table,
        ylabel="Final occupied-region variance (mean of 3 repeats, lower=better)",
        title="Final variance\nred dot = winning beta per map",
        winner_label="winner (lowest final variance)",
        ylim_bottom=2.0,
    )
    _, _ = draw_panel(
        ax2, auc_mean_table,
        ylabel="Occupied-region variance AUC (mean of 3 repeats, lower=better)",
        ylim_bottom=4.0,
        title="Variance-drop AUC\nred dot = winning beta per map",
        winner_label="winner (lowest AUC)",
    )

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, title="beta", loc="upper center", ncol=len(BETAS) + 1, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("CMA-ES beta sweep, maps 51-55 (maxiter=20, 200 timesteps)", fontsize=14, y=1.14)
    fig.tight_layout()

    plot_path = OUT_DIR / "beta_sweep_maps51-55_combined_final_and_auc.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
