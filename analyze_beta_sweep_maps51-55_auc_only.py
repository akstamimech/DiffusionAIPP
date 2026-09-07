"""
Single-panel version of analyze_beta_sweep_maps51-55_combined.py: just the
step-averaged occupied-region variance panel (the one labeled "AUC" in the
combined figure - see that script's occupied_variance_final_and_auc, which
divides the trapezoidal-rule integral (over the timestep index column, not
wall-clock seconds) by elapsed steps, so it's really a per-step average of the
variance curve, not a raw area-under-curve).
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


def occupied_variance_step_avg(traj_path, mask, xs_c, ys_c, cov_c):
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
    if len(variance_curve) <= 1 or timesteps[-1] <= timesteps[0]:
        return float(variance_curve[-1])
    return float(np.trapezoid(variance_curve, timesteps) / (timesteps[-1] - timesteps[0]))


def draw_panel(ax, mean_table, ylabel, title, winner_label, ylim_bottom=0.0):
    n_betas = len(BETAS)
    bar_width = 0.8 / n_betas
    x = np.arange(len(MAP_IDS))
    cmap = plt.get_cmap("viridis")
    for i, beta in enumerate(BETAS):
        values = [mean_table[map_id].get(beta, np.nan) for map_id in MAP_IDS]
        ax.bar(x + i * bar_width, values, width=bar_width, align="edge",
               label=f"beta={beta}", color=cmap(i / max(1, n_betas - 1)))

    ax.set_ylim(bottom=ylim_bottom)
    winner_xs, winner_ys = [], []
    for j, map_id in enumerate(MAP_IDS):
        best_beta = min(mean_table[map_id], key=mean_table[map_id].get)
        best_idx = BETAS.index(best_beta)
        best_val = mean_table[map_id][best_beta]
        winner_xs.append(x[j] + best_idx * bar_width + bar_width / 2)
        winner_ys.append((ylim_bottom + best_val) / 2.0)
    ax.scatter(winner_xs, winner_ys, marker="o", s=90, color="red", zorder=5,
               edgecolors="white", linewidths=1.2, label=winner_label)

    ax.set_xticks(x + bar_width * n_betas / 2)
    ax.set_xticklabels([str(m_id) for m_id in MAP_IDS], fontsize=11)
    ax.set_xlabel("map", fontsize=10, color="#52514e")
    ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.3)


def main():
    runs = discover_runs()
    print(f"Discovered {len(runs)} runs")
    if not runs:
        raise SystemExit("No runs found - has the sweep finished?")

    xs_c, ys_c, cov_c, coords_c = m.build_coarse_grid()
    mask_cache = {}

    results = defaultdict(lambda: defaultdict(list))

    for run in runs:
        map_id = run["map_id"]
        if map_id not in mask_cache:
            mask_cache[map_id], _ = m._important_mask_for_map(map_id, coords_c)
        mask = mask_cache[map_id]

        step_avg_val = occupied_variance_step_avg(run["traj_path"], mask, xs_c, ys_c, cov_c)
        results[map_id][run["beta"]].append(step_avg_val)
        print(f"map={map_id} beta={run['beta']} repeat={run['repeat']}: step_avg={step_avg_val:.4f}")

    mean_table = {
        map_id: {beta: float(np.mean(results[map_id][beta])) for beta in BETAS if beta in results[map_id]}
        for map_id in MAP_IDS
    }

    # Narrow/tall, sized for one IEEE column (~3.5in wide) - short, bold axis label and a compact
    # 3-column legend instead of the wide combined figure's single-row 6-across layout.
    fig, ax = plt.subplots(figsize=(4.2, 5.2))
    draw_panel(
        ax, mean_table,
        ylabel="Step-avg. variance",
        title="Step-averaged occupied variance",
        winner_label="winner",
        ylim_bottom=4.0,
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="beta", loc="upper center", ncol=3,
               fontsize=8.5, title_fontsize=9, bbox_to_anchor=(0.5, 1.22), frameon=False)
    fig.tight_layout()

    plot_path = OUT_DIR / "beta_sweep_maps51-55_step_avg_variance.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
