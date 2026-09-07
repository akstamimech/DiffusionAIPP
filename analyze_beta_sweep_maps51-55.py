"""
For the beta sweep on maps 51-55 (sweep_cmaes_beta_multimap.py, maxiter=20,
3 repeats/map/beta, TIMEALLOTED=200): computes ground-truth occupied-region
variance drop per run (coarse-grid Kalman replay, same methodology as
important_region_variance_from_trajectories.py), averages over the 3
repeats per (map, beta), and produces a matplotlib grouped-bar figure plus a
LaTeX table (winning beta bolded per map row) showing that the best beta is
map-dependent, not a single universal winner.
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


def occupied_variance_final(traj_path, mask, xs_c, ys_c, cov_c):
    traj = np.loadtxt(traj_path, delimiter=",", skiprows=1)
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    P = cov_c.copy()
    for row in traj[1:]:
        _, _, x, y, z = row[0], row[1], row[2], row[3], row[4]
        fov = m.fov_grid_points(x, y, z, xs_c, ys_c, angle_of_view=m.ANGLE_OF_VIEW)
        if fov:
            sensor, block_ids = m.build_sensor_matrix(fov, z, xs_c, ys_c, return_block_ids=True)
            if sensor.shape[0] > 0:
                R = m.noise_model(z)
                sensor_c, _, R_c = m.compress_shared_sensor_rows(sensor, None, R, block_ids)
                P = m._covariance_step(P, sensor_c, R_c)
    return float(np.diag(P)[mask].sum())


def main():
    runs = discover_runs()
    print(f"Discovered {len(runs)} runs")
    if not runs:
        raise SystemExit("No runs found - has the sweep finished?")

    xs_c, ys_c, cov_c, coords_c = m.build_coarse_grid()
    mask_cache = {}

    # results[map_id][beta] = list of final occupied variances across repeats
    results = defaultdict(lambda: defaultdict(list))
    initial_variance = {}

    for run in runs:
        map_id = run["map_id"]
        if map_id not in mask_cache:
            mask_cache[map_id], _ = m._important_mask_for_map(map_id, coords_c)
        mask = mask_cache[map_id]
        if map_id not in initial_variance:
            initial_variance[map_id] = float(np.diag(cov_c)[mask].sum())

        final_var = occupied_variance_final(run["traj_path"], mask, xs_c, ys_c, cov_c)
        results[map_id][run["beta"]].append(final_var)
        print(f"map={map_id} beta={run['beta']} repeat={run['repeat']}: final_occupied_var={final_var:.4f}")

    # mean_table[map_id][beta] = mean final occupied variance across repeats
    mean_table = {
        map_id: {beta: float(np.mean(results[map_id][beta])) for beta in BETAS if beta in results[map_id]}
        for map_id in MAP_IDS
    }

    # --- matplotlib grouped bar chart ---
    fig, ax = plt.subplots(figsize=(11, 6))
    n_betas = len(BETAS)
    bar_width = 0.8 / n_betas
    x = np.arange(len(MAP_IDS))
    cmap = plt.get_cmap("viridis")
    for i, beta in enumerate(BETAS):
        values = [mean_table[map_id].get(beta, np.nan) for map_id in MAP_IDS]
        # align="edge": bar spans exactly [x+i*bar_width, x+i*bar_width+bar_width],
        # so the bin's true horizontal center is base + bar_width/2 - matplotlib's
        # default align="center" would have made that +bar_width/2 land on the
        # bar's right edge instead, which is why the dot was off before.
        ax.bar(x + i * bar_width, values, width=bar_width, align="edge",
               label=f"beta={beta}", color=cmap(i / max(1, n_betas - 1)))

    # mark the winning beta per map with a dot at the midpoint of its bin
    # (horizontally centered on the bar, vertically at half the bar's height)
    ax.set_ylim(bottom=0)
    winner_xs, winner_ys = [], []
    for j, map_id in enumerate(MAP_IDS):
        best_beta = min(mean_table[map_id], key=mean_table[map_id].get)
        best_idx = BETAS.index(best_beta)
        best_val = mean_table[map_id][best_beta]
        winner_xs.append(x[j] + best_idx * bar_width + bar_width / 2)
        winner_ys.append(best_val / 2.0)
    ax.scatter(winner_xs, winner_ys, marker="o", s=90, color="red", zorder=5,
               edgecolors="white", linewidths=1.2, label="winner (lowest final variance)")

    ax.set_xticks(x + bar_width * n_betas / 2)
    ax.set_xticklabels([f"map {m_id}" for m_id in MAP_IDS])
    ax.set_ylabel("Final occupied-region variance (mean of 3 repeats, lower=better)")
    ax.set_title("CMA-ES beta sweep, maps 51-55 (maxiter=20, 200 timesteps)\nred dot marks the winning beta per map (lowest final variance)")
    ax.legend(title="beta", loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path = OUT_DIR / "beta_sweep_maps51-55_grouped_bar.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")

    # --- LaTeX table, winning beta bolded per row ---
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Final occupied-region variance (mean of 3 repeats) by map and $\beta$, CMA-ES, maxiter=20, 200 timesteps. Bold = best (lowest) $\beta$ per map.}")
    lines.append(r"\label{tab:beta-sweep-maps51-55}")
    col_spec = "l" + "c" * len(BETAS)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = "Map & " + " & ".join(rf"$\beta={b}$" for b in BETAS) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for map_id in MAP_IDS:
        best_beta = min(mean_table[map_id], key=mean_table[map_id].get)
        cells = []
        for beta in BETAS:
            val = mean_table[map_id].get(beta, float("nan"))
            cell = f"{val:.2f}"
            if beta == best_beta:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{map_id} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    latex_table = "\n".join(lines)

    table_path = OUT_DIR / "beta_sweep_maps51-55_table.tex"
    table_path.write_text(latex_table, encoding="utf-8")
    print(f"Wrote {table_path}")
    print()
    print(latex_table)

    print()
    print("Winning beta per map:")
    for map_id in MAP_IDS:
        best_beta = min(mean_table[map_id], key=mean_table[map_id].get)
        print(f"  map {map_id}: beta={best_beta} (final_occupied_var={mean_table[map_id][best_beta]:.4f})")


if __name__ == "__main__":
    main()
