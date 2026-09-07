"""
For the maxiter sweep on maps 51-55 (sweep_cmaes_maxiter_multimap.py, beta=1,
3 repeats/map/maxiter, TIMEALLOTED=200): computes ground-truth occupied-region
variance drop per run (coarse-grid Kalman replay, same methodology as
important_region_variance_from_trajectories.py) and pulls real per-run
wall-clock time (process_wall_seconds, already recorded by hpc_sweep_common,
ENFORCE_MIN_STEP_TIME off so this is genuine compute time). Produces a
matplotlib grouped-bar figure + LaTeX table (per map, winning maxiter bolded),
plus a second, map-independent summary table of mean quality vs. mean
wall-clock time per maxiter value, since compute cost doesn't depend on map
content but quality does.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
SWEEP_ROOT = SCRIPT_DIR / "results_cmaes_maxiter_sweep_maps51-55" / "planner_outputs"
SUMMARY_CSV = SCRIPT_DIR / "results_cmaes_maxiter_sweep_maps51-55" / "summaries" / "cmaes_maxiter_multimap_summary.csv"
OUT_DIR = SCRIPT_DIR / "results_cmaes_maxiter_sweep_maps51-55" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIR_PATTERN = re.compile(
    r"^classic_map_grf_(?P<mapid>\d+)_viz_hpc_cmaes_maxiter(?P<maxiter>\d+)_map\d+_repeat(?P<repeat>\d+)_seed\d+$"
)
MAP_IDS = [51, 52, 53, 54, 55]
MAXITERS = [2, 8, 20, 45]


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
            "maxiter": int(match.group("maxiter")),
            "repeat": int(match.group("repeat")),
            "traj_path": str(traj_path),
        })
    return runs


def occupied_variance_auc(traj_path, mask, xs_c, ys_c, cov_c):
    """Replays the full occupied-region variance curve (not just the final
    value) and returns its WALL-CLOCK-time-averaged trapezoidal AUC (using
    the executed_trajectory.csv's own recorded wall_time_seconds column, not
    timestep index) - i.e. the time-averaged variance over however long this
    particular run actually took in real seconds. This is deliberately
    per-run self-contained (each run normalizes by its own total wall-clock
    duration), not a shared-clock alignment across runs of different total
    duration - that's the case the earlier "don't use wall-clock AUC when
    sweeping a compute-time-affecting parameter" caution applies to, not this
    one, since here time-efficiency itself is exactly the thing being asked
    for."""
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

    wall_times = np.array(wall_times)
    variance_curve = np.array(variance_curve)
    if len(variance_curve) <= 1 or wall_times[-1] <= wall_times[0]:
        return float(variance_curve[-1])
    return float(np.trapezoid(variance_curve, wall_times) / (wall_times[-1] - wall_times[0]))


def load_wall_times():
    # (map_id, maxiter, repeat) -> process_wall_seconds, straight from
    # hpc_sweep_common's own timing, independent of the replay analysis.
    wall_times = {}
    with SUMMARY_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["map_id"]), int(row["maxiter"]), int(row["repeat"]))
            wall_times[key] = float(row["process_wall_seconds"])
    return wall_times


def main():
    runs = discover_runs()
    print(f"Discovered {len(runs)} runs")
    if not runs:
        raise SystemExit("No runs found - has the sweep finished?")

    wall_times = load_wall_times()
    xs_c, ys_c, cov_c, coords_c = m.build_coarse_grid()
    mask_cache = {}

    results = defaultdict(lambda: defaultdict(list))  # results[map_id][maxiter] = [final_var, ...]
    wall_by_maxiter = defaultdict(list)  # wall_by_maxiter[maxiter] = [process_wall_seconds, ...] (all maps/repeats)

    for run in runs:
        map_id = run["map_id"]
        if map_id not in mask_cache:
            mask_cache[map_id], _ = m._important_mask_for_map(map_id, coords_c)
        mask = mask_cache[map_id]

        auc = occupied_variance_auc(run["traj_path"], mask, xs_c, ys_c, cov_c)
        results[map_id][run["maxiter"]].append(auc)

        wall_key = (map_id, run["maxiter"], run["repeat"])
        wall_seconds = wall_times.get(wall_key)
        if wall_seconds is not None:
            wall_by_maxiter[run["maxiter"]].append(wall_seconds)

        print(f"map={map_id} maxiter={run['maxiter']} repeat={run['repeat']}: "
              f"occupied_variance_auc={auc:.4f} wall={wall_seconds}")

    mean_table = {
        map_id: {mi: float(np.mean(results[map_id][mi])) for mi in MAXITERS if mi in results[map_id]}
        for map_id in MAP_IDS
    }

    # --- matplotlib grouped bar chart (per-map quality) ---
    fig, ax = plt.subplots(figsize=(11, 6))
    n = len(MAXITERS)
    bar_width = 0.8 / n
    x = np.arange(len(MAP_IDS))
    cmap = plt.get_cmap("viridis")
    for i, mi in enumerate(MAXITERS):
        values = [mean_table[map_id].get(mi, np.nan) for map_id in MAP_IDS]
        ax.bar(x + i * bar_width, values, width=bar_width, label=f"maxiter={mi}", color=cmap(i / max(1, n - 1)))

    winner_xs, winner_ys = [], []
    for j, map_id in enumerate(MAP_IDS):
        best_mi = min(mean_table[map_id], key=mean_table[map_id].get)
        best_idx = MAXITERS.index(best_mi)
        best_val = mean_table[map_id][best_mi]
        winner_xs.append(x[j] + best_idx * bar_width + bar_width / 2)
        winner_ys.append(best_val / 2.0)  # centered vertically within the bar, not floating above it
    ax.scatter(winner_xs, winner_ys, marker="o", s=90, color="red", zorder=5,
               edgecolors="white", linewidths=1.2, label="winner (lowest AUC)")

    ax.set_xticks(x + bar_width * (n - 1) / 2)
    ax.set_xticklabels([f"map {m_id}" for m_id in MAP_IDS])
    ax.set_ylabel("Occupied-region variance AUC, wall-clock-time-averaged (mean of 3 repeats, lower=better)")
    ax.set_title("CMA-ES maxiter sweep, maps 51-55 (beta=1, 200 timesteps)\nred dot marks the winning maxiter per map (lowest AUC)")
    ax.legend(title="maxiter", loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path = OUT_DIR / "maxiter_sweep_maps51-55_grouped_bar.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")

    # --- second figure: quality vs. wall-clock time tradeoff ---
    mean_var_by_maxiter = {mi: float(np.mean([mean_table[map_id][mi] for map_id in MAP_IDS if mi in mean_table[map_id]])) for mi in MAXITERS}
    mean_wall_by_maxiter = {mi: float(np.mean(wall_by_maxiter[mi])) for mi in MAXITERS if wall_by_maxiter[mi]}
    std_wall_by_maxiter = {mi: float(np.std(wall_by_maxiter[mi])) for mi in MAXITERS if wall_by_maxiter[mi]}

    fig2, ax2 = plt.subplots(figsize=(7.5, 6))
    xs_time = [mean_wall_by_maxiter[mi] for mi in MAXITERS]
    ys_var = [mean_var_by_maxiter[mi] for mi in MAXITERS]
    ax2.plot(xs_time, ys_var, marker="o", color="tab:blue", linewidth=2)
    for mi, xt, yv in zip(MAXITERS, xs_time, ys_var):
        ax2.annotate(f"maxiter={mi}", (xt, yv), textcoords="offset points", xytext=(8, 5), fontsize=10)
    ax2.set_xlabel("Mean process wall-clock time per run (s)")
    ax2.set_ylabel("Mean wall-clock-time-averaged occupied variance AUC across maps 51-55 (lower=better)")
    ax2.set_title("CMA-ES: quality (AUC) vs. compute-time tradeoff by maxiter")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    tradeoff_path = OUT_DIR / "maxiter_sweep_maps51-55_quality_vs_time.png"
    fig2.savefig(tradeoff_path, dpi=150)
    print(f"Wrote {tradeoff_path}")

    # --- LaTeX table 1: per-map quality, winning maxiter bolded ---
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Occupied-region variance AUC (mean of 3 repeats) by map and maxiter, CMA-ES, beta=1, 200 timesteps. Bold = best (lowest) maxiter per map.}")
    lines.append(r"\label{tab:maxiter-sweep-maps51-55}")
    col_spec = "l" + "c" * len(MAXITERS)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = "Map & " + " & ".join(rf"maxiter={mi}" for mi in MAXITERS) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for map_id in MAP_IDS:
        best_mi = min(mean_table[map_id], key=mean_table[map_id].get)
        cells = []
        for mi in MAXITERS:
            val = mean_table[map_id].get(mi, float("nan"))
            cell = f"{val:.2f}"
            if mi == best_mi:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{map_id} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    table1 = "\n".join(lines)
    table1_path = OUT_DIR / "maxiter_sweep_maps51-55_table.tex"
    table1_path.write_text(table1, encoding="utf-8")
    print(f"Wrote {table1_path}")

    # --- LaTeX table 2: map-independent quality vs. time summary ---
    lines2 = []
    lines2.append(r"\begin{table}[h]")
    lines2.append(r"\centering")
    lines2.append(r"\caption{Mean occupied-region variance AUC and mean real wall-clock time per run by maxiter, averaged across maps 51-55 and 3 repeats each (beta=1, 200 timesteps, ENFORCE\_MIN\_STEP\_TIME off).}")
    lines2.append(r"\label{tab:maxiter-sweep-time-tradeoff}")
    lines2.append(r"\begin{tabular}{lccc}")
    lines2.append(r"\toprule")
    lines2.append(r"Maxiter & Mean occupied variance AUC (wall-clock-integrated) & Mean wall-clock (s) & Std wall-clock (s) \\")
    lines2.append(r"\midrule")
    best_mi_overall = min(mean_var_by_maxiter, key=mean_var_by_maxiter.get)
    for mi in MAXITERS:
        var_cell = f"{mean_var_by_maxiter[mi]:.2f}"
        if mi == best_mi_overall:
            var_cell = rf"\textbf{{{var_cell}}}"
        lines2.append(
            f"{mi} & {var_cell} & {mean_wall_by_maxiter.get(mi, float('nan')):.1f} & "
            f"{std_wall_by_maxiter.get(mi, float('nan')):.1f} \\\\"
        )
    lines2.append(r"\bottomrule")
    lines2.append(r"\end{tabular}")
    lines2.append(r"\end{table}")
    table2 = "\n".join(lines2)
    table2_path = OUT_DIR / "maxiter_sweep_maps51-55_time_tradeoff_table.tex"
    table2_path.write_text(table2, encoding="utf-8")
    print(f"Wrote {table2_path}")

    print()
    print(table1)
    print()
    print(table2)

    print()
    print("Winning maxiter per map:")
    for map_id in MAP_IDS:
        best_mi = min(mean_table[map_id], key=mean_table[map_id].get)
        print(f"  map {map_id}: maxiter={best_mi} (occupied_variance_auc={mean_table[map_id][best_mi]:.4f})")

    print()
    print("Mean quality(AUC)/time per maxiter (all maps):")
    for mi in MAXITERS:
        print(f"  maxiter={mi}: mean_auc={mean_var_by_maxiter[mi]:.4f} mean_wall={mean_wall_by_maxiter.get(mi, float('nan')):.1f}s")


if __name__ == "__main__":
    main()
