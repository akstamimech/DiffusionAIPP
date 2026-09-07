"""
Compares the diffusion and ImitateTrans ("Imitation") execution-horizon sweep
results (maps 51-59, wallclock=100s, ENFORCE_MIN_STEP_TIME=1) across the four
metrics reported in their respective tables. Reads both the per-horizon
averaged analysis CSVs and the per-(map,horizon) raw CSVs written by
analyze_diffusion_horizon_maps51-59.py and analyze_imitatetrans_horizon_maps51-59.py.

Each point shows: the mean marker and line (as before), a thin error bar
spanning the min-max range across the 9 maps at that horizon, and a small
jittered scatter of the 9 individual per-map values so the underlying spread
- not just its range - is visible.

Palette: dataviz skill's validated categorical slots 1 (blue, Diffusion) and
2 (orange, Imitation) - documented in the skill's reference palette as
passing all pairwise CVD/contrast gates in light mode.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DIFFUSION_CSV = SCRIPT_DIR / "results_diffusion_horizon_maps51-59_analysis.csv"
IMITATE_CSV = SCRIPT_DIR / "results_imitate_horizon_maps51-59_analysis.csv"
DIFFUSION_PER_RUN_CSV = SCRIPT_DIR / "results_diffusion_horizon_maps51-59_per_run.csv"
IMITATE_PER_RUN_CSV = SCRIPT_DIR / "results_imitate_horizon_maps51-59_per_run.csv"
OUT_DIR = SCRIPT_DIR / "results_horizon_diffusion_vs_imitate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COLOR_DIFFUSION = "#2a78d6"   # categorical slot 1 (blue)
COLOR_IMITATION = "#eb6834"   # categorical slot 2 (orange)

# Horizontal offsets (in horizon units) so each architecture's per-map scatter
# cluster sits beside, not on top of, the mean marker/line at the true horizon.
SCATTER_SIDE_OFFSET = 1.6
SCATTER_JITTER_WIDTH = 0.7

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK_PRIMARY,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load_csv(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(float(r["horizon"])))
    return rows


def series(rows, key):
    return [float(r[key]) for r in rows]


def group_by_horizon(per_run_rows, key):
    groups = defaultdict(list)
    for r in per_run_rows:
        groups[int(float(r["horizon"]))].append(float(r[key]))
    return groups


def style_axes(ax, ylabel, title):
    ax.set_xlabel("Execution horizon", color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(length=0)


def _scatter_x(horizon, n, side_sign):
    if n <= 1:
        return np.array([horizon + side_sign * SCATTER_SIDE_OFFSET])
    jitter = np.linspace(-SCATTER_JITTER_WIDTH / 2, SCATTER_JITTER_WIDTH / 2, n)
    return horizon + side_sign * SCATTER_SIDE_OFFSET + jitter


def _plot_one_series(ax, horizons, groups, color, label, side_sign):
    means = [float(np.mean(groups[h])) for h in horizons]
    mins = [float(np.min(groups[h])) for h in horizons]
    maxs = [float(np.max(groups[h])) for h in horizons]
    lower_err = [max(0.0, m - lo) for m, lo in zip(means, mins)]
    upper_err = [max(0.0, hi - m) for hi, m in zip(means, maxs)]

    for h in horizons:
        vals = groups[h]
        xs = _scatter_x(h, len(vals), side_sign)
        ax.scatter(xs, vals, color=color, s=16, alpha=0.35, linewidths=0, zorder=2)

    ax.errorbar(
        horizons, means, yerr=[lower_err, upper_err],
        color=color, ecolor=color, elinewidth=1.4, capsize=4, capthick=1.4,
        alpha=0.9, fmt="none", zorder=3,
    )
    ax.plot(
        horizons, means, color=color, linewidth=2,
        marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=1,
        label=label, zorder=4,
    )
    return means, mins, maxs


def plot_metric(ax, horizons, diff_groups, imit_groups, ylabel, title, pct=False, log=False):
    _plot_one_series(ax, horizons, diff_groups, COLOR_DIFFUSION, "Diffusion", side_sign=-1)
    _plot_one_series(ax, horizons, imit_groups, COLOR_IMITATION, "Imitation", side_sign=+1)
    style_axes(ax, ylabel, title)
    ax.set_xticks(horizons)
    ax.set_xlim(horizons[0] - 3.2, horizons[-1] + 3.2)
    if pct:
        ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.0f}%")
    if log:
        ax.set_yscale("log")


def main():
    diff_rows = load_csv(DIFFUSION_CSV)
    imit_rows = load_csv(IMITATE_CSV)
    horizons = [int(float(r["horizon"])) for r in diff_rows]
    assert horizons == [int(float(r["horizon"])) for r in imit_rows], "horizon grids must match"

    with DIFFUSION_PER_RUN_CSV.open(newline="") as f:
        diff_per_run = list(csv.DictReader(f))
    with IMITATE_PER_RUN_CSV.open(newline="") as f:
        imit_per_run = list(csv.DictReader(f))

    metrics = [
        ("occupied_trP_pct_drop", "Occupied Tr(P) drop (%)", "Occupied variance drop (higher = better)", True, False, "pct_drop"),
        ("final_occupied_rmse", "Final occupied RMSE", "Final occupied RMSE (lower = better)", False, False, "final_rmse"),
        ("occupied_trP_auc", "Occupied Tr(P)-AUC", "Time-averaged occupied variance (lower = better)", False, False, "trP_auc"),
        ("total_replan_time", "Total replanning time (s, log scale)", "Total replanning cost (lower = cheaper)", False, True, "replan_time"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
    fig.suptitle(
        "Diffusion vs. Imitation: execution-horizon sweep (maps 51-59, 100 s wall-clock)\n"
        "markers = mean across 9 maps, bars = min-max range, dots = individual maps",
        color=INK_PRIMARY, fontsize=12.5, y=0.99,
    )
    flat_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for ax, (key, ylabel, title, pct, log, tag) in zip(flat_axes, metrics):
        diff_groups = group_by_horizon(diff_per_run, key)
        imit_groups = group_by_horizon(imit_per_run, key)
        plot_metric(ax, horizons, diff_groups, imit_groups, ylabel, title, pct=pct, log=log)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, -0.01), labelcolor=INK_SECONDARY,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    combined_path = OUT_DIR / "diffusion_vs_imitation_horizon_sweep.png"
    fig.savefig(combined_path, dpi=160)
    print(f"Wrote {combined_path}")

    for key, ylabel, title, pct, log, tag in metrics:
        diff_groups = group_by_horizon(diff_per_run, key)
        imit_groups = group_by_horizon(imit_per_run, key)
        fig_i, ax_i = plt.subplots(figsize=(5.8, 4.7))
        plot_metric(ax_i, horizons, diff_groups, imit_groups, ylabel, title, pct=pct, log=log)
        ax_i.legend(frameon=False, labelcolor=INK_SECONDARY, loc="best")
        fig_i.tight_layout()
        out_path = OUT_DIR / f"horizon_sweep_{tag}.png"
        fig_i.savefig(out_path, dpi=160)
        plt.close(fig_i)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
