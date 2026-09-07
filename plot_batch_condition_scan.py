"""
Plots results_multimodal_branch_test/batch_condition_scan.csv - the 20-
condition diffusion multimodality scan (test_multimodal_branch_batch.py).

Two panels:
  (a) win fraction for the winning branch per condition (0.5 = perfect
      50/50 split, 1.0 = total collapse onto one branch) - shows where each
      condition sits on the collapse spectrum.
  (b) mean distance to the winning branch, normalized by the branch-to-branch
      distance itself (>1.0 means the samples are, on average, farther from
      their "closer" branch than the two branches are from each other - i.e.
      not really hitting anything specific, as opposed to a low ratio
      indicating genuine accurate reproduction of that mode).

Conditions are sorted by win fraction so panel (a) reads as a spectrum from
most-mixed to most-collapsed, with panel (b) plotted in the same order.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "results_multimodal_branch_test" / "batch_condition_scan.csv"
OUT_PATH = SCRIPT_DIR / "results_multimodal_branch_test" / "batch_condition_scan.png"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COLOR_WIN_FRAC = "#2a78d6"    # categorical slot 1 (blue)
COLOR_DIST_RATIO = "#eb6834"  # categorical slot 2 (orange)
COLOR_COLLAPSED = "#e34948"   # categorical slot 8 (red) - marks fully-collapsed conditions

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK_PRIMARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style_axes(ax, ylabel, title):
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(length=0, colors=INK_MUTED)


def main():
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    records = []
    for r in rows:
        n = int(r["n_samples"])
        wins_winner = max(int(r["wins_branch0"]), int(r["wins_branch1"]))
        win_frac = wins_winner / n
        d0, d1 = float(r["mean_dist_branch0"]), float(r["mean_dist_branch1"])
        winner_dist = d0 if int(r["winner"]) == 0 else d1
        branch_dist = float(r["branch_dist"])
        records.append({
            "label": f"map{r['map_id']} ts{r['timestep']}",
            "win_frac": win_frac,
            "dist_ratio": winner_dist / branch_dist,
            "collapsed": r["collapsed"] == "True",
        })

    records.sort(key=lambda x: x["win_frac"])
    labels = [rec["label"] for rec in records]
    win_fracs = [rec["win_frac"] for rec in records]
    dist_ratios = [rec["dist_ratio"] for rec in records]
    colors_a = [COLOR_COLLAPSED if rec["collapsed"] else COLOR_WIN_FRAC for rec in records]
    x = np.arange(len(records))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)

    axes[0].bar(x, win_fracs, color=colors_a, zorder=3)
    axes[0].axhline(0.5, color=INK_MUTED, linewidth=1, linestyle="--", zorder=2)
    axes[0].axhline(1.0, color=COLOR_COLLAPSED, linewidth=1, linestyle=":", zorder=2)
    axes[0].set_ylim(0.4, 1.05)
    style_axes(axes[0], "Win fraction (winning branch)", "How collapsed is each condition? (0.5 = even split, 1.0 = total collapse)")

    axes[1].bar(x, dist_ratios, color=COLOR_DIST_RATIO, zorder=3)
    axes[1].axhline(1.0, color=INK_MUTED, linewidth=1, linestyle="--", zorder=2,
                     label="branch-to-branch distance (samples this far = not hitting anything specific)")
    style_axes(axes[1], "Mean dist. to winning branch / branch-to-branch dist.",
               "Is the \"winning\" branch actually being reproduced accurately?")
    axes[1].legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5)

    fig.suptitle(
        "sparse_trans_waypoints_epoch_1000.pth - 20 multi-branch conditions, 30 diffusion samples each",
        fontsize=12.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=160)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
