"""
Updates checkpoint_sweep_time_to_80pct_completion.png with the extended-wallclock rerun
results (rerun_failed_80pct_checkpoints_summary.csv, WALLCLOCK_SECONDS=400) for the 13
runs that never reached 80% occupied variance reduction within the original 150s cap.

Three visual categories now, distinguished by marker style (not just color), since
blending them together without a visual cue would misrepresent "took a while" and "reached
within the original budget" as the same thing:
  - filled circle : reached 80% reduction within the original 150s
  - open circle    : did NOT reach it within 150s, but did within the extended 400s budget
                      (same seed/checkpoint - this is genuinely how long that run needed,
                      not a different draw)
  - x at the bottom: still had not reached 80% reduction even at 400s

Diffusion epoch 10's mean/std now uses all 3 repeats (two within 150s, one needing the
extended budget) rather than the partial 2-of-3 mean from the first version.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PER_RUN_CSV = SCRIPT_DIR / "checkpoint_sweep_time_to_80pct_completion_per_run.csv"
EXTENDED_CSV = SCRIPT_DIR / "rerun_failed_80pct_checkpoints_summary.csv"

COLOR_DIFFUSION = "tab:red"
COLOR_IMITATION = "tab:purple"


def load_original():
    with PER_RUN_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    # (planner, epoch, repeat) -> time or None
    values = {}
    for r in rows:
        key = (r["planner"], int(r["epoch"]), int(r["repeat"]))
        values[key] = float(r["time_to_80pct_completion_s"]) if r["time_to_80pct_completion_s"] != "" else None
    return values


def load_extended():
    with EXTENDED_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    extended = {}
    for r in rows:
        key = (r["planner"], int(r["epoch"]))
        extended[key] = {
            "status": r["status"],
            "time": float(r["time_to_80pct_reduction_s"]) if r.get("time_to_80pct_reduction_s") not in (None, "") else None,
        }
    return extended


def main():
    original = load_original()
    extended = load_extended()

    # repeat index used by the extended rerun for each (planner, epoch) - diffusion used
    # repeat 2 (the one that failed); ImitateTrans has only repeat 0.
    extended_repeat = defaultdict(lambda: 0)
    extended_repeat[("diffusion", 10)] = 2

    all_keys = sorted({(p, e) for (p, e, _r) in original})
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    plotted = defaultdict(lambda: {"epoch": [], "mean": [], "std": [], "extended_mask": []})
    zero_reach = defaultdict(list)

    for planner, epoch in all_keys:
        repeats = sorted(r for (p, e, r) in original if p == planner and e == epoch)
        times = []
        was_extended = False
        for rep in repeats:
            t = original[(planner, epoch, rep)]
            if t is not None:
                times.append(t)
                continue
            # This repeat failed within 150s - see if the extended rerun resolved it.
            ext = extended.get((planner, epoch))
            if ext is not None and rep == extended_repeat[(planner, epoch)]:
                was_extended = True
                if ext["status"] == "reached":
                    times.append(ext["time"])
                # else: still not reached even at 400s - contributes no value
        if times:
            plotted[planner]["epoch"].append(epoch)
            plotted[planner]["mean"].append(np.mean(times))
            plotted[planner]["std"].append(np.std(times) if len(times) > 1 else 0.0)
            plotted[planner]["extended_mask"].append(was_extended)
        else:
            zero_reach[planner].append(epoch)

    for planner, color, label in [("diffusion", COLOR_DIFFUSION, "Diffusion"), ("imitation", COLOR_IMITATION, "ImitateTrans")]:
        d = plotted[planner]
        if not d["epoch"]:
            continue
        epochs = np.array(d["epoch"])
        means = np.array(d["mean"])
        stds = np.array(d["std"])
        ext_mask = np.array(d["extended_mask"])

        n_repeats = 3 if planner == "diffusion" else 1
        run_label = "3 seeds" if planner == "diffusion" else "1 run"
        ax.plot(epochs, means, color=color, linewidth=1.5, zorder=2, label=f"{label} ({run_label})")
        ax.errorbar(epochs[~ext_mask], means[~ext_mask], yerr=stds[~ext_mask], marker="o",
                    color=color, capsize=3, linestyle="none", zorder=3, markersize=7)
        if ext_mask.any():
            ax.errorbar(epochs[ext_mask], means[ext_mask], yerr=stds[ext_mask], marker="o",
                        markerfacecolor="white", markeredgecolor=color, markeredgewidth=2,
                        ecolor=color, capsize=3, linestyle="none", zorder=4, markersize=8)

    # Legend entries for the marker meanings (separate from the per-planner color legend).
    from matplotlib.lines import Line2D
    marker_legend = [
        Line2D([0], [0], marker="o", color="gray", linestyle="none", markersize=7, label="reached within 150s"),
        Line2D([0], [0], marker="o", color="gray", linestyle="none", markersize=8,
               markerfacecolor="white", markeredgewidth=2, label="reached only within extended 400s"),
        Line2D([0], [0], marker="x", color="gray", linestyle="none", markersize=9, markeredgewidth=2,
               label="still not reached even at 400s"),
    ]

    y_min, y_max = ax.get_ylim()
    marker_y = y_min - 0.08 * (y_max - y_min)
    ax.set_ylim(marker_y - 0.05 * (y_max - y_min), y_max)
    for planner, color in [("diffusion", COLOR_DIFFUSION), ("imitation", COLOR_IMITATION)]:
        if zero_reach[planner]:
            ax.scatter(zero_reach[planner], [marker_y] * len(zero_reach[planner]), marker="x", s=60,
                       color=color, linewidth=2, zorder=4)

    ax.set_xlabel("Training epoch (checkpoint)")
    ax.set_ylabel("Time to 80% occupied variance reduction (s)")
    ax.set_title(
        "Time to 80% occupied variance completion vs. training epoch (map 59)\n"
        "open markers = needed the extended 400s budget to get there at all",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)
    color_legend = ax.legend(loc="upper right", fontsize=8)
    ax.add_artist(color_legend)
    ax.legend(handles=marker_legend, loc="lower right", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    out_path = SCRIPT_DIR / "checkpoint_sweep_time_to_80pct_completion.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")

    print("\nDiffusion epoch 10 updated stats:", plotted["diffusion"]["mean"][plotted["diffusion"]["epoch"].index(10)]
          if 10 in plotted["diffusion"]["epoch"] else "N/A")
    print("Still-zero-reach epochs after extension:")
    for planner in ("diffusion", "imitation"):
        if zero_reach[planner]:
            print(f"  {planner}: {zero_reach[planner]}")


if __name__ == "__main__":
    main()
