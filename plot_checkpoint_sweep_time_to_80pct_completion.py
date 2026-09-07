"""
Time to reach 80% occupied variance completion (occupied_variance_fraction_left <= 0.20,
i.e. an 80% reduction from the initial value) vs. training epoch, for the current data for
both planners:
  - Diffusion: checkpoint_sweep_summary.csv (150s missions, 3 repeats/epoch)
  - ImitateTrans: checkpoint_sweep_imitatetrans_150s_v2_summary.csv (150s missions,
    EXECUTION_CHUNK=10, 1 deterministic run/epoch - the current/latest ImitateTrans sweep)

Both datasets are the same 150s mission length, so this is a matched comparison. Replayed
via important_region_variance_from_trajectories.reconstruct_one (Kalman replay from the
already-saved trajectories, no re-simulation) - fraction_left is important_variance(t) /
important_variance(0) along each trajectory.

Time-to-event with possible right-censoring: a run may never reach 80% reduction within
150s. Every epoch is shown on the x-axis regardless of reach rate - an epoch where NO run
reached the threshold gets an explicit marker at the bottom of the axis rather than being
silently dropped from the line (which would be indistinguishable from "not measured" and
was wrong in an earlier draft of this plot). Partial reach rates (some but not all repeats
reached it) get a percentage label instead.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
DIFFUSION_CSV = SCRIPT_DIR / "checkpoint_sweep_summary.csv"
IMITATE_CSV = SCRIPT_DIR / "checkpoint_sweep_imitatetrans_150s_v2_summary.csv"
SELECTED_MAP = 59
WALLCLOCK_CAP = 150.0
THRESHOLD = 0.20  # <=20% remaining = 80% reduction

COLOR_DIFFUSION = "tab:red"
COLOR_IMITATION = "tab:purple"


def load_runs(path, planner_key, has_repeat):
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    runs = []
    for r in rows:
        run = {
            "planner": planner_key,
            "epoch": int(r["epoch"]),
            "map_id": SELECTED_MAP,
            "traj_path": str(Path(r["output_dir"]) / f"map_{SELECTED_MAP}_executed_trajectory.csv"),
        }
        if has_repeat:
            run["repeat"] = int(r["repeat"])
        runs.append(run)
    return runs


def time_to_threshold(run):
    result = m.reconstruct_one(run)
    wall_time = result["wall_time_seconds"]
    important_var = result["important_variance"]
    within_cap = wall_time <= WALLCLOCK_CAP
    wt = wall_time[within_cap]
    iv = important_var[within_cap]
    if wt.size == 0:
        return None
    frac_left = iv / iv[0]
    hit = np.where(frac_left <= THRESHOLD)[0]
    if hit.size == 0:
        return None
    return float(wt[hit[0]])


def main():
    diff_runs = load_runs(DIFFUSION_CSV, "diffusion", has_repeat=True)
    imit_runs = load_runs(IMITATE_CSV, "imitation", has_repeat=False)
    print(f"Replaying {len(diff_runs)} Diffusion + {len(imit_runs)} ImitateTrans trajectories on map {SELECTED_MAP}...")

    times_by_planner_epoch = defaultdict(lambda: defaultdict(list))
    n_total_by_planner_epoch = defaultdict(lambda: defaultdict(int))
    per_run_rows = []
    for i, run in enumerate(diff_runs + imit_runs):
        t = time_to_threshold(run)
        n_total_by_planner_epoch[run["planner"]][run["epoch"]] += 1
        if t is not None:
            times_by_planner_epoch[run["planner"]][run["epoch"]].append(t)
        per_run_rows.append([run["planner"], run["epoch"], run.get("repeat", 0), t if t is not None else ""])
        if (i + 1) % 20 == 0:
            print(f"  replayed {i + 1}/{len(diff_runs) + len(imit_runs)} trajectories", flush=True)

    per_run_csv = SCRIPT_DIR / "checkpoint_sweep_time_to_80pct_completion_per_run.csv"
    with per_run_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["planner", "epoch", "repeat", "time_to_80pct_completion_s"])
        writer.writerows(per_run_rows)
    print(f"Wrote {per_run_csv}")

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    summary_lines = []

    plot_data = {}
    for planner, color, label in [
        ("diffusion", COLOR_DIFFUSION, "Diffusion (150s, 3 seeds)"),
        ("imitation", COLOR_IMITATION, "ImitateTrans (150s, 1 run)"),
    ]:
        epochs = sorted(n_total_by_planner_epoch[planner])
        means, stds, reached_frac = [], [], []
        for e in epochs:
            vals = times_by_planner_epoch[planner][e]
            n_tot = n_total_by_planner_epoch[planner][e]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0.0)
            reached_frac.append(len(vals) / n_tot)
            summary_lines.append(
                f"  {planner} epoch {e}: {len(vals)}/{n_tot} reached 80% reduction within {WALLCLOCK_CAP:.0f}s"
                + (f", mean time = {np.mean(vals):.1f}s" if vals else " - NONE reached")
            )
        plot_data[planner] = (np.array(epochs), np.array(means), np.array(stds), np.array(reached_frac), color, label)

    for planner, (epochs_arr, means_arr, stds_arr, reached_frac, color, label) in plot_data.items():
        valid = ~np.isnan(means_arr)
        ax.errorbar(epochs_arr[valid], means_arr[valid], yerr=stds_arr[valid], marker="o",
                    color=color, label=label, capsize=3, linewidth=1.8, zorder=3)
        for x, y, frac in zip(epochs_arr[valid], means_arr[valid], reached_frac[valid]):
            if frac < 1.0:
                ax.annotate(f"{frac:.0%}", (x, y), textcoords="offset points",
                            xytext=(0, 10 if planner == "diffusion" else -14),
                            fontsize=7, color=color, ha="center")

    y_min, y_max = ax.get_ylim()
    marker_y = y_min - 0.08 * (y_max - y_min)
    ax.set_ylim(marker_y - 0.05 * (y_max - y_min), y_max)
    for planner, (epochs_arr, means_arr, stds_arr, reached_frac, color, label) in plot_data.items():
        zero_reach = epochs_arr[np.isnan(means_arr)]
        if zero_reach.size:
            planner_name = "Diffusion" if planner == "diffusion" else "ImitateTrans"
            ax.scatter(zero_reach, [marker_y] * len(zero_reach), marker="x", s=60,
                       color=color, linewidth=2, zorder=4, label=f"{planner_name}: 0% reached")

    ax.set_xlabel("Training epoch (checkpoint)")
    ax.set_ylabel("Time to 80% occupied variance reduction (s)")
    ax.set_title(
        "Time to 80% occupied variance completion vs. training epoch (map 59, both 150s missions)\n"
        "% labels = partial reach rate; × at bottom = 0% of runs reached the threshold at all",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path = SCRIPT_DIR / "checkpoint_sweep_time_to_80pct_completion.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    print()
    for line in summary_lines:
        print(line)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
