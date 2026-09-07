"""
Time to reach 90% occupied variance completion (occupied_variance_fraction_left <= 0.10,
i.e. a 90% reduction from the initial value) vs. training epoch, for the current data for
both planners:
  - Diffusion: checkpoint_sweep_summary.csv (150s missions, 3 repeats/epoch)
  - ImitateTrans: checkpoint_sweep_imitatetrans_150s_v2_summary.csv (150s missions,
    EXECUTION_CHUNK=10, 1 deterministic run/epoch - the current/latest ImitateTrans sweep)

Both datasets are now the same 150s mission length, so this is a matched comparison unlike
the earlier 150s-vs-100s plot. Reconstructed via the same Kalman-replay machinery as the
other checkpoint-sweep analyses (important_region_variance_from_trajectories.reconstruct_
one) - no re-simulation needed, fraction_left is derived from important_variance(t) /
important_variance(0) along each already-saved trajectory.

This is a time-to-event metric with possible right-censoring (a run might never reach 90%
reduction within 150s) - same handling as the earlier results_hpc time-to-threshold
analysis: non-reached runs are excluded from the mean/line, not silently imputed, and the
reached-fraction is reported and printed so a thin line isn't mistaken for a complete one.
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
THRESHOLD = 0.10  # <=10% remaining = 90% reduction

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

    per_run_csv = SCRIPT_DIR / "checkpoint_sweep_time_to_90pct_completion_per_run.csv"
    with per_run_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["planner", "epoch", "repeat", "time_to_90pct_completion_s"])
        writer.writerows(per_run_rows)
    print(f"Wrote {per_run_csv}")

    def series(planner):
        epochs = sorted(times_by_planner_epoch[planner])
        means, stds, reached_frac = [], [], []
        for e in epochs:
            vals = times_by_planner_epoch[planner][e]
            n_total = n_total_by_planner_epoch[planner][e]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0.0)
            reached_frac.append(len(vals) / n_total)
            print(f"  {planner} epoch {e}: {len(vals)}/{n_total} reached 90% reduction within {WALLCLOCK_CAP:.0f}s"
                  + (f", mean time = {np.mean(vals):.1f}s" if vals else " - NONE reached"))
        return np.array(epochs), np.array(means), np.array(stds), np.array(reached_frac)

    de, dm, ds, d_reached = series("diffusion")
    ie, im, istd, i_reached = series("imitation")

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.errorbar(de, dm, yerr=ds, marker="o", color=COLOR_DIFFUSION,
                label="Diffusion (150s, mean ± std, 3 seeds)", capsize=3, linewidth=1.8)
    ax.plot(ie, im, marker="o", color=COLOR_IMITATION,
            label="ImitateTrans (150s, single deterministic run)", linewidth=1.8)

    # Flag any epoch where not every run reached the threshold, since a partial mean would
    # otherwise look identical to a complete one.
    for x, y, frac in zip(de, dm, d_reached):
        if frac < 1.0:
            ax.annotate(f"{frac:.0%}", (x, y), textcoords="offset points", xytext=(0, 8),
                        fontsize=7, color=COLOR_DIFFUSION, ha="center")
    for x, y, frac in zip(ie, im, i_reached):
        if frac < 1.0:
            ax.annotate(f"{frac:.0%}", (x, y), textcoords="offset points", xytext=(0, -12),
                        fontsize=7, color=COLOR_IMITATION, ha="center")

    ax.set_xlabel("Training epoch (checkpoint)")
    ax.set_ylabel("Time to 90% occupied variance reduction (s)")
    ax.set_title(
        f"Time to 90% occupied variance completion vs. training epoch (map {SELECTED_MAP}, both 150s missions)\n"
        "% labels shown only where not every run reached the threshold",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = SCRIPT_DIR / "checkpoint_sweep_time_to_90pct_completion.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
