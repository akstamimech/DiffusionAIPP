"""
Occupied-region variance drop at mission end (150s), per checkpoint epoch, Diffusion vs
ImitateTrans - the same "later-stage strength" framing as
plot_checkpoint_sweep_late_stage.py, but using occupied variance drop instead of task
completion.

Neither checkpoint sweep script logs occupied-region variance directly (Diffusionplanner_
singlemap.py / ImitateTrans_singlemap.py only log whole-grid `global_variance` per
timestep - see their rmse_over_time.csv header). Re-running 96 simulations just to get a
different metric isn't necessary though: the Kalman covariance update depends only on
sensor position/altitude (from the already-saved executed_trajectory.csv), never on the
measured values, so the full covariance trajectory - and hence occupied variance at any
timestep - can be replayed exactly from the recorded poses alone. This reuses
important_region_variance_from_trajectories.py's replay machinery directly rather than
reimplementing it.

Occupied variance drop = value at t=0 (before any measurement) minus value at the last
timestep with wall_time_seconds <= 150 (some runs' last logged row is a fraction of a
second past 150 since ENFORCE_MIN_STEP_TIME is a floor, not a ceiling - truncated for a
clean, consistent 150s comparison, same reasoning as the earlier results_hpc mission-length
fix this session).
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
IMITATE_CSV = SCRIPT_DIR / "checkpoint_sweep_imitatetrans_summary.csv"
SELECTED_MAP = 59
WALLCLOCK_CAP = 150.0
LATE_STAGE_MIN_EPOCH = 200

COLOR_DIFFUSION = "tab:red"
COLOR_IMITATION = "tab:purple"


def load_runs(path, planner_key):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        {"planner": planner_key, "epoch": int(r["epoch"]), "repeat": int(r["repeat"]),
         "map_id": SELECTED_MAP, "traj_path": str(Path(r["output_dir"]) / f"map_{SELECTED_MAP}_executed_trajectory.csv")}
        for r in rows if r["status"] == "ok"
    ]


def occupied_variance_drop(run):
    result = m.reconstruct_one(run)
    wall_time = result["wall_time_seconds"]
    important_var = result["important_variance"]
    within_cap = wall_time <= WALLCLOCK_CAP
    initial = float(important_var[0])
    final = float(important_var[within_cap][-1]) if within_cap.any() else initial
    return initial - final


def main():
    diff_runs = load_runs(DIFFUSION_CSV, "diffusion")
    imit_runs = load_runs(IMITATE_CSV, "imitation")
    print(f"Replaying {len(diff_runs)} Diffusion + {len(imit_runs)} ImitateTrans trajectories on map {SELECTED_MAP}...")

    drops_by_planner_epoch = defaultdict(lambda: defaultdict(list))
    per_run_rows = []
    for i, run in enumerate(diff_runs + imit_runs):
        drop = occupied_variance_drop(run)
        drops_by_planner_epoch[run["planner"]][run["epoch"]].append(drop)
        per_run_rows.append([run["planner"], run["epoch"], run["repeat"], drop])
        if (i + 1) % 20 == 0:
            print(f"  replayed {i + 1}/{len(diff_runs) + len(imit_runs)} trajectories", flush=True)

    per_run_csv = SCRIPT_DIR / "checkpoint_sweep_occupied_variance_drop_per_run.csv"
    with per_run_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["planner", "epoch", "repeat", "occupied_variance_drop"])
        writer.writerows(per_run_rows)
    print(f"Wrote {per_run_csv}")

    def series(planner, min_epoch=0):
        epochs = sorted(e for e in drops_by_planner_epoch[planner] if e >= min_epoch)
        means = np.array([np.mean(drops_by_planner_epoch[planner][e]) for e in epochs])
        stds = np.array([np.std(drops_by_planner_epoch[planner][e]) for e in epochs])
        return np.array(epochs), means, stds

    # --- Full-range view ---
    de_all, dm_all, ds_all = series("diffusion")
    ie_all, im_all, istd_all = series("imitation")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.errorbar(de_all, dm_all, yerr=ds_all, marker="o", color=COLOR_DIFFUSION, label="Diffusion", capsize=3, linewidth=1.8)
    ax.errorbar(ie_all, im_all, yerr=istd_all, marker="o", color=COLOR_IMITATION, label="ImitateTrans", capsize=3, linewidth=1.8)
    ax.set_xlabel("Training epoch (checkpoint)")
    ax.set_ylabel("Occupied variance drop at 150s")
    ax.set_title(f"Occupied variance drop vs. training epoch (map {SELECTED_MAP}, mean ± std across 3 seeds)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    full_path = SCRIPT_DIR / "checkpoint_sweep_occupied_variance_drop.png"
    fig.savefig(full_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {full_path}")

    # --- Late-stage view: zoomed + gap panel, same framing as plot_checkpoint_sweep_late_stage.py ---
    de, dm, ds = series("diffusion", LATE_STAGE_MIN_EPOCH)
    ie, im, istd = series("imitation", LATE_STAGE_MIN_EPOCH)
    im_at_de = np.interp(de, ie, im)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    ax = axes[0]
    ax.fill_between(de, im_at_de, dm, where=(dm >= im_at_de), color=COLOR_DIFFUSION, alpha=0.12, zorder=1)
    ax.errorbar(de, dm, yerr=ds, marker="o", color=COLOR_DIFFUSION, label="Diffusion", capsize=3, linewidth=1.8, zorder=3)
    ax.errorbar(ie, im, yerr=istd, marker="o", color=COLOR_IMITATION, label="ImitateTrans", capsize=3, linewidth=1.8, zorder=3)
    ax.set_xlabel("Training epoch (checkpoint)")
    ax.set_ylabel("Occupied variance drop at 150s")
    ax.set_title(f"Occupied variance drop, epoch ≥ {LATE_STAGE_MIN_EPOCH}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    ax2 = axes[1]
    gap = dm - im_at_de
    ax2.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax2.bar(de, gap, width=(de[1] - de[0]) * 0.6 if len(de) > 1 else 40, color=COLOR_DIFFUSION, alpha=0.6)
    ax2.set_xlabel("Training epoch (checkpoint)")
    ax2.set_ylabel("Occupied variance drop gap (Diffusion − ImitateTrans)")
    ax2.set_title("Diffusion's margin over ImitateTrans")
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"Later-training comparison (epoch ≥ {LATE_STAGE_MIN_EPOCH}): map {SELECTED_MAP}, mean ± std across 3 seeds",
        fontsize=12,
    )
    fig.tight_layout()
    late_path = SCRIPT_DIR / "checkpoint_sweep_late_stage_occupied_variance_drop.png"
    fig.savefig(late_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {late_path}")
    print(f"Gap stats: mean={gap.mean():.2f} min={gap.min():.2f} max={gap.max():.2f}")


if __name__ == "__main__":
    main()
