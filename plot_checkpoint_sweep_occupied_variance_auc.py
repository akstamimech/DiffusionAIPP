"""
Occupied-region variance AUC (not just the endpoint drop) at 150s, per checkpoint epoch,
Diffusion vs ImitateTrans - same trajectory replay as
plot_checkpoint_sweep_occupied_variance_drop.py, but integrating the occupied-variance
curve itself (trapz against real wall_time_seconds, truncated to <=150s) rather than just
reading off its endpoints.

Same convention as "occupied variance AUC" everywhere else this session (evalmetrics.
compute_variance_time_metrics, the CMA-ES maxiter sweep table): the raw (non-time-
normalized) integral, in variance*seconds - LOWER is better (variance stayed lower for
more of the mission), not a "drop" curve (which would be higher-is-better and just an
additive shift of this same quantity: auc_of_drop = initial_variance*150 - this_value).

Prints and saves a mean +/- std table per (planner, epoch) across the 3 repeat seeds, plus
the same full-range and late-stage (epoch >= 200) plots as the sibling scripts.
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


def occupied_variance_auc(run):
    result = m.reconstruct_one(run)
    wall_time = result["wall_time_seconds"]
    important_var = result["important_variance"]
    within_cap = wall_time <= WALLCLOCK_CAP
    wt = wall_time[within_cap]
    iv = important_var[within_cap]
    if wt.size < 2:
        return 0.0
    return float(np.trapezoid(iv, wt))


def main():
    diff_runs = load_runs(DIFFUSION_CSV, "diffusion")
    imit_runs = load_runs(IMITATE_CSV, "imitation")
    print(f"Replaying {len(diff_runs)} Diffusion + {len(imit_runs)} ImitateTrans trajectories on map {SELECTED_MAP}...")

    auc_by_planner_epoch = defaultdict(lambda: defaultdict(list))
    per_run_rows = []
    for i, run in enumerate(diff_runs + imit_runs):
        auc = occupied_variance_auc(run)
        auc_by_planner_epoch[run["planner"]][run["epoch"]].append(auc)
        per_run_rows.append([run["planner"], run["epoch"], run["repeat"], auc])
        if (i + 1) % 20 == 0:
            print(f"  replayed {i + 1}/{len(diff_runs) + len(imit_runs)} trajectories", flush=True)

    per_run_csv = SCRIPT_DIR / "checkpoint_sweep_occupied_variance_auc_per_run.csv"
    with per_run_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["planner", "epoch", "repeat", "occupied_variance_auc"])
        writer.writerows(per_run_rows)
    print(f"Wrote {per_run_csv}")

    # --- mean +/- std summary table ---
    summary_rows = []
    for planner in ("diffusion", "imitation"):
        for epoch in sorted(auc_by_planner_epoch[planner]):
            vals = np.array(auc_by_planner_epoch[planner][epoch])
            summary_rows.append({
                "planner": planner,
                "epoch": epoch,
                "n": len(vals),
                "mean_occupied_variance_auc": float(vals.mean()),
                "std_occupied_variance_auc": float(vals.std()),
            })
    summary_csv = SCRIPT_DIR / "checkpoint_sweep_occupied_variance_auc_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_csv}")

    print(f"\n{'planner':<10} {'epoch':>6} {'n':>3} {'mean':>12} {'std':>10}")
    for row in summary_rows:
        print(f"{row['planner']:<10} {row['epoch']:>6} {row['n']:>3} "
              f"{row['mean_occupied_variance_auc']:>12.2f} {row['std_occupied_variance_auc']:>10.2f}")

    def series(planner, min_epoch=0):
        epochs = sorted(e for e in auc_by_planner_epoch[planner] if e >= min_epoch)
        means = np.array([np.mean(auc_by_planner_epoch[planner][e]) for e in epochs])
        stds = np.array([np.std(auc_by_planner_epoch[planner][e]) for e in epochs])
        return np.array(epochs), means, stds

    # --- Full-range view ---
    de_all, dm_all, ds_all = series("diffusion")
    ie_all, im_all, istd_all = series("imitation")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.errorbar(de_all, dm_all, yerr=ds_all, marker="o", color=COLOR_DIFFUSION, label="Diffusion", capsize=3, linewidth=1.8)
    ax.errorbar(ie_all, im_all, yerr=istd_all, marker="o", color=COLOR_IMITATION, label="ImitateTrans", capsize=3, linewidth=1.8)
    ax.set_xlabel("Training epoch (checkpoint)")
    ax.set_ylabel("Occupied variance AUC at 150s (lower = better)")
    ax.set_title(f"Occupied variance AUC vs. training epoch (map {SELECTED_MAP}, mean ± std across 3 seeds)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    full_path = SCRIPT_DIR / "checkpoint_sweep_occupied_variance_auc.png"
    fig.savefig(full_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {full_path}")

    # --- Late-stage view: zoomed + gap panel ---
    de, dm, ds = series("diffusion", LATE_STAGE_MIN_EPOCH)
    ie, im, istd = series("imitation", LATE_STAGE_MIN_EPOCH)
    im_at_de = np.interp(de, ie, im)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    ax = axes[0]
    # Lower is better here, so shade where ImitateTrans is WORSE (higher) than Diffusion.
    ax.fill_between(de, dm, im_at_de, where=(im_at_de >= dm), color=COLOR_DIFFUSION, alpha=0.12, zorder=1)
    ax.errorbar(de, dm, yerr=ds, marker="o", color=COLOR_DIFFUSION, label="Diffusion", capsize=3, linewidth=1.8, zorder=3)
    ax.errorbar(ie, im, yerr=istd, marker="o", color=COLOR_IMITATION, label="ImitateTrans", capsize=3, linewidth=1.8, zorder=3)
    ax.set_xlabel("Training epoch (checkpoint)")
    ax.set_ylabel("Occupied variance AUC at 150s (lower = better)")
    ax.set_title(f"Occupied variance AUC, epoch ≥ {LATE_STAGE_MIN_EPOCH}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    ax2 = axes[1]
    gap = im_at_de - dm  # positive = Diffusion better (lower AUC)
    ax2.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax2.bar(de, gap, width=(de[1] - de[0]) * 0.6 if len(de) > 1 else 40, color=COLOR_DIFFUSION, alpha=0.6)
    ax2.set_xlabel("Training epoch (checkpoint)")
    ax2.set_ylabel("AUC gap (ImitateTrans − Diffusion; positive = Diffusion better)")
    ax2.set_title("Diffusion's margin over ImitateTrans")
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"Later-training comparison (epoch ≥ {LATE_STAGE_MIN_EPOCH}): map {SELECTED_MAP}, mean ± std across 3 seeds",
        fontsize=12,
    )
    fig.tight_layout()
    late_path = SCRIPT_DIR / "checkpoint_sweep_late_stage_occupied_variance_auc.png"
    fig.savefig(late_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {late_path}")
    print(f"Gap stats (positive = Diffusion better): mean={gap.mean():.2f} min={gap.min():.2f} max={gap.max():.2f}")


if __name__ == "__main__":
    main()
