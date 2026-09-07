"""
Diffusion's occupied variance AUC (150s missions, 3-repeat mean +/- std, already computed
in checkpoint_sweep_occupied_variance_auc_summary.csv) plotted against ImitateTrans's
occupied variance AUC from its NEW sweep (100s missions, single deterministic run per
checkpoint - checkpoint_sweep_imitatetrans_100s_summary.csv).

The two curves are integrated over different wallclock budgets (150s vs 100s) - by
explicit instruction this comparison is being made anyway, so both axis label and legend
say so directly rather than silently presenting them as matched.
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
DIFFUSION_AUC_CSV = SCRIPT_DIR / "checkpoint_sweep_occupied_variance_auc_summary.csv"
IMITATE_100S_CSV = SCRIPT_DIR / "checkpoint_sweep_imitatetrans_100s_summary.csv"
SELECTED_MAP = 59
IMITATE_WALLCLOCK_CAP = 100.0

COLOR_DIFFUSION = "tab:red"
COLOR_IMITATION = "tab:purple"


def load_diffusion_auc():
    with DIFFUSION_AUC_CSV.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["planner"] == "diffusion"]
    rows.sort(key=lambda r: int(r["epoch"]))
    epochs = np.array([int(r["epoch"]) for r in rows])
    means = np.array([float(r["mean_occupied_variance_auc"]) for r in rows])
    stds = np.array([float(r["std_occupied_variance_auc"]) for r in rows])
    return epochs, means, stds


def compute_imitatetrans_auc():
    with IMITATE_100S_CSV.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    rows.sort(key=lambda r: int(r["epoch"]))

    epochs, aucs = [], []
    for r in rows:
        epoch = int(r["epoch"])
        traj_path = Path(r["output_dir"]) / f"map_{SELECTED_MAP}_executed_trajectory.csv"
        run = {"map_id": SELECTED_MAP, "traj_path": str(traj_path)}
        result = m.reconstruct_one(run)
        wall_time = result["wall_time_seconds"]
        important_var = result["important_variance"]
        within_cap = wall_time <= IMITATE_WALLCLOCK_CAP
        wt = wall_time[within_cap]
        iv = important_var[within_cap]
        auc = float(np.trapezoid(iv, wt)) if wt.size >= 2 else 0.0
        epochs.append(epoch)
        aucs.append(auc)
        print(f"  epoch {epoch}: occupied_variance_auc (100s) = {auc:.2f}", flush=True)
    return np.array(epochs), np.array(aucs)


def main():
    de, dm, ds = load_diffusion_auc()
    print(f"Loaded Diffusion AUC (150s, mean +/- std across 3 seeds): {len(de)} epochs")

    print("Replaying ImitateTrans 100s-sweep trajectories...")
    ie, iauc = compute_imitatetrans_auc()

    per_run_csv = SCRIPT_DIR / "checkpoint_sweep_imitatetrans_100s_occupied_variance_auc.csv"
    with per_run_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "occupied_variance_auc_100s"])
        writer.writerows(zip(ie.tolist(), iauc.tolist()))
    print(f"Wrote {per_run_csv}")

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.errorbar(de, dm, yerr=ds, marker="o", color=COLOR_DIFFUSION,
                label="Diffusion (150s missions, mean ± std, 3 seeds)", capsize=3, linewidth=1.8)
    ax.plot(ie, iauc, marker="o", color=COLOR_IMITATION,
            label="ImitateTrans (100s missions, single deterministic run)", linewidth=1.8)
    ax.set_xlabel("Training epoch (checkpoint)")
    ax.set_ylabel("Occupied variance AUC (lower = better)")
    ax.set_title(
        f"Occupied variance AUC vs. training epoch, map {SELECTED_MAP}\n"
        "NOTE: integrated over different mission lengths (Diffusion 150s vs ImitateTrans 100s)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = SCRIPT_DIR / "checkpoint_sweep_diffusion150s_vs_imitatetrans100s_occupied_variance_auc.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
