"""
Compares the two checkpoint sweeps (checkpoint_sweep_summary.csv = Diffusion,
checkpoint_sweep_imitatetrans_summary.csv = ImitateTrans): mean +/- std across the 3
repeat-seeds at each swept training epoch, for task completion % and occupied RMSE AUC.
Both sweeps used the same 3 seeds and the same held-out map (59), full 150s missions -
this is the checkpoint-level, multi-seed, full-mission comparison, not the noisy
single-sample training-time periodic eval from earlier.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DIFFUSION_CSV = SCRIPT_DIR / "checkpoint_sweep_summary.csv"
IMITATE_CSV = SCRIPT_DIR / "checkpoint_sweep_imitatetrans_summary.csv"

COLOR_DIFFUSION = "tab:red"
COLOR_IMITATION = "tab:purple"


def load_by_epoch(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_epoch = defaultdict(list)
    for r in rows:
        by_epoch[int(r["epoch"])].append(r)
    return by_epoch


def series(by_epoch, key):
    epochs = sorted(by_epoch)
    means = [np.mean([float(r[key]) for r in by_epoch[e]]) for e in epochs]
    stds = [np.std([float(r[key]) for r in by_epoch[e]]) for e in epochs]
    return np.array(epochs), np.array(means), np.array(stds)


def main():
    diff_by_epoch = load_by_epoch(DIFFUSION_CSV)
    imit_by_epoch = load_by_epoch(IMITATE_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    for key, ylabel, title, ax in [
        ("task_completion_pct", "Task completion (%)", "Task completion vs. training epoch", axes[0]),
        ("occupied_rmse_auc", "Occupied RMSE AUC", "Occupied RMSE AUC vs. training epoch", axes[1]),
    ]:
        de, dm, ds = series(diff_by_epoch, key)
        ie, im, istd = series(imit_by_epoch, key)
        ax.errorbar(de, dm, yerr=ds, marker="o", color=COLOR_DIFFUSION, label="Diffusion",
                    capsize=3, linewidth=1.8)
        ax.errorbar(ie, im, yerr=istd, marker="o", color=COLOR_IMITATION, label="ImitateTrans",
                    capsize=3, linewidth=1.8)
        ax.set_xlabel("Training epoch (checkpoint)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle("Checkpoint sweep: full 150s missions, map 59, mean ± std across 3 seeds", fontsize=12)
    fig.tight_layout()
    out_path = SCRIPT_DIR / "checkpoint_sweep_diffusion_vs_imitatetrans.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
