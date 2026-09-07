"""
Late-training-focused view of the checkpoint sweep comparison, built to make Diffusion's
actual advantage visible instead of buried under ImitateTrans's early climb.

Restricted to epoch >= 200: this is where ImitateTrans's dramatic initial recovery from
its mode-averaged epoch-10 start has essentially finished and it settles into its noisy
plateau regime (the cutoff is chosen from where that qualitative shift happens in the raw
per-epoch numbers, not picked to flatter either curve). Task completion only - the
occupied-RMSE-AUC metric doesn't show a real separation between the two planners even in
this range (see checkpoint_sweep_diffusion_vs_imitatetrans.png), so including it here
would misrepresent this as a bigger across-the-board win than the data supports.

Left panel: task completion, y-axis tightened to where the two curves actually sit, with
the gap between them shaded. Right panel: the gap itself (Diffusion - ImitateTrans) per
epoch, which is what actually demonstrates the "later-stage strength" claim - a
consistently positive margin, not just a visually favourable crop.
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
LATE_STAGE_MIN_EPOCH = 200


def load_by_epoch(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_epoch = defaultdict(list)
    for r in rows:
        by_epoch[int(r["epoch"])].append(r)
    return by_epoch


def series(by_epoch, key, min_epoch):
    epochs = sorted(e for e in by_epoch if e >= min_epoch)
    means = np.array([np.mean([float(r[key]) for r in by_epoch[e]]) for e in epochs])
    stds = np.array([np.std([float(r[key]) for r in by_epoch[e]]) for e in epochs])
    return np.array(epochs), means, stds


def main():
    diff_by_epoch = load_by_epoch(DIFFUSION_CSV)
    imit_by_epoch = load_by_epoch(IMITATE_CSV)

    de, dm, ds = series(diff_by_epoch, "task_completion_pct", LATE_STAGE_MIN_EPOCH)
    ie, im, istd = series(imit_by_epoch, "task_completion_pct", LATE_STAGE_MIN_EPOCH)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    ax = axes[0]
    # Shade the gap between the two curves - only well-defined where the epoch grids
    # match, so interpolate ImitateTrans (denser grid) onto Diffusion's epoch points.
    im_at_de = np.interp(de, ie, im)
    ax.fill_between(de, im_at_de, dm, where=(dm >= im_at_de), color=COLOR_DIFFUSION, alpha=0.12, zorder=1)
    ax.errorbar(de, dm, yerr=ds, marker="o", color=COLOR_DIFFUSION, label="Diffusion",
                capsize=3, linewidth=1.8, zorder=3)
    ax.errorbar(ie, im, yerr=istd, marker="o", color=COLOR_IMITATION, label="ImitateTrans",
                capsize=3, linewidth=1.8, zorder=3)
    ax.set_xlabel("Training epoch (checkpoint)")
    ax.set_ylabel("Task completion (%)")
    ax.set_title(f"Task completion, epoch ≥ {LATE_STAGE_MIN_EPOCH}")
    ax.set_ylim(65, 102)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    ax2 = axes[1]
    gap = dm - im_at_de
    ax2.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax2.bar(de, gap, width=(de[1] - de[0]) * 0.6 if len(de) > 1 else 40, color=COLOR_DIFFUSION, alpha=0.6)
    ax2.set_xlabel("Training epoch (checkpoint)")
    ax2.set_ylabel("Task completion gap (Diffusion − ImitateTrans, pp)")
    ax2.set_title("Diffusion's margin over ImitateTrans")
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"Later-training comparison (epoch ≥ {LATE_STAGE_MIN_EPOCH}): map 59, mean ± std across 3 seeds",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = SCRIPT_DIR / "checkpoint_sweep_late_stage_diffusion_advantage.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Gap stats (pp): mean={gap.mean():.1f} min={gap.min():.1f} max={gap.max():.1f}")


if __name__ == "__main__":
    main()
