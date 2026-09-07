"""
Time to 80% occupied variance reduction: Diffusion (existing, complete dataset - 150s
original + extended-to-400s reruns for the one repeat that needed it) vs. ImitateTrans's
NEW sweep under the current (reverted) horizon/EXECUTION_CHUNK settings
(sweep_imitatetrans_stop_at_80pct_summary.csv, single run per checkpoint, 600s ceiling).

Diffusion keeps the filled/open-circle distinction from the earlier plot (open = needed
the extended 400s budget). ImitateTrans is single-run per checkpoint under a generous 600s
ceiling already, so it only needs filled circle (reached) vs. X (never reached even at
600s) - no separate "extended" category this time.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DIFF_ORIGINAL_CSV = SCRIPT_DIR / "checkpoint_sweep_time_to_80pct_completion_per_run.csv"
DIFF_EXTENDED_CSV = SCRIPT_DIR / "rerun_failed_80pct_checkpoints_summary.csv"
IMITATE_CSV = SCRIPT_DIR / "sweep_imitatetrans_stop_at_80pct_summary.csv"

COLOR_DIFFUSION = "tab:red"
COLOR_IMITATION = "tab:purple"


def load_diffusion():
    with DIFF_ORIGINAL_CSV.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["planner"] == "diffusion"]
    original = {}
    for r in rows:
        key = (int(r["epoch"]), int(r["repeat"]))
        original[key] = float(r["time_to_80pct_completion_s"]) if r["time_to_80pct_completion_s"] != "" else None

    with DIFF_EXTENDED_CSV.open(newline="") as f:
        ext_rows = [r for r in csv.DictReader(f) if r["planner"] == "diffusion"]
    extended = {}
    for r in ext_rows:
        t = r.get("time_to_80pct_reduction_s")
        extended[int(r["epoch"])] = float(t) if t not in (None, "") else None
    extended_repeat = {10: 2}  # the repeat index that needed the extended rerun

    epochs = sorted({e for (e, _r) in original})
    means, stds, ext_mask = [], [], []
    for epoch in epochs:
        repeats = sorted(r for (e, r) in original if e == epoch)
        times = []
        was_extended = False
        for rep in repeats:
            t = original[(epoch, rep)]
            if t is not None:
                times.append(t)
            elif rep == extended_repeat.get(epoch):
                was_extended = True
                if extended.get(epoch) is not None:
                    times.append(extended[epoch])
        means.append(np.mean(times))
        stds.append(np.std(times))
        ext_mask.append(was_extended)

    return np.array(epochs), np.array(means), np.array(stds), np.array(ext_mask)


def load_imitatetrans():
    with IMITATE_CSV.open(newline="") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["status"] in ("reached", "not_reached") and int(r["epoch"]) != 1500
        ]  # epoch 1500 dropped - no matching Diffusion checkpoint (that series stops at 1400)
    rows.sort(key=lambda r: int(r["epoch"]))

    epochs, times, not_reached = [], [], []
    for r in rows:
        epochs.append(int(r["epoch"]))
        if r["status"] == "reached":
            times.append(float(r["time_to_80pct_reduction_s"]))
        else:
            times.append(np.nan)
            not_reached.append(int(r["epoch"]))
    return np.array(epochs), np.array(times), not_reached


def main():
    de, dm, ds, d_ext = load_diffusion()
    ie, it, i_not_reached = load_imitatetrans()

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    # Diffusion - single style throughout, no open/filled distinction
    ax.plot(de, dm, color=COLOR_DIFFUSION, linewidth=1.5, zorder=2, label="Diffusion")
    ax.errorbar(de, dm, yerr=ds, marker="o", color=COLOR_DIFFUSION,
                capsize=3, linestyle="none", zorder=3, markersize=7)

    # ImitateTrans (new horizon sweep)
    i_valid = ~np.isnan(it)
    ax.plot(ie[i_valid], it[i_valid], color=COLOR_IMITATION, linewidth=1.5, zorder=2,
             label="ImitateTrans")
    ax.scatter(ie[i_valid], it[i_valid], marker="o", color=COLOR_IMITATION, s=50, zorder=3)

    y_min, y_max = ax.get_ylim()
    marker_y = y_max + 0.08 * (y_max - y_min)
    ax.set_ylim(y_min, marker_y + 0.05 * (y_max - y_min))
    if i_not_reached:
        ax.scatter(i_not_reached, [marker_y] * len(i_not_reached), marker="x", s=60,
                   color=COLOR_IMITATION, linewidth=2, zorder=4)

    ax.set_xlabel("Training epoch (checkpoint)")
    ax.set_ylabel("Wall-time (s)")
    ax.set_title("Wall time vs. training epoch on held out set", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.text(0.5, 0.005, "× = hit the held-out ceiling without reaching 80% reduction",
              ha="center", fontsize=8, style="italic", color=COLOR_IMITATION)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out_path = SCRIPT_DIR / "diffusion_vs_imitatetrans_new_horizon_80pct.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")

    print(f"\nDiffusion: {len(de)}/{len(de)} epochs reached 80% ({d_ext.sum()} needed the extended budget)")
    print(f"ImitateTrans: {i_valid.sum()}/{len(ie)} epochs reached 80% within 600s; not reached: {i_not_reached}")
    print(f"Diffusion mean time (all epochs): {dm.mean():.1f}s")
    print(f"ImitateTrans mean time (reached epochs only): {np.nanmean(it):.1f}s")


if __name__ == "__main__":
    main()
