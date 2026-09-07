"""
Executed-position heatmap for the map-200 (synthetic four-corner) planner
comparison - companion to run_map51_planner_heatmaps.py's FOV time-spent
heatmap, reconstructed to match the same style previously produced in this
project (map_51_*_time_spent_heatmap.csv / map_51_planner_time_spent_heatmaps.png,
whose generating script no longer exists on disk, but whose CSV schema -
"x,y,time_spent_steps" - and colorbar range confirm the methodology below:
unlike the FOV heatmap, which paints the whole sensor-footprint radius around
each pose, this counts only the exact grid cell the vehicle's own (x,y)
position occupied at each timestep - executed positions are already snapped
to the xs/ys grid by dynamics_3d, so this is a direct index match, no
radius/interpolation involved).

Reuses run_map51_planner_heatmaps.py's map-loading/output-dir helpers so this
points at the exact same trajectory files that script's run just wrote.

Palette: same as the FOV heatmap - grayscale ground truth background (cmap
"Greys"), magma overlay for revisit counts (vmin=1 so single-visit cells fade
into the background, only revisited cells pop), plus the raw executed path
drawn as a thin black line so the flight pattern itself remains legible.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_map51_planner_heatmaps import (
    PLANNERS,
    load_grid,
    output_dir_for,
    read_trace,
)

SCRIPT_DIR = Path(__file__).resolve().parent
VIZ_DIR = SCRIPT_DIR / "Vizualization"


def position_heatmap(trace, xs, ys):
    heat = np.zeros((len(ys), len(xs)), dtype=float)
    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}
    for row in trace:
        xi = x_idx.get(float(row["x"]))
        yi = y_idx.get(float(row["y"]))
        if xi is not None and yi is not None:
            heat[yi, xi] += 1.0
    return heat


def save_position_heatmap_csv(path, xs, ys, heat):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "time_spent_steps"])
        for row_idx, y in enumerate(ys):
            for col_idx, x in enumerate(xs):
                writer.writerow([x, y, int(heat[row_idx, col_idx])])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", type=int, default=200)
    parser.add_argument("--maptype", default="grf")
    parser.add_argument("--timealloted", type=int, default=3000)
    parser.add_argument("--wallclock-seconds", type=float, default=300.0)
    parser.add_argument("--run-tag", default="corners_start5050")
    parser.add_argument("--important-threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    comparison_dir = VIZ_DIR / "map51_planner_heatmaps"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    map_path, xs, ys, ground_truth = load_grid(args.map_id, args.maptype)
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    map_min = float(np.nanmin(ground_truth))
    map_max = float(np.nanmax(ground_truth))

    heatmaps = {}
    traces = {}
    max_count = 1.0
    for planner in PLANNERS:
        output_dir = output_dir_for(planner, args.maptype, args.map_id, args.run_tag)
        trajectory_path, trace = read_trace(output_dir, args.map_id)
        heat = position_heatmap(trace, xs, ys)
        heatmaps[planner["key"]] = heat
        traces[planner["key"]] = trace
        max_count = max(max_count, float(np.max(heat)))
        heat_csv = comparison_dir / f"map_{args.map_id}_{planner['key']}_time_spent_heatmap.csv"
        save_position_heatmap_csv(heat_csv, xs, ys, heat)
        print(f"[{planner['key']}] {trajectory_path} -> {heat_csv} (max revisits {int(heat.max())})")

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.6), sharex=True, sharey=True, constrained_layout=True)
    im = None
    for ax, planner in zip(axes, PLANNERS):
        ax.imshow(
            ground_truth,
            origin="lower",
            extent=extent,
            cmap="Greys",
            vmin=map_min,
            vmax=map_max,
            alpha=0.55,
            aspect="equal",
        )
        trace = traces[planner["key"]]
        ax.plot(trace["x"], trace["y"], color="black", linewidth=0.7, alpha=0.9, zorder=2)
        heat = np.ma.masked_where(heatmaps[planner["key"]] <= 0, heatmaps[planner["key"]])
        im = ax.imshow(
            heat,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=1,
            vmax=max_count,
            alpha=0.82,
            aspect="equal",
            zorder=3,
        )
        ax.set_title(planner["label"])
        ax.set_xlabel("x")
        ax.grid(False)
    axes[0].set_ylabel("y")
    assert im is not None
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("Time spent at location (simulation steps)")
    fig.suptitle(
        f"Map {args.map_id} GRF executed-position heatmaps "
        f"({args.wallclock_seconds:g}s wall clock, {args.timealloted} timestep cap)",
        fontsize=14,
    )

    plot_path = comparison_dir / f"map_{args.map_id}_planner_time_spent_heatmaps.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"Read map: {map_path}")
    print(f"Wrote position heatmap plot: {plot_path}")


if __name__ == "__main__":
    main()
