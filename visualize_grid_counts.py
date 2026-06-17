import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CSV_DIR = Path(r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\Aksha\OneDrive\Desktop\Documents\Playground\grid_count_visualizations")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize x,y,count grid-count CSV maps as heatmaps."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help="Directory containing *_grid_counts.csv files.",
    )
    parser.add_argument(
        "--pattern",
        default="*_grid_counts.csv",
        help='Glob pattern inside --csv-dir, e.g. "map_10_*_grid_counts.csv".',
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Visualize one specific CSV file instead of using --pattern.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG visualizations are saved.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Plot log1p(count) to make weak blobs easier to see.",
    )
    parser.add_argument(
        "--show-zero",
        action="store_true",
        help="Keep zero-count cells visible instead of masking them dark.",
    )
    return parser.parse_args()


def load_grid_counts(csv_path):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{csv_path} does not look like an x,y,count CSV.")

    xs = np.unique(data[:, 0])
    ys = np.unique(data[:, 1])
    x_to_idx = {x: i for i, x in enumerate(xs)}
    y_to_idx = {y: i for i, y in enumerate(ys)}

    grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for x, y, count in data[:, :3]:
        grid[y_to_idx[y], x_to_idx[x]] = count

    return xs, ys, grid


def plot_grid_counts(csv_path, output_dir, log_scale=False, show_zero=False):
    xs, ys, grid = load_grid_counts(csv_path)
    plot_grid = np.log1p(grid) if log_scale else grid.copy()
    if not show_zero:
        plot_grid = np.ma.masked_where(grid <= 0, plot_grid)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_log" if log_scale else ""
    output_path = output_dir / f"{csv_path.stem}{suffix}.png"

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(
        plot_grid,
        origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_title(csv_path.stem)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    cbar_label = "log1p(count)" if log_scale else "count"
    fig.colorbar(image, ax=ax, label=cbar_label)

    nonzero = grid[grid > 0]
    stats_text = (
        f"sum={grid.sum():.0f}\n"
        f"max={grid.max():.0f}\n"
        f"nonzero={nonzero.size}/{grid.size}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def iter_csv_paths(args):
    if args.csv is not None:
        yield args.csv
        return

    yield from sorted(args.csv_dir.glob(args.pattern))


def main():
    args = parse_args()
    csv_paths = list(iter_csv_paths(args))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs matched {args.csv_dir / args.pattern}")

    for csv_path in csv_paths:
        output_path = plot_grid_counts(
            csv_path,
            args.output_dir,
            log_scale=args.log_scale,
            show_zero=args.show_zero,
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
