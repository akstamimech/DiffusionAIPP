import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
VIZ_DIR = SCRIPT_DIR / "Vizualization"
DIFFUSION_PLANNER = SCRIPT_DIR / "Diffusionplanner_singlemap.py"
IMITATION_PLANNER = SCRIPT_DIR / "ImitateTrans_singlemap.py"
DEFAULT_DATASET = SCRIPT_DIR / "CMAES_beamsearch_dataset_3d_synthetic_final.pt"
SUMMARY_KEYWORDS = (
    "Initial total variance",
    "Final total variance",
    "Variance reduction",
    "Task Completion",
    "Global RMSE",
    "Occupied RMSE",
)


def run_planner(script_path, env):
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPT_DIR),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        if any(keyword in line for keyword in SUMMARY_KEYWORDS):
            print(line)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"{script_path.name} failed with exit code {result.returncode}")


def read_trace(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)
    return data


def threshold_tag(value):
    return str(value).replace(".", "p")


def trace_metrics(data):
    t = data["timestep"]
    metrics = {}
    for field in ("global_rmse", "occupied_rmse", "global_variance"):
        y = data[field]
        metrics[field] = {
            "final": float(y[-1]),
            "mean": float(y.mean()),
            "auc": float(np.trapezoid(y, t)),
        }
    metrics["variance_drop_from_step1"] = float(data["global_variance"][0] - data["global_variance"][-1])
    return metrics


def make_comparison_plot(map_id, diffusion_data, imitation_data, output_dir, utility_threshold):
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_label = threshold_tag(utility_threshold)
    plot_path = output_dir / f"map_{map_id}_threshold_{threshold_label}_diffusion_vs_imitation_comparison.png"
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10), sharex=True)
    series = [
        ("global_rmse", "Global RMSE"),
        ("occupied_rmse", "Occupied RMSE"),
        ("global_variance", "Global Variance"),
    ]
    for ax, (field, title) in zip(axes, series):
        ax.plot(diffusion_data["timestep"], diffusion_data[field], label="Diffusion", linewidth=2)
        ax.plot(imitation_data["timestep"], imitation_data[field], label="ImitateTrans", linewidth=2)
        ax.set_ylabel(title)
        ax.set_title(f"{title} (lower is better)")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Simulation timestep")
    axes[0].legend(loc="best")
    fig.suptitle(
        f"Map {map_id} GRF: Diffusion vs ImitateTrans (utility threshold {utility_threshold:g})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def compare_map(map_id, args):
    common_env = os.environ.copy()
    common_env.update(
        {
            "SELECTED_MAP": str(map_id),
            "MAPTYPE": args.maptype,
            "EXECUTION_CHUNK": str(args.execution_chunk),
            "WALLCLOCK_SECONDS": str(args.wallclock_seconds),
            "TIMEALLOTED": str(args.timealloted),
            "UTILITY_THRESHOLD": str(args.utility_threshold),
            "SKIP_VIZ": "1",
            "ENFORCE_MIN_STEP_TIME": "0",
            "SENSORNOISE_SEED": str(args.sensornoise_seed),
        }
    )

    diffusion_env = common_env.copy()
    diffusion_env.update(
        {
            "ETA": str(args.eta),
            "RUN_SEED": str(args.diffusion_seed + map_id),
            "DIFFUSION_DATASET_PATH": str(args.dataset_path),
        }
    )
    print(f"[map {map_id}] Diffusion")
    run_planner(DIFFUSION_PLANNER, diffusion_env)
    diffusion_dir = VIZ_DIR / f"diffusion_map_{args.maptype}_{map_id}_viz"
    diffusion_csv = diffusion_dir / f"map_{map_id}_rmse_over_time.csv"
    threshold_label = threshold_tag(args.utility_threshold)
    diffusion_named_csv = diffusion_dir / f"map_{map_id}_rmse_over_time_threshold_{threshold_label}_diff.csv"
    shutil.copy2(diffusion_csv, diffusion_named_csv)

    imitation_env = common_env.copy()
    print(f"[map {map_id}] ImitateTrans")
    run_planner(IMITATION_PLANNER, imitation_env)
    imitation_dir = VIZ_DIR / f"imitate_trans_map_{args.maptype}_{map_id}_viz"
    imitation_csv = imitation_dir / f"map_{map_id}_rmse_over_time.csv"
    imitation_named_csv = imitation_dir / f"map_{map_id}_rmse_over_time_threshold_{threshold_label}_imitation.csv"
    shutil.copy2(imitation_csv, imitation_named_csv)

    diffusion_data = read_trace(diffusion_named_csv)
    imitation_data = read_trace(imitation_named_csv)
    comparison_dir = VIZ_DIR / "diffusion_vs_imitation_grf_comparison"
    plot_path = make_comparison_plot(
        map_id,
        diffusion_data,
        imitation_data,
        comparison_dir,
        args.utility_threshold,
    )

    diffusion_metrics = trace_metrics(diffusion_data)
    imitation_metrics = trace_metrics(imitation_data)
    row = {
        "map_id": map_id,
        "diffusion_csv": str(diffusion_named_csv),
        "imitation_csv": str(imitation_named_csv),
        "plot_path": str(plot_path),
    }
    for prefix, metrics in (("diffusion", diffusion_metrics), ("imitation", imitation_metrics)):
        row[f"{prefix}_final_global_rmse"] = metrics["global_rmse"]["final"]
        row[f"{prefix}_mean_global_rmse"] = metrics["global_rmse"]["mean"]
        row[f"{prefix}_final_occupied_rmse"] = metrics["occupied_rmse"]["final"]
        row[f"{prefix}_mean_occupied_rmse"] = metrics["occupied_rmse"]["mean"]
        row[f"{prefix}_final_global_variance"] = metrics["global_variance"]["final"]
        row[f"{prefix}_mean_global_variance"] = metrics["global_variance"]["mean"]
        row[f"{prefix}_variance_drop_from_step1"] = metrics["variance_drop_from_step1"]
    row["diffusion_minus_imitation_final_global_rmse"] = (
        row["diffusion_final_global_rmse"] - row["imitation_final_global_rmse"]
    )
    row["diffusion_minus_imitation_final_occupied_rmse"] = (
        row["diffusion_final_occupied_rmse"] - row["imitation_final_occupied_rmse"]
    )
    row["diffusion_minus_imitation_variance_drop"] = (
        row["diffusion_variance_drop_from_step1"] - row["imitation_variance_drop_from_step1"]
    )
    return row


def write_summary(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", type=int, nargs="+", default=list(range(50, 56)))
    parser.add_argument("--maptype", default="grf")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--execution-chunk", type=int, default=20)
    parser.add_argument("--utility-threshold", type=float, default=0.5)
    parser.add_argument("--wallclock-seconds", type=float, default=0.0)
    parser.add_argument("--timealloted", type=int, default=300)
    parser.add_argument("--sensornoise-seed", type=int, default=123)
    parser.add_argument("--diffusion-seed", type=int, default=17000)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    rows = [compare_map(map_id, args) for map_id in args.maps]
    summary_path = (
        VIZ_DIR
        / "diffusion_vs_imitation_grf_comparison"
        / f"maps_{min(args.maps)}_{max(args.maps)}_threshold_{threshold_tag(args.utility_threshold)}_comparison_summary.csv"
    )
    write_summary(rows, summary_path)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
