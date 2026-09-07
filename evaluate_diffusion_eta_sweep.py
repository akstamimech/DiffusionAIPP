import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PLANNER = SCRIPT_DIR / "Diffusionplanner_singlemap.py"
DATASET_NAME = "CMAES_beamsearch_dataset_3d_synthetic_final.pt"
INITIAL_VARIANCE_RE = re.compile(r"Initial total variance:\s*([0-9.eE+-]+)")
SUMMARY_LINE_RE = re.compile(
    r"(Initial total variance|Final total variance|Variance reduction|Task Completion|Global RMSE|Occupied RMSE)"
)


def read_metrics(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)
    return data


def write_summary(summary_path, timesteps, mean_drop, lower_drop, upper_drop, counts):
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestep",
                "mean_variance_drop",
                "lower_variance_drop",
                "upper_variance_drop",
                "n_runs",
            ]
        )
        for row in zip(timesteps, mean_drop, lower_drop, upper_drop, counts):
            writer.writerow(row)


def print_planner_output(stdout, stderr, show_full=False):
    if show_full:
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
        return

    for line in stdout.splitlines():
        if SUMMARY_LINE_RE.search(line):
            print(line)
    for line in stderr.splitlines():
        if "Traceback" in line or "Error" in line or "Exception" in line:
            print(line, file=sys.stderr)


def summarize_runs(run_records, output_dir, selected_map, maptype):
    by_eta = {}
    for record in run_records:
        key = (record["eta"], record["execution_chunk"])
        by_eta.setdefault(key, []).append(record)

    combined_paths = []
    plt.figure(figsize=(10, 6))
    for (eta, execution_chunk), records in sorted(by_eta.items()):
        max_timestep = max(int(np.max(r["metrics"]["timestep"])) for r in records)
        timesteps = np.arange(1, max_timestep + 1, dtype=int)
        drops = np.full((len(records), len(timesteps)), np.nan, dtype=float)

        for run_idx, record in enumerate(records):
            metrics = record["metrics"]
            timestep = metrics["timestep"].astype(int)
            variance_drop = record["initial_variance"] - metrics["global_variance"]
            drops[run_idx, timestep - 1] = variance_drop

        mean_drop = np.nanmean(drops, axis=0)
        lower_drop = np.nanmin(drops, axis=0)
        upper_drop = np.nanmax(drops, axis=0)
        counts = np.sum(~np.isnan(drops), axis=0).astype(int)

        safe_eta = str(eta).replace(".", "p")
        summary_path = (
            output_dir
            / f"map_{selected_map}_{maptype}_eta_{safe_eta}_horizon_{execution_chunk}_variance_drop_summary.csv"
        )
        write_summary(summary_path, timesteps, mean_drop, lower_drop, upper_drop, counts)
        combined_paths.append(summary_path)

        plt.plot(timesteps, mean_drop, label=f"eta={eta:g}, horizon={execution_chunk}")
        plt.fill_between(timesteps, lower_drop, upper_drop, alpha=0.18)

    plt.xlabel("Simulation timestep")
    plt.ylabel("Variance drop from initial covariance")
    plt.title(f"Map {selected_map} ({maptype}) diffusion ETA sweep")
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / f"map_{selected_map}_{maptype}_eta_variance_drop_bands.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return combined_paths, plot_path


def main():
    parser = argparse.ArgumentParser(
        description="Run Diffusionplanner_singlemap.py repeatedly for ETA values and summarize variance-drop bands."
    )
    parser.add_argument("--selected-map", type=int, default=int(os.environ.get("SELECTED_MAP", "58")))
    parser.add_argument("--maptype", default=os.environ.get("MAPTYPE", "grf"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--etas", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--base-seed", type=int, default=17000)
    parser.add_argument("--wallclock-seconds", type=float, default=0.0)
    parser.add_argument("--timealloted", type=int, default=300)
    parser.add_argument(
        "--execution-chunks",
        type=int,
        nargs="+",
        default=[int(os.environ.get("EXECUTION_CHUNK", "10"))],
    )
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--show-planner-output", action="store_true")
    args = parser.parse_args()

    sweep_output_dir = SCRIPT_DIR / "Vizualization" / f"diffusion_eta_sweep_map_{args.maptype}_{args.selected_map}"
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = args.dataset_path
    if dataset_path is None:
        for candidate in (SCRIPT_DIR / DATASET_NAME, SCRIPT_DIR / "Diffusion" / DATASET_NAME):
            if candidate.exists():
                dataset_path = candidate
                break
    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError(
            "Could not find the diffusion dataset used for normalization. "
            "Pass it explicitly with --dataset-path."
        )

    run_records = []
    for chunk_index, execution_chunk in enumerate(args.execution_chunks):
        for eta_index, eta in enumerate(args.etas):
            for run_index in range(args.runs):
                seed = args.base_seed + chunk_index * 10000 + eta_index * 1000 + run_index
                safe_eta = str(eta).replace(".", "p")
                run_tag = f"eta_{safe_eta}_horizon_{execution_chunk}_run_{run_index + 1}_seed_{seed}"
                planner_output_dir = (
                    SCRIPT_DIR
                    / "Vizualization"
                    / f"diffusion_map_{args.maptype}_{args.selected_map}_viz_{run_tag}"
                )
                env = os.environ.copy()
                env.update(
                    {
                        "ETA": str(eta),
                        "RUN_SEED": str(seed),
                        "SENSORNOISE_SEED": str(seed),
                        "SELECTED_MAP": str(args.selected_map),
                        "MAPTYPE": args.maptype,
                        "EXECUTION_CHUNK": str(execution_chunk),
                        "SKIP_VIZ": "1",
                        "WALLCLOCK_SECONDS": str(args.wallclock_seconds),
                        "TIMEALLOTED": str(args.timealloted),
                        "DIFFUSION_DATASET_PATH": str(dataset_path),
                        "RUN_OUTPUT_TAG": run_tag,
                    }
                )

                print(
                    f"[eta={eta:g} horizon={execution_chunk} run={run_index + 1}/{args.runs}] seed={seed}",
                    flush=True,
                )
                result = subprocess.run(
                    [sys.executable, str(PLANNER)],
                    cwd=str(SCRIPT_DIR),
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                print_planner_output(
                    result.stdout,
                    result.stderr,
                    show_full=args.show_planner_output,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Planner failed for eta={eta:g}, horizon={execution_chunk}, "
                        f"run={run_index + 1} with exit code {result.returncode}"
                    )

                match = INITIAL_VARIANCE_RE.search(result.stdout)
                if match is None:
                    raise RuntimeError("Could not parse initial variance from planner output.")
                initial_variance = float(match.group(1))

                source_csv = planner_output_dir / f"map_{args.selected_map}_rmse_over_time.csv"
                if not source_csv.exists():
                    raise FileNotFoundError(f"Expected planner metrics CSV was not created: {source_csv}")

                run_csv = (
                    sweep_output_dir
                    / f"map_{args.selected_map}_{args.maptype}_eta_{safe_eta}_horizon_{execution_chunk}_run_{run_index + 1}_rmse_over_time.csv"
                )
                shutil.copy2(source_csv, run_csv)
                run_records.append(
                    {
                        "eta": eta,
                        "execution_chunk": execution_chunk,
                        "run_index": run_index,
                        "seed": seed,
                        "initial_variance": initial_variance,
                        "metrics": read_metrics(run_csv),
                    }
                )

    summary_paths, plot_path = summarize_runs(
        run_records,
        sweep_output_dir,
        args.selected_map,
        args.maptype,
    )
    print("Wrote summaries:")
    for path in summary_paths:
        print(f"  {path}")
    print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
