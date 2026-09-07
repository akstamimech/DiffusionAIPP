import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
VIZ_DIR = SCRIPT_DIR / "Vizualization"
DEFAULT_MAPS = [50, 51, 52, 53, 54]
MAPTYPE = "grf"
TIMEALLOTED = 3000
WALLCLOCK_SECONDS = 300
UTILITY_THRESHOLD = 0.5
RUN_TAG = "mission300_steps3000_avg5"
INITIAL_VARIANCE = 130.05


PLANNER_SPECS = [
    ("greedy", "GreedyGradient", "greedygradient_singlemap.py", "greedygradient_map"),
    ("lawnmower", "Lawnmower", "lawnmower_singlemap.py", "lawnmower_map"),
    ("cmaes", "CMA-ES", "CMAES_classic_singlemap.py", "classic_map"),
    ("diffusion", "Diffusion", "Diffusionplanner_singlemap.py", "diffusion_map"),
    ("imitation", "ImitateTrans", "ImitateTrans_singlemap.py", "imitate_trans_map"),
]


SUMMARY_KEYWORDS = (
    "Initial total variance",
    "Final total variance",
    "Variance reduction",
    "Task Completion",
    "Global RMSE",
    "Occupied RMSE",
    "flight ended early",
)


def planner_for(key, map_id, args):
    spec = next(item for item in PLANNER_SPECS if item[0] == key)
    output_dir = (
        VIZ_DIR / f"{spec[3]}_{args.maptype}_{map_id}_viz_{args.run_tag}"
    )
    planner = {
        "key": spec[0],
        "label": spec[1],
        "script": SCRIPT_DIR / spec[2],
        "output_dir": output_dir,
    }
    if key == "diffusion":
        planner["extra_env"] = {
            "ETA": str(args.eta),
            "RUN_SEED": str(args.diffusion_seed + map_id),
            "DIFFUSION_DATASET_PATH": str(args.dataset_path),
        }
    return planner


def metrics_path(planner, map_id):
    return planner["output_dir"] / f"map_{map_id}_rmse_over_time.csv"


def log_path(planner, map_id):
    return planner["output_dir"] / f"map_{map_id}_{planner['key']}_stdout.log"


def run_planner(planner, map_id, args):
    planner["output_dir"].mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "SELECTED_MAP": str(map_id),
            "MAPTYPE": args.maptype,
            "TIMEALLOTED": str(args.timealloted),
            "WALLCLOCK_SECONDS": str(args.wallclock_seconds),
            "EXECUTION_CHUNK": str(args.execution_chunk),
            "UTILITY_THRESHOLD": str(args.utility_threshold),
            "SENSORNOISE_SEED": str(args.sensornoise_seed),
            "ENFORCE_MIN_STEP_TIME": "1",
            "SKIP_VIZ": "1",
            "RUN_OUTPUT_TAG": args.run_tag,
            "PRINT_FLIGHT_PLAN": "0",
            "PRINT_WAYPOINT_EVENTS": "0",
        }
    )
    env.update(planner.get("extra_env", {}))

    this_log_path = log_path(planner, map_id)
    print(f"[map {map_id} {planner['key']}] running {planner['script'].name}")
    with this_log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-u", str(planner["script"])],
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            if any(keyword in line for keyword in SUMMARY_KEYWORDS):
                print(line.rstrip())
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"{planner['script'].name} failed with exit code {return_code}; see {this_log_path}"
        )


def ensure_outputs(args):
    for map_id in args.maps:
        for key, _label, _script, _prefix in PLANNER_SPECS:
            planner = planner_for(key, map_id, args)
            path = metrics_path(planner, map_id)
            if args.reuse_existing and path.exists():
                print(f"[map {map_id} {key}] reusing {path}")
            else:
                run_planner(planner, map_id, args)


def read_metrics(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)
    return data


def parse_task_completion(path):
    if not path.exists():
        return np.nan
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Task Completion = ([0-9.]+)%", text)
    return float(matches[-1]) if matches else np.nan


def summarize_runs(args):
    out_dir = VIZ_DIR / "multimap_all_planner_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    time_grid = np.linspace(0.0, args.wallclock_seconds, 301)
    curves = {
        key: {
            "global": [],
            "occupied": [],
            "variance": [],
        }
        for key, _label, _script, _prefix in PLANNER_SPECS
    }

    for map_id in args.maps:
        for key, label, _script, _prefix in PLANNER_SPECS:
            planner = planner_for(key, map_id, args)
            data = read_metrics(metrics_path(planner, map_id))
            wall_time = np.concatenate(([0.0], np.asarray(data["wall_time_seconds"], dtype=float)))
            global_drop = np.concatenate(
                ([0.0], float(data["global_rmse"][0]) - np.asarray(data["global_rmse"], dtype=float))
            )
            occupied_drop = np.concatenate(
                ([0.0], float(data["occupied_rmse"][0]) - np.asarray(data["occupied_rmse"], dtype=float))
            )
            variance_drop = np.concatenate(
                ([0.0], INITIAL_VARIANCE - np.asarray(data["global_variance"], dtype=float))
            )
            curves[key]["global"].append(np.interp(time_grid, wall_time, global_drop))
            curves[key]["occupied"].append(np.interp(time_grid, wall_time, occupied_drop))
            curves[key]["variance"].append(np.interp(time_grid, wall_time, variance_drop))

            rows.append(
                {
                    "map_id": map_id,
                    "planner": key,
                    "label": label,
                    "steps_completed": int(data["timestep"][-1]),
                    "wall_time_seconds": float(data["wall_time_seconds"][-1]),
                    "initial_global_variance": INITIAL_VARIANCE,
                    "final_global_variance": float(data["global_variance"][-1]),
                    "variance_drop": INITIAL_VARIANCE - float(data["global_variance"][-1]),
                    "initial_global_rmse_logged": float(data["global_rmse"][0]),
                    "final_global_rmse": float(data["global_rmse"][-1]),
                    "global_rmse_drop_logged": float(data["global_rmse"][0])
                    - float(data["global_rmse"][-1]),
                    "initial_occupied_rmse_logged": float(data["occupied_rmse"][0]),
                    "final_occupied_rmse": float(data["occupied_rmse"][-1]),
                    "occupied_rmse_drop_logged": float(data["occupied_rmse"][0])
                    - float(data["occupied_rmse"][-1]),
                    "task_completion_percent": parse_task_completion(log_path(planner, map_id)),
                    "metrics_csv": str(metrics_path(planner, map_id)),
                    "log_path": str(log_path(planner, map_id)),
                }
            )

    rows_path = (
        out_dir
        / f"maps_{min(args.maps)}_{max(args.maps)}_all_planners_rows.csv"
    )
    with rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    means = []
    for key, label, _script, _prefix in PLANNER_SPECS:
        planner_rows = [row for row in rows if row["planner"] == key]
        means.append(
            {
                "planner": key,
                "label": label,
                "maps": len(planner_rows),
                "mean_steps_completed": float(np.mean([row["steps_completed"] for row in planner_rows])),
                "mean_final_global_variance": float(np.mean([row["final_global_variance"] for row in planner_rows])),
                "std_final_global_variance": float(np.std([row["final_global_variance"] for row in planner_rows])),
                "mean_variance_drop": float(np.mean([row["variance_drop"] for row in planner_rows])),
                "mean_final_global_rmse": float(np.mean([row["final_global_rmse"] for row in planner_rows])),
                "mean_global_rmse_drop_logged": float(
                    np.mean([row["global_rmse_drop_logged"] for row in planner_rows])
                ),
                "mean_final_occupied_rmse": float(np.mean([row["final_occupied_rmse"] for row in planner_rows])),
                "mean_occupied_rmse_drop_logged": float(
                    np.mean([row["occupied_rmse_drop_logged"] for row in planner_rows])
                ),
                "mean_task_completion_percent": float(
                    np.nanmean([row["task_completion_percent"] for row in planner_rows])
                ),
            }
        )

    means_path = (
        out_dir
        / f"maps_{min(args.maps)}_{max(args.maps)}_all_planners_means.csv"
    )
    with means_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(means[0].keys()))
        writer.writeheader()
        writer.writerows(means)

    plot_path = (
        out_dir
        / f"maps_{min(args.maps)}_{max(args.maps)}_all_planners_mean_drops_over_walltime.png"
    )
    colors = {
        "greedy": "tab:blue",
        "lawnmower": "tab:orange",
        "cmaes": "tab:green",
        "diffusion": "tab:red",
        "imitation": "tab:purple",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), constrained_layout=True)
    panels = [
        ("global", "Global RMSE Drop", "First logged - current RMSE"),
        ("occupied", "Occupied RMSE Drop", "First logged - current RMSE"),
        ("variance", "Variance Drop", "Initial total variance - current variance"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for key, label, _script, _prefix in PLANNER_SPECS:
            values = np.vstack(curves[key][metric])
            mean = values.mean(axis=0)
            lower = np.percentile(values, 25, axis=0)
            upper = np.percentile(values, 75, axis=0)
            ax.plot(time_grid, mean, linewidth=2, label=label, color=colors[key])
            ax.fill_between(time_grid, lower, upper, color=colors[key], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Wall-clock time (s)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, args.wallclock_seconds)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle(
        (
            f"GRF maps {min(args.maps)}-{max(args.maps)} planner drops over wall-clock time "
            f"(mean over {len(args.maps)} maps)"
        ),
        fontsize=14,
    )
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"Wrote rows: {rows_path}")
    print(f"Wrote means: {means_path}")
    print(f"Wrote plot: {plot_path}")
    return rows_path, means_path, plot_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", type=int, nargs="+", default=DEFAULT_MAPS)
    parser.add_argument("--maptype", default=MAPTYPE)
    parser.add_argument("--timealloted", type=int, default=TIMEALLOTED)
    parser.add_argument("--wallclock-seconds", type=float, default=WALLCLOCK_SECONDS)
    parser.add_argument("--execution-chunk", type=int, default=20)
    parser.add_argument("--utility-threshold", type=float, default=UTILITY_THRESHOLD)
    parser.add_argument("--sensornoise-seed", type=int, default=123)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--diffusion-seed", type=int, default=17000)
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=SCRIPT_DIR / "CMAES_beamsearch_dataset_3d_synthetic_final.pt",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.summary_only:
        ensure_outputs(args)
    summarize_runs(args)


if __name__ == "__main__":
    main()
