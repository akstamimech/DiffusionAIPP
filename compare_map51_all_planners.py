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
OUT_DIR = VIZ_DIR / "map51_all_planner_comparison"
RUN_TAG = "mission300_steps3000"
MAP_ID = 51
MAPTYPE = "grf"
TIMEALLOTED = 3000
WALLCLOCK_SECONDS = 300
UTILITY_THRESHOLD = 0.5


PLANNERS = [
    {
        "key": "greedy",
        "label": "GreedyGradient",
        "script": SCRIPT_DIR / "greedygradient_singlemap.py",
        "output_dir": VIZ_DIR / f"greedygradient_map_{MAPTYPE}_{MAP_ID}_viz_{RUN_TAG}",
    },
    {
        "key": "lawnmower",
        "label": "Lawnmower",
        "script": SCRIPT_DIR / "lawnmower_singlemap.py",
        "output_dir": VIZ_DIR / f"lawnmower_map_{MAPTYPE}_{MAP_ID}_viz_{RUN_TAG}",
    },
    {
        "key": "cmaes",
        "label": "CMA-ES",
        "script": SCRIPT_DIR / "CMAES_classic_singlemap.py",
        "output_dir": VIZ_DIR / f"classic_map_{MAPTYPE}_{MAP_ID}_viz_{RUN_TAG}",
    },
    {
        "key": "diffusion",
        "label": "Diffusion",
        "script": SCRIPT_DIR / "Diffusionplanner_singlemap.py",
        "output_dir": VIZ_DIR / f"diffusion_map_{MAPTYPE}_{MAP_ID}_viz_{RUN_TAG}",
        "extra_env": {
            "ETA": "1.0",
            "RUN_SEED": "17051",
            "DIFFUSION_DATASET_PATH": str(
                SCRIPT_DIR / "CMAES_beamsearch_dataset_3d_synthetic_final.pt"
            ),
        },
    },
    {
        "key": "imitation",
        "label": "ImitateTrans",
        "script": SCRIPT_DIR / "ImitateTrans_singlemap.py",
        "output_dir": VIZ_DIR / f"imitate_trans_map_{MAPTYPE}_{MAP_ID}_viz_{RUN_TAG}",
    },
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


def run_planner(planner):
    planner["output_dir"].mkdir(parents=True, exist_ok=True)
    log_path = planner["output_dir"] / f"map_{MAP_ID}_{planner['key']}_stdout.log"
    env = os.environ.copy()
    env.update(
        {
            "SELECTED_MAP": str(MAP_ID),
            "MAPTYPE": MAPTYPE,
            "TIMEALLOTED": str(TIMEALLOTED),
            "WALLCLOCK_SECONDS": str(WALLCLOCK_SECONDS),
            "EXECUTION_CHUNK": "20",
            "UTILITY_THRESHOLD": str(UTILITY_THRESHOLD),
            "SENSORNOISE_SEED": "123",
            "ENFORCE_MIN_STEP_TIME": "1",
            "SKIP_VIZ": "1",
            "RUN_OUTPUT_TAG": RUN_TAG,
            "PRINT_FLIGHT_PLAN": "0",
        }
    )
    env.update(planner.get("extra_env", {}))
    print(f"[{planner['key']}] running {planner['script'].name}")
    with log_path.open("w", encoding="utf-8") as log_file:
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
            f"{planner['script'].name} failed with exit code {return_code}; see {log_path}"
        )


def ensure_outputs():
    for planner in PLANNERS:
        metrics_path = planner["output_dir"] / f"map_{MAP_ID}_rmse_over_time.csv"
        if metrics_path.exists():
            print(f"[{planner['key']}] reusing {metrics_path}")
        else:
            run_planner(planner)


def read_metrics(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)
    return data


def parse_task_completion(log_path):
    if not log_path.exists():
        return np.nan
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Task Completion = ([0-9.]+)%", text)
    return float(matches[-1]) if matches else np.nan


def summarize():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    initial_variance = 130.05
    for planner in PLANNERS:
        metrics_path = planner["output_dir"] / f"map_{MAP_ID}_rmse_over_time.csv"
        log_path = planner["output_dir"] / f"map_{MAP_ID}_{planner['key']}_stdout.log"
        data = read_metrics(metrics_path)
        initial_global_rmse = float(data["global_rmse"][0])
        final_global_rmse = float(data["global_rmse"][-1])
        initial_occupied_rmse = float(data["occupied_rmse"][0])
        final_occupied_rmse = float(data["occupied_rmse"][-1])
        final_variance = float(data["global_variance"][-1])
        rows.append(
            {
                "planner": planner["key"],
                "label": planner["label"],
                "steps_completed": int(data["timestep"][-1]),
                "wall_time_seconds": float(data["wall_time_seconds"][-1]),
                "initial_global_variance": initial_variance,
                "final_global_variance": final_variance,
                "variance_drop": initial_variance - final_variance,
                "initial_global_rmse_logged": initial_global_rmse,
                "final_global_rmse": final_global_rmse,
                "global_rmse_drop_logged": initial_global_rmse - final_global_rmse,
                "initial_occupied_rmse_logged": initial_occupied_rmse,
                "final_occupied_rmse": final_occupied_rmse,
                "occupied_rmse_drop_logged": initial_occupied_rmse - final_occupied_rmse,
                "task_completion_percent": parse_task_completion(log_path),
                "metrics_csv": str(metrics_path),
                "log_path": str(log_path),
            }
        )

    summary_path = OUT_DIR / f"map_{MAP_ID}_all_planners_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = OUT_DIR / f"map_{MAP_ID}_all_planners_variance_rmse_drop.png"
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)
    axes[0].bar(x, [row["variance_drop"] for row in rows])
    axes[0].set_title("Variance Drop")
    axes[0].set_ylabel("Initial - final total variance")
    axes[1].bar(x, [row["global_rmse_drop_logged"] for row in rows])
    axes[1].set_title("Global RMSE Drop")
    axes[1].set_ylabel("First logged - final RMSE")
    axes[2].bar(x, [row["occupied_rmse_drop_logged"] for row in rows])
    axes[2].set_title("Occupied RMSE Drop")
    axes[2].set_ylabel("First logged - final RMSE")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(
        f"Map {MAP_ID} GRF planner comparison ({WALLCLOCK_SECONDS}s wall clock, {TIMEALLOTED} step cap)",
        fontsize=14,
    )
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    timeseries_path = OUT_DIR / f"map_{MAP_ID}_all_planners_drops_over_walltime.png"
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), constrained_layout=True)
    for planner in PLANNERS:
        metrics_path = planner["output_dir"] / f"map_{MAP_ID}_rmse_over_time.csv"
        data = read_metrics(metrics_path)
        wall_time = data["wall_time_seconds"]
        global_rmse_drop = data["global_rmse"][0] - data["global_rmse"]
        occupied_rmse_drop = data["occupied_rmse"][0] - data["occupied_rmse"]
        variance_drop = initial_variance - data["global_variance"]
        axes[0].plot(wall_time, global_rmse_drop, linewidth=2, label=planner["label"])
        axes[1].plot(wall_time, occupied_rmse_drop, linewidth=2, label=planner["label"])
        axes[2].plot(wall_time, variance_drop, linewidth=2, label=planner["label"])

    axes[0].set_title("Global RMSE Drop")
    axes[0].set_ylabel("First logged - current RMSE")
    axes[1].set_title("Occupied RMSE Drop")
    axes[1].set_ylabel("First logged - current RMSE")
    axes[2].set_title("Variance Drop")
    axes[2].set_ylabel("Initial total variance - current variance")
    for ax in axes:
        ax.set_xlabel("Wall-clock time (s)")
        ax.set_xlim(0, WALLCLOCK_SECONDS)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle(
        f"Map {MAP_ID} GRF planner drops over wall-clock time",
        fontsize=14,
    )
    fig.savefig(timeseries_path, dpi=180)
    plt.close(fig)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote plot: {plot_path}")
    print(f"Wrote time-series plot: {timeseries_path}")
    return rows, summary_path, plot_path, timeseries_path


def main():
    ensure_outputs()
    summarize()


if __name__ == "__main__":
    main()
