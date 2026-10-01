import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

"""
Execution-chunk sweep for Diffusionplanner_singlemap.py and ImitateTrans_singlemap.py.

Runs each script once per EXECUTION_CHUNK_VALUES entry (as a subprocess, current
settings otherwise unchanged - same selected_map/MAPTYPE/checkpoint/beta/alpha/
utility_threshold each script already defaults to), then consolidates every run's
own map_{N}_rmse_over_time.csv output plus a few stdout-only figures (initial
variance, task completion, per-replan timing) into one CSV and one comparison plot.

Two "current settings" are deliberately overridden for the sweep rather than left
at their interactive-demo defaults:
- WALLCLOCK_SECONDS -> unconstrained (0). At the default 90s budget + the
  per-step pacing floor, a small execution_chunk (more frequent, costlier
  replans) could get cut off earlier than a large one, confounding the
  comparison with an artifact of wall-clock truncation rather than genuine
  execution_chunk quality differences. Unconstrained guarantees every run
  completes the same 200 timesteps.
- ENFORCE_MIN_STEP_TIME -> off (0). That pacing floor exists to simulate
  realistic flight timing for an interactive demo; it only adds dead time
  (theoretical minimum 200 * 0.333s = 66.7s of pure sleep per run) in a batch
  sweep and isn't part of what's being measured.
- SKIP_VIZ -> on (1). Per-run gifs/videos aren't needed for the sweep and are
  the slowest part of each run; SKIP_VIZ was added to both scripts (mirroring
  greedygradient_singlemap.py's existing convention) specifically to make this
  practical.
"""

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

EXECUTION_CHUNK_VALUES = [5, 10, 15, 20, 25, 30]
MAP_ID = 44
MAPTYPE = "NAIP"

PLANNERS = {
    "Diffusion": {
        "script": SCRIPT_DIR / "Diffusionplanner_singlemap.py",
        "output_dir": SCRIPT_DIR / "Vizualization" / f"diffusion_map_{MAPTYPE}_{MAP_ID}_viz",
    },
    "Imitation": {
        "script": SCRIPT_DIR / "ImitateTrans_singlemap.py",
        "output_dir": SCRIPT_DIR / "Vizualization" / f"imitate_trans_map_{MAPTYPE}_{MAP_ID}_viz",
    },
}

RESULTS_DIR = SCRIPT_DIR / "sweep_results"
RESULTS_DIR.mkdir(exist_ok=True)
CONSOLIDATED_CSV = RESULTS_DIR / "execution_chunk_sweep_map44.csv"
CONSOLIDATED_PLOT = RESULTS_DIR / "execution_chunk_sweep_map44.png"

INITIAL_VARIANCE_RE = re.compile(r"Initial total variance:\s*([-\d.]+)")
TASK_COMPLETION_RE = re.compile(r"Task Completion = ([\d.]+)%")
REPLAN_TIME_RE = re.compile(r"Replanning time:\s*([\d.]+) seconds")

FIELDNAMES = [
    "planner",
    "execution_chunk",
    "num_timesteps_completed",
    "initial_variance",
    "final_variance",
    "variance_reduction",
    "final_global_rmse",
    "final_occupied_rmse",
    "global_rmse_auc",
    "global_rmse_mean",
    "occupied_rmse_auc",
    "occupied_rmse_mean",
    "task_completion_pct",
    "num_replans",
    "mean_replan_seconds",
    "sim_wall_time_seconds",
    "subprocess_wall_time_seconds",
]


def run_one(script_path, execution_chunk):
    env = dict(os.environ)
    env["EXECUTION_CHUNK"] = str(execution_chunk)
    env["SELECTED_MAP"] = str(MAP_ID)
    env["MAPTYPE"] = MAPTYPE
    env["WALLCLOCK_SECONDS"] = "0"
    env["ENFORCE_MIN_STEP_TIME"] = "0"
    env["SKIP_VIZ"] = "1"

    start = time.time()
    result = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(SCRIPT_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise RuntimeError(
            f"{script_path.name} (execution_chunk={execution_chunk}) failed with "
            f"exit code {result.returncode}"
        )

    return result.stdout, elapsed


def parse_stdout(stdout):
    initial_variance_match = INITIAL_VARIANCE_RE.search(stdout)
    task_completion_match = TASK_COMPLETION_RE.search(stdout)
    replan_times = [float(x) for x in REPLAN_TIME_RE.findall(stdout)]
    return {
        "initial_variance": float(initial_variance_match.group(1)) if initial_variance_match else float("nan"),
        "task_completion_pct": float(task_completion_match.group(1)) if task_completion_match else float("nan"),
        "num_replans": len(replan_times),
        "mean_replan_seconds": float(np.mean(replan_times)) if replan_times else float("nan"),
    }


def read_run_csv(output_dir, map_id):
    csv_path = output_dir / f"map_{map_id}_rmse_over_time.csv"
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    timestep, wall_time_seconds, global_rmse, occupied_rmse, global_variance = data.T
    return {
        "num_timesteps_completed": int(timestep[-1]),
        "final_global_rmse": float(global_rmse[-1]),
        "final_occupied_rmse": float(occupied_rmse[-1]),
        "final_variance": float(global_variance[-1]),
        "global_rmse_auc": float(np.trapezoid(global_rmse, dx=1.0)),
        "global_rmse_mean": float(np.mean(global_rmse)),
        "occupied_rmse_auc": float(np.trapezoid(occupied_rmse, dx=1.0)),
        "occupied_rmse_mean": float(np.mean(occupied_rmse)),
        "sim_wall_time_seconds": float(wall_time_seconds[-1]),
    }


def main():
    rows = []

    for planner_name, config in PLANNERS.items():
        for execution_chunk in EXECUTION_CHUNK_VALUES:
            print(f"[{planner_name}] execution_chunk={execution_chunk}: running...", flush=True)
            stdout, subprocess_elapsed = run_one(config["script"], execution_chunk)

            csv_metrics = read_run_csv(config["output_dir"], MAP_ID)
            stdout_metrics = parse_stdout(stdout)

            row = {
                "planner": planner_name,
                "execution_chunk": execution_chunk,
                "subprocess_wall_time_seconds": round(subprocess_elapsed, 2),
                **csv_metrics,
                **stdout_metrics,
            }
            row["variance_reduction"] = row["initial_variance"] - row["final_variance"]
            rows.append(row)

            # Archive this run's raw per-timestep CSV under a unique name so it
            # isn't overwritten by the next execution_chunk value's run.
            archive_path = RESULTS_DIR / f"{planner_name.lower()}_ec{execution_chunk}_map{MAP_ID}_rmse_over_time.csv"
            src_csv = config["output_dir"] / f"map_{MAP_ID}_rmse_over_time.csv"
            archive_path.write_bytes(src_csv.read_bytes())

            print(
                f"  done in {subprocess_elapsed:.1f}s | final_variance={row['final_variance']:.4f} "
                f"| final_occupied_rmse={row['final_occupied_rmse']:.4f} "
                f"| task_completion={row['task_completion_pct']:.2f}%",
                flush=True,
            )

    with open(CONSOLIDATED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nConsolidated CSV: {CONSOLIDATED_CSV}")

    build_plot(rows)
    print(f"Consolidated plot: {CONSOLIDATED_PLOT}")


def build_plot(rows):
    # Fixed categorical colors per planner (identity, not magnitude) - matches
    # the blue=Diffusion / orange=Imitation convention already used earlier in
    # this project's own variance-drop-over-epoch plot.
    colors = {"Diffusion": "tab:blue", "Imitation": "tab:orange"}

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    for planner_name in PLANNERS:
        planner_rows = sorted(
            (r for r in rows if r["planner"] == planner_name),
            key=lambda r: r["execution_chunk"],
        )
        chunks = [r["execution_chunk"] for r in planner_rows]
        color = colors[planner_name]

        axes[0].plot(
            chunks,
            [r["final_variance"] for r in planner_rows],
            marker="o",
            color=color,
            label=planner_name,
        )
        axes[1].plot(
            chunks,
            [r["final_occupied_rmse"] for r in planner_rows],
            marker="o",
            color=color,
            label=planner_name,
        )
        axes[2].plot(
            chunks,
            [r["sim_wall_time_seconds"] for r in planner_rows],
            marker="o",
            color=color,
            label=planner_name,
        )

    axes[0].set_title("Final global variance")
    axes[0].set_ylabel("Final variance")

    axes[1].set_title("Final occupied RMSE")
    axes[1].set_ylabel("Final occupied RMSE")

    axes[2].set_title("Simulated wall-clock time")
    axes[2].set_ylabel("Wall time (s)")

    for axis in axes:
        axis.set_xlabel("execution_chunk")
        axis.set_xticks(EXECUTION_CHUNK_VALUES)
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle(f"execution_chunk sweep - map {MAP_ID} ({MAPTYPE})", fontsize=14)
    figure.savefig(CONSOLIDATED_PLOT, dpi=150, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
