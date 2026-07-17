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
Map-id sweep for Diffusionplanner_singlemap.py and ImitateTrans_singlemap.py.

Runs each script once per map_id in MAP_IDS (as a subprocess, current settings
otherwise unchanged - each script's own selected/default execution_chunk, beta,
alpha, utility_threshold, checkpoint, MAPTYPE), then consolidates every run's
own map_{N}_rmse_over_time.csv output plus a stdout-only figure (initial
variance) into one CSV and one comparison plot for variance drop and RMSE drop.

Two things are deliberately overridden from "current settings" for this sweep:
- WALLCLOCK_SECONDS -> 200, per the request ("of 200 seconds each").
- ENFORCE_MIN_STEP_TIME -> off (0). That pacing floor exists to simulate
  realistic flight timing for an interactive demo; with it on, WALLCLOCK_SECONDS
  measures the same real compute clock but harder maps (more/slower replans)
  could get truncated to fewer simulated timesteps than easier ones, confounding
  the map-to-map comparison with an artifact of wall-clock pacing rather than
  genuine per-map difficulty/quality differences (same reasoning as the earlier
  execution_chunk sweep). SKIP_VIZ is also turned on so gifs aren't rendered for
  all 14 runs.

execution_chunk is intentionally left unset (per-script current default: 10 for
Diffusion, 20 for Imitation) since the request was "current settings."
"""

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

MAP_IDS = [39, 40, 41, 42, 43, 44, 45]
MAPTYPE = "NAIP"
WALLCLOCK_SECONDS = 200

PLANNERS = {
    "Diffusion": {
        "script": SCRIPT_DIR / "Diffusionplanner_singlemap.py",
        "output_dir_template": "diffusion_map_{maptype}_{map_id}_viz",
    },
    "Imitation": {
        "script": SCRIPT_DIR / "ImitateTrans_singlemap.py",
        "output_dir_template": "imitate_trans_map_{maptype}_{map_id}_viz",
    },
}

RESULTS_DIR = SCRIPT_DIR / "sweep_results"
RESULTS_DIR.mkdir(exist_ok=True)
CONSOLIDATED_CSV = RESULTS_DIR / "map_sweep_39_45.csv"
CONSOLIDATED_PLOT = RESULTS_DIR / "map_sweep_39_45.png"

INITIAL_VARIANCE_RE = re.compile(r"Initial total variance:\s*([-\d.]+)")
TASK_COMPLETION_RE = re.compile(r"Task Completion = ([\d.]+)%")
REPLAN_TIME_RE = re.compile(r"Replanning time:\s*([\d.]+) seconds")

FIELDNAMES = [
    "planner",
    "map_id",
    "num_timesteps_completed",
    "initial_variance",
    "final_variance",
    "variance_drop",
    "final_global_rmse",
    "final_occupied_rmse",
    "global_rmse_drop",
    "occupied_rmse_drop",
    "global_rmse_auc",
    "occupied_rmse_auc",
    "task_completion_pct",
    "num_replans",
    "mean_replan_seconds",
    "sim_wall_time_seconds",
    "subprocess_wall_time_seconds",
]


def run_one(script_path, map_id):
    env = dict(os.environ)
    env["SELECTED_MAP"] = str(map_id)
    env["MAPTYPE"] = MAPTYPE
    env["WALLCLOCK_SECONDS"] = str(WALLCLOCK_SECONDS)
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
            f"{script_path.name} (map_id={map_id}) failed with exit code {result.returncode}"
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
        "initial_global_rmse": float(global_rmse[0]),
        "initial_occupied_rmse": float(occupied_rmse[0]),
        "global_rmse_auc": float(np.trapezoid(global_rmse, dx=1.0)),
        "occupied_rmse_auc": float(np.trapezoid(occupied_rmse, dx=1.0)),
        "sim_wall_time_seconds": float(wall_time_seconds[-1]),
    }


def main():
    rows = []

    for planner_name, config in PLANNERS.items():
        for map_id in MAP_IDS:
            print(f"[{planner_name}] map_id={map_id}: running...", flush=True)
            stdout, subprocess_elapsed = run_one(config["script"], map_id)

            output_dir = SCRIPT_DIR / "Vizualization" / config["output_dir_template"].format(
                maptype=MAPTYPE, map_id=map_id
            )
            csv_metrics = read_run_csv(output_dir, map_id)
            stdout_metrics = parse_stdout(stdout)

            row = {
                "planner": planner_name,
                "map_id": map_id,
                "subprocess_wall_time_seconds": round(subprocess_elapsed, 2),
                **csv_metrics,
                **stdout_metrics,
            }
            row["variance_drop"] = row["initial_variance"] - row["final_variance"]
            row["global_rmse_drop"] = row["initial_global_rmse"] - row["final_global_rmse"]
            row["occupied_rmse_drop"] = row["initial_occupied_rmse"] - row["final_occupied_rmse"]
            rows.append(row)

            # Archive this run's raw per-timestep CSV under a unique name.
            archive_path = RESULTS_DIR / f"{planner_name.lower()}_map{map_id}_rmse_over_time.csv"
            src_csv = output_dir / f"map_{map_id}_rmse_over_time.csv"
            archive_path.write_bytes(src_csv.read_bytes())

            print(
                f"  done in {subprocess_elapsed:.1f}s ({row['num_timesteps_completed']} timesteps) | "
                f"variance_drop={row['variance_drop']:.4f} | "
                f"global_rmse_drop={row['global_rmse_drop']:.4f} | "
                f"occupied_rmse_drop={row['occupied_rmse_drop']:.4f} | "
                f"task_completion={row['task_completion_pct']:.2f}%",
                flush=True,
            )

    fieldnames = [f for f in FIELDNAMES if f in rows[0]] + [
        k for k in rows[0] if k not in FIELDNAMES
    ]
    with open(CONSOLIDATED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nConsolidated CSV: {CONSOLIDATED_CSV}")

    build_plot(rows)
    print(f"Consolidated plot: {CONSOLIDATED_PLOT}")


def build_plot(rows):
    # Fixed categorical colors per planner (identity, not magnitude) - matches
    # the blue=Diffusion / orange=Imitation convention used earlier this project.
    colors = {"Diffusion": "tab:blue", "Imitation": "tab:orange"}

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    for planner_name in PLANNERS:
        planner_rows = sorted(
            (r for r in rows if r["planner"] == planner_name),
            key=lambda r: r["map_id"],
        )
        map_ids = [r["map_id"] for r in planner_rows]
        color = colors[planner_name]

        axes[0].plot(map_ids, [r["variance_drop"] for r in planner_rows], marker="o", color=color, label=planner_name)
        axes[1].plot(map_ids, [r["global_rmse_drop"] for r in planner_rows], marker="o", color=color, label=planner_name)
        axes[2].plot(map_ids, [r["occupied_rmse_drop"] for r in planner_rows], marker="o", color=color, label=planner_name)

    axes[0].set_title("Variance drop (initial - final)")
    axes[0].set_ylabel("Variance drop")

    axes[1].set_title("Global RMSE drop (initial - final)")
    axes[1].set_ylabel("Global RMSE drop")

    axes[2].set_title("Occupied RMSE drop (initial - final)")
    axes[2].set_ylabel("Occupied RMSE drop")

    for axis in axes:
        axis.set_xlabel("map_id")
        axis.set_xticks(MAP_IDS)
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle(f"Map sweep (id {MAP_IDS[0]}-{MAP_IDS[-1]}) - {WALLCLOCK_SECONDS}s per run", fontsize=14)
    figure.savefig(CONSOLIDATED_PLOT, dpi=150, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
