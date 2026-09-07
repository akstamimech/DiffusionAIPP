import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", str(SCRIPT_DIR / "results_hpc"))).resolve()
PLANNER_OUTPUT_ROOT = RESULTS_ROOT / "planner_outputs"
LOG_ROOT = RESULTS_ROOT / "logs"
SUMMARY_ROOT = RESULTS_ROOT / "summaries"


@dataclass(frozen=True)
class PlannerConfig:
    planner: str
    script_name: str
    output_prefix: str
    extra_env: dict = field(default_factory=dict)


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _mission_grid():
    map_start = _env_int("MAP_START", 0)
    map_end = _env_int("MAP_END", 150)
    repeats = _env_int("REPEATS_PER_MAP", 5)
    return [(map_id, repeat) for map_id in range(map_start, map_end + 1) for repeat in range(repeats)]


def _task_subset(items):
    task_id = os.environ.get("SWEEP_TASK_INDEX", os.environ.get("SLURM_ARRAY_TASK_ID"))
    if task_id is None:
        return items
    index = int(task_id)
    if index < 0 or index >= len(items):
        raise SystemExit(f"Task index {index} is outside 0..{len(items) - 1}")
    return [items[index]]


def _seed_for(map_id, repeat):
    base_seed = _env_int("BASE_SEED", 1000)
    return base_seed + map_id * 100 + repeat


def _trapz_mean(y, x):
    if len(y) == 0:
        return np.nan
    if len(y) == 1 or x[-1] <= x[0]:
        return float(y[-1])
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def _trajectory_summary(path):
    if not path.exists():
        return {
            "trajectory_rows": 0,
            "altitude_min": np.nan,
            "altitude_mean": np.nan,
            "altitude_max": np.nan,
            "altitude_final": np.nan,
            "altitude_auc_mean": np.nan,
            "path_length_xy": np.nan,
            "path_length_3d": np.nan,
        }

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    wall = data[:, 1]
    xyz = data[:, 2:5]
    diffs = np.diff(xyz, axis=0)
    xy_diffs = np.diff(xyz[:, :2], axis=0)
    altitude = xyz[:, 2]
    return {
        "trajectory_rows": int(data.shape[0]),
        "altitude_min": float(np.min(altitude)),
        "altitude_mean": float(np.mean(altitude)),
        "altitude_max": float(np.max(altitude)),
        "altitude_final": float(altitude[-1]),
        "altitude_auc_mean": _trapz_mean(altitude, wall),
        "path_length_xy": float(np.sum(np.linalg.norm(xy_diffs, axis=1))) if len(xy_diffs) else 0.0,
        "path_length_3d": float(np.sum(np.linalg.norm(diffs, axis=1))) if len(diffs) else 0.0,
    }


def _metrics_summary(path):
    if not path.exists():
        return {
            "metric_rows": 0,
            "sim_steps": 0,
            "wall_time_final": np.nan,
            "global_rmse_first": np.nan,
            "global_rmse_final": np.nan,
            "global_rmse_drop": np.nan,
            "occupied_rmse_first": np.nan,
            "occupied_rmse_final": np.nan,
            "occupied_rmse_drop": np.nan,
            "global_variance_first": np.nan,
            "global_variance_final": np.nan,
            "global_variance_drop": np.nan,
            "global_rmse_auc_mean": np.nan,
            "occupied_rmse_auc_mean": np.nan,
            "global_variance_auc_mean": np.nan,
        }

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    timestep = data[:, 0]
    wall = data[:, 1]
    global_rmse = data[:, 2]
    occupied_rmse = data[:, 3]
    global_variance = data[:, 4]
    return {
        "metric_rows": int(data.shape[0]),
        "sim_steps": int(timestep[-1]),
        "wall_time_final": float(wall[-1]),
        "global_rmse_first": float(global_rmse[0]),
        "global_rmse_final": float(global_rmse[-1]),
        "global_rmse_drop": float(global_rmse[0] - global_rmse[-1]),
        "occupied_rmse_first": float(occupied_rmse[0]),
        "occupied_rmse_final": float(occupied_rmse[-1]),
        "occupied_rmse_drop": float(occupied_rmse[0] - occupied_rmse[-1]),
        "global_variance_first": float(global_variance[0]),
        "global_variance_final": float(global_variance[-1]),
        "global_variance_drop": float(global_variance[0] - global_variance[-1]),
        "global_rmse_auc_mean": _trapz_mean(global_rmse, wall),
        "occupied_rmse_auc_mean": _trapz_mean(occupied_rmse, wall),
        "global_variance_auc_mean": _trapz_mean(global_variance, wall),
    }


def _array_task_id():
    return os.environ.get("SWEEP_TASK_INDEX", os.environ.get("SLURM_ARRAY_TASK_ID"))


def _write_summary(planner, rows, task_id=None):
    if task_id is None:
        SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
        summary_path = SUMMARY_ROOT / f"{planner}_summary.csv"
    else:
        task_summary_root = SUMMARY_ROOT / "tasks"
        task_summary_root.mkdir(parents=True, exist_ok=True)
        summary_path = task_summary_root / f"{planner}_task{task_id}.csv"
    if not rows:
        return summary_path
    fieldnames = list(rows[0].keys())
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def _run_one(config, map_id, repeat):
    seed = _seed_for(map_id, repeat)
    run_tag = f"hpc_{config.planner}_map{map_id}_repeat{repeat}_seed{seed}"
    maptype = os.environ.get("MAPTYPE", "grf")
    output_dir = PLANNER_OUTPUT_ROOT / f"{config.output_prefix}_map_{maptype}_{map_id}_viz_{run_tag}"
    log_path = LOG_ROOT / f"{config.planner}_map{map_id}_repeat{repeat}_seed{seed}.log"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    PLANNER_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "SELECTED_MAP": str(map_id),
            "MAPTYPE": os.environ.get("MAPTYPE", "grf"),
            "TIMEALLOTED": os.environ.get("TIMEALLOTED", "3000"),
            "WALLCLOCK_SECONDS": os.environ.get("WALLCLOCK_SECONDS", "300"),
            "UTILITY_THRESHOLD": os.environ.get("UTILITY_THRESHOLD", "0.5"),
            "PLANNING_HORIZON": os.environ.get("PLANNING_HORIZON", "8"),
            "EXECUTION_CHUNK": os.environ.get("EXECUTION_CHUNK", "40"),
            "SENSORNOISE_SEED": str(seed),
            "RUN_SEED": str(seed),
            "CMA_SEED": str(seed),
            "ENFORCE_MIN_STEP_TIME": "1",
            "SKIP_VIZ": os.environ.get("SKIP_VIZ", "1"),
            "RUN_OUTPUT_TAG": run_tag,
            "CSV_DIR": os.environ.get("CSV_DIR", str(SCRIPT_DIR / "csv")),
            "RESULTS_ROOT": str(PLANNER_OUTPUT_ROOT),
            "DIFFUSION_CHECKPOINT": os.environ.get(
                "DIFFUSION_CHECKPOINT", str(SCRIPT_DIR / "checkpoints" / "current_best.pth")
            ),
            "IMITATE_CHECKPOINT": os.environ.get(
                "IMITATE_CHECKPOINT", str(SCRIPT_DIR / "checkpoints" / "imitate_best.pth")
            ),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "1"),
        }
    )
    env.update(config.extra_env)

    python_exe = os.environ.get("PYTHON_EXECUTABLE", sys.executable)
    cmd = [python_exe, "-u", str(SCRIPT_DIR / config.script_name)]
    started = time.time()
    with log_path.open("w", newline="") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - started

    metrics_path = output_dir / f"map_{map_id}_rmse_over_time.csv"
    trajectory_path = output_dir / f"map_{map_id}_executed_trajectory.csv"
    row = {
        "planner": config.planner,
        "map_id": map_id,
        "repeat": repeat,
        "seed": seed,
        "return_code": proc.returncode,
        "process_wall_seconds": elapsed,
        "output_dir": str(output_dir),
        "log_file": str(log_path),
    }
    row.update(_metrics_summary(metrics_path))
    row.update(_trajectory_summary(trajectory_path))
    if proc.returncode != 0:
        row["status"] = "failed"
    elif not metrics_path.exists():
        row["status"] = "missing_metrics"
    else:
        row["status"] = "ok"
    return row


def run_sweep(config):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    task_id = _array_task_id()
    items = _task_subset(_mission_grid())
    rows = []
    for map_id, repeat in items:
        print(f"[{config.planner}] map={map_id} repeat={repeat}", flush=True)
        row = _run_one(config, map_id, repeat)
        rows.append(row)
        _write_summary(config.planner, rows, task_id=task_id)
        if row["return_code"] != 0 and os.environ.get("STOP_ON_FAILURE", "0") == "1":
            raise SystemExit(row["return_code"])
    summary_path = _write_summary(config.planner, rows, task_id=task_id)
    print(f"[{config.planner}] wrote {summary_path}", flush=True)
