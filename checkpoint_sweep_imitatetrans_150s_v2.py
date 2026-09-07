"""
Redo of checkpoint_sweep_imitatetrans_100s.py: WALLCLOCK_SECONDS has been changed back to
150 in ImitateTrans_singlemap.py, but EXECUTION_CHUNK is still 10 (not reverted to the
original sweep's 20) - confirmed by reading the script fresh, not assumed. So this is a
genuinely third distinct configuration, different from both the original
checkpoint_sweep_imitatetrans.py (150s, EXECUTION_CHUNK=20, 3 repeats) and
checkpoint_sweep_imitatetrans_100s.py (100s, EXECUTION_CHUNK=10, 1 repeat) - hence its own
script, output CSV, and RUN_OUTPUT_TAG prefix (ckpt_sweep_150s_v2) so none of the three
overwrite each other's output directories.

Same as the 100s version otherwise: one run per checkpoint (ImitateTrans is deterministic),
4-way parallel with single-threaded BLAS per subprocess, only IMITATE_CHECKPOINT overridden
per run.

Writes checkpoint_sweep_imitatetrans_150s_v2_summary.csv, one row per checkpoint.
"""
import csv
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
VIZ_DIR = SCRIPT_DIR / "Vizualization"
LOG_DIR = SCRIPT_DIR / "checkpoint_sweep_imitatetrans_150s_v2_logs"
LOG_DIR.mkdir(exist_ok=True)
SUMMARY_CSV = SCRIPT_DIR / "checkpoint_sweep_imitatetrans_150s_v2_summary.csv"

EPOCHS = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
SEED = 90001  # single run per checkpoint - ImitateTrans is deterministic, see docstring
RUN_TAG_PREFIX = "ckpt_sweep_150s_v2"
MAX_WORKERS = 4  # "a little" parallelism on a 12-core machine, single-threaded BLAS per worker
BLAS_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_write_lock = threading.Lock()

MAPTYPE = os.environ.get("MAPTYPE", "grf")
SELECTED_MAP = int(os.environ.get("SELECTED_MAP", 59))

LOG_LINE_PATTERNS = {
    "initial_total_variance": r"Initial total variance:\s*([\-0-9.]+)",
    "final_total_variance": r"Final total variance:\s*([\-0-9.]+)",
    "variance_reduction": r"Variance reduction:\s*([\-0-9.]+)",
    "gained_utility": r"Gained Utility = ([\-0-9.]+)",
    "total_utility": r"Total Utility = ([\-0-9.]+)",
    "task_completion_pct": r"Task Completion = ([\-0-9.]+)%",
    "coverage_efficiency": r"Coverage Efficiency = ([\-0-9.]+)",
    "final_global_rmse": r"Global RMSE = ([\-0-9.]+), Occupied RMSE",
    "final_occupied_rmse": r"Occupied RMSE = ([\-0-9.]+)\s*$",
    "global_rmse_auc": r"Global RMSE AUC = ([\-0-9.]+), Mean = ([\-0-9.]+)",
    "occupied_rmse_auc": r"Occupied RMSE AUC = ([\-0-9.]+), Mean = ([\-0-9.]+)",
    "flight_ended_early": r"flight ended early",
}


def parse_log(text):
    out = {}
    for key, pattern in LOG_LINE_PATTERNS.items():
        if key in ("global_rmse_auc", "occupied_rmse_auc"):
            m = re.search(pattern, text, re.MULTILINE)
            if m:
                out[key] = float(m.group(1))
                out[key.replace("_auc", "_mean")] = float(m.group(2))
        elif key == "flight_ended_early":
            out[key] = bool(re.search(pattern, text))
        else:
            m = re.search(pattern, text, re.MULTILINE)
            if m:
                out[key] = float(m.group(1))
    return out


def run_one(epoch):
    checkpoint_path = CHECKPOINT_DIR / f"imitate_trans_waypoints_epoch_{epoch}.pth"
    run_tag = f"{RUN_TAG_PREFIX}_epoch{epoch}"
    output_dir = VIZ_DIR / f"imitate_trans_map_{MAPTYPE}_{SELECTED_MAP}_viz_{run_tag}"
    log_path = LOG_DIR / f"{run_tag}.log"

    env = os.environ.copy()
    env.update(BLAS_THREAD_ENV)
    env.update(
        {
            "IMITATE_CHECKPOINT": str(checkpoint_path),
            "RUN_SEED": str(SEED),
            "RUN_OUTPUT_TAG": run_tag,
            "SKIP_VIZ": "1",
        }
    )

    start = time.time()
    print(f"[epoch {epoch}] starting, log -> {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-u", str(SCRIPT_DIR / "ImitateTrans_singlemap.py")],
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
        return_code = process.wait()
    elapsed = time.time() - start

    row = {
        "epoch": epoch,
        "seed": SEED,
        "return_code": return_code,
        "wall_clock_elapsed_s": elapsed,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
    }

    if return_code != 0:
        print(f"[epoch {epoch}] FAILED (exit {return_code}) after {elapsed:.1f}s", flush=True)
        row["status"] = "failed"
        return row

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    row.update(parse_log(log_text))

    metrics_csv = output_dir / f"map_{SELECTED_MAP}_rmse_over_time.csv"
    if metrics_csv.exists():
        data = np.genfromtxt(metrics_csv, delimiter=",", names=True)
        if data.ndim == 0:
            data = np.asarray([data], dtype=data.dtype)
        row["steps_completed"] = int(data["timestep"][-1])
        row["final_wall_time_from_csv"] = float(data["wall_time_seconds"][-1])
        row["final_global_variance_from_csv"] = float(data["global_variance"][-1])

    row["status"] = "ok"
    print(
        f"[epoch {epoch}] done in {elapsed:.1f}s - "
        f"task_completion={row.get('task_completion_pct', 'NA')}%, "
        f"variance_reduction={row.get('variance_reduction', 'NA')}",
        flush=True,
    )
    return row


def write_summary(rows_by_epoch):
    rows = [rows_by_epoch[e] for e in sorted(rows_by_epoch)]
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing():
    if not SUMMARY_CSV.exists():
        return {}
    with SUMMARY_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    existing = {}
    for r in rows:
        if r.get("status") == "ok" and int(r["seed"]) == SEED:
            existing[int(r["epoch"])] = r
    return existing


def main():
    existing = load_existing()
    todo = [e for e in EPOCHS if e not in existing]
    print(f"Checkpoint sweep (ImitateTrans, 150s current settings, EXECUTION_CHUNK=10): {len(EPOCHS)} epochs total, "
          f"{len(existing)} already done, {len(todo)} to run, {MAX_WORKERS}-way parallel", flush=True)
    print(f"SELECTED_MAP={SELECTED_MAP} MAPTYPE={MAPTYPE} WALLCLOCK_SECONDS=150 (both left at ImitateTrans_singlemap.py's own current defaults)", flush=True)

    rows_by_epoch = dict(existing)
    if todo:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(run_one, epoch): epoch for epoch in todo}
            for future in as_completed(futures):
                row = future.result()
                with _write_lock:
                    rows_by_epoch[row["epoch"]] = row
                    write_summary(rows_by_epoch)

    print(f"\nSweep complete. Wrote {SUMMARY_CSV}", flush=True)
    n_failed = sum(1 for r in rows_by_epoch.values() if r["status"] != "ok")
    if n_failed:
        print(f"WARNING: {n_failed} of {len(rows_by_epoch)} runs failed - see checkpoint_sweep_imitatetrans_150s_v2_logs/", flush=True)


if __name__ == "__main__":
    main()
