"""
Wrapper around ImitateTrans_singlemap.py (unmodified - see docstring note below) that
determines, for each checkpoint, how long it takes to reach 80% occupied variance
reduction (fraction_left <= 0.20), for the CURRENT script settings (horizon/EXECUTION_
CHUNK/WALLCLOCK_SECONDS changed since the last sweep - read fresh at run time via the
smoke test, not assumed).

LIMITATION, stated directly: ImitateTrans_singlemap.py writes no incremental state during
a run (its executed_trajectory.csv / rmse_over_time.csv are only written once, after the
whole simulation loop finishes) and prints nothing about occupied variance along the way.
So this wrapper cannot literally interrupt a running simulation the instant it crosses 80%
without instrumenting the script itself, which was explicitly ruled out. What it does
instead: runs each checkpoint to whatever WALLCLOCK_SECONDS ImitateTrans_singlemap.py
itself currently defaults to (600s, not overridden here - see the "only override the
checkpoint" note below), then replays the resulting trajectory (important_region_variance_from_trajectories.
reconstruct_one - same Kalman-replay approach used throughout this session, no
re-simulation) to find the exact wall-clock time it FIRST crossed 80% reduction. This
produces the same number "stop at 80%" would have reported; it just doesn't save wall-
clock time on runs that cross early, since the underlying process still runs to the
ceiling regardless.

One run per checkpoint (ImitateTrans is deterministic - established earlier this session).
Parallelized: bounded worker pool + single-threaded BLAS per subprocess (OMP/OPENBLAS/MKL/
NUMEXPR/VECLIB_NUM_THREADS=1), same pattern as the other checkpoint sweeps.

Writes sweep_imitatetrans_stop_at_80pct_summary.csv.
"""
import csv
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
VIZ_DIR = SCRIPT_DIR / "Vizualization"
LOG_DIR = SCRIPT_DIR / "sweep_imitatetrans_stop_at_80pct_logs"
LOG_DIR.mkdir(exist_ok=True)
SUMMARY_CSV = SCRIPT_DIR / "sweep_imitatetrans_stop_at_80pct_summary.csv"

EPOCHS = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
SEED = 90001
RUN_TAG_PREFIX = "stop_at_80pct"
THRESHOLD = 0.20  # <=20% remaining = 80% reduction
# WALLCLOCK_SECONDS is deliberately NOT overridden below - it stays at whatever
# ImitateTrans_singlemap.py's own current default is (600s as of this run, confirmed by
# reading the script fresh), matching the "only override the checkpoint" convention used
# for every other sweep this session.
MAX_WORKERS = 4
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


def run_one(epoch):
    checkpoint_path = CHECKPOINT_DIR / f"imitate_trans_waypoints_epoch_{epoch}.pth"
    run_tag = f"{RUN_TAG_PREFIX}_epoch{epoch}"
    output_dir = VIZ_DIR / f"imitate_trans_map_{MAPTYPE}_{SELECTED_MAP}_viz_{run_tag}"
    log_path = LOG_DIR / f"{run_tag}.log"

    env = os.environ.copy()
    env.update(BLAS_THREAD_ENV)
    env.update({
        "IMITATE_CHECKPOINT": str(checkpoint_path),
        "RUN_SEED": str(SEED),
        "RUN_OUTPUT_TAG": run_tag,
        "SKIP_VIZ": "1",
    })

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
        for line in process.stdout:
            log_file.write(line)
        return_code = process.wait()
    elapsed = time.time() - start

    row = {
        "epoch": epoch, "seed": SEED, "return_code": return_code,
        "wall_clock_elapsed_s": elapsed, "output_dir": str(output_dir), "log_path": str(log_path),
    }

    if return_code != 0:
        print(f"[epoch {epoch}] FAILED (exit {return_code}) after {elapsed:.1f}s", flush=True)
        row["status"] = "failed"
        return row

    traj_path = output_dir / f"map_{SELECTED_MAP}_executed_trajectory.csv"
    result = m.reconstruct_one({"map_id": SELECTED_MAP, "traj_path": str(traj_path)})
    wall_time = result["wall_time_seconds"]
    important_var = result["important_variance"]
    frac_left = important_var / important_var[0]
    hit = np.where(frac_left <= THRESHOLD)[0]

    row["mission_end_wall_time_s"] = float(wall_time[-1])
    if hit.size:
        row["time_to_80pct_reduction_s"] = float(wall_time[hit[0]])
        row["status"] = "reached"
    else:
        row["time_to_80pct_reduction_s"] = ""
        row["status"] = "not_reached"

    if row["status"] == "reached":
        msg = f"reached 80% at {row['time_to_80pct_reduction_s']:.1f}s"
    else:
        msg = f"NOT REACHED (ran to {row['mission_end_wall_time_s']:.1f}s)"
    print(f"[epoch {epoch}] done in {elapsed:.1f}s - {msg}", flush=True)
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


def main():
    print(f"Sweep: {len(EPOCHS)} ImitateTrans checkpoints, WALLCLOCK_SECONDS left at script default, "
          f"{MAX_WORKERS}-way parallel, SELECTED_MAP={SELECTED_MAP}", flush=True)
    rows_by_epoch = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, epoch): epoch for epoch in EPOCHS}
        for future in as_completed(futures):
            row = future.result()
            with _write_lock:
                rows_by_epoch[row["epoch"]] = row
                write_summary(rows_by_epoch)

    print(f"\nSweep complete. Wrote {SUMMARY_CSV}")
    n_reached = sum(1 for r in rows_by_epoch.values() if r.get("status") == "reached")
    n_not_reached = sum(1 for r in rows_by_epoch.values() if r.get("status") == "not_reached")
    n_failed = sum(1 for r in rows_by_epoch.values() if r.get("status") == "failed")
    print(f"{n_reached} reached, {n_not_reached} did not reach it, {n_failed} failed")


if __name__ == "__main__":
    main()
