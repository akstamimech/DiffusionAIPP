"""
Same sweep as checkpoint_sweep_diffusionplanner.py, but for ImitateTrans_singlemap.py:
runs it once per (checkpoint epoch, repeat) pair, every other current script default left
unchanged (SELECTED_MAP, TIMEALLOTED, WALLCLOCK_SECONDS, EXECUTION_CHUNK, etc. all left at
whatever ImitateTrans_singlemap.py itself currently defaults to - only IMITATE_CHECKPOINT,
RUN_SEED, and RUN_OUTPUT_TAG are overridden per run).

Same REPEAT_SEEDS as the diffusion sweep are reused here too (90001/90002/90003), for the
same paired-design reason, and so that the same seed means the same sensor-noise/start
conditions across BOTH sweeps too if the two are ever compared directly.

SKIP_VIZ=1 for every run, same reasoning as the diffusion sweep.

Writes checkpoint_sweep_imitatetrans_summary.csv incrementally, one row per completed run.
"""
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
VIZ_DIR = SCRIPT_DIR / "Vizualization"
LOG_DIR = SCRIPT_DIR / "checkpoint_sweep_imitatetrans_logs"
LOG_DIR.mkdir(exist_ok=True)
SUMMARY_CSV = SCRIPT_DIR / "checkpoint_sweep_imitatetrans_summary.csv"

EPOCHS = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
REPEAT_SEEDS = [90001, 90002, 90003]
RUN_TAG_PREFIX = "ckpt_sweep"

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


def run_one(epoch, seed, repeat_idx):
    checkpoint_path = CHECKPOINT_DIR / f"imitate_trans_waypoints_epoch_{epoch}.pth"
    run_tag = f"{RUN_TAG_PREFIX}_epoch{epoch}_repeat{repeat_idx}"
    output_dir = VIZ_DIR / f"imitate_trans_map_{MAPTYPE}_{SELECTED_MAP}_viz_{run_tag}"
    log_path = LOG_DIR / f"{run_tag}.log"

    env = os.environ.copy()
    env.update(
        {
            "IMITATE_CHECKPOINT": str(checkpoint_path),
            "RUN_SEED": str(seed),
            "RUN_OUTPUT_TAG": run_tag,
            "SKIP_VIZ": "1",
        }
    )

    start = time.time()
    print(f"[epoch {epoch} repeat {repeat_idx} seed {seed}] starting, log -> {log_path}", flush=True)
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
        "repeat": repeat_idx,
        "seed": seed,
        "return_code": return_code,
        "wall_clock_elapsed_s": elapsed,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
    }

    if return_code != 0:
        print(f"[epoch {epoch} repeat {repeat_idx}] FAILED (exit {return_code}) after {elapsed:.1f}s", flush=True)
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
        f"[epoch {epoch} repeat {repeat_idx}] done in {elapsed:.1f}s - "
        f"task_completion={row.get('task_completion_pct', 'NA')}%, "
        f"variance_reduction={row.get('variance_reduction', 'NA')}",
        flush=True,
    )
    return row


def write_summary(rows):
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print(f"Checkpoint sweep (ImitateTrans): {len(EPOCHS)} epochs x {len(REPEAT_SEEDS)} repeats = {len(EPOCHS) * len(REPEAT_SEEDS)} runs", flush=True)
    print(f"SELECTED_MAP={SELECTED_MAP} MAPTYPE={MAPTYPE} (both left at ImitateTrans_singlemap.py's own defaults)", flush=True)
    rows = []
    for epoch in EPOCHS:
        for repeat_idx, seed in enumerate(REPEAT_SEEDS):
            row = run_one(epoch, seed, repeat_idx)
            rows.append(row)
            write_summary(rows)  # rewritten after every run so progress isn't lost
    print(f"\nSweep complete. Wrote {SUMMARY_CSV}", flush=True)
    n_failed = sum(1 for r in rows if r["status"] != "ok")
    if n_failed:
        print(f"WARNING: {n_failed} of {len(rows)} runs failed - see checkpoint_sweep_imitatetrans_logs/", flush=True)


if __name__ == "__main__":
    main()
