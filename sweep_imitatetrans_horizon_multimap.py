import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from hpc_sweep_common import PlannerConfig, _run_one, _write_summary

# ImitateTrans (direct waypoint-regression, non-diffusion) execution-horizon
# sweep across several maps, one run per (map, horizon) combination,
# wall-clock-budgeted, ENFORCE_MIN_STEP_TIME forced ON - same experiment as
# sweep_diffusion_horizon_multimap.py, different architecture (single forward
# pass instead of iterative denoising).
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HORIZON_VALUES = (10, 20, 30, 40)
DEFAULT_MAP_IDS = tuple(range(51, 60))  # 51..59


def _parse_int_list(env_name, default):
    raw = os.environ.get(env_name)
    if not raw:
        return list(default)
    values = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise SystemExit(f"{env_name} did not contain any ints")
    return values


def _run_horizon_variant(payload):
    map_id, horizon, repeat, wallclock, checkpoint = payload
    variant = f"imitate_horizon{horizon}"
    started = time.time()
    row = _run_one(
        PlannerConfig(
            planner=variant,
            script_name="ImitateTrans_singlemap.py",
            output_prefix="imitate_trans",
            extra_env={
                "EXECUTION_CHUNK": str(horizon),
                "WALLCLOCK_SECONDS": str(wallclock),
                "ENFORCE_MIN_STEP_TIME": "1",
                "IMITATE_CHECKPOINT": checkpoint,
            },
        ),
        map_id,
        repeat,
    )
    row["planner"] = "imitate"
    row["planner_variant"] = variant
    row["sweep"] = "horizon_multimap"
    row["execution_horizon"] = horizon
    row["repeat"] = repeat
    row["wallclock_fixed"] = wallclock
    row["wrapper_wall_seconds"] = time.time() - started
    return row


def main():
    map_ids = _parse_int_list("IMITATE_HORIZON_MAP_IDS", DEFAULT_MAP_IDS)
    horizons = _parse_int_list("IMITATE_HORIZON_VALUES", DEFAULT_HORIZON_VALUES)
    wallclock = os.environ.get("IMITATE_HORIZON_WALLCLOCK", "100")
    repeats = int(os.environ.get("REPEATS_PER_HORIZON", "1"))
    workers = int(os.environ.get("PARALLEL_WORKERS", "1"))
    checkpoint = os.environ.get(
        "IMITATE_CHECKPOINT", str(SCRIPT_DIR / "checkpoints" / "Imitate_best.pth")
    )

    payloads = [
        (map_id, horizon, repeat, wallclock, checkpoint)
        for map_id in map_ids
        for horizon in horizons
        for repeat in range(repeats)
    ]
    total = len(payloads)
    print(
        f"[imitate horizon multimap sweep] {total} runs: {len(map_ids)} maps x {len(horizons)} horizons x "
        f"{repeats} repeats, wallclock={wallclock}s, enforce_min_step_time=1, checkpoint={checkpoint}, workers={workers}",
        flush=True,
    )

    rows = []
    completed = 0
    sweep_start = time.time()
    if workers <= 1:
        for payload in payloads:
            row = _run_horizon_variant(payload)
            rows.append(row)
            completed += 1
            print(
                f"[imitate horizon multimap sweep] ({completed}/{total}) map={row['map_id']} "
                f"horizon={row['execution_horizon']} repeat={row['repeat']} wall={row['process_wall_seconds']:.1f}s "
                f"elapsed_total={time.time()-sweep_start:.0f}s",
                flush=True,
            )
            _write_summary("imitate_horizon_multimap", rows, task_id=None)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_horizon_variant, payload): payload for payload in payloads}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                completed += 1
                print(
                    f"[imitate horizon multimap sweep] ({completed}/{total}) map={row['map_id']} "
                    f"horizon={row['execution_horizon']} repeat={row['repeat']} wall={row['process_wall_seconds']:.1f}s "
                    f"elapsed_total={time.time()-sweep_start:.0f}s",
                    flush=True,
                )
                _write_summary("imitate_horizon_multimap", rows, task_id=None)

    summary_path = _write_summary("imitate_horizon_multimap", rows, task_id=None)
    print(f"[imitate horizon multimap sweep] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
