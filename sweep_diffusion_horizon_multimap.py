import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from hpc_sweep_common import PlannerConfig, _run_one, _write_summary

# Diffusion planner execution-horizon sweep across several maps, one run per
# (map, horizon) combination, wall-clock-budgeted (not timestep-budgeted) so
# the receding-horizon replanning cadence is compared under a fixed real-time
# flight budget. ENFORCE_MIN_STEP_TIME is forced ON (unlike the CMA-ES maxiter
# sweeps, which forced it off) so each simulated step is floored to real
# flight speed - this run is about how execution horizon changes real-time
# planning effectiveness, not raw compute time.
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
    map_id, horizon, repeat, wallclock, eta, checkpoint = payload
    variant = f"diffusion_horizon{horizon}"
    started = time.time()
    row = _run_one(
        PlannerConfig(
            planner=variant,
            script_name="Diffusionplanner_singlemap.py",
            output_prefix="diffusion",
            extra_env={
                "EXECUTION_CHUNK": str(horizon),
                "WALLCLOCK_SECONDS": str(wallclock),
                "ENFORCE_MIN_STEP_TIME": "1",
                "ETA": eta,
                "DIFFUSION_CHECKPOINT": checkpoint,
            },
        ),
        map_id,
        repeat,
    )
    row["planner"] = "diffusion"
    row["planner_variant"] = variant
    row["sweep"] = "horizon_multimap"
    row["execution_horizon"] = horizon
    row["repeat"] = repeat
    row["wallclock_fixed"] = wallclock
    row["wrapper_wall_seconds"] = time.time() - started
    return row


def main():
    map_ids = _parse_int_list("DIFF_HORIZON_MAP_IDS", DEFAULT_MAP_IDS)
    horizons = _parse_int_list("DIFF_HORIZON_VALUES", DEFAULT_HORIZON_VALUES)
    wallclock = os.environ.get("DIFF_HORIZON_WALLCLOCK", "100")
    eta = os.environ.get("ETA", "0.0")
    repeats = int(os.environ.get("REPEATS_PER_HORIZON", "1"))
    workers = int(os.environ.get("PARALLEL_WORKERS", "1"))
    checkpoint = os.environ.get(
        "DIFFUSION_CHECKPOINT", str(SCRIPT_DIR / "checkpoints" / "current_best_updated_bc.pth")
    )

    payloads = [
        (map_id, horizon, repeat, wallclock, eta, checkpoint)
        for map_id in map_ids
        for horizon in horizons
        for repeat in range(repeats)
    ]
    total = len(payloads)
    print(
        f"[diffusion horizon multimap sweep] {total} runs: {len(map_ids)} maps x {len(horizons)} horizons x "
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
                f"[diffusion horizon multimap sweep] ({completed}/{total}) map={row['map_id']} "
                f"horizon={row['execution_horizon']} repeat={row['repeat']} wall={row['process_wall_seconds']:.1f}s "
                f"elapsed_total={time.time()-sweep_start:.0f}s",
                flush=True,
            )
            _write_summary("diffusion_horizon_multimap", rows, task_id=None)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_horizon_variant, payload): payload for payload in payloads}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                completed += 1
                print(
                    f"[diffusion horizon multimap sweep] ({completed}/{total}) map={row['map_id']} "
                    f"horizon={row['execution_horizon']} repeat={row['repeat']} wall={row['process_wall_seconds']:.1f}s "
                    f"elapsed_total={time.time()-sweep_start:.0f}s",
                    flush=True,
                )
                _write_summary("diffusion_horizon_multimap", rows, task_id=None)

    summary_path = _write_summary("diffusion_horizon_multimap", rows, task_id=None)
    print(f"[diffusion horizon multimap sweep] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
