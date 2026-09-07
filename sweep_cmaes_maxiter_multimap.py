import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from hpc_sweep_common import PlannerConfig, _run_one, _write_summary

# Maxiter sweep, fixed CMA-ES objective, across several maps with
# REPEATS_PER_MAXITER independent seeded repeats per map/maxiter combo.
# Tracks process_wall_seconds (real elapsed time per run, ENFORCE_MIN_STEP_TIME
# off so this is genuine compute time, not padded) explicitly, so
# quality-vs-time tradeoffs across maxiter values are visible, not just
# quality alone. Runs the (map, maxiter, repeat) grid concurrently via a
# process pool; each subprocess already gets OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1
# from hpc_sweep_common._run_one.
DEFAULT_MAXITER_VALUES = (2, 8, 20, 45)
DEFAULT_MAP_IDS = (91, 92, 93, 94, 95)


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


def _run_maxiter_variant(payload):
    map_id, maxiter, repeat, beta, timesteps = payload
    variant = f"cmaes_maxiter{maxiter}"
    started = time.time()
    row = _run_one(
        PlannerConfig(
            planner=variant,
            script_name="CMAES_classic_singlemap.py",
            output_prefix="classic",
            extra_env={
                "TIMEALLOTED": timesteps,
                "WALLCLOCK_SECONDS": "0",
                "ENFORCE_MIN_STEP_TIME": "0",
                "CMA_PREDICTIVE_MAXITER": str(maxiter),
                "BETA": str(beta),
            },
        ),
        map_id,
        repeat,
    )
    row["planner"] = "cmaes"
    row["planner_variant"] = variant
    row["sweep"] = "maxiter_multimap"
    row["maxiter"] = maxiter
    row["repeat"] = repeat
    row["beta_fixed"] = beta
    row["timesteps_fixed"] = timesteps
    row["wrapper_wall_seconds"] = time.time() - started
    return row


def main():
    map_ids = _parse_int_list("MAXITER_SWEEP_MAP_IDS", DEFAULT_MAP_IDS)
    beta = os.environ.get("MAXITER_SWEEP_BETA", "1")
    timesteps = os.environ.get("MAXITER_SWEEP_TIMESTEPS", "250")
    maxiters = _parse_int_list("CMA_MAXITER_VALUES", DEFAULT_MAXITER_VALUES)
    repeats = int(os.environ.get("REPEATS_PER_MAXITER", "1"))
    workers = int(os.environ.get("PARALLEL_WORKERS", "1"))

    payloads = [
        (map_id, maxiter, repeat, beta, timesteps)
        for map_id in map_ids
        for maxiter in maxiters
        for repeat in range(repeats)
    ]
    total = len(payloads)
    print(
        f"[cmaes maxiter multimap sweep] {total} runs: {len(map_ids)} maps x {len(maxiters)} maxiters x "
        f"{repeats} repeats, beta={beta}, fixed_timesteps={timesteps}, workers={workers}",
        flush=True,
    )

    rows = []
    completed = 0
    sweep_start = time.time()
    if workers <= 1:
        for payload in payloads:
            row = _run_maxiter_variant(payload)
            rows.append(row)
            completed += 1
            print(
                f"[cmaes maxiter multimap sweep] ({completed}/{total}) map={row['map_id']} maxiter={row['maxiter']} "
                f"repeat={row['repeat']} wall={row['process_wall_seconds']:.1f}s "
                f"elapsed_total={time.time()-sweep_start:.0f}s",
                flush=True,
            )
            _write_summary("cmaes_maxiter_multimap", rows, task_id=None)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_maxiter_variant, payload): payload for payload in payloads}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                completed += 1
                print(
                    f"[cmaes maxiter multimap sweep] ({completed}/{total}) map={row['map_id']} maxiter={row['maxiter']} "
                    f"repeat={row['repeat']} wall={row['process_wall_seconds']:.1f}s "
                    f"elapsed_total={time.time()-sweep_start:.0f}s",
                    flush=True,
                )
                _write_summary("cmaes_maxiter_multimap", rows, task_id=None)

    summary_path = _write_summary("cmaes_maxiter_multimap", rows, task_id=None)
    print(f"[cmaes maxiter multimap sweep] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
