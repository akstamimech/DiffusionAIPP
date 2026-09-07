import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from hpc_sweep_common import PlannerConfig, _run_one, _write_summary

# Same beta sweep as sweep_cmaes_beta.py (fixed CMA-ES objective), repeated
# across several maps (with REPEATS_PER_BETA independent seeded repeats per
# map/beta combo) and averaged, to see whether any single beta value wins
# consistently or whether - as found earlier - the winner is map-dependent
# noise. Runs the (map, beta, repeat) grid concurrently via a process pool;
# each subprocess already gets OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 from
# hpc_sweep_common._run_one, so PARALLEL_WORKERS concurrent single-threaded
# runs should not oversubscribe cores the way naive parallelism would.
DEFAULT_BETA_VALUES = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
DEFAULT_MAP_IDS = (91, 92, 93, 94, 95)


def _parse_float_list(env_name, default):
    raw = os.environ.get(env_name)
    if not raw:
        return list(default)
    values = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise SystemExit(f"{env_name} did not contain any floats")
    return values


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


def _run_beta_variant(payload):
    map_id, beta, repeat, maxiter, timesteps = payload
    variant = f"cmaes_beta{beta}"
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
                "CMA_PREDICTIVE_MAXITER": maxiter,
                "BETA": str(beta),
            },
        ),
        map_id,
        repeat,
    )
    row["planner"] = "cmaes"
    row["planner_variant"] = variant
    row["sweep"] = "beta_multimap"
    row["beta"] = beta
    row["repeat"] = repeat
    row["cma_predictive_maxiter"] = maxiter
    row["timesteps_fixed"] = timesteps
    row["wrapper_wall_seconds"] = time.time() - started
    return row


def main():
    map_ids = _parse_int_list("BETA_SWEEP_MAP_IDS", DEFAULT_MAP_IDS)
    maxiter = os.environ.get("BETA_SWEEP_MAXITER", "20")
    timesteps = os.environ.get("BETA_SWEEP_TIMESTEPS", "250")
    betas = _parse_float_list("CMA_BETA_VALUES", DEFAULT_BETA_VALUES)
    repeats = int(os.environ.get("REPEATS_PER_BETA", "1"))
    workers = int(os.environ.get("PARALLEL_WORKERS", "1"))

    payloads = [
        (map_id, beta, repeat, maxiter, timesteps)
        for map_id in map_ids
        for beta in betas
        for repeat in range(repeats)
    ]
    total = len(payloads)
    print(
        f"[cmaes beta multimap sweep] {total} runs: {len(map_ids)} maps x {len(betas)} betas x "
        f"{repeats} repeats, maxiter={maxiter}, fixed_timesteps={timesteps}, workers={workers}",
        flush=True,
    )

    rows = []
    completed = 0
    sweep_start = time.time()
    if workers <= 1:
        for payload in payloads:
            row = _run_beta_variant(payload)
            rows.append(row)
            completed += 1
            print(
                f"[cmaes beta multimap sweep] ({completed}/{total}) map={row['map_id']} beta={row['beta']} "
                f"repeat={row['repeat']} wall={row['process_wall_seconds']:.1f}s "
                f"elapsed_total={time.time()-sweep_start:.0f}s",
                flush=True,
            )
            _write_summary("cmaes_beta_multimap", rows, task_id=None)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_beta_variant, payload): payload for payload in payloads}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                completed += 1
                print(
                    f"[cmaes beta multimap sweep] ({completed}/{total}) map={row['map_id']} beta={row['beta']} "
                    f"repeat={row['repeat']} wall={row['process_wall_seconds']:.1f}s "
                    f"elapsed_total={time.time()-sweep_start:.0f}s",
                    flush=True,
                )
                _write_summary("cmaes_beta_multimap", rows, task_id=None)

    summary_path = _write_summary("cmaes_beta_multimap", rows, task_id=None)
    print(f"[cmaes beta multimap sweep] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
