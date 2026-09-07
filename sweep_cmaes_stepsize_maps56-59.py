import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from hpc_sweep_common import PlannerConfig, _run_one, _write_summary

# Same 5-point step-size sweep as sweep_cmaes_stepsize_map55.py, extended to
# maps 56-59 (single run each, no repeats, matching the original map-55
# methodology) so the map-55 result can be checked for generalization rather
# than trusted on its own - the exact lesson beta and the earlier step-size
# comparison already taught this session.
STEPSIZE_SETS = [
    ("1.5_1.2", 1.5, 1.2),
    ("6.0_4.8", 6.0, 4.8),
    ("10_4", 10.0, 4.0),
    ("20_8", 20.0, 8.0),
    ("40_16", 40.0, 16.0),
]
MAP_IDS = [56, 57, 58, 59]


def _run_variant(payload):
    map_id, name, xy, z, maxiter, beta, timesteps = payload
    variant = f"cmaes_stepsize_{name}"
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
                "BETA": beta,
                "CMA_STEP_SIZE_XY": str(xy),
                "CMA_STEP_SIZE_Z": str(z),
            },
        ),
        map_id,
        0,
    )
    row["planner"] = "cmaes"
    row["planner_variant"] = variant
    row["sweep"] = "stepsize_maps56-59"
    row["step_xy"] = xy
    row["step_z"] = z
    row["cma_predictive_maxiter"] = maxiter
    row["beta_fixed"] = beta
    row["timesteps_fixed"] = timesteps
    row["wrapper_wall_seconds"] = time.time() - started
    return row


def main():
    maxiter = os.environ.get("STEPSIZE_MAXITER", "20")
    beta = os.environ.get("STEPSIZE_BETA", "1")
    timesteps = os.environ.get("STEPSIZE_TIMESTEPS", "200")
    workers = int(os.environ.get("PARALLEL_WORKERS", "8"))

    payloads = [
        (map_id, name, xy, z, maxiter, beta, timesteps)
        for map_id in MAP_IDS
        for name, xy, z in STEPSIZE_SETS
    ]
    total = len(payloads)
    print(f"[cmaes stepsize maps56-59 sweep] {total} runs: {len(MAP_IDS)} maps x {len(STEPSIZE_SETS)} "
          f"stepsizes, maxiter={maxiter}, beta={beta}, fixed_timesteps={timesteps}, workers={workers}",
          flush=True)

    rows = []
    completed = 0
    sweep_start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_variant, payload): payload for payload in payloads}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            completed += 1
            print(f"[cmaes stepsize maps56-59 sweep] ({completed}/{total}) map={row['map_id']} "
                  f"stepsize=({row['step_xy']},{row['step_z']}) wall={row['process_wall_seconds']:.1f}s "
                  f"elapsed_total={time.time()-sweep_start:.0f}s", flush=True)
            _write_summary("cmaes_stepsize_maps56-59", rows, task_id=None)

    summary_path = _write_summary("cmaes_stepsize_maps56-59", rows, task_id=None)
    print(f"[cmaes stepsize maps56-59 sweep] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
