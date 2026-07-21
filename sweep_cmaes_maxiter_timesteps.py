import os

from hpc_sweep_common import PlannerConfig, _array_task_id, _run_one, _write_summary


# Popovic-style CMA-ES budget grid: effective evaluations ~= maxiter * popsize
# (popsize=12 by default), since maxiter binds before maxfevals at these sizes.
DEFAULT_MAXITER_VALUES = (2, 4, 8, 16, 32, 45)
DEFAULT_MAPS = (91, 92)


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
        raise SystemExit(f"{env_name} did not contain any integers")
    return values


def _grid():
    maps = _parse_int_list("MAXITER_SWEEP_MAP_IDS", DEFAULT_MAPS)
    maxiters = _parse_int_list("CMA_MAXITER_VALUES", DEFAULT_MAXITER_VALUES)
    repeats = int(os.environ.get("REPEATS_PER_MAXITER", "1"))
    return [
        (map_id, maxiter, repeat)
        for map_id in maps
        for maxiter in maxiters
        for repeat in range(repeats)
    ]


def _task_subset(items):
    task_id = _array_task_id()
    if task_id is None:
        return items
    index = int(task_id)
    if index < 0 or index >= len(items):
        raise SystemExit(f"Task index {index} is outside 0..{len(items) - 1}")
    return [items[index]]


def main():
    task_id = _array_task_id()
    rows = []
    # Fixed simulation-timestep budget, not wall-clock: WALLCLOCK_SECONDS=0 makes
    # CMAES_classic_singlemap.py's flight loop unconstrained by real time, so it
    # always runs exactly TIMEALLOTED timesteps regardless of how slow a given
    # maxiter's replanning calls are.
    timesteps = os.environ.get("MAXITER_SWEEP_TIMESTEPS", "200")

    for map_id, maxiter, repeat in _task_subset(_grid()):
        variant = f"cmaes_maxiter{maxiter}"
        print(
            f"[cmaes maxiter sweep] map={map_id} maxiter={maxiter} repeat={repeat} "
            f"fixed_timesteps={timesteps} (wall-clock unconstrained)",
            flush=True,
        )
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
                },
            ),
            map_id,
            repeat,
        )
        row["planner"] = "cmaes"
        row["planner_variant"] = variant
        row["sweep"] = "cma_predictive_maxiter"
        row["cma_predictive_maxiter"] = maxiter
        row["timesteps_fixed"] = timesteps
        row["repeat"] = repeat
        rows.append(row)
        _write_summary("cmaes_maxiter_timesteps", rows, task_id=task_id)
        if row["return_code"] != 0 and os.environ.get("STOP_ON_FAILURE", "0") == "1":
            raise SystemExit(row["return_code"])

    summary_path = _write_summary("cmaes_maxiter_timesteps", rows, task_id=task_id)
    print(f"[cmaes maxiter sweep] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
