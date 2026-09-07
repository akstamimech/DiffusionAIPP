import os

from hpc_sweep_common import PlannerConfig, _run_one, _write_summary

# Compares the current CMA_STEP_SIZE_XY/Z default (1.5/1.2) against the
# Popovic-et-al.-scaling-derived target (~10/~4, from their best (3,3,4)
# step size in a 30x30m domain, scaled to this codebase's ~92m usable XY
# extent / 30m Z range), across several maps, under the now-fixed CMA-ES
# objective (cost normalization + aligned boundary penalty + densified
# lattice) - this exact comparison was never run under the fixed objective;
# the 1.5/1.2 default was only ever validated against the old, broken one.
CONDITIONS = (
    ("stepsize_default", {"CMA_STEP_SIZE_XY": "1.5", "CMA_STEP_SIZE_Z": "1.2"}),
    ("stepsize_popovic_scaled", {"CMA_STEP_SIZE_XY": "10", "CMA_STEP_SIZE_Z": "4"}),
)
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


def main():
    rows = []
    map_ids = _parse_int_list("STEPSIZE_SWEEP_MAP_IDS", DEFAULT_MAP_IDS)
    maxiter = os.environ.get("STEPSIZE_SWEEP_MAXITER", "20")
    timesteps = os.environ.get("STEPSIZE_SWEEP_TIMESTEPS", "250")

    for map_id in map_ids:
        for name, extra in CONDITIONS:
            variant = f"cmaes_stepsize_{name}"
            print(
                f"[cmaes stepsize sweep] map={map_id} condition={name} maxiter={maxiter} "
                f"fixed_timesteps={timesteps} (wall-clock unconstrained) env={extra}",
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
                        "CMA_PREDICTIVE_MAXITER": maxiter,
                        **extra,
                    },
                ),
                map_id,
                0,
            )
            row["planner"] = "cmaes"
            row["planner_variant"] = variant
            row["sweep"] = "stepsize_multimap"
            row["condition"] = name
            row["cma_predictive_maxiter"] = maxiter
            row["timesteps_fixed"] = timesteps
            rows.append(row)
            _write_summary("cmaes_stepsize_multimap", rows, task_id=None)
            if row["return_code"] != 0 and os.environ.get("STOP_ON_FAILURE", "0") == "1":
                raise SystemExit(row["return_code"])

    summary_path = _write_summary("cmaes_stepsize_multimap", rows, task_id=None)
    print(f"[cmaes stepsize sweep] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
