import os

from hpc_sweep_common import PlannerConfig, _run_one, _write_summary


DEFAULT_BETA_VALUES = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)


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


def main():
    rows = []
    map_id = int(os.environ.get("BETA_SWEEP_MAP_ID", "91"))
    maxiter = os.environ.get("BETA_SWEEP_MAXITER", "2")
    timesteps = os.environ.get("BETA_SWEEP_TIMESTEPS", "250")
    betas = _parse_float_list("CMA_BETA_VALUES", DEFAULT_BETA_VALUES)

    for beta in betas:
        variant = f"cmaes_beta{beta}"
        print(
            f"[cmaes beta sweep] map={map_id} beta={beta} maxiter={maxiter} "
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
                    "CMA_PREDICTIVE_MAXITER": maxiter,
                    "BETA": str(beta),
                },
            ),
            map_id,
            0,
        )
        row["planner"] = "cmaes"
        row["planner_variant"] = variant
        row["sweep"] = "beta"
        row["beta"] = beta
        row["cma_predictive_maxiter"] = maxiter
        row["timesteps_fixed"] = timesteps
        rows.append(row)
        _write_summary("cmaes_beta", rows, task_id=None)
        if row["return_code"] != 0 and os.environ.get("STOP_ON_FAILURE", "0") == "1":
            raise SystemExit(row["return_code"])

    summary_path = _write_summary("cmaes_beta", rows, task_id=None)
    print(f"[cmaes beta sweep] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
