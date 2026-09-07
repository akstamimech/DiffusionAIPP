import os

from hpc_sweep_common import PlannerConfig, _run_one, _write_summary

# Single-map (55), single-run-per-config sigma sweep, spanning the current
# sub-pixel default (1.5/1.2, smaller than the step=2.0m grid resolution)
# through and past the Popovic-scaling-derived target (10/4), to see whether
# an effect is flat, threshold-like, or monotonic - not just two isolated
# points.
STEPSIZE_SETS = [
    ("1.5_1.2", 1.5, 1.2),
    ("6.0_4.8", 6.0, 4.8),
    ("10_4", 10.0, 4.0),
    ("20_8", 20.0, 8.0),
    ("40_16", 40.0, 16.0),
]
MAP_ID = 55


def main():
    maxiter = os.environ.get("STEPSIZE_MAP55_MAXITER", "20")
    beta = os.environ.get("STEPSIZE_MAP55_BETA", "1")
    timesteps = os.environ.get("STEPSIZE_MAP55_TIMESTEPS", "200")

    rows = []
    for name, xy, z in STEPSIZE_SETS:
        variant = f"cmaes_stepsize_{name}"
        print(
            f"[cmaes stepsize map55 sweep] xy={xy} z={z} maxiter={maxiter} beta={beta} "
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
                    "BETA": beta,
                    "CMA_STEP_SIZE_XY": str(xy),
                    "CMA_STEP_SIZE_Z": str(z),
                },
            ),
            MAP_ID,
            0,
        )
        row["planner"] = "cmaes"
        row["planner_variant"] = variant
        row["sweep"] = "stepsize_map55"
        row["step_xy"] = xy
        row["step_z"] = z
        row["cma_predictive_maxiter"] = maxiter
        row["beta_fixed"] = beta
        row["timesteps_fixed"] = timesteps
        rows.append(row)
        _write_summary("cmaes_stepsize_map55", rows, task_id=None)
        if row["return_code"] != 0 and os.environ.get("STOP_ON_FAILURE", "0") == "1":
            raise SystemExit(row["return_code"])

    summary_path = _write_summary("cmaes_stepsize_map55", rows, task_id=None)
    print(f"[cmaes stepsize map55 sweep] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
