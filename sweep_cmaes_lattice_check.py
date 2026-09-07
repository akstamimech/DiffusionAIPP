import os

from hpc_sweep_common import PlannerConfig, _run_one, _write_summary

# Checks whether CMA-ES refinement actually beats the raw grid search once the
# low-altitude lattice tier is densified (build_pyramid_lattice_3d, 5x5->9x9).
# Three conditions, one map, fixed timesteps, wall-clock unconstrained (same
# methodology as sweep_cmaes_beta.py):
#   lattice_only  - CMAES_REFINE=0, i.e. Popovic et al. (2020)'s "Lattice" row
#   cma_maxiter2  - CMA-ES refinement, CMA_PREDICTIVE_MAXITER=2
#   cma_maxiter20 - CMA-ES refinement, CMA_PREDICTIVE_MAXITER=20
CONDITIONS = (
    ("lattice_only", {"CMAES_REFINE": "0"}),
    ("cma_maxiter2", {"CMAES_REFINE": "1", "CMA_PREDICTIVE_MAXITER": "2"}),
    ("cma_maxiter20", {"CMAES_REFINE": "1", "CMA_PREDICTIVE_MAXITER": "20"}),
)


def main():
    rows = []
    map_id = int(os.environ.get("LATTICE_CHECK_MAP_ID", "91"))
    timesteps = os.environ.get("LATTICE_CHECK_TIMESTEPS", "250")

    for name, extra in CONDITIONS:
        variant = f"cmaes_lattice_{name}"
        print(
            f"[cmaes lattice check] map={map_id} condition={name} "
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
                    **extra,
                },
            ),
            map_id,
            0,
        )
        row["planner"] = "cmaes"
        row["planner_variant"] = variant
        row["sweep"] = "lattice_check"
        row["condition"] = name
        row["timesteps_fixed"] = timesteps
        rows.append(row)
        _write_summary("cmaes_lattice_check", rows, task_id=None)
        if row["return_code"] != 0 and os.environ.get("STOP_ON_FAILURE", "0") == "1":
            raise SystemExit(row["return_code"])

    summary_path = _write_summary("cmaes_lattice_check", rows, task_id=None)
    print(f"[cmaes lattice check] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
