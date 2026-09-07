from hpc_sweep_common import PlannerConfig, run_sweep


if __name__ == "__main__":
    run_sweep(
        PlannerConfig(
            planner="cmaes",
            script_name="CMAES_classic_singlemap.py",
            output_prefix="classic",
        )
    )

