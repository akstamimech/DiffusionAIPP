from hpc_sweep_common import PlannerConfig, run_sweep


if __name__ == "__main__":
    run_sweep(
        PlannerConfig(
            planner="lawnmower",
            script_name="lawnmower_singlemap.py",
            output_prefix="lawnmower",
        )
    )

