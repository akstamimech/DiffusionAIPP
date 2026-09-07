from hpc_sweep_common import PlannerConfig, run_sweep


if __name__ == "__main__":
    run_sweep(
        PlannerConfig(
            planner="greedy",
            script_name="greedygradient_singlemap.py",
            output_prefix="greedygradient",
        )
    )

