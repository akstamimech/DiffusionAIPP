from hpc_sweep_common import PlannerConfig, run_sweep


if __name__ == "__main__":
    run_sweep(
        PlannerConfig(
            planner="realgreedy",
            script_name="realgreedy_singlemap.py",
            output_prefix="greedy",
            extra_env={
                "PLANNING_HORIZON": "1",
                "EXECUTION_CHUNK": "5",
            },
        )
    )

