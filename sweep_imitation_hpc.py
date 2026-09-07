from hpc_sweep_common import PlannerConfig, run_sweep


if __name__ == "__main__":
    run_sweep(
        PlannerConfig(
            planner="imitation",
            script_name="ImitateTrans_singlemap.py",
            output_prefix="imitate_trans",
        )
    )

