import os

from hpc_sweep_common import PlannerConfig, run_sweep


if __name__ == "__main__":
    run_sweep(
        PlannerConfig(
            planner="diffusion",
            script_name="Diffusionplanner_singlemap.py",
            output_prefix="diffusion",
            extra_env={
                "ETA": os.environ.get("ETA", "0.0"),
            },
        )
    )
