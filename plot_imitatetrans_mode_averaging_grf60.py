"""
Finds a genuinely multimodal example in dataset_grf_60.pt (a beam-search
imitation/diffusion training set spanning GRF maps 0-59): a condition_id
group where several recorded expert (CMA-ES beam) continuations share the
EXACT same conditioning state (current_position, current_mean, current_var)
but diverge into meaningfully different trajectories - i.e., true multiple
modes for one input, not sampling noise.

Then runs ImitateTrans_singlemap.py's actual one-shot inference
(sample_imitation_trajectory) on that identical conditioning state, using
whatever IMITATE_CHECKPOINT is currently the live default (not overridden
here), to show its single deterministic output landing somewhere between the
recorded modes rather than committing to any one of them - the classic
mode-averaging failure of a non-multimodal regressor trained on multimodal
data.
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PNG = SCRIPT_DIR / "imitatetrans_mode_averaging_grf60.png"

DATASET_PATH = SCRIPT_DIR / "dataset_grf_60.pt"
CONDITION_ID = 5703001000  # map 57, ts 1: cleanest of the high-divergence 3-mode groups
                            # (found by scanning for max trajectory divergence, then
                            # re-ranking by mean arc length / visual cleanliness)

MODE_COLORS = ["#00E5FF", "#39FF14", "#FFA500", "#FF00E6"]
IMITATE_COLOR = "#FF3B3B"


def find_group(dataset, condition_id):
    idx = (dataset["condition_id"] == condition_id).nonzero(as_tuple=True)[0]
    return idx.tolist()


def main():
    dataset = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    idx = find_group(dataset, CONDITION_ID)
    n_modes = len(idx)
    map_id = int(dataset["map_id"][idx[0]].item())
    timestep = int(dataset["timestep"][idx[0]].item())
    print(f"Condition group: {n_modes} modes, map_id={map_id}, timestep={timestep}, rows={idx}")

    current_position = dataset["current_position"][idx[0]].numpy()  # (3,)
    current_mean = dataset["current_mean"][idx[0]].numpy()  # (51,51)
    current_var = dataset["current_var"][idx[0]].numpy()  # (51,51)
    current_util = dataset["current_util"][idx[0]].numpy()  # (51,51)
    heading_velocities = dataset["initial_heading_velocity"][idx].numpy()
    mean_heading_velocity = heading_velocities.mean(axis=0)

    mode_trajs = dataset["trajectories"][idx].numpy()  # (n_modes, 3, 41)

    print(f"Shared current_position (x,y,z) = {current_position}")
    print(f"Mode endpoints (x,y):")
    for i in range(n_modes):
        print(f"  mode {i+1}: {mode_trajs[i, 0, -1]:.1f}, {mode_trajs[i, 1, -1]:.1f}")

    import os
    os.environ.setdefault("SELECTED_MAP", str(map_id))
    os.environ.setdefault("MAPTYPE", "grf")
    from gaussianprocesstraining import initialize_gp
    import ImitateTrans_singlemap as it_sim

    print(f"Loading ImitateTrans checkpoint (current live default): {it_sim.imitate_checkpoint_path}")
    model = it_sim.load_imitate_model(checkpoint_path=it_sim.imitate_checkpoint_path)

    gp, X_test, mean0, cov0, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()

    dense_traj, control_waypoints = it_sim.sample_imitation_trajectory(
        model,
        current_position=current_position,
        current_mean=current_mean,
        current_var=current_var,
        current_heading_velocity=mean_heading_velocity,
        grid_step=step,
        bounds=(xmin, xmax, ymin, ymax, it_sim.ZMIN, it_sim.ZMAX),
    )
    print(f"ImitateTrans predicted endpoint (x,y): {dense_traj[0, -1]:.1f}, {dense_traj[1, -1]:.1f}")

    for i in range(n_modes):
        dist = np.hypot(
            dense_traj[0, -1] - mode_trajs[i, 0, -1],
            dense_traj[1, -1] - mode_trajs[i, 1, -1],
        )
        print(f"  distance to mode {i+1} endpoint: {dist:.1f}")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        current_util.reshape(len(ys), len(xs)), origin="lower", cmap="viridis",
        extent=[xmin, xmax, ymin, ymax], interpolation="bicubic", zorder=1,
    )
    ax.contour(
        X, Y, current_util.reshape(X.shape), levels=[0.0],
        colors="white", linewidths=1.2, linestyles="--", alpha=0.7, zorder=2,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("importance (utility)")

    stroke = [pe.withStroke(linewidth=4, foreground="black")]

    for i in range(n_modes):
        ax.plot(
            mode_trajs[i, 0], mode_trajs[i, 1],
            color=MODE_COLORS[i % len(MODE_COLORS)], linewidth=2.6, alpha=0.95,
            path_effects=stroke, zorder=3, label=f"recorded mode {i+1} (dataset)",
        )

    ax.plot(
        dense_traj[0], dense_traj[1],
        color=IMITATE_COLOR, linewidth=3.2, linestyle=(0, (5, 2)),
        path_effects=stroke, zorder=4, label="ImitateTrans prediction",
    )

    ax.scatter(
        [current_position[0]], [current_position[1]], s=180, color="white",
        edgecolors="black", linewidths=1.6, zorder=5, marker="o",
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"ImitateTrans mode averaging - map {map_id} (grf), timestep {timestep}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
