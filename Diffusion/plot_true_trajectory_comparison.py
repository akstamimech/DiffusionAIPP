from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import MLPdiffusion as diffusion


DATASET_PATH = SCRIPT_DIR / "CMAES_classic_betasweep_dataset.pt"
CHECKPOINT_CANDIDATES = [
    SCRIPT_DIR / "checkpoints" / "mlp_control_waypoints_final.pth",
    *sorted(
        (SCRIPT_DIR / "checkpoints").glob("mlp_control_waypoints_epoch_*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ),
]
NUM_PLOTS = 6
GRID_SHAPE = (2, 3)


def resolve_checkpoint():
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find an MLP control-waypoint checkpoint.")


def load_control_waypoints():
    data_dict = torch.load(DATASET_PATH, map_location="cpu")
    control_waypoints = data_dict["control_waypoints"].float().permute(0, 2, 1)
    conditions = data_dict["current_position"].float()

    means = data_dict["current_mean"].float()
    vars = data_dict["current_var"].float()
    log_vars = torch.log1p(vars)

    means = (means - diffusion.mean_center.cpu()) / diffusion.mean_scale.cpu()
    log_vars = (log_vars - diffusion.log_var_center.cpu()) / diffusion.log_var_scale.cpu()
    meanvarmaps = torch.stack([means, log_vars], dim=1)

    return control_waypoints, conditions, meanvarmaps


def choose_indices(num_samples, num_plots):
    if num_samples <= num_plots:
        return list(range(num_samples))

    raw = torch.linspace(0, num_samples - 1, steps=num_plots)
    return [int(i.item()) for i in raw.round().long()]


def load_model(checkpoint_path):
    model = diffusion.NoisePredictor().to(diffusion.device)
    payload = torch.load(checkpoint_path, map_location=diffusion.device)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    diffusion.model = model
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


@torch.no_grad()
def sample_control_waypoints(meanvar_map, current_position):
    traj = torch.randn((1, diffusion.NUM_COORDS, diffusion.TRAJ_SIZE), device=diffusion.device)
    meanvar_map = meanvar_map.to(diffusion.device)
    current_position = current_position.to(diffusion.device)
    return diffusion.ddim_sample(traj, meanvar_map, current_position)[0].cpu()


def plot_control_waypoint_grid(control_waypoints, conditions, meanvarmaps, indices, checkpoint_path):
    fig, axes = plt.subplots(*GRID_SHAPE, figsize=(12, 8))
    axes = axes.flatten()

    for ax, idx in zip(axes, tqdm(indices, desc="Sampling")):
        truth = control_waypoints[idx]
        sampled = sample_control_waypoints(
            meanvarmaps[idx:idx + 1],
            conditions[idx:idx + 1] / diffusion.SCALE_FACTOR,
        )

        sampled_world = diffusion.denormalize_control_waypoints(sampled)

        ax.plot(truth[0], truth[1], marker="o", linewidth=2, label="Ground truth control")
        ax.plot(sampled_world[0], sampled_world[1], marker="x", linewidth=2, label="Generated control")
        ax.scatter(truth[0, 0], truth[1, 0], s=70, label="GT first")
        ax.scatter(truth[0, -1], truth[1, -1], s=70, marker="s", label="GT last")
        pos = conditions[idx]
        ax.set_title(f"Sample {idx} @ ({pos[0]:.0f}, {pos[1]:.0f})")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    for ax in axes[len(indices):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(
        f"Ground truth vs DDIM-generated control waypoints from {checkpoint_path.name}",
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()


def main():
    checkpoint_path = resolve_checkpoint()
    control_waypoints, conditions, meanvarmaps = load_control_waypoints()
    indices = choose_indices(len(control_waypoints), NUM_PLOTS)
    load_model(checkpoint_path)

    print(f"Using dataset: {DATASET_PATH}")
    print(f"Control waypoint tensor shape: {tuple(control_waypoints.shape)}")
    print(f"Condition tensor shape: {tuple(conditions.shape)}")
    print(f"Mean/var conditioning tensor shape: {tuple(meanvarmaps.shape)}")
    print(f"Showing indices: {indices}")

    plot_control_waypoint_grid(control_waypoints, conditions, meanvarmaps, indices, checkpoint_path)


if __name__ == "__main__":
    main()
