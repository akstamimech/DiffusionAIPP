from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import MLPdiffusion as diffusion


CHECKPOINT_PATH = (
    r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\Diffusion\checkpoints\mlp_spiral_diffusion_traj_final.pth"
)
DATASET_PATH = (
    r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\Diffusion\spiral_trajectory_dataset.pt"
)


@torch.no_grad()
def sample_trajectory(model):
    model.eval()
    traj = torch.randn(
        (1, diffusion.NUM_COORDS, diffusion.TRAJ_SIZE),
        device=diffusion.device,
    )

    for i in range(diffusion.T - 1, -1, -1):
        t = torch.full((1,), i, dtype=torch.long, device=diffusion.device)
        traj = diffusion.sample_timestep(traj, t)

    return traj[0].cpu()


def load_ground_truth():
    data_dict = torch.load(DATASET_PATH, map_location="cpu")
    return data_dict["trajectories"][0].float() / diffusion.SCALE_FACTOR


def plot_comparison(sampled_traj, ground_truth):
    sampled_xy = (sampled_traj * diffusion.SCALE_FACTOR).clamp(0.0, diffusion.SCALE_FACTOR)
    truth_xy = (ground_truth * diffusion.SCALE_FACTOR).clamp(0.0, diffusion.SCALE_FACTOR)

    plt.figure(figsize=(7, 7))
    plt.plot(truth_xy[0], truth_xy[1], marker="o", linewidth=2, label="Ground truth")
    plt.plot(sampled_xy[0], sampled_xy[1], marker="x", linewidth=2, label="Sampled")
    plt.scatter(truth_xy[0, 0], truth_xy[1, 0], s=80, label="Start")
    plt.title("MLP Diffusion Trajectory vs Ground Truth")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, diffusion.SCALE_FACTOR)
    plt.ylim(0, diffusion.SCALE_FACTOR)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    model = diffusion.NoisePredictor().to(diffusion.device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=diffusion.device)
    model.load_state_dict(state_dict)

    diffusion.model = model

    ground_truth = load_ground_truth()
    sampled_traj = sample_trajectory(model)
    plot_comparison(sampled_traj, ground_truth)


if __name__ == "__main__":
    data_dict = torch.load(DATASET_PATH, map_location="cpu")
    print(data_dict["trajectories"].shape)
    quit()
    main()
