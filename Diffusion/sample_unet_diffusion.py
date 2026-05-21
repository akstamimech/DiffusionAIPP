import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import UNETdiffusion as diffusion


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(
        checkpoint_dir.glob("unet_diffusion_epoch_*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No unet_diffusion_epoch_*.pth checkpoints found in {checkpoint_dir}"
        )
    return checkpoints[0]


@torch.no_grad()
def sample_trajectory(model, device):
    model.eval()
    traj = torch.randn((1, diffusion.NUM_COORDS, diffusion.TRAJ_SIZE), device=device)
    return diffusion.ddim_sample(traj)


def load_model(checkpoint_path, device):
    model = diffusion.UNet().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def plot_sample(sample, output_path=None, truth_index=0):
    start_position = diffusion.start_positions[truth_index].cpu()
    sample_world = diffusion.reconstruct_waypoints(sample[0].cpu(), start_position)
    truth_world = diffusion.reconstruct_waypoints(
        diffusion.trajectories[truth_index].cpu(),
        start_position,
    )

    plt.figure(figsize=(6, 6))
    plt.axis("equal")
    plt.grid(True)
    plt.plot(sample_world[0], sample_world[1], marker="o", label="Generated")
    plt.plot(truth_world[0], truth_world[1], marker="x", label="Ground truth")
    plt.legend()
    plt.title("UNet Diffusion Sample")

    if output_path is None:
        plt.show()
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved sample plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--truth-index", type=int, default=0)
    args = parser.parse_args()

    device = diffusion.device
    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    print(f"Loading checkpoint: {checkpoint_path}")

    model = load_model(checkpoint_path, device)
    diffusion.model = model
    sample = sample_trajectory(model, device)
    plot_sample(sample, output_path=args.output, truth_index=args.truth_index)


if __name__ == "__main__":
    main()
