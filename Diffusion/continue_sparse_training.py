import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import SparseDiffusion as diffusion


SCRIPT_DIR = Path(__file__).resolve().parent


def make_dataloader(dataset, batch_size):
    requested_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if requested_workers <= 0:
        requested_workers = 2 if torch.cuda.is_available() else 0
    num_workers = min(requested_workers, 4) if torch.cuda.is_available() else 0

    kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True

    return DataLoader(dataset, **kwargs)


def resolve_checkpoint_path(path_text):
    path = Path(path_text)
    if path.is_file():
        return path

    checkpoint_path = diffusion.CHECKPOINT_DIR / path_text
    if checkpoint_path.is_file():
        return checkpoint_path

    script_path = SCRIPT_DIR / path_text
    if script_path.is_file():
        return script_path

    raise FileNotFoundError(f"Could not find checkpoint: {path_text}")


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=diffusion.device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        previous_loss = checkpoint.get("loss")
    else:
        model.load_state_dict(checkpoint)
        start_epoch = 0
        previous_loss = None
        print("Loaded model weights only; optimizer state and epoch were not present.")

    return start_epoch, previous_loss


def train_from_checkpoint(
    checkpoint_path,
    target_epochs,
    batch_size,
    lr,
    save_every,
):
    model = diffusion.NoisePredictor().to(diffusion.device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    start_epoch, previous_loss = load_checkpoint(model, optimizer, checkpoint_path)
    diffusion.model = model

    if target_epochs <= start_epoch:
        raise ValueError(
            f"target_epochs={target_epochs} must be greater than checkpoint epoch {start_epoch}"
        )

    dataset = diffusion.TrajectoryDataset(
        diffusion.trajectories,
        diffusion.weights,
        diffusion.meanvarmarkermaps,
        diffusion.conditions,
    )
    dataloader = make_dataloader(dataset, batch_size)

    print(f"Loaded checkpoint: {checkpoint_path}", flush=True)
    print(f"Checkpoint epoch: {start_epoch}", flush=True)
    if previous_loss is not None:
        print(f"Checkpoint loss: {previous_loss}", flush=True)
    print(f"Continuing to epoch: {target_epochs}", flush=True)
    print(f"Using device: {diffusion.device}", flush=True)

    model.train()
    loss_vals = []

    for epoch in range(start_epoch, target_epochs):
        print(f"Epoch {epoch + 1}/{target_epochs}", flush=True)
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            traj, current_position, meanvarmarker_map, batch_weights = batch
            model_device = next(model.parameters()).device
            traj = traj.to(model_device, non_blocking=True)
            current_position = current_position.to(model_device, non_blocking=True)
            meanvarmarker_map = meanvarmarker_map.to(model_device, non_blocking=True)
            batch_weights = batch_weights.to(model_device, non_blocking=True)

            t = torch.randint(0, diffusion.T, (traj.shape[0],), device=traj.device).long()
            loss = diffusion.get_loss(
                model,
                traj,
                t,
                meanvarmarker_map,
                current_position,
                weights=batch_weights,
            )
            loss_vals.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}", flush=True)

        if (epoch + 1) % save_every == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }
            output_path = diffusion.CHECKPOINT_DIR / f"sparse_waypoints_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, output_path)
            print(f"Checkpoint saved: {output_path}", flush=True)

        print(f"Epoch {epoch + 1} complete", flush=True)

    if loss_vals:
        loss_plot_path = diffusion.PLOT_DIR / f"sparse_continue_loss_epoch_{target_epochs}.png"
        window = min(200, len(loss_vals))
        plt.figure()
        if window > 1:
            kernel = np.ones(window, dtype=np.float32) / window
            moving_avg = np.convolve(np.asarray(loss_vals, dtype=np.float32), kernel, mode="valid")
            plt.plot(loss_vals, alpha=0.18, label="Raw continued training loss")
            plt.plot(range(window - 1, len(loss_vals)), moving_avg, label=f"{window}-step moving average")
        else:
            plt.plot(loss_vals, label="Continued training loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Sparse diffusion continued training loss")
        plt.legend()
        plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved continued loss plot to {loss_plot_path}", flush=True)

    final_sample_path = diffusion.PLOT_DIR / f"sparse_continue_sample_epoch_{target_epochs}.png"
    diffusion.sample_plot_traj(final_sample_path)
    print(f"Saved continued sample plot to {final_sample_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Continue SparseDiffusion training from a checkpoint.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path. Can be absolute, relative to this script, or a filename in checkpoints/.",
    )
    parser.add_argument(
        "--target-epochs",
        type=int,
        default=diffusion.EPOCHS,
        help="Final epoch number to train to, not additional epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=diffusion.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_from_checkpoint(
        checkpoint_path=resolve_checkpoint_path(args.checkpoint),
        target_epochs=args.target_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_every=args.save_every,
    )
