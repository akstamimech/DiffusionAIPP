from pathlib import Path

import matplotlib.pyplot as plt
import torch


DATASET_CANDIDATES = [
    Path(r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\Diffusion\trajectory_dataset.pt"),
]
NUM_PLOTS = 6
GRID_SHAPE = (2, 3)


def resolve_dataset_path():
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find trajectory_dataset.pt in the expected locations.")


def load_dataset(dataset_path):
    data_dict = torch.load(dataset_path, map_location="cpu")
    return (
        data_dict["trajectories"].float(),
        data_dict["current_position"].float(),
        data_dict["current_mean"].float(),
    )


def choose_indices(num_samples, num_plots):
    if num_samples <= num_plots:
        return list(range(num_samples))

    raw = torch.linspace(0, num_samples - 1, steps=num_plots)
    return [int(i.item()) for i in raw.round().long()]


def plot_grid(trajectories, conditions, means, indices, dataset_path):
    fig, axes = plt.subplots(*GRID_SHAPE, figsize=(13, 8))
    axes = axes.flatten()
    image_artist = None

    for ax, idx in zip(axes, indices):
        traj = trajectories[idx]
        mean_map = means[idx]
        pos = conditions[idx]
        plot_traj = torch.cat([pos.unsqueeze(0).T, traj], dim=1)

        image_artist = ax.imshow(
            mean_map,
            origin="lower",
            extent=[0, 100, 0, 100],
            cmap="viridis",
            aspect="equal",
        )
        ax.plot(plot_traj[0], plot_traj[1], color="white", marker="o", linewidth=2, label="Trajectory")
        ax.scatter(pos[0], pos[1], color="red", s=70, label="Current position")
        ax.scatter(plot_traj[0, -1], plot_traj[1, -1], color="orange", s=70, marker="s", label="Trajectory end")
        ax.set_title(f"Sample {idx}")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)

    for ax in axes[len(indices):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(right=0.82, top=0.90, wspace=0.28, hspace=0.30)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.84, 0.93))
    cbar_ax = fig.add_axes([0.86, 0.18, 0.02, 0.62])
    fig.colorbar(image_artist, cax=cbar_ax, label="Mean map value")
    fig.suptitle(f"Dataset trajectories over mean maps from {dataset_path.name}", fontsize=13)
    plt.show()


def main():
    dataset_path = resolve_dataset_path()
    trajectories, conditions, means = load_dataset(dataset_path)
    indices = choose_indices(len(trajectories), NUM_PLOTS)

    print(f"Using dataset: {dataset_path}")
    print(f"Trajectory tensor shape: {tuple(trajectories.shape)}")
    print(f"Condition tensor shape: {tuple(conditions.shape)}")
    print(f"Mean tensor shape: {tuple(means.shape)}")
    print(f"Showing indices: {indices}")

    plot_grid(trajectories, conditions, means, indices, dataset_path)


if __name__ == "__main__":
    main()
