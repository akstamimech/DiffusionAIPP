from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colors


# DATASET_PATH = Path(__file__).resolve().parent / "parallel_randomstart_dataset.pt"
DATASET_PATH = Path(__file__).resolve().parent / "FINAL_NAIP_DATASET.pt"

# (map_id, start_index, timestep) together identify one decision point ("node")
# along one of DataCollector_3D_randomstart_multimodal.py's STARTS_PER_MAP=3
# parallel chains. Usually one row is recorded there (the winning branch); when
# alternate near-tied branches were also kept, multiple rows share the same
# node - those are what this script visualizes as candidate modes.
SELECTED_MAP = 10
SELECTED_START_INDEX = 3  # which of the 3 parallel chains (0, 1, 2)
SELECTED_ROUND = 3 # position along that chain (dataset's "timestep" field)
TOP_N = 4  # BRANCH_COUNT during collection was 4, so a node never has more candidates than this

REQUIRED_KEYS = {
    "trajectories",
    "control_waypoints",
    "current_position",
    "map_id",
    "start_index",
    "timestep",
    "parent_beam_index",
    "variance_correction",
    "RMSE_correction",
}


def load_dataset(path):
    path = Path(path)
    try:
        dataset = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        dataset = torch.load(path, map_location="cpu")

    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset):
    if not isinstance(dataset, dict):
        raise TypeError(
            f"Expected a dictionary of tensors, got {type(dataset).__name__}."
        )

    missing = REQUIRED_KEYS - set(dataset)
    if missing:
        raise KeyError(f"Dataset is missing required keys: {sorted(missing)}")

    sample_count = len(dataset["map_id"])
    for key in REQUIRED_KEYS:
        value = dataset[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Dataset key {key!r} must contain a tensor.")
        if len(value) != sample_count:
            raise ValueError(
                f"Dataset key {key!r} has {len(value)} samples; "
                f"expected {sample_count}."
            )

    expected_shapes = {
        "trajectories": (3, None),
        "control_waypoints": (3, None),
        "current_position": (3,),
    }
    for key, expected in expected_shapes.items():
        actual = tuple(dataset[key].shape[1:])
        shape_matches = len(actual) == len(expected) and all(
            wanted is None or got == wanted
            for got, wanted in zip(actual, expected)
        )
        if not shape_matches:
            raise ValueError(
                f"{key} must have sample shape {expected}; got {actual}."
            )


def select_samples(dataset, map_id, start_index, round_idx, top_n):
    validate_dataset(dataset)
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}.")

    matches = (
        (dataset["map_id"] == map_id)
        & (dataset["start_index"] == start_index)
        & (dataset["timestep"] == round_idx)
    )
    matching_indices = torch.nonzero(matches, as_tuple=False).flatten()
    if matching_indices.numel() == 0:
        raise ValueError(
            f"No samples found for map_id={map_id}, start_index={start_index}, "
            f"round={round_idx}."
        )

    # Rank by variance_correction, not RMSE_correction: variance reduction is
    # what run_chain_and_record() actually used to pick the winning branch
    # (parent_beam_index==0) during collection - RMSE_correction is kept only
    # as a diagnostic signal. Ranking on it here could show a different branch
    # as "best" than the one that actually won and advanced the chain.
    corrections = dataset["variance_correction"][matching_indices]
    order = torch.argsort(corrections, descending=True)
    selected_indices = matching_indices[order[:top_n]]

    return {
        "indices": selected_indices,
        "trajectories": dataset["trajectories"][selected_indices],
        "control_waypoints": dataset["control_waypoints"][selected_indices],
        "current_position": dataset["current_position"][selected_indices],
        "variance_correction": dataset["variance_correction"][selected_indices],
        "rmse_correction": dataset["RMSE_correction"][selected_indices],
        "parent_beam_index": dataset["parent_beam_index"][selected_indices],
        "current_mean": dataset["current_mean"][selected_indices],
    }


def _color_normalizer(values):
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if np.isclose(minimum, maximum):
        padding = max(abs(minimum) * 0.05, 1e-6)
        minimum -= padding
        maximum += padding
    return colors.Normalize(vmin=minimum, vmax=maximum)


def _set_equal_xy_limits(axis_3d, axis_xy, trajectories, starts):
    x_values = np.concatenate((trajectories[:, 0, :].ravel(), starts[:, 0]))
    y_values = np.concatenate((trajectories[:, 1, :].ravel(), starts[:, 1]))
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    span = max(x_max - x_min, y_max - y_min, 1.0)
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    padding = 0.08 * span
    half_span = 0.5 * span + padding

    for axis in (axis_3d, axis_xy):
        axis.set_xlim(x_mid - half_span, x_mid + half_span)
        axis.set_ylim(y_mid - half_span, y_mid + half_span)

    axis_xy.set_aspect("equal", adjustable="box")


def build_figure(selected, map_id, start_index, round_idx):
    trajectories = selected["trajectories"].detach().cpu().numpy()
    controls = selected["control_waypoints"].detach().cpu().numpy()
    starts = selected["current_position"].detach().cpu().numpy()
    variance_corrections = selected["variance_correction"].detach().cpu().numpy()
    rmse_corrections = selected["rmse_correction"].detach().cpu().numpy()
    parent_beam_indices = selected["parent_beam_index"].detach().cpu().numpy()
    indices = selected["indices"].detach().cpu().numpy()
    condition_map = selected["current_mean"][0].detach().cpu().numpy()
    height, width = condition_map.shape
    X = np.linspace(0.0, 100.0, width)
    Y = np.linspace(0.0, 100.0, height)

    x, y = np.meshgrid(X, Y)
    z = np.full_like(x, 0.0)

    # Color by variance_correction (the actual winner-selection criterion) so
    # the branch drawn brightest here is the same one parent_beam_index==0
    # identifies as the winner.
    normalizer = _color_normalizer(variance_corrections)
    colormap = plt.get_cmap("viridis")
    figure = plt.figure(figsize=(15, 7), constrained_layout=True)
    axis_3d = figure.add_subplot(1, 2, 1, projection="3d", computed_zorder = False)
    axis_xy = figure.add_subplot(1, 2, 2)
   
    mean_map_normalizer = colors.Normalize(
    vmin=condition_map.min(),
    vmax=condition_map.max(),
    )
    heatmap_colors = cm.Greys(mean_map_normalizer(condition_map))
    

   



    for order, (trajectory, control, start, var_corr, rmse_corr, beam_index, sample_index) in enumerate(
        zip(
            trajectories,
            controls,
            starts,
            variance_corrections,
            rmse_corrections,
            parent_beam_indices,
            indices,
        ),
        start=1,
    ):
        color = colormap(normalizer(var_corr))
        is_winner = beam_index == 0
        label = (
            f"#{order} sample {sample_index}{' [WINNER]' if is_winner else ''} "
            f"(var drop={var_corr:.4f}, rmse drop={rmse_corr:.5f})"
        )

        axis_3d.plot(
            trajectory[0],
            trajectory[1],
            trajectory[2],
            color=color,
            linewidth=3.2 if is_winner else 1.6,
            label=label,
        )

        # axis_3d.contour(x, y, condition_map, levels = 30, zdir = "z", offset = 10.0, cmap = "Greys", alpha = 0.65)
        

        axis_3d.plot_surface(
            x,
            y,
            z,
            facecolors=heatmap_colors,
            shade=False,
            alpha=0.75,
            antialiased=False,
            rstride=1,
            cstride=1,
        )

        axis_3d.scatter(
            control[0],
            control[1],
            control[2],
            color=[color],
            marker="o",
            s=24,
            edgecolors="black",
            linewidths=0.4,
        )
        axis_3d.scatter(
            start[0],
            start[1],
            start[2],
            color=[color],
            marker="*",
            s=110,
            edgecolors="black",
            linewidths=0.7,
        )

        axis_xy.plot(
            trajectory[0],
            trajectory[1],
            color=color,
            linewidth=3.2 if is_winner else 1.6,
        )
        axis_xy.scatter(
            control[0],
            control[1],
            color=[color],
            marker="o",
            s=24,
            edgecolors="black",
            linewidths=0.4,
        )
        axis_xy.scatter(
            start[0],
            start[1],
            color=[color],
            marker="*",
            s=110,
            edgecolors="black",
            linewidths=0.7,
        )
        axis_xy.annotate(
            f"#{order}",
            (trajectory[0, -1], trajectory[1, -1]),
            xytext=(4, 4),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
        )

    _set_equal_xy_limits(axis_3d, axis_xy, trajectories, starts)

    axis_3d.set_title("3D candidate action trajectories")
    axis_3d.set_xlabel("X")
    axis_3d.set_ylabel("Y")
    axis_3d.set_zlabel("Altitude")
    axis_3d.legend(loc="upper left", fontsize=8)

    axis_xy.set_title("Top-down view")
    axis_xy.set_xlabel("X")
    axis_xy.set_ylabel("Y")
    axis_xy.grid(alpha=0.25)
    axis_3d.set_zlim(0.0, 40.0)
    axis_3d.set_xlim(0.0, 100.0)
    axis_3d.set_ylim(0.0, 100.0)

    figure.suptitle(
        f"CMAES dataset candidates: map {map_id}, chain/start {start_index}, round {round_idx}",
        fontsize=14,
    )
    scalar_mappable = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    scalar_mappable.set_array(variance_corrections)
    colorbar = figure.colorbar(
        scalar_mappable,
        ax=[axis_3d, axis_xy],
        fraction=0.025,
        pad=0.03,
    )
    colorbar.set_label("Variance correction")
    return figure


def main():
    dataset = load_dataset(DATASET_PATH)
    selected = select_samples(
        dataset,
        map_id=SELECTED_MAP,
        start_index=SELECTED_START_INDEX,
        round_idx=SELECTED_ROUND,
        top_n=TOP_N,
    )

    print(
        f"Plotting {len(selected['indices'])} candidate branch(es) for "
        f"map {SELECTED_MAP}, chain/start {SELECTED_START_INDEX}, round {SELECTED_ROUND}."
    )
    for order, (sample_index, var_corr, rmse_corr, beam_index) in enumerate(
        zip(
            selected["indices"],
            selected["variance_correction"],
            selected["rmse_correction"],
            selected["parent_beam_index"],
        ),
        start=1,
    ):
        winner_tag = " (WINNER)" if beam_index.item() == 0 else ""
        print(
            f"  #{order}: dataset sample {sample_index.item()}{winner_tag}, "
            f"variance correction={var_corr.item():.6f}, "
            f"RMSE correction={rmse_corr.item():.6f}"
        )

    build_figure(
        selected,
        map_id=SELECTED_MAP,
        start_index=SELECTED_START_INDEX,
        round_idx=SELECTED_ROUND,
    )
    plt.show()


if __name__ == "__main__":
    main()
