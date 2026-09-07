"""
Multimodal-vs-mode-averaging test using dataset_grf_60.pt (the dataset
Diffusion and ImitateTrans were trained on).

Finds a specific training condition (map 0, timestep 0, condition_id
3000000) where exactly 2 of up to 4 candidate branch trajectories were kept
during data collection - a genuine bimodal decision point (the two saved
branches' control waypoints differ by an L2 norm of ~99.7, not near-
duplicates). Both branches share the EXACT same starting position, GP mean
field, and GP variance field (verified via torch.allclose on the dataset
tensors) - only the resulting trajectory differs.

Re-initializes the diffusion and ImitateTrans models at that exact
condition (position/mean/var/heading pulled directly from the dataset, no
resimulation needed since sample_diffusion_trajectory/sample_imitation_trajectory
only need those as inputs) and draws multiple independent rounds from each:
  - Diffusion samples fresh noise every call, so repeated rounds can land on
    either mode (or diverge from both).
  - ImitateTrans is a deterministic regression model (no injected noise,
    eval mode) - repeated rounds should be numerically identical every time,
    which is itself the "mode averaging" signature: a single point estimate
    that can only ever be one path, most likely somewhere between the two
    training modes rather than matching either one.

Outputs a 2D (x,y) overlay plot and a per-round CSV recording each sample's
control waypoints and its L2 distance to each of the two dataset branches.
"""
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gaussianprocesstraining import initialize_gp
import Diffusionplanner_singlemap as diffplan
import ImitateTrans_singlemap as imitplan

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "dataset_grf_60.pt"
CONDITION_ID = 200002000  # map_id=2, timestep=2, 2-of-4 branches saved, branch dist=96.2
N_DIFFUSION_ROUNDS = 80
N_IMITATION_ROUNDS = 4  # deterministic - run a few times to demonstrate zero variance
OUT_DIR = SCRIPT_DIR / "results_multimodal_branch_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
COLOR_BRANCH_A = "#0b0b0b"    # solid black - dataset branch A (ground truth)
COLOR_BRANCH_B = "#898781"    # muted gray - dataset branch B (ground truth)
COLOR_DIFFUSION = "#2a78d6"   # categorical slot 1 (blue)
COLOR_IMITATION = "#eb6834"   # categorical slot 2 (orange)


def waypoint_distance(a, b):
    return float(np.linalg.norm(a.reshape(-1) - b.reshape(-1)))


def main():
    dataset = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    cond_mask = dataset["condition_id"] == CONDITION_ID
    idxs = cond_mask.nonzero(as_tuple=True)[0].tolist()
    if len(idxs) != 2:
        raise SystemExit(f"Expected exactly 2 saved branches for condition_id={CONDITION_ID}, found {len(idxs)}")

    map_id = int(dataset["map_id"][idxs[0]])
    timestep = int(dataset["timestep"][idxs[0]])
    assert torch.allclose(dataset["current_position"][idxs[0]], dataset["current_position"][idxs[1]])
    assert torch.allclose(dataset["current_mean"][idxs[0]], dataset["current_mean"][idxs[1]])
    assert torch.allclose(dataset["current_var"][idxs[0]], dataset["current_var"][idxs[1]])
    print(f"condition_id={CONDITION_ID}: map_id={map_id} timestep={timestep} branch rows={idxs} (position/mean/var confirmed identical across branches)")

    current_position = dataset["current_position"][idxs[0]].numpy()
    current_mean = dataset["current_mean"][idxs[0]].numpy()
    current_var = dataset["current_var"][idxs[0]].numpy()
    heading = dataset["initial_heading_velocity"][idxs[0]].numpy()

    branch_waypoints = [dataset["control_waypoints"][i].numpy() for i in idxs]  # each [3,8]
    branch_dense = [dataset["trajectories"][i].numpy() for i in idxs]  # each [3,41]
    branch_dist = waypoint_distance(branch_waypoints[0], branch_waypoints[1])
    print(f"L2 distance between the two dataset branches' control waypoints: {branch_dist:.2f}")

    gp, X_test, mean0, cov0, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()
    bounds = (xmin, xmax, ymin, ymax, diffplan.ZMIN, diffplan.ZMAX)

    diffusion_model = diffplan.load_diffusion_model(checkpoint_path=diffplan.diffusion_path)
    imitate_model = imitplan.load_imitate_model(checkpoint_path=imitplan.imitate_checkpoint_path)
    print(f"diffusion checkpoint: {diffplan.diffusion_path}")
    print(f"imitation checkpoint: {imitplan.imitate_checkpoint_path}")

    rows = []
    diffusion_dense = []
    for r in range(N_DIFFUSION_ROUNDS):
        dense, waypoints = diffplan.sample_diffusion_trajectory(
            diffusion_model,
            current_position=tuple(current_position),
            current_mean=current_mean,
            current_var=current_var,
            current_heading_velocity=heading,
            grid_step=step,
            bounds=bounds,
        )
        diffusion_dense.append(dense)
        rows.append({
            "model": "diffusion",
            "round": r,
            "dist_to_branch_A": waypoint_distance(waypoints, branch_waypoints[0]),
            "dist_to_branch_B": waypoint_distance(waypoints, branch_waypoints[1]),
            "control_waypoints": waypoints.tolist(),
        })

    imitation_dense = []
    for r in range(N_IMITATION_ROUNDS):
        dense, waypoints = imitplan.sample_imitation_trajectory(
            imitate_model,
            current_position=tuple(current_position),
            current_mean=current_mean,
            current_var=current_var,
            current_heading_velocity=heading,
            grid_step=step,
            bounds=bounds,
        )
        imitation_dense.append(dense)
        rows.append({
            "model": "imitation",
            "round": r,
            "dist_to_branch_A": waypoint_distance(waypoints, branch_waypoints[0]),
            "dist_to_branch_B": waypoint_distance(waypoints, branch_waypoints[1]),
            "control_waypoints": waypoints.tolist(),
        })

    imitation_spread = max(
        waypoint_distance(
            np.array(rows[len(rows) - N_IMITATION_ROUNDS + i]["control_waypoints"]),
            np.array(rows[len(rows) - N_IMITATION_ROUNDS]["control_waypoints"]),
        )
        for i in range(N_IMITATION_ROUNDS)
    )
    print(f"Max pairwise distance among the {N_IMITATION_ROUNDS} ImitateTrans rounds: {imitation_spread:.6f} (near-zero confirms determinism)")

    diff_dists_A = [r["dist_to_branch_A"] for r in rows if r["model"] == "diffusion"]
    diff_dists_B = [r["dist_to_branch_B"] for r in rows if r["model"] == "diffusion"]
    imit_dists_A = [r["dist_to_branch_A"] for r in rows if r["model"] == "imitation"]
    imit_dists_B = [r["dist_to_branch_B"] for r in rows if r["model"] == "imitation"]
    print(f"Diffusion: dist-to-A mean={np.mean(diff_dists_A):.2f} min={np.min(diff_dists_A):.2f} | dist-to-B mean={np.mean(diff_dists_B):.2f} min={np.min(diff_dists_B):.2f}")
    print(f"Imitation: dist-to-A mean={np.mean(imit_dists_A):.2f} | dist-to-B mean={np.mean(imit_dists_B):.2f}")

    csv_path = OUT_DIR / f"map{map_id}_condition{CONDITION_ID}_branch_rounds.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["model", "round", "dist_to_branch_A", "dist_to_branch_B", "control_waypoints"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            row_out["control_waypoints"] = repr(row["control_waypoints"])
            writer.writerow(row_out)
    print(f"Wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.plot(branch_dense[0][0], branch_dense[0][1], color=COLOR_BRANCH_A, linewidth=3, zorder=6, label="Dataset branch A (kept)")
    ax.plot(branch_dense[1][0], branch_dense[1][1], color=COLOR_BRANCH_B, linewidth=3, linestyle="--", zorder=6, label="Dataset branch B (kept)")
    for i, dense in enumerate(diffusion_dense):
        ax.plot(dense[0], dense[1], color=COLOR_DIFFUSION, linewidth=1.3, alpha=0.55, zorder=4,
                 label="Diffusion samples" if i == 0 else None)
    for i, dense in enumerate(imitation_dense):
        ax.plot(dense[0], dense[1], color=COLOR_IMITATION, linewidth=1.6, alpha=0.7, zorder=5,
                 label="ImitateTrans samples" if i == 0 else None)
    ax.scatter([current_position[0]], [current_position[1]], color=INK_PRIMARY, s=90, zorder=7,
               marker="*", label="Start (shared condition)")
    ax.set_xlabel("x (m)", color=INK_SECONDARY)
    ax.set_ylabel("y (m)", color=INK_SECONDARY)
    ax.set_title(
        f"Map {map_id}, condition_id={CONDITION_ID} (timestep {timestep}): "
        f"{N_DIFFUSION_ROUNDS} diffusion rounds vs. {N_IMITATION_ROUNDS} ImitateTrans rounds\n"
        "against the 2 kept dataset branches from this exact GP state",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    plot_path = OUT_DIR / f"map{map_id}_condition{CONDITION_ID}_branch_rounds.png"
    fig.savefig(plot_path, dpi=160)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
