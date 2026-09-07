"""
Batch version of test_multimodal_branch_grf60.py: instead of one condition at
a time, scans a spread of ~20 two-branch conditions (map_id, timestep pairs
with exactly 2 of up to 4 candidate trajectories kept) and checks, for each,
whether diffusion's samples split across both branches or collapse onto one -
looking specifically for a systematic pattern in *which* branch wins (e.g.
correlated with parent_beam_index, i.e. whether it's the original
unregularized CMA-ES solution vs. a diversity-regularized alternate; or with
the per-branch training weight/RMSE_correction).

Loads the diffusion model once and reuses it across all conditions (unlike
the single-condition script, which reloads per run) - this is the only
change needed to make a ~20-condition sweep tractable.
"""
import csv
from pathlib import Path

import numpy as np
import torch

from gaussianprocesstraining import initialize_gp
import Diffusionplanner_singlemap as diffplan

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "dataset_grf_60.pt"
N_DIFFUSION_ROUNDS = 30
N_CONDITIONS = 20
MIN_BRANCH_DIST = 40.0  # skip near-duplicate branch pairs - not informative for a collapse question
OUT_DIR = SCRIPT_DIR / "results_multimodal_branch_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def waypoint_distance(a, b):
    return float(np.linalg.norm(a.reshape(-1) - b.reshape(-1)))


def select_conditions(dataset, n_conditions, min_branch_dist):
    cond = dataset["condition_id"].tolist()
    groups = {}
    for i, c in enumerate(cond):
        groups.setdefault(c, []).append(i)
    two_traj_conds = [c for c, idxs in groups.items() if len(idxs) == 2]

    candidates = []
    for c in two_traj_conds:
        idxs = groups[c]
        cw0 = dataset["control_waypoints"][idxs[0]].numpy()
        cw1 = dataset["control_waypoints"][idxs[1]].numpy()
        dist = waypoint_distance(cw0, cw1)
        if dist < min_branch_dist:
            continue
        candidates.append((c, idxs, dist))

    candidates.sort(key=lambda x: x[0])
    if len(candidates) <= n_conditions:
        return candidates
    stride = len(candidates) / n_conditions
    picked = [candidates[int(i * stride)] for i in range(n_conditions)]
    return picked


def main():
    dataset = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    conditions = select_conditions(dataset, N_CONDITIONS, MIN_BRANCH_DIST)
    print(f"Selected {len(conditions)} conditions (min_branch_dist={MIN_BRANCH_DIST}, {N_DIFFUSION_ROUNDS} samples each)", flush=True)

    gp, X_test, mean0, cov0, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()
    bounds = (xmin, xmax, ymin, ymax, diffplan.ZMIN, diffplan.ZMAX)

    diffusion_model = diffplan.load_diffusion_model(checkpoint_path=diffplan.diffusion_path)
    print(f"diffusion checkpoint: {diffplan.diffusion_path}", flush=True)

    rows = []
    for cond_id, idxs, branch_dist in conditions:
        map_id = int(dataset["map_id"][idxs[0]])
        timestep = int(dataset["timestep"][idxs[0]])
        current_position = dataset["current_position"][idxs[0]].numpy()
        current_mean = dataset["current_mean"][idxs[0]].numpy()
        current_var = dataset["current_var"][idxs[0]].numpy()
        heading = dataset["initial_heading_velocity"][idxs[0]].numpy()

        branch_waypoints = [dataset["control_waypoints"][i].numpy() for i in idxs]
        branch_weight = [float(dataset["variance_correction"][i]) for i in idxs]
        branch_rmse_corr = [float(dataset["RMSE_correction"][i]) for i in idxs]
        branch_parent_idx = [int(dataset["parent_beam_index"][i]) for i in idxs]

        dist0_list, dist1_list = [], []
        for _ in range(N_DIFFUSION_ROUNDS):
            _dense, waypoints = diffplan.sample_diffusion_trajectory(
                diffusion_model,
                current_position=tuple(current_position),
                current_mean=current_mean,
                current_var=current_var,
                current_heading_velocity=heading,
                grid_step=step,
                bounds=bounds,
            )
            dist0_list.append(waypoint_distance(waypoints, branch_waypoints[0]))
            dist1_list.append(waypoint_distance(waypoints, branch_waypoints[1]))

        dist0 = np.array(dist0_list)
        dist1 = np.array(dist1_list)
        wins0 = int((dist0 < dist1).sum())
        wins1 = N_DIFFUSION_ROUNDS - wins0
        winner = 0 if wins0 >= wins1 else 1
        collapsed = (wins0 == 0 or wins1 == 0)

        row = {
            "condition_id": cond_id,
            "map_id": map_id,
            "timestep": timestep,
            "branch_dist": branch_dist,
            "n_samples": N_DIFFUSION_ROUNDS,
            "wins_branch0": wins0,
            "wins_branch1": wins1,
            "winner": winner,
            "collapsed": collapsed,
            "branch0_weight": branch_weight[0],
            "branch1_weight": branch_weight[1],
            "branch0_rmse_corr": branch_rmse_corr[0],
            "branch1_rmse_corr": branch_rmse_corr[1],
            "branch0_parent_idx": branch_parent_idx[0],
            "branch1_parent_idx": branch_parent_idx[1],
            "winner_parent_idx": branch_parent_idx[winner],
            "winner_has_lower_parent_idx": branch_parent_idx[winner] < branch_parent_idx[1 - winner],
            "winner_has_higher_weight": branch_weight[winner] > branch_weight[1 - winner],
            "winner_has_lower_rmse_corr": branch_rmse_corr[winner] < branch_rmse_corr[1 - winner],
            "mean_dist_branch0": float(dist0.mean()),
            "mean_dist_branch1": float(dist1.mean()),
        }
        rows.append(row)
        print(
            f"map={map_id:3d} ts={timestep:2d} cond={cond_id:9d} branch_dist={branch_dist:6.1f} "
            f"wins=[{wins0:2d},{wins1:2d}] {'COLLAPSED' if collapsed else 'mixed':9s} "
            f"parent_idx=[{branch_parent_idx[0]},{branch_parent_idx[1]}] winner_parent_idx={branch_parent_idx[winner]} "
            f"weight=[{branch_weight[0]:.2f},{branch_weight[1]:.2f}]",
            flush=True,
        )

    out_csv = OUT_DIR / "batch_condition_scan.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}", flush=True)

    n = len(rows)
    n_collapsed = sum(r["collapsed"] for r in rows)
    n_lower_parent = sum(r["winner_has_lower_parent_idx"] for r in rows)
    n_higher_weight = sum(r["winner_has_higher_weight"] for r in rows)
    n_lower_rmse = sum(r["winner_has_lower_rmse_corr"] for r in rows)
    print(f"\n=== Summary across {n} conditions ===")
    print(f"Fully collapsed (0/{N_DIFFUSION_ROUNDS} or {N_DIFFUSION_ROUNDS}/{N_DIFFUSION_ROUNDS} split): {n_collapsed}/{n}")
    print(f"Winning branch has the LOWER parent_beam_index (i.e. is the original, non-diversity-regularized CMA-ES solution): {n_lower_parent}/{n}")
    print(f"Winning branch has the HIGHER training weight (variance_correction): {n_higher_weight}/{n}")
    print(f"Winning branch has the LOWER RMSE_correction (better reconstruction): {n_lower_rmse}/{n}")


if __name__ == "__main__":
    main()
