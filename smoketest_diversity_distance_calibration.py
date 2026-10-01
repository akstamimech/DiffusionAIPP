"""
Diagnostic: how different are the CMA_SOLUTIONS_PER_BRANCH plain CMA-ES
solutions (same warm start, different seeds, NO diversity term) at real belief
states from a real chain, and how many does the production selection keep?
Prints per candidate its masked variance reduction relative to the best, per
pair the xyz RMS distance and the Jaccard distance of the important-cell
footprints, and how many candidates survive the neighbourhood cut and the
Jaccard filter. Doesn't touch the production collector file; duplicates a thin
slice of run_chain_and_record's round-advance loop so it can call
simulate_candidate directly and see ALL candidates.
"""
import os
import sys
from pathlib import Path

import numpy as np

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import DataCollector_3D_randomstart_CMAESregularized as dc

SELECTED_MAP = int(os.environ.get("SMOKETEST_MAP", "5"))
ROUNDS = int(os.environ.get("SMOKETEST_ROUNDS", "3"))


def pairwise_xyz_rms(path_a, path_b):
    a = dc.resample_trajectory(path_a, dc.target_trajectory_len)
    b = dc.resample_trajectory(path_b, dc.target_trajectory_len)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def main():
    pts = dc.load_map(SELECTED_MAP)
    _, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, _ = dc.initialize_gp(
        sigma2=dc.GP_KERNEL_SIGMA2, lengthscale=dc.GP_KERNEL_LENGTHSCALE,
    )
    true_map_flat = dc.build_true_map_flat(pts, X_test)
    start_index, (start_cx, start_cy) = list(
        enumerate(dc.starts_for_map(SELECTED_MAP, xmin, xmax, ymin, ymax))
    )[0]

    mu = np.full(X_test.shape[0], dc.utility_threshold - 0.1, dtype=float)
    P = cov.copy()
    cx, cy, cz = start_cx, start_cy, dc.INIT_ALTITUDE
    samplestep = dc.step
    rng = np.random.default_rng(
        dc.SENSORNOISE_SEED + SELECTED_MAP * 100_000_000 + start_index * 1_000_000
    )
    warmup_goal_x = float(np.clip(cx + dc.WARMUP_OFFSET, xmin + dc.START_MARGIN, xmax - dc.START_MARGIN))
    warmup_goal_y = float(np.clip(cy + dc.WARMUP_OFFSET, ymin + dc.START_MARGIN, ymax - dc.START_MARGIN))
    cx, cy, cz, mu, P = dc.warmup_rollout(
        cx, cy, cz, mu, P, true_map_flat, xs, ys, xmin, xmax, ymin, ymax, rng,
        goal_x=warmup_goal_x, goal_y=warmup_goal_y,
    )
    print(f"CMA_SOLUTIONS_PER_BRANCH={dc.CMA_SOLUTIONS_PER_BRANCH}, CMA_GENERATIONS={dc.CMA_GENERATIONS}, "
          f"NEIGHBOURHOOD_THRESHOLD={dc.NEIGHBOURHOOD_THRESHOLD}, JACCARD_MIN_DISTANCE={dc.JACCARD_MIN_DISTANCE}, "
          f"CMA_STEP_SIZE_XY={dc.CMA_STEP_SIZE_XY}, CMA_STEP_SIZE_Z={dc.CMA_STEP_SIZE_Z}\n")

    for round_idx in range(ROUNDS):
        branch_idx = 0
        results = dc.simulate_candidate(
            dc.BETA, cx, cy, cz, mu, P, pts, true_map_flat, xs, ys, xmin, xmax, ymin, ymax,
            samplestep,
            rng_seed=dc.make_round_seed(SELECTED_MAP, start_index, round_idx, branch_idx, purpose=0),
            planner_seed=dc.make_round_seed(SELECTED_MAP, start_index, round_idx, branch_idx, purpose=1),
            variant_seeds=[
                dc.make_round_seed(SELECTED_MAP, start_index, round_idx, branch_idx, purpose=2 + i)
                for i in range(max(0, dc.CMA_SOLUTIONS_PER_BRANCH - 1))
            ],
        )
        paths = [r["spline_path"] for r in results]
        utility = dc.importance_filter(mu, P, dc.BETA, threshold=dc.utility_threshold)
        importance_mask = utility > 0
        footprints = [dc.path_footprint(p, xs, ys, importance_mask) for p in paths]
        baseline_var = float(np.sum(np.diag(P)[importance_mask]))
        gains = [baseline_var - float(np.sum(np.diag(r["final_P"])[importance_mask])) for r in results]
        best_gain = max(gains)

        print(f"round {round_idx}: reduction vs best = " + ", ".join(f"{g / best_gain:.2f}" for g in gains))
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                print(f"  seed {i} vs seed {j}: RMS xyz = {pairwise_xyz_rms(paths[i], paths[j]):6.2f} m, "
                      f"Jaccard = {dc.jaccard_distance(footprints[i], footprints[j]):.3f}")
        winner_i = int(np.argmax(gains))
        near_opt = [i for i, g in enumerate(gains) if i == winner_i or g / best_gain >= dc.NEIGHBOURHOOD_THRESHOLD]
        kept = dc.select_distinct_modes(
            [{"spline_path": p, "variance_correction": g} for p, g in zip(paths, gains)],
            near_opt, winner_i, importance_mask, xs, ys,
        )
        print(f"  kept: {len(near_opt)} within {dc.NEIGHBOURHOOD_THRESHOLD:.0%} of best, {len(kept)} after Jaccard >= {dc.JACCARD_MIN_DISTANCE}")
        best = results[winner_i]
        cx, cy, cz = best["update_cx"], best["update_cy"], best["update_cz"]
        mu, P = best["update_mu"], best["update_P"]

    print("\nDone.")


if __name__ == "__main__":
    main()
