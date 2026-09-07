"""
Diagnostic: how far apart do the CMA_SOLUTIONS_PER_BRANCH=3 CMA-ES variants
(original + 2 diversity-regularized) actually land, in practice, at real
belief states from a real chain - compared against
CMA_DIVERSITY_DISTANCE_THRESHOLD? Doesn't touch the production collector
file; duplicates a thin slice of run_chain_and_record's round-advance loop
so it can call simulate_candidate directly and see ALL 3 variants (not just
the ones the separate, unrelated variance-based NEIGHBOURHOOD_THRESHOLD
happens to keep).
"""
import sys
from pathlib import Path

import numpy as np

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import DataCollector_3D_randomstart_CMAESregularized as dc

SELECTED_MAP = 91
ROUNDS = 5


def pairwise_xyz_rms(path_a, path_b):
    a = dc.resample_trajectory(path_a, dc.target_trajectory_len)
    b = dc.resample_trajectory(path_b, dc.target_trajectory_len)
    d2 = np.sum((a - b) ** 2, axis=1)  # squared euclidean per point, xyz
    return float(np.sqrt(np.mean(d2)))  # RMS euclidean distance per point


def main():
    pts = dc.load_map(SELECTED_MAP)
    _, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, _ = dc.initialize_gp(
        sigma2=dc.GP_KERNEL_SIGMA2, lengthscale=dc.GP_KERNEL_LENGTHSCALE,
    )
    true_map_flat = dc.build_true_map_flat(pts, X_test)
    start_index, (start_cx, start_cy) = list(
        enumerate(dc.starts_for_map(SELECTED_MAP, xmin, xmax, ymin, ymax))
    )[0]

    mean_field = np.full(X_test.shape[0], dc.utility_threshold + 0.1, dtype=float)
    mu = mean_field.copy()
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

    threshold_rms = dc.CMA_DIVERSITY_DISTANCE_THRESHOLD  # what the code effectively enforces as an RMS xy distance
    mse_threshold = dc.CMA_DIVERSITY_DISTANCE_THRESHOLD ** 2
    print(f"CMA_DIVERSITY_DISTANCE_THRESHOLD={dc.CMA_DIVERSITY_DISTANCE_THRESHOLD} "
          f"(mse_threshold={mse_threshold}), CMA_STEP_SIZE_XY={dc.CMA_STEP_SIZE_XY}, "
          f"CMA_STEP_SIZE_Z={dc.CMA_STEP_SIZE_Z}\n")

    for round_idx in range(ROUNDS):
        branch_idx = 0
        results = dc.simulate_candidate(
            dc.BETA, cx, cy, cz, mu, P, pts, true_map_flat, xs, ys, xmin, xmax, ymin, ymax,
            samplestep,
            rng_seed=dc.make_round_seed(SELECTED_MAP, start_index, round_idx, branch_idx, purpose=0),
            planner_seed=dc.make_round_seed(SELECTED_MAP, start_index, round_idx, branch_idx, purpose=1),
            regularized_seeds=[
                dc.make_round_seed(SELECTED_MAP, start_index, round_idx, branch_idx, purpose=2 + i)
                for i in range(max(0, dc.CMA_SOLUTIONS_PER_BRANCH - 1))
            ],
        )
        paths = [r["spline_path"] for r in results]
        variants = [r["cma_variant"] for r in results]

        print(f"round {round_idx}:")
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                rms = pairwise_xyz_rms(paths[i], paths[j])
                cleared = "CLEARS threshold (counts as diverse)" if rms > threshold_rms else "under threshold (penalized as too similar)"
                print(f"  {variants[i]} vs {variants[j]}: RMS xyz distance = {rms:6.2f} m -> {cleared}")

        # advance using the variance-winner, same rule run_chain_and_record uses
        winner = max(results, key=lambda r: 0)  # placeholder, replaced below
        # compute masked variance exactly as run_chain_and_record does, to pick the same winner
        utility = dc.importance_filter(mu, P, dc.BETA, threshold=dc.utility_threshold)
        importance_mask = utility > 0
        baseline_variance = float(np.sum(np.diag(P)[importance_mask]))
        best = max(results, key=lambda r: baseline_variance - float(np.sum(np.diag(r["final_P"])[importance_mask])))
        cx, cy, cz = best["update_cx"], best["update_cy"], best["update_cz"]
        mu, P = best["update_mu"], best["update_P"]

    print("\nDone.")


if __name__ == "__main__":
    main()
