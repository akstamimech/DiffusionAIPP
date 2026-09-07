"""
Diagnostic: across many real rounds of a real chain, which of the
CMA_SOLUTIONS_PER_BRANCH=3 CMA-ES variants (cma_original, cma_regularized_1,
cma_regularized_2) actually wins each round - i.e. achieves the highest
REALIZED masked-variance reduction, the same criterion run_chain_and_record
uses to pick the winner and to decide which branches are kept as multimodal
alternates? Does a diversity-regularized variant (found by penalizing
closeness to earlier solutions in the same round) ever come back better than
the plain (unregularized) original CMA-ES pass, not just different?
"""
import sys
from pathlib import Path

import numpy as np

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import DataCollector_3D_randomstart_CMAESregularized as dc

SELECTED_MAP = 91
ROUNDS = 15


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

    win_counts = {}
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

        utility = dc.importance_filter(mu, P, dc.BETA, threshold=dc.utility_threshold)
        importance_mask = utility > 0
        baseline_variance = float(np.sum(np.diag(P)[importance_mask]))

        scored = []
        for r in results:
            variance_correction = baseline_variance - float(np.sum(np.diag(r["final_P"])[importance_mask]))
            scored.append((r["cma_variant"], variance_correction))
        scored.sort(key=lambda t: -t[1])

        winner_name, winner_score = scored[0]
        win_counts[winner_name] = win_counts.get(winner_name, 0) + 1
        gap_to_second = winner_score - scored[1][1]
        rel_gap = (gap_to_second / winner_score * 100) if winner_score != 0 else float("nan")
        print(
            f"round {round_idx:2d}: ranking = {[(name, f'{v:.4f}') for name, v in scored]} "
            f"-> winner={winner_name}, margin over 2nd place = {gap_to_second:.4f} ({rel_gap:.1f}%)"
        )

        best = max(results, key=lambda r: baseline_variance - float(np.sum(np.diag(r["final_P"])[importance_mask])))
        cx, cy, cz = best["update_cx"], best["update_cy"], best["update_cz"]
        mu, P = best["update_mu"], best["update_P"]

    print(f"\nWin counts over {ROUNDS} rounds: {win_counts}")


if __name__ == "__main__":
    main()
