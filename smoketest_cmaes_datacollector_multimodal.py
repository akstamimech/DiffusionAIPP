"""
Smoke test for DataCollector_3D_randomstart_CMAESregularized.py after this
session's CMA-ES objective/penalty/lattice fixes: runs exactly one chain, a
few rounds, on one map - by calling the collector's own (unmodified)
run_chain_and_record directly with an explicit rank_limit override, rather
than editing its module-level RANKLIM/STARTS_PER_MAP/mapcount constants used
by the real HPC collection job.

Any round where more than one of the CMA_SOLUTIONS_PER_BRANCH=3 CMA-ES
variants (1 original + 2 diversity-regularized) survive the
NEIGHBOURHOOD_THRESHOLD cut (i.e. a genuinely multimodal round - multiple
meaningfully different trajectories achieving comparable real variance
reduction from the same belief state) gets its own overlay plot.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import DataCollector_3D_randomstart_CMAESregularized as dc

SELECTED_MAP = 91
ROUNDS = 5
OUT_DIR = script_dir / "smoketest_cmaes_datacollector_out"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pts = dc.load_map(SELECTED_MAP)
    _, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, _ = dc.initialize_gp(
        sigma2=dc.GP_KERNEL_SIGMA2,
        lengthscale=dc.GP_KERNEL_LENGTHSCALE,
    )
    true_map_flat = dc.build_true_map_flat(pts, X_test)

    start_index, (start_cx, start_cy) = list(
        enumerate(dc.starts_for_map(SELECTED_MAP, xmin, xmax, ymin, ymax))
    )[0]
    print(f"Map {SELECTED_MAP}, single chain, start {start_index} = ({start_cx:.1f}, {start_cy:.1f}), {ROUNDS} rounds")

    dataset = dc.make_empty_dataset()
    dc.run_chain_and_record(
        dataset,
        SELECTED_MAP,
        start_index,
        start_cx,
        start_cy,
        pts,
        true_map_flat,
        xs,
        ys,
        X,
        xmin,
        xmax,
        ymin,
        ymax,
        X_test,
        cov,
        mpi_rank=0,
        mpi_size=1,
        rank_limit=ROUNDS,
        pool=None,
    )

    n = len(dataset["trajectories"])
    print(f"Recorded {n} samples across {ROUNDS} rounds")

    # Group recorded samples by parent_beam_id (one group per round's kept
    # branch/CMA-variant set) to find genuinely multimodal rounds.
    groups = {}
    for i in range(n):
        pid = dataset["parent_beam_id"][i]
        groups.setdefault(pid, []).append(i)

    multimodal_groups = {pid: idxs for pid, idxs in groups.items() if len(idxs) > 1}
    print(f"{len(multimodal_groups)} of {len(groups)} rounds kept >1 variant (multimodal)")

    if not multimodal_groups:
        print("No multimodal rounds found in this smoke test (all rounds converged "
              "to a single dominant solution) - nothing to visualize.")
        return

    for plot_idx, (pid, idxs) in enumerate(sorted(multimodal_groups.items())):
        idxs = sorted(idxs, key=lambda i: dataset["parent_beam_index"][i])
        round_idx = dataset["timestep"][idxs[0]]
        current_pos = dataset["current_position"][idxs[0]]

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap="YlGn", s=14, alpha=0.6, linewidths=0)
        plt.colorbar(sc, ax=ax, label="ground-truth field value")

        colors = plt.get_cmap("tab10")
        for j, i in enumerate(idxs):
            traj = dataset["trajectories"][i]  # (target_len, 3): x, y, z
            label = "winner (executed)" if dataset["parent_beam_index"][i] == 0 else f"alternate {dataset['parent_beam_index'][i]}"
            ax.plot(traj[:, 0], traj[:, 1], color=colors(j), linewidth=2.0, marker="o", markersize=2, label=label)

        ax.scatter([current_pos[0]], [current_pos[1]], color="black", marker="*", s=180, zorder=5, label="start of round")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Map {SELECTED_MAP}, round {round_idx}: {len(idxs)} kept CMA-ES variants (multimodal)")
        ax.legend(fontsize=8, loc="best")
        ax.set_aspect("equal")
        fig.tight_layout()

        out_path = OUT_DIR / f"multimodal_map{SELECTED_MAP}_round{round_idx}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
