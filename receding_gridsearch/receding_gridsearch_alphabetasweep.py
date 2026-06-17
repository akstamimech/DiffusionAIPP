import itertools
from zipfile import Path
import numpy as np
import matplotlib.pyplot as plt
from gaussianprocesstraining import utility_function, kalman_update, initialize_gp, grid_search
from evalmetrics import compute_task_completion


step = 2.0
timealloted = 100
beta = 0.1
alpha = 0.1
utility_threshold = 0.0
planning_horizon = 8
action_horizon = 3  # this is more like replanning horizon
selected_map = 22
path = r"/scratch/ajain3/aipp/csv"

# Multi-map evaluation settings
mapcount = 30
initial_map = 2
plot_output_path = "alpha_beta_task_completion_heatmap.png"
best_configs_output_path = "best_alpha_beta_by_map.txt"

# Sweep settings
run_alpha_beta_sweep = True
alpha_min = 0.1
alpha_max = 0.6
beta_min = 0.1
beta_max = 0.6
num_alpha_values = 10
num_beta_values = 10


"""
Evaluation copy of receding_gridsearch.py.

If run_alpha_beta_sweep is True, the script evaluates multiple alpha/beta
combinations over a range of maps and finds the best (alpha, beta) pair for each
map individually after timealloted timesteps. The best configurations are
printed and saved ordered by map number.
"""


def receding_horizon_planner(cx, cy, sorted_util_values, alpha, horizon=planning_horizon):
    flight_plan = []
    for _ in range(horizon + 1):
        updated_utils = []
        for util, (x, y) in sorted_util_values:
            dist = np.hypot(x - cx, y - cy)

            if dist == 0:
                dist = 1e-6

            new_util = util * np.exp(-alpha * dist)
            updated_utils.append((new_util, (x, y)))

        sorted_util_values = sorted(updated_utils, key=lambda x: x[0], reverse=True)

        best_coord = sorted_util_values[0][1]
        best_coord = (
            step * np.round(best_coord[0] / step),
            step * np.round(best_coord[1] / step),
        )

        flight_plan.append(best_coord)
        cx, cy = best_coord
        sorted_util_values.pop(0)

    return flight_plan


def waypoint(cx, cy, goal_x, goal_y, step):
    dx = goal_x - cx
    dy = goal_y - cy
    dist = np.hypot(dx, dy)

    if dist <= step:
        return 0.0, 0.0, True

    grad_x = dx / dist
    grad_y = dy / dist

    return grad_x, grad_y, False


def dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax, buffer=step * 2):
    noise = np.random.randint(-1, 2, size=2)
    cx = np.clip(cx + grad_x * samplestep + noise[0], xmin + buffer, xmax - buffer)
    cy = np.clip(cy + grad_y * samplestep + noise[1], ymin + buffer, ymax - buffer)
    cx = step * np.round(cx / step)
    cy = step * np.round(cy / step)
    return cx, cy


def run_rollout(alpha_value, beta_value, map_id):
    data = np.loadtxt(rf"{path}/map_{map_id}_blob_grid_counts.csv", delimiter=",", skiprows=1)

    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()

    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]

    N = X_test.shape[0]
    mu = mean.copy()
    P = cov.copy()
    R = 1e-6

    pos_history = []
    lateral_coverage = step * 2
    samplestep = 4.0

    cx, cy = 20.0, 20.0
    grad_x, grad_y = 0.0, 0.0
    pos_history.append((cx, cy))

    initial_total_variance = np.sum(np.diag(P))
    utility = utility_function(mu, P, utility_threshold, beta_value)
    flight_plan = []

    for ts in range(0, timealloted):
        if ts <= 10:
            grad_x, grad_y, _ = waypoint(cx, cy, goal_x=80.0, goal_y=80.0, step=step)
            cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
            pos_history.append((cx, cy))
        else:
            if ts == 11:
                flight_plan = receding_horizon_planner(
                    cx, cy, grid_search(X, Y, cx, cy, [utility]), alpha=alpha_value, horizon=planning_horizon
                )

            grad_x, grad_y, waypoint_reached = waypoint(
                cx, cy, goal_x=flight_plan[0][0], goal_y=flight_plan[0][1], step=step
            )
            if waypoint_reached:
                flight_plan.pop(0)
                if len(flight_plan) > 0:
                    grad_x, grad_y, _ = waypoint(
                        cx, cy, goal_x=flight_plan[0][0], goal_y=flight_plan[0][1], step=step
                    )

            if len(flight_plan) <= action_horizon:
                flight_plan = receding_horizon_planner(
                    cx, cy, grid_search(X, Y, cx, cy, [utility]), alpha=alpha_value, horizon=planning_horizon
                )

            cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
            pos_history.append((cx, cy))

        fov = [
            (x, y)
            for x in np.arange(cx - lateral_coverage, cx + lateral_coverage + 1e-9, step)
            for y in np.arange(cy - lateral_coverage, cy + lateral_coverage + 1e-9, step)
        ]

        sensor = np.zeros((len(fov), N))

        for i, (x_meas, y_meas) in enumerate(fov):
            idx = np.where(
                np.isclose(X_test[:, 0], x_meas) & np.isclose(X_test[:, 1], y_meas)
            )[0]

            if len(idx) == 0:
                continue

            idx = idx[0]
            sensor[i, idx] = 1.0

        measurement_list = []
        for x_fov, y_fov in fov:
            idx = np.where(
                np.isclose(pts[:, 0], x_fov) & np.isclose(pts[:, 1], y_fov)
            )[0]

            if idx.size > 0:
                measurement_list.append(pts[idx[0], 2])
            else:
                measurement_list.append(0.0)

        z_meas = np.array(measurement_list)
        mu, P = kalman_update(mu, P, sensor, z_meas, R)
        utility = utility_function(mu, P, utility_threshold, beta=beta_value)

    final_total_variance = np.sum(np.diag(P))
    metrics = compute_task_completion(
        pos_history=pos_history,
        pts=pts,
        xs=xs,
        ys=ys,
        step=step,
        lateral_coverage=lateral_coverage,
        xmin=xmin,
        ymin=ymin,
    )

    return {
        "alpha": alpha_value,
        "beta": beta_value,
        "map_id": map_id,
        "task_completion": metrics["task_completion"],
        "gained_true_utility": metrics["gained_true_utility"],
        "total_true_utility": metrics["total_true_utility"],
        "initial_total_variance": float(initial_total_variance),
        "final_total_variance": float(final_total_variance),
        "variance_reduction": float(initial_total_variance - final_total_variance),
        "num_positions": len(pos_history),
    }


def get_sweep_values():
    alpha_values = np.linspace(alpha_min, alpha_max, num_alpha_values)
    beta_values = np.linspace(beta_min, beta_max, num_beta_values)
    return alpha_values.tolist(), beta_values.tolist()


def get_map_ids():
    return list(range(initial_map, initial_map + mapcount))


def find_best_config_per_map(all_rollout_results, map_ids):
    best_by_map = []
    for map_id in map_ids:
        map_results = [item for item in all_rollout_results if item["map_id"] == map_id]
        best_result = max(map_results, key=lambda item: item["task_completion"])
        best_by_map.append(best_result)
    best_by_map.sort(key=lambda item: item["map_id"])
    return best_by_map


def print_best_config(result):
    print(
        f"Map {result['map_id']}: alpha={result['alpha']:.3f}, beta={result['beta']:.3f} | "
        f"task completion={result['task_completion']:.4%} | "
        f"captured={result['gained_true_utility']:.4f}/{result['total_true_utility']:.4f} | "
        f"variance reduction={result['variance_reduction']:.4f}"
    )


def save_best_configs(best_by_map):
    output_path = Path(best_configs_output_path)
    lines = []
    for result in best_by_map:
        lines.append(
            f"Map {result['map_id']}: alpha={result['alpha']:.6f}, beta={result['beta']:.6f}, "
            f"task_completion={result['task_completion']:.6f}, "
            f"gained_true_utility={result['gained_true_utility']:.6f}, "
            f"total_true_utility={result['total_true_utility']:.6f}, "
            f"variance_reduction={result['variance_reduction']:.6f}"
        )
    output_path.write_text("".join(lines) + "")
    print(f"Saved best per-map alpha/beta configurations to {output_path}")


def summarize_results(alpha_value, beta_value, rollout_results):
    avg_task_completion = float(np.mean([item["task_completion"] for item in rollout_results]))
    avg_gained_true_utility = float(np.mean([item["gained_true_utility"] for item in rollout_results]))
    avg_total_true_utility = float(np.mean([item["total_true_utility"] for item in rollout_results]))
    avg_variance_reduction = float(np.mean([item["variance_reduction"] for item in rollout_results]))

    return {
        "alpha": alpha_value,
        "beta": beta_value,
        "avg_task_completion": avg_task_completion,
        "avg_gained_true_utility": avg_gained_true_utility,
        "avg_total_true_utility": avg_total_true_utility,
        "avg_variance_reduction": avg_variance_reduction,
        "num_maps": len(rollout_results),
        "per_map_results": rollout_results,
    }


def print_summary(result):
    print(
        f"alpha={result['alpha']:.3f}, beta={result['beta']:.3f} | "
        f"avg task completion={result['avg_task_completion']:.4%} over {result['num_maps']} maps | "
        f"avg captured={result['avg_gained_true_utility']:.4f}/{result['avg_total_true_utility']:.4f} | "
        f"avg variance reduction={result['avg_variance_reduction']:.4f}"
    )


def save_heatmap(summary_results, alpha_values, beta_values):
    heatmap = np.zeros((len(beta_values), len(alpha_values)), dtype=np.float32)

    for result in summary_results:
        alpha_idx = alpha_values.index(result["alpha"])
        beta_idx = beta_values.index(result["beta"])
        heatmap[beta_idx, alpha_idx] = result["avg_task_completion"]

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=[min(alpha_values), max(alpha_values), min(beta_values), max(beta_values)],
        cmap="viridis",
    )
    ax.set_title(f"Average Task Completion over {mapcount} Maps")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Average task completion")
    fig.tight_layout()
    fig.savefig(plot_output_path, dpi=200)
    plt.close(fig)
    print(f"Saved heatmap to {plot_output_path}")


if __name__ == "__main__":
    if run_alpha_beta_sweep:
        alpha_values, beta_values = get_sweep_values()
        map_ids = get_map_ids()
        all_rollout_results = []

        print(
            f"Running alpha/beta sweep across {len(map_ids)} maps with "
            f"{len(alpha_values) * len(beta_values)} parameter combinations"
        )

        for map_id in map_ids:
            print(f"Evaluating map {map_id}...")
            for alpha_value, beta_value in itertools.product(alpha_values, beta_values):
                result = run_rollout(alpha_value, beta_value, map_id)
                all_rollout_results.append(result)

        best_by_map = find_best_config_per_map(all_rollout_results, map_ids)
        print("Best alpha/beta configuration for each map:")
        for result in best_by_map:
            print_best_config(result)
        save_best_configs(best_by_map)
    else:
        result = run_rollout(alpha, beta, selected_map)
        print(
            f"Single rollout on map {selected_map} for {timealloted} timesteps | "
            f"task completion={result['task_completion']:.4%} | "
            f"captured={result['gained_true_utility']:.4f}/{result['total_true_utility']:.4f} | "
            f"variance reduction={result['variance_reduction']:.4f}"
        )
