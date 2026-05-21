import numpy as np


def compute_task_completion(pos_history, pts, xs, ys, step, lateral_coverage, xmin=None, ymin=None):
    """Compute unique true utility coverage over a rollout.

    Args:
        pos_history: Sequence of sampled positions [(x, y), ...].
        pts: Array with columns [x, y, true_count].
        xs, ys: Grid coordinate axes used by the planner.
        step: Grid spacing.
        lateral_coverage: Half-width of the square sensor footprint.
        xmin, ymin: Optional grid origin. Defaults to xs.min(), ys.min().

    Returns:
        dict with gained_true_utility, total_true_utility, task_completion,
        true_map, and observed_mask.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    pts = np.asarray(pts)

    if xmin is None:
        xmin = float(xs.min())
    if ymin is None:
        ymin = float(ys.min())

    true_map = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for x, y, count in pts:
        xi = int(round((x - xmin) / step))
        yi = int(round((y - ymin) / step))
        if 0 <= xi < len(xs) and 0 <= yi < len(ys):
            true_map[yi, xi] = count

    observed_mask = np.zeros_like(true_map, dtype=bool)

    for cx, cy in pos_history:
        for x in np.arange(cx - lateral_coverage, cx + lateral_coverage + 1e-9, step):
            for y in np.arange(cy - lateral_coverage, cy + lateral_coverage + 1e-9, step):
                xi = int(round((x - xmin) / step))
                yi = int(round((y - ymin) / step))
                if 0 <= xi < len(xs) and 0 <= yi < len(ys):
                    observed_mask[yi, xi] = True

    gained_true_utility = float(true_map[observed_mask].sum())
    total_true_utility = float(true_map.sum())
    task_completion = gained_true_utility / total_true_utility if total_true_utility > 0 else 0.0

    return {
        "gained_true_utility": gained_true_utility,
        "total_true_utility": total_true_utility,
        "task_completion": task_completion,
        "true_map": true_map,
        "observed_mask": observed_mask,
    }


def compute_reconstruction_rmse(mu, pts, xs, ys, step, xmin=None, ymin=None):
    """Compare a posterior mean map against true grid counts.

    Args:
        mu: Posterior mean as either a flat vector or a grid-shaped array.
        pts: Array with columns [x, y, true_count].
        xs, ys: Grid coordinate axes used by the planner.
        step: Grid spacing.
        xmin, ymin: Optional grid origin. Defaults to xs.min(), ys.min().

    Returns:
        dict with global_rmse, occupied_rmse, weighted_rmse, true_map,
        mean_map, and occupied_mask.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    pts = np.asarray(pts)

    if xmin is None:
        xmin = float(xs.min())
    if ymin is None:
        ymin = float(ys.min())

    true_map = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for x, y, count in pts:
        xi = int(round((x - xmin) / step))
        yi = int(round((y - ymin) / step))
        if 0 <= xi < len(xs) and 0 <= yi < len(ys):
            true_map[yi, xi] = count

    mean_map = np.asarray(mu, dtype=np.float32)
    if mean_map.ndim == 1:
        mean_map = mean_map.reshape(len(ys), len(xs))
    elif mean_map.shape != true_map.shape:
        raise ValueError(
            f"mu must be flat or have shape {true_map.shape}, got {mean_map.shape}"
        )

    error = mean_map - true_map
    global_rmse = float(np.sqrt(np.mean(error ** 2)))

    occupied_mask = true_map > 0
    if np.any(occupied_mask):
        occupied_rmse = float(np.sqrt(np.mean(error[occupied_mask] ** 2)))
        weighted_rmse = float(
            np.sqrt(np.sum(true_map * error ** 2) / np.sum(true_map))
        )
    else:
        occupied_rmse = 0.0
        weighted_rmse = 0.0

    return {
        "global_rmse": global_rmse,
        "occupied_rmse": occupied_rmse,
        "weighted_rmse": weighted_rmse,
        "true_map": true_map,
        "mean_map": mean_map,
        "occupied_mask": occupied_mask,
    }


def compute_rmse_time_metrics(rmse_values, dt=1.0):
    """Summarize an RMSE-over-time curve.

    Lower mean_rmse and auc_rmse indicate faster reconstruction error reduction.
    """
    rmse_values = np.asarray(rmse_values, dtype=float)
    if rmse_values.size == 0:
        return {
            "final_rmse": 0.0,
            "mean_rmse": 0.0,
            "auc_rmse": 0.0,
        }

    return {
        "final_rmse": float(rmse_values[-1]),
        "mean_rmse": float(np.mean(rmse_values)),
        "auc_rmse": float(np.trapezoid(rmse_values, dx=dt)),
    }


def compute_coverage_efficiency(task_completion, ts):
    """Compute coverage efficiency as task completion divided by time steps."""
    if ts > 0:
        return task_completion / ts
    else:
        return 0.0
    
def compute_spatial_efficiency(pos_history, task_completion, ):
    """Compute spatial efficiency as task completion divided by unique positions."""
    unique_positions = set((round(x, 2), round(y, 2)) for x, y in pos_history)
    num_unique_positions = len(unique_positions)
    if num_unique_positions > 0:
        return task_completion / num_unique_positions
    else:
        return 0.0
