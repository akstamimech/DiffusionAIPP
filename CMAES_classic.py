import numpy as np
import matplotlib

matplotlib.use("Agg")

from gaussianprocesstraining import (
    importance_filter,
    grid_measure,
    next_best_waypoint,
    cma_es_refine_waypoints,
    build_spline_trajectory,
    create_plots_and_gifs,
    kalman_update,
    initialize_gp,
)
from evalmetrics import compute_task_completion, compute_rmse_time_metrics, compute_reconstruction_rmse


step = 2.0
timealloted = 60
beta = 1.5
alpha = 1.0
utility_threshold = 0.5
planning_horizon = 8
initial_map = 0
mapcount = 1
samples_per_segment = 5
execution_chunk = 8
save_visualizations = True


def real_receding_horizon_planner(cx, cy, mu, P, xs, ys, utility_threshold, beta, planning_horizon, alpha=0.1):
    flight_plan = []
    filtered_utility = importance_filter(mu, P, beta, threshold=utility_threshold)
    utility_at_gridpoints = grid_measure(filtered_utility, xs, ys)
    curr_x, curr_y = cx, cy

    for _ in range(planning_horizon):
        best_waypoint = next_best_waypoint(utility_at_gridpoints, curr_x, curr_y, alpha=alpha)
        flight_plan.append(best_waypoint)
        curr_x, curr_y = best_waypoint
        utility_at_gridpoints = [
            (util, point) for util, point in utility_at_gridpoints if point != best_waypoint
        ]

    return cma_es_refine_waypoints(flight_plan, mu, P, xs, ys, cx, cy, beta, utility_threshold)


def waypoint(cx, cy, goal_x, goal_y, step):
    dx = goal_x - cx
    dy = goal_y - cy
    dist = np.hypot(dx, dy)

    if dist <= step:
        return 0.0, 0.0, True

    grad_x = dx / dist
    grad_y = dy / dist
    return grad_x, grad_y, False


def dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax, buffer=None):
    if buffer is None:
        buffer = step * 2

    cx = np.clip(cx + grad_x * samplestep, xmin + buffer, xmax - buffer)
    cy = np.clip(cy + grad_y * samplestep, ymin + buffer, ymax - buffer)
    cx = step * np.round(cx / step)
    cy = step * np.round(cy / step)
    return cx, cy


def run_map(selected_map):
    csv_path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv"
    data = np.loadtxt(rf"{csv_path}/map_{selected_map}_blob_grid_counts.csv", delimiter=",", skiprows=1)

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

    mu_history = []
    P_history = []
    step_numbers = []
    grad_history = []
    pos_history = []
    utility_history = []
    sorted_util_values_list = []
    planned_path_history = []
    control_waypoint_history = []
    rmse_metrics_list = []

    mu_history.append(mu.copy())
    P_history.append(P.copy())
    step_numbers.append(0)

    initial_utility = importance_filter(mu, P, beta, threshold=utility_threshold)
    utility_history.append(initial_utility.copy())
    planned_path_history.append([])
    control_waypoint_history.append([])

    lateral_coverage = step * 2
    samplestep = 4.0

    cx, cy = 20.0, 20.0
    grad_x, grad_y = 0.0, 0.0
    pos_history.append((cx, cy))

    initial_var_field = np.diag(P).reshape(X.shape)
    gy0, gx0 = np.gradient(initial_var_field, Y[:, 0], X[0, :])
    grad_history.append((gx0, gy0))
    sorted_util_values_list.append(grid_measure(initial_utility, xs, ys))

    initial_total_variance = np.sum(np.diag(P))
    print(f"Map {selected_map}: Initial total variance: {initial_total_variance:.4f}")

    utility = initial_utility
    control_waypoints = []
    spline_path = []
    spline_idx = 0
    executed_since_replan = 0

    for ts in range(0, timealloted):
        if ts <= 5:
            grad_x, grad_y, waypoint_reached = waypoint(cx, cy, goal_x=80.0, goal_y=80.0, step=step)
            cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
            pos_history.append((cx, cy))
            step_numbers.append(ts + 1)
            planned_path_history.append([])
            control_waypoint_history.append([])
        else:
            if ts == 6 or executed_since_replan >= execution_chunk or spline_idx >= len(spline_path):
                control_waypoints = real_receding_horizon_planner(
                    cx, cy, mu, P, xs, ys, utility_threshold, beta, planning_horizon, alpha=alpha
                )
                spline_path = build_spline_trajectory(
                    cx, cy, control_waypoints, samples_per_segment=samples_per_segment
                )
                spline_idx = 0
                executed_since_replan = 0
                print(
                    f"Map {selected_map}: Replanning with {len(control_waypoints)} control waypoints "
                    f"and {len(spline_path)} spline coordinates."
                )
                rmse_metrics = compute_reconstruction_rmse(mu, pts, xs, ys, step)
                rmse_metrics_list.append(rmse_metrics["global_rmse"])


            if spline_idx < len(spline_path):
                goal_x, goal_y = spline_path[spline_idx]
                grad_x, grad_y, waypoint_reached = waypoint(
                    cx, cy, goal_x=goal_x, goal_y=goal_y, step=step
                )

                if waypoint_reached:
                    spline_idx += 1
                    if spline_idx < len(spline_path):
                        goal_x, goal_y = spline_path[spline_idx]
                        grad_x, grad_y, _ = waypoint(
                            cx, cy, goal_x=goal_x, goal_y=goal_y, step=step
                        )
                    else:
                        grad_x, grad_y = 0.0, 0.0

                cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
                pos_history.append((cx, cy))
                step_numbers.append(ts + 1)
                executed_since_replan += 1
                planned_path_history.append(list(spline_path[spline_idx:]))
                control_waypoint_history.append(list(control_waypoints))
            else:
                pos_history.append((cx, cy))
                step_numbers.append(ts + 1)
                planned_path_history.append([])
                control_waypoint_history.append([])

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
        utility = importance_filter(mu, P, beta, threshold=utility_threshold)

        mu_history.append(mu.copy())
        P_history.append(P.copy())
        utility_history.append(utility.copy())
        sorted_util_values_list.append(grid_measure(utility, xs, ys))

        var_field = np.diag(P).reshape(X.shape)
        gy, gx = np.gradient(var_field, Y[:, 0], X[0, :])
        grad_history.append((gx, gy))

    final_variance = np.sum(np.diag(P))
    variance_delta = initial_total_variance - final_variance
    
    print(f"Map {selected_map}: Final total variance: {final_variance:.4f}")
    print(f"Map {selected_map}: Variance reduction: {variance_delta:.4f}")


    if save_visualizations:
        output_dir = (
            r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts"
            rf"\classic_map_{selected_map}_viz"
        )
        create_plots_and_gifs(
            output_dir,
            mu_history,
            P_history,
            step_numbers,
            grad_history,
            pos_history,
            utility_history,
            sorted_util_values_list,
            X,
            Y,
            xs,
            ys,
            cx,
            cy,
            lateral_coverage,
            xmin,
            xmax,
            ymin,
            ymax,
            plot_utility=True,
            plot_grad=False,
            planned_path_history=planned_path_history,
            control_waypoint_history=control_waypoint_history,
        )

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

    rmse_final = compute_rmse_time_metrics(rmse_metrics_list)


    print(
        f"Map {selected_map}: Gained Utility = {metrics['gained_true_utility']:.4f}, "
        f"Total Utility = {metrics['total_true_utility']:.4f}, "
        f"Task Completion = {metrics['task_completion']:.4%}",
        f"RMSE-AUC = {rmse_final['auc_rmse']}"
    )

    return (
        selected_map,
        metrics["gained_true_utility"],
        metrics["total_true_utility"],
        metrics["task_completion"],
        rmse_final["auc_rmse"]
    )


if __name__ == "__main__":
    metriclist = []

    for selected_map in range(initial_map, initial_map + mapcount):
        metriclist.append(run_map(selected_map))

    print("\nSummary of results:")
    for map_id, _, _, _, rmse in metriclist:
        print(
            f"Map {map_id}: RMSE-AUC = {rmse}"
        )

    avg_completion = np.mean([completion for _, _, _, completion, _ in metriclist])
    print(f"\nAverage Task Completion = {avg_completion:.4%}")
