import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import matplotlib.pyplot as plt
import os
from pathlib import Path
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
from gaussianprocesstraining import utility_function, sampler, create_plots_and_gifs, kalman_update, initialize_gp, grid_search
from gaussianprocesstraining import importance_filter, grid_measure, next_best_waypoint, cma_es_refine_waypoints, build_spline_trajectory
from evalmetrics import compute_task_completion, compute_reconstruction_rmse, compute_rmse_time_metrics
import torch


step = 2.0
timealloted = 100
beta = 0.8
alpha = 0.02
utility_threshold = 0.5
planning_horizon = 8
# action_horizon = 3  # this is more like replanning horizon
selected_map = int(os.environ.get("SELECTED_MAP", 1))
samples_per_segment = 5
execution_chunk = 20
SENSORNOISE_SEED = 123
rng = np.random.default_rng(SENSORNOISE_SEED + selected_map)
MAPTYPE = os.environ.get("MAPTYPE", "multiblob") #choose between "multiblob" and "halffield" or "blob" or "(nothing)"


#TOTAL PLANNED = PLANNING HORIZON * SPLINE SAMPLES PER SEGMENT
#TOTAL EXECUTED = EXECUTION CHUNK

"""
Single-map classic grid-search control script.
Set `selected_map` above to run one map and generate visualizations afterward.
"""


"""
Planning_horizon = 8
filtered variances = importance_filter(mu, P, beta, threshold=utility_threshold)
variances_at_gridpoints = grid_measure(filtered_variances, X, Y, xs, ys)
waypoint_plan =[]
curr_x, curr_y = cx, cy
for i in range(1, planning_horizon+1):
    best_waypoint = next_best_waypoint(variances_at_gridpoints, curr_x, curr_y)
    waypoint_plan.append(best_waypoint)
    curr_x, curr_y = best_waypoint

optim_waypoint_plan = CMA_ES(waypoint_plan, utility_function) 

"""


def real_receding_horizon_planner(
    cx,
    cy,
    mu,
    P,
    xs,
    ys,
    utility_threshold,
    beta,
    planning_horizon,
    alpha=0.1,
    R=None,
    lateral_coverage=None,
):

    flight_plan = []
    filtered_utility = importance_filter(mu, P, beta, threshold=utility_threshold)
    utility_at_gridpoints = grid_measure(filtered_utility, xs, ys)
    curr_x, curr_y = cx, cy

    for _ in range(planning_horizon):
        best_waypoint = next_best_waypoint(utility_at_gridpoints, curr_x, curr_y, alpha = alpha)
        flight_plan.append(best_waypoint)
        curr_x, curr_y = best_waypoint
        utility_at_gridpoints = [
            (util, point) for util, point in utility_at_gridpoints if point != best_waypoint
        ]
    
    control_waypoints = cma_es_refine_waypoints(
        flight_plan,
        mu,
        P,
        xs,
        ys,
        cx,
        cy,
        beta,
        utility_threshold,
        R=R,
        lateral_coverage=lateral_coverage,
        predictive_variance=True,
    )

    



    return control_waypoints




# alpha is weight for how costly distance is
# def receding_horizon_planner(cx, cy, sorted_util_values, alpha, horizon=planning_horizon):
#     flight_plan = []
#     for _ in range(horizon + 1):
#         updated_utils = []
#         for util, (x, y) in sorted_util_values:
#             dist = np.hypot(x - cx, y - cy)

#             if dist == 0:
#                 dist = 1e-6

#             new_util = util * np.exp(-alpha * dist)
#             updated_utils.append((new_util, (x, y)))

#         sorted_util_values = sorted(updated_utils, key=lambda x: x[0], reverse=True)

#         best_coord = sorted_util_values[0][1]
#         best_coord = (
#             step * np.round(best_coord[0] / step),
#             step * np.round(best_coord[1] / step),
#         )

#         flight_plan.append(best_coord)

#         cx, cy = best_coord
#         sorted_util_values.pop(0)

#     return flight_plan


"""
Assume a very simple waypoint based receding horizon. We aren't even considering dynamics yet.

This is wrong! This is just a greedy planner. Receding horizon needs to consider total gain over n steps.
"""
# def true_receding_horizon(cx, cy, sorted_util_values, alpha=alpha, horizon=planning_horizon):
#     ...


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
    # noise = np.random.randint(-1, 2, size=2)
    # cx = np.clip(cx + grad_x * samplestep + noise[0], xmin + buffer, xmax - buffer)
    # cy = np.clip(cy + grad_y * samplestep + noise[1], ymin + buffer, ymax - buffer)
    cx = np.clip(cx + grad_x * samplestep, xmin + buffer, xmax - buffer)
    cy = np.clip(cy + grad_y * samplestep, ymin + buffer, ymax - buffer)
    cx = step * np.round(cx / step)
    cy = step * np.round(cy / step)
    return cx, cy


if __name__ == "__main__":
    csv_path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv"
    output_dir = Path(__file__).resolve().parent / f"classic_map_{selected_map}_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.loadtxt(rf"{csv_path}/map_{selected_map}_{MAPTYPE}_grid_counts.csv", delimiter=",", skiprows=1)

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
    R = 500  #NOISE VARIABLES

    mu_history = []
    P_history = []
    step_numbers = []
    grad_history = []
    pos_history = []
    utility_history = []
    sorted_util_values_list = []
    planned_path_history = []
    control_waypoint_history = []

    mu_history.append(mu.copy())
    P_history.append(P.copy())
    step_numbers.append(0)

    initial_utility = importance_filter(mu, P, beta, threshold=utility_threshold)
    utility_history.append(initial_utility.copy())
    planned_path_history.append([])
    control_waypoint_history.append([])

    save_every = 5
    lateral_coverage = step * 2
    samplestep = step

    cx, cy = 20.0, 20.0
    grad_x, grad_y = 0.0, 0.0
    pos_history.append((cx, cy))

    initial_var_field = np.diag(P).reshape(X.shape)
    gy0, gx0 = np.gradient(initial_var_field, Y[:, 0], X[0, :])
    grad_history.append((gx0, gy0))
    sorted_util_values_list.append(grid_measure(initial_utility, xs, ys))

    initial_total_variance = np.sum(np.diag(P))
    print(f"Initial total variance: {initial_total_variance:.4f}")

    utility = initial_utility
    control_waypoints = []
    spline_path = []
    spline_idx = 0
    executed_since_replan = 0
    
    
    rmselist = []
    weighted_rmselist = []
    global_rmselist = []

    for ts in range(0, timealloted):
        if ts <= 1:
            grad_x, grad_y, waypoint_reached = waypoint(cx, cy, goal_x=80.0, goal_y=80.0, step=step)
            cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
            pos_history.append((cx, cy))
            step_numbers.append(ts + 1)
            planned_path_history.append([])
            control_waypoint_history.append([])
        else:
            if ts == 2 or executed_since_replan >= execution_chunk or spline_idx >= len(spline_path):
                control_waypoints = real_receding_horizon_planner(
                    cx,
                    cy,
                    mu,
                    P,
                    xs,
                    ys,
                    utility_threshold,
                    beta,
                    planning_horizon,
                    alpha=alpha,
                    R=R,
                    lateral_coverage=lateral_coverage,
                )
                spline_path = build_spline_trajectory(
                    cx, cy, control_waypoints, samples_per_segment=samples_per_segment
                )
                spline_idx = 0
                executed_since_replan = 0
                print("Replanning...")
                print("Control waypoints:", control_waypoints)

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

        z_meas = np.array(measurement_list) + rng.normal(0, np.sqrt(R), size=len(fov))

        mu, P = kalman_update(mu, P, sensor, z_meas, R)
        utility = importance_filter(mu, P, beta, threshold=utility_threshold)

        mu_history.append(mu.copy())
        P_history.append(P.copy())
        utility_history.append(utility.copy())
        sorted_util_values_list.append(grid_measure(utility, xs, ys))

        var_field = np.diag(P).reshape(X.shape)
        gy, gx = np.gradient(var_field, Y[:, 0], X[0, :])
        grad_history.append((gx, gy))

        util = utility.reshape(len(ys), len(xs))
        # print(util, "shape:", util.shape)


        reconstruction_metrics = compute_reconstruction_rmse(
        mu=mu,
        pts=pts,
        xs=xs,
        ys=ys,
        step=step,
        xmin=xmin,
        ymin=ymin,
        )

        rmselist.append(reconstruction_metrics['occupied_rmse'])
        weighted_rmselist.append(reconstruction_metrics['weighted_rmse'])
        global_rmselist.append(reconstruction_metrics['global_rmse'])

    rmse_trace = np.column_stack([global_rmselist, rmselist, weighted_rmselist])
    np.savetxt(
        output_dir / f"map_{selected_map}_rmse_over_time.csv",
        rmse_trace,
        delimiter=",",
        header="global_rmse,occupied_rmse,weighted_rmse",
        comments="",
    )

    plt.figure()
    plt.plot(global_rmselist, label="Global RMSE")
    plt.plot(rmselist, label="Occupied RMSE")
    plt.plot(weighted_rmselist, label="Weighted RMSE")
    plt.xlabel("Timestep")
    plt.ylabel("RMSE")
    plt.title(f"Map {selected_map} - RMSE over Time")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_rmse_over_time.png")
    plt.close()

    

    final_variance = np.sum(np.diag(P))
    print(f"Final total variance: {final_variance:.4f}")
    variance_delta = initial_total_variance - final_variance
    print(f"Variance reduction: {variance_delta:.4f}")

    if os.environ.get("SKIP_VIZ", "0") != "1":
        create_plots_and_gifs(
            str(output_dir),
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
    print(
        f"Map {selected_map}: Gained Utility = {metrics['gained_true_utility']:.4f}, "
        f"Total Utility = {metrics['total_true_utility']:.4f}, "
        f"Task Completion = {metrics['task_completion']:.4%}"
    )

    
    print(
        f"Map {selected_map}: Global RMSE = {reconstruction_metrics['global_rmse']:.4f}, "
        f"Occupied RMSE = {reconstruction_metrics['occupied_rmse']:.4f}, "
        f"Weighted RMSE = {reconstruction_metrics['weighted_rmse']:.4f}"
    )

    global_rmse_time = compute_rmse_time_metrics(global_rmselist)
    occupied_rmse_time = compute_rmse_time_metrics(rmselist)
    weighted_rmse_time = compute_rmse_time_metrics(weighted_rmselist)
    print(
        f"Map {selected_map}: Global RMSE AUC = {global_rmse_time['auc_rmse']:.4f}, "
        f"Mean = {global_rmse_time['mean_rmse']:.4f}"
    )
    print(
        f"Map {selected_map}: Occupied RMSE AUC = {occupied_rmse_time['auc_rmse']:.4f}, "
        f"Mean = {occupied_rmse_time['mean_rmse']:.4f}"
    )
    print(
        f"Map {selected_map}: Weighted RMSE AUC = {weighted_rmse_time['auc_rmse']:.4f}, "
        f"Mean = {weighted_rmse_time['mean_rmse']:.4f}"
    )
