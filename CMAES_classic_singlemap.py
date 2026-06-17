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
from gaussianprocesstraining import noise_model, utility_function, sampler, create_plots_and_gifs, kalman_update, initialize_gp, grid_search
from gaussianprocesstraining import importance_filter, grid_measure, next_best_waypoint, cma_es_refine_waypoints_3d, build_spline_trajectory_3d, fov_lateral_radius, resolution_block_size, grid_search_3d
from gaussianprocesstraining import (
    build_correlated_noise_covariance,
    build_sensor_matrix,
    noise_model,
    sample_correlated_sensor_noise,
)
from evalmetrics import compute_task_completion, compute_reconstruction_rmse, compute_rmse_time_metrics
import torch
import time


step = 2.0
timealloted = 150
beta = 1.0
alpha = 0.02
utility_threshold = 0.3
planning_horizon = 8
# action_horizon = 3  # this is more like replanning horizon
selected_map = int(os.environ.get("SELECTED_MAP", 17))
samples_per_segment = 5
execution_chunk = 10
SENSORNOISE_SEED = 123
rng = np.random.default_rng(SENSORNOISE_SEED + selected_map)
MAPTYPE = os.environ.get("MAPTYPE", "NAIP") #choose between "multiblob" and "halffield" or "blob" or "(nothing)" or "NAIP"




##MAKING ALTITUDE A THING
INIT_ALTITUDE = 10.0  # Example fixed altitude for all waypoints
ZMIN = 10.0
ZMAX = 40.0


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
    cz,
    mu,
    P,
    xs,
    ys,
    utility_threshold,
    beta,
    planning_horizon,
    alpha=0.1,
):
    flight_plan_3d = grid_search_3d(
        mu,
        P,
        xs,
        ys,
        start_pose=(cx, cy, cz),
        beta=beta,
        utility_threshold=utility_threshold,
        planning_horizon=planning_horizon,
        zmin=ZMIN,
        zmax=ZMAX,
        alpha=alpha,
    )
    
    control_waypoints = cma_es_refine_waypoints_3d(
        flight_plan_3d,
        mu,
        P,
        xs,
        ys,
        cx,
        cy,
        cz,
        beta,
        utility_threshold,
        ZMIN,
        ZMAX,
        predictive_variance=True,
    )

    return control_waypoints



"""
Assume a very simple waypoint based planner. We aren't even considering dynamics yet.

This is wrong! This is just a greedy planner. Receding horizon needs to consider total gain over n steps.
"""
# def true_receding_horizon(cx, cy, sorted_util_values, alpha=alpha, horizon=planning_horizon):
#     ...



'''
the waypoint() function converts the current position and a goal position into a unit vector direction for the dynamics() function
'''

def waypoint(cx, cy, goal_x, goal_y, step):

    dx = goal_x - cx
    dy = goal_y - cy
    dist = np.hypot(dx, dy)

    if dist <= step:
        return 0.0, 0.0, True

    grad_x = dx / dist
    grad_y = dy / dist

    return grad_x, grad_y, False


def waypoint_3d(cx, cy, cz, goal_x, goal_y, goal_z, step):
    displacement = np.array([
        goal_x - cx,
        goal_y - cy,
        goal_z - cz,
    ])

    distance = np.linalg.norm(displacement)

    if distance <= step:
        return 0.0, 0.0, 0.0, True

    direction = displacement / distance
    return direction[0], direction[1], direction[2], False

def dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax, buffer=step * 2):


    # noise = np.random.randint(-1, 2, size=2)
    # cx = np.clip(cx + grad_x * samplestep + noise[0], xmin + buffer, xmax - buffer)
    # cy = np.clip(cy + grad_y * samplestep + noise[1], ymin + buffer, ymax - buffer)
    cx = np.clip(cx + grad_x * samplestep, xmin + buffer, xmax - buffer)
    cy = np.clip(cy + grad_y * samplestep, ymin + buffer, ymax - buffer)
    cx = step * np.round(cx / step)
    cy = step * np.round(cy / step)
    return cx, cy

def dynamics_3d(
    cx, cy, cz,
    grad_x, grad_y, grad_z,
    samplestep,
    xmin, xmax, ymin, ymax,
    zmin, zmax,
    buffer,
):
    cx = np.clip(cx + grad_x * samplestep, xmin + buffer, xmax - buffer)
    cy = np.clip(cy + grad_y * samplestep, ymin + buffer, ymax - buffer)
    cz = np.clip(cz + grad_z * samplestep, zmin, zmax)

    cx = step * np.round(cx / step)
    cy = step * np.round(cy / step)
    cz = step * np.round(cz / step)

    return cx, cy, cz


def compute_fov(cz, xs, ys, angle_of_view = 60, step=step, cx=None, cy=None): 
    
    radius = fov_lateral_radius(cz, angle_of_view)

    visible_xs = xs[
        (xs >= cx - radius) &
        (xs <= cx + radius)
    ]
    visible_ys = ys[
        (ys >= cy - radius) &
        (ys <= cy + radius)
    ]


    return [
        (float(x), float(y))
        for x in visible_xs
        for y in visible_ys
        ]





if __name__ == "__main__":
    csv_path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv"
    output_dir = Path(__file__).resolve().parent / "Vizualization" / f"classic_map_{selected_map}_viz"
    # output_dir = Path(__file__).resolve().parent / "Vizualization" / f"classic_map_NAIP_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.loadtxt(rf"{csv_path}/map_{selected_map}_{MAPTYPE}_grid_counts.csv", delimiter=",", skiprows=1)
    # data = np.loadtxt(rf"{csv_path}/safecast_poly_normalized_multiblob_grid_counts.csv", delimiter=",", skiprows=1)
    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()

    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]

    N = X_test.shape[0]
    true_map_flat = np.zeros(N, dtype=float)
    for x_true, y_true, value in pts:
        idx = np.where(
            np.isclose(X_test[:, 0], x_true)
            & np.isclose(X_test[:, 1], y_true)
        )[0]
        if idx.size > 0:
            true_map_flat[idx[0]] = value

    # mu = mean.copy()
    mean = np.full(X_test.shape[0], utility_threshold - 0.1) ##ATTEMPT
    mu = mean.copy()
    P = cov.copy()
    R = noise_model(INIT_ALTITUDE)   # measurement noise variance based on altitude

    mu_history = []
    P_history = []
    step_numbers = []
    grad_history = []
    pos_history = []
    utility_history = []
    sorted_util_values_list = []
    planned_path_history = []
    control_waypoint_history = []
    pose_history = [] 

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

    cx, cy, cz = 4.0, 4.0, INIT_ALTITUDE
    grad_x, grad_y, grad_z = 0.0, 0.0, 0.0
    pos_history.append((cx, cy))
    pose_history.append((cx, cy, cz))
    fov = compute_fov(cz = cz, xs = xs, ys = ys, angle_of_view=60, step=step, cx=cx, cy=cy)

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
    global_rmselist = []

    for ts in range(0, timealloted):
        if ts <= 1:
            grad_x, grad_y, waypoint_reached = waypoint(cx, cy, goal_x=80.0, goal_y=80.0, step=step)
            cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
            pos_history.append((cx, cy))
            pose_history.append((cx, cy, cz))
            step_numbers.append(ts + 1)
            planned_path_history.append([])
            control_waypoint_history.append([])
        else:
            if ts == 2 or executed_since_replan >= execution_chunk or spline_idx >= len(spline_path):
                start_time = time.time()
                control_waypoints = real_receding_horizon_planner(
                    cx,
                    cy,
                    cz,
                    mu,
                    P,
                    xs,
                    ys,
                    utility_threshold,
                    beta,
                    planning_horizon,
                    alpha=alpha,
                )
                end_time = time.time()
                spline_path = build_spline_trajectory_3d(
                    cx, cy, cz, control_waypoints, samples_per_segment=samples_per_segment
                )
                spline_idx = 0
                executed_since_replan = 0

                print("Replanning...")
                print(f"Replanning time: {end_time - start_time:.4f} seconds")
                print("Control waypoints:", control_waypoints)

            if spline_idx < len(spline_path):
                goal_x, goal_y, goal_z = spline_path[spline_idx]
                grad_x, grad_y, grad_z, waypoint_reached = waypoint_3d(
                    cx, cy, cz, goal_x=goal_x, goal_y=goal_y, goal_z=goal_z, step=step
                )

                if waypoint_reached:
                    spline_idx += 1
                    if spline_idx < len(spline_path):
                        goal_x, goal_y, goal_z = spline_path[spline_idx]
                        grad_x, grad_y, grad_z, _ = waypoint_3d(
                            cx, cy, cz, goal_x=goal_x, goal_y=goal_y, goal_z=goal_z, step=step
                        )
                    else:
                        grad_x, grad_y, grad_z = 0.0, 0.0, 0.0

                cx, cy, cz = dynamics_3d(cx, cy, cz, grad_x, grad_y, grad_z, samplestep, xmin, xmax, ymin, ymax, ZMIN, ZMAX, buffer=step/2)
                pos_history.append((cx, cy))
                pose_history.append((cx, cy, cz))
                step_numbers.append(ts + 1)
                executed_since_replan += 1
                planned_path_history.append([(x, y) for x, y, _ in spline_path[spline_idx:]])
                control_waypoint_history.append([(x, y) for x, y, _ in control_waypoints])
            else:
                pos_history.append((cx, cy))
                pose_history.append((cx, cy, cz))
                step_numbers.append(ts + 1)
                planned_path_history.append([])
                control_waypoint_history.append([])

        fov = compute_fov(cz = cz, xs = xs, ys = ys, angle_of_view=60, step=step, cx=cx, cy=cy)

        sensor, sensor_block_ids = build_sensor_matrix(
            fov, cz, xs, ys, return_block_ids=True
        )

        R = noise_model(cz)  # Update measurement noise based on current altitude
        z_meas = sensor @ true_map_flat
        z_meas += sample_correlated_sensor_noise(sensor_block_ids, R, rng)
        R_cov = build_correlated_noise_covariance(sensor_block_ids, R)
        mu, P = kalman_update(
            mu, P, sensor, z_meas, R_cov, block_ids=sensor_block_ids
        )
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
        utility_threshold=utility_threshold,
        xmin=xmin,
        ymin=ymin,
        )

        rmselist.append(reconstruction_metrics['occupied_rmse'])
        global_rmselist.append(reconstruction_metrics['global_rmse'])

    rmse_trace = np.column_stack([global_rmselist, rmselist])
    np.savetxt(
        output_dir / f"map_{selected_map}_rmse_over_time.csv",
        rmse_trace,
        delimiter=",",
        header="global_rmse,occupied_rmse",
        comments="",
    )

    plt.figure()
    plt.plot(global_rmselist, label="Global RMSE")
    plt.plot(rmselist, label="Occupied RMSE")
    plt.xlabel("Timestep")
    plt.ylabel("RMSE")
    plt.title(f"Map {selected_map} - RMSE over Time")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_rmse_over_time.png")
    plt.close()

    

    final_variance = np.sum(np.diag(P))
    # print(f"Final total variance: {final_variance:.4f}")
    variance_delta = initial_total_variance - final_variance
    # print(f"Variance reduction: {variance_delta:.4f}")

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
            pose_history=pose_history,
            angle_of_view=60.0,
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
        f"Occupied RMSE = {reconstruction_metrics['occupied_rmse']:.4f}"
    )

    global_rmse_time = compute_rmse_time_metrics(global_rmselist)
    occupied_rmse_time = compute_rmse_time_metrics(rmselist)
    print(
        f"Map {selected_map}: Global RMSE AUC = {global_rmse_time['auc_rmse']:.4f}, "
        f"Mean = {global_rmse_time['mean_rmse']:.4f}"
    )
    print(
        f"Map {selected_map}: Occupied RMSE AUC = {occupied_rmse_time['auc_rmse']:.4f}, "
        f"Mean = {occupied_rmse_time['mean_rmse']:.4f}"
    )
